"""Hybrid retrieval (BM25 multilingual + BGE-M3 kNN, fused with RRF) + optional rerank + cited RAG answers.
Run: uvicorn api:app --app-dir pipeline --port 8000
Env: OLLAMA_URL (default http://localhost:11434), LLM_MODEL (default qwen2.5:7b-instruct)
"""
import os, requests
from functools import lru_cache
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer, CrossEncoder

INDEX = "dz_records"
OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
RRF_K = 60

app = FastAPI(title="DZ DSpace discovery")
es = OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

@lru_cache
def embedder(): return SentenceTransformer("BAAI/bge-m3")
@lru_cache
def reranker(): return CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

def _filters(year_from, year_to, repo):
    f = []
    if year_from or year_to:
        f.append({"range": {"year": {k: v for k, v in (("gte", year_from), ("lte", year_to)) if v}}})
    if repo: f.append({"term": {"repo": repo}})
    return f

def bm25(q, size, filt):
    fields = ["title^3", "title.ar^3", "title.fr^3", "title.en^3",
              "abstract", "abstract.ar", "abstract.fr", "abstract.en", "subjects^2", "creators^2"]
    body = {"size": size, "_source": {"excludes": ["embedding"]},
            "query": {"bool": {"must": {"multi_match": {"query": q, "fields": fields, "type": "most_fields"}},
                               "filter": filt}}}
    return es.search(index=INDEX, body=body)["hits"]["hits"]

def knn(q, size, filt):
    vec = embedder().encode(q, normalize_embeddings=True).tolist()
    body = {"size": size, "_source": {"excludes": ["embedding"]},
            "query": {"knn": {"embedding": {"vector": vec, "k": size, "filter": {"bool": {"filter": filt}}}}}}
    return es.search(index=INDEX, body=body)["hits"]["hits"]

def rrf(*rankings):
    score, docs = {}, {}
    for ranking in rankings:
        for r, h in enumerate(ranking, 1):
            score[h["_id"]] = score.get(h["_id"], 0) + 1 / (RRF_K + r)
            docs[h["_id"]] = h
    return [dict(docs[i]["_source"], id=i, score=s) for i, s in sorted(score.items(), key=lambda x: -x[1])]

def retrieve(q, k=8, pool=50, rerank=True, year_from=None, year_to=None, repo=None, mode="hybrid"):
    filt = _filters(year_from, year_to, repo)
    if mode == "bm25": results = rrf(bm25(q, pool, filt))
    elif mode == "dense": results = rrf(knn(q, pool, filt))
    else: results = rrf(bm25(q, pool, filt), knn(q, pool, filt))
    results = results[:pool]
    if rerank and results:
        scores = reranker().predict([(q, f"{r['title']}\n{r.get('abstract','')}") for r in results])
        for r, s in zip(results, scores): r["score"] = float(s)
        results.sort(key=lambda r: -r["score"])
    return results[:k]

@app.get("/search")
def search(q: str, k: int = 10, rerank: bool = True, mode: str = Query("hybrid", pattern="^(hybrid|bm25|dense)$"),
           year_from: int | None = None, year_to: int | None = None, repo: str | None = None):
    return retrieve(q, k, rerank=rerank, mode=mode, year_from=year_from, year_to=year_to, repo=repo)

class Ask(BaseModel):
    question: str
    k: int = 6
    year_from: int | None = None
    year_to: int | None = None
    repo: str | None = None

PROMPT = """You are a scholarly discovery assistant for Algerian university repositories.
Answer the question ONLY from the numbered sources below. Cite sources inline like [1], [2].
Answer in the same language as the question. If the sources do not contain enough evidence,
say so explicitly and do not invent facts. Be concise.

Sources:
{sources}

Question: {question}
Answer:"""

@app.post("/ask")
def ask(a: Ask):
    hits = retrieve(a.question, a.k, year_from=a.year_from, year_to=a.year_to, repo=a.repo)
    contexts = [f"{h['title']} ({h.get('year')}; {', '.join(h['creators'][:2])})\n{(h.get('abstract') or '')[:1200]}"
                for h in hits]
    sources = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
    r = requests.post(f"{OLLAMA}/api/chat", timeout=300, json={
        "model": LLM, "stream": False, "options": {"temperature": 0.1},
        "messages": [{"role": "user", "content": PROMPT.format(sources=sources, question=a.question)}]})
    r.raise_for_status()
    return {"answer": r.json()["message"]["content"], "contexts": contexts,
            "sources": [{"n": i, "title": h["title"], "url": h.get("url"), "year": h.get("year"),
                         "repo": h["repo"]} for i, h in enumerate(hits, 1)]}


@app.get("/facets")
def facets():
    """Repositories and year range, to populate the UI filters."""
    r = es.search(index=INDEX, body={"size": 0, "aggs": {
        "repos": {"terms": {"field": "repo", "size": 50}},
        "years": {"stats": {"field": "year"}}}})["aggregations"]
    return {"repos": [{"id": b["key"], "count": b["doc_count"]} for b in r["repos"]["buckets"]],
            "year_min": r["years"].get("min"), "year_max": r["years"].get("max")}

UI_DIR = Path(__file__).resolve().parent.parent / "ui"

@app.get("/", include_in_schema=False)
def home(): return FileResponse(UI_DIR / "index.html")

app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
