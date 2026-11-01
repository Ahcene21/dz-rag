"""Generation evaluation for /ask: RAGAS faithfulness + answer relevancy (local open-source judge)
plus deterministic citation checks and a human-review sheet.

Input : eval/queries.csv  (qid,lang,text)  -- same file as evaluate.py
Output: eval/results/rag_answers.jsonl, rag_metrics.csv, human_review.csv

Judge LLM and embeddings are local/open-source: an Ollama model through its OpenAI-compatible
endpoint, and BGE-M3 via sentence-transformers. Use a judge at least as strong as the generator
(e.g. qwen2.5:14b-instruct judging qwen2.5:7b-instruct); report the judge model in the article.

Usage:
  Dependencies (tested together): pip install "ragas==0.4.3" "langchain-openai<1" "langchain-core<1" \
      "langchain-community==0.3.27" sentence-transformers requests
  (newer langchain-community breaks ragas 0.4.3's import of ChatVertexAI)
  python eval/rag_eval.py generate --api http://localhost:8000 --k 6
  python eval/rag_eval.py score --judge qwen2.5:14b-instruct
  python eval/rag_eval.py human --n 60 --seed 7
"""
import argparse, csv, json, pathlib, random, re
import requests

EVAL = pathlib.Path(__file__).parent
OUT = EVAL / "results"
ANS = OUT / "rag_answers.jsonl"
CITE = re.compile(r"\[(\d+)\]")
# phrases signalling a (desired) abstention; extend per language after reading real outputs
ABSTAIN = ["insufficient evidence", "not enough", "do not contain", "ne contiennent pas",
           "pas assez", "غير كافية", "لا تحتوي", "لا تكفي"]

def read_queries():
    with open(EVAL / "queries.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def cmd_generate(a):
    """Call /ask, then re-fetch the exact contexts the answer was built from."""
    OUT.mkdir(exist_ok=True)
    with open(ANS, "w", encoding="utf-8") as f:
        for q in read_queries():
            r = requests.post(f"{a.api}/ask", json={"question": q["text"], "k": a.k}, timeout=600)
            r.raise_for_status()
            d = r.json()
            f.write(json.dumps({"qid": q["qid"], "lang": q["lang"], "question": q["text"],
                                "answer": d["answer"], "sources": d["sources"],
                                "contexts": d["contexts"]}, ensure_ascii=False) + "\n")
            print(q["qid"], "ok")

def citation_stats(row):
    n_src = len(row["sources"])
    cited = [int(x) for x in CITE.findall(row["answer"])]
    valid = [c for c in cited if 1 <= c <= n_src]
    sentences = [s for s in re.split(r"(?<=[.!?؟])\s+", row["answer"].strip()) if len(s) > 20]
    with_cite = sum(1 for s in sentences if CITE.search(s))
    abstained = any(p in row["answer"].lower() for p in ABSTAIN)
    return {"n_citations": len(cited),
            "citation_validity": len(valid) / len(cited) if cited else None,  # cited index exists
            "cited_sentence_ratio": with_cite / len(sentences) if sentences else None,
            "abstained": abstained}

def cmd_score(a):
    from langchain_openai import ChatOpenAI
    from ragas import SingleTurnSample, EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.metrics import Faithfulness, ResponseRelevancy

    rows = [json.loads(l) for l in open(ANS, encoding="utf-8")]
    judge = LangchainLLMWrapper(ChatOpenAI(model=a.judge, base_url=f"{a.ollama}/v1",
                                           api_key="ollama", temperature=0, timeout=600))
    emb = HuggingFaceEmbeddings(model="BAAI/bge-m3")

    scorable = [r for r in rows if r["contexts"] and not citation_stats(r)["abstained"]]
    ds = EvaluationDataset([SingleTurnSample(user_input=r["question"], response=r["answer"],
                                             retrieved_contexts=r["contexts"]) for r in scorable])
    res = evaluate(ds, metrics=[Faithfulness(llm=judge), ResponseRelevancy(llm=judge, embeddings=emb)])
    df = res.to_pandas()
    per = {r["qid"]: (float(df.iloc[i]["faithfulness"]), float(df.iloc[i]["answer_relevancy"]))
           for i, r in enumerate(scorable)}

    table = []
    for r in rows:
        c = citation_stats(r)
        f, ar = per.get(r["qid"], (None, None))
        table.append({"qid": r["qid"], "lang": r["lang"], "faithfulness": f, "answer_relevancy": ar, **c})
    with open(OUT / "rag_metrics.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None
    print(f"judge={a.judge}  answers={len(rows)}  scored={len(scorable)}")
    for g in ["all"] + sorted({t["lang"] for t in table}):
        sub = [t for t in table if g == "all" or t["lang"] == g]
        print(f"{g:>4} n={len(sub):<3} faith={mean(t['faithfulness'] for t in sub)} "
              f"relev={mean(t['answer_relevancy'] for t in sub)} "
              f"cite_valid={mean(t['citation_validity'] for t in sub)} "
              f"cited_sent={mean(t['cited_sentence_ratio'] for t in sub)} "
              f"abstain={mean(float(t['abstained']) for t in sub)}")

def cmd_human(a):
    """Stratified sample for manual annotation (needed to validate the LLM judge)."""
    rows = [json.loads(l) for l in open(ANS, encoding="utf-8")]
    random.seed(a.seed)
    by_lang = {}
    for r in rows: by_lang.setdefault(r["lang"], []).append(r)
    per = max(1, a.n // len(by_lang))
    sample = [r for rs in by_lang.values() for r in random.sample(rs, min(per, len(rs)))]
    with open(OUT / "human_review.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "lang", "question", "answer", "sources", "claims_total", "claims_supported",
                    "citation_correct(0/1)", "answer_addresses_question(0-2)", "notes"])
        for r in sample:
            src = "\n".join(f"[{s['n']}] {s['title']} {s.get('url') or ''}" for s in r["sources"])
            w.writerow([r["qid"], r["lang"], r["question"], r["answer"], src, "", "", "", "", ""])
    print(f"{len(sample)} answers -> {OUT/'human_review.csv'}")
    print("Human faithfulness = claims_supported / claims_total. Report Pearson/Spearman vs RAGAS faithfulness.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("generate"); g.add_argument("--api", default="http://localhost:8000")
    g.add_argument("--k", type=int, default=6); g.set_defaults(f=cmd_generate)
    s = sp.add_parser("score"); s.add_argument("--judge", default="qwen2.5:14b-instruct")
    s.add_argument("--ollama", default="http://localhost:11434"); s.set_defaults(f=cmd_score)
    h = sp.add_parser("human"); h.add_argument("--n", type=int, default=60)
    h.add_argument("--seed", type=int, default=7); h.set_defaults(f=cmd_human)
    a = ap.parse_args(); a.f(a)
