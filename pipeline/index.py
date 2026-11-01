"""Embed harvested records (title + abstract) with BGE-M3 and index into OpenSearch.
Usage: python pipeline/index.py [--data data] [--recreate]
"""
import argparse, glob, hashlib, json, re
from opensearchpy import OpenSearch, helpers
from sentence_transformers import SentenceTransformer

INDEX, DIM = "dz_records", 1024
os_client = lambda: OpenSearch(hosts=[{"host": "localhost", "port": 9200}])

MAPPING = {
  "settings": {"index": {"knn": True}},
  "mappings": {"properties": {
    "title":    {"type": "text", "fields": {"ar": {"type": "text", "analyzer": "arabic"},
                 "fr": {"type": "text", "analyzer": "french"}, "en": {"type": "text", "analyzer": "english"}}},
    "abstract": {"type": "text", "fields": {"ar": {"type": "text", "analyzer": "arabic"},
                 "fr": {"type": "text", "analyzer": "french"}, "en": {"type": "text", "analyzer": "english"}}},
    "creators": {"type": "text"}, "subjects": {"type": "text"},
    "year": {"type": "integer"}, "repo": {"type": "keyword"}, "type": {"type": "keyword"},
    "lang": {"type": "keyword"}, "url": {"type": "keyword"},
    "embedding": {"type": "knn_vector", "dimension": DIM,
                  "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"}},
  }},
}

def clean(s): return re.sub(r"\s+", " ", s or "").strip()
def L(x): return [str(i).strip() for i in (x or []) if i and str(i).strip()]  # drop None / empty entries

def to_doc(r):
    title = clean((L(r.get("title")) or [""])[0])
    abstract = clean(" ".join(L(r.get("description"))))
    if not title: return None
    m = re.search(r"(19|20)\d{2}", " ".join(L(r.get("date"))))
    urls = [i for i in L(r.get("identifier")) if i.startswith("http")]
    return {"title": title, "abstract": abstract,
            "creators": L(r.get("creator")), "subjects": L(r.get("subject")),
            "year": int(m.group(0)) if m else None, "repo": r["repo"],
            "type": (L(r.get("type")) or [None])[0], "lang": (L(r.get("language")) or [None])[0],
            "url": urls[0] if urls else None, "_id": hashlib.sha1(r["oai_id"].encode()).hexdigest()}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data")
    ap.add_argument("--recreate", action="store_true"); ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    es = os_client()
    if a.recreate and es.indices.exists(index=INDEX): es.indices.delete(index=INDEX)
    if not es.indices.exists(index=INDEX): es.indices.create(index=INDEX, body=MAPPING)

    docs = []
    for f in glob.glob(f"{a.data}/*/records.jsonl"):
        for line in open(f, encoding="utf-8"):
            d = to_doc(json.loads(line))
            if d: docs.append(d)
    print("records to index:", len(docs))

    model = SentenceTransformer("BAAI/bge-m3")
    for i in range(0, len(docs), a.batch):
        batch = docs[i:i + a.batch]
        vecs = model.encode([f"{d['title']}\n{d['abstract']}" for d in batch],
                            normalize_embeddings=True)
        actions = []
        for d, v in zip(batch, vecs):
            src = {k: x for k, x in d.items() if k != "_id"}; src["embedding"] = v.tolist()
            actions.append({"_index": INDEX, "_id": d["_id"], "_source": src})
        helpers.bulk(es, actions)
        print(f"indexed {min(i + a.batch, len(docs))}/{len(docs)}")
    es.indices.refresh(index=INDEX)

if __name__ == "__main__":
    main()
