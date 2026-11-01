"""Retrieval evaluation with ranx for the DZ DSpace discovery API.

Files (in eval/):
  queries.csv : qid,lang,text            (lang in ar|fr|en; the query language)
  qrels.csv   : qid,doc_id,rel           (doc_id = OpenSearch _id returned as "id" by /search; rel 0-3)

Commands:
  python eval/evaluate.py pool  --api http://localhost:8000 --depth 20 --out eval/pool.csv
      -> union of top-`depth` results from every configuration, for human relevance judgment
         (fill the empty `rel` column, then save as eval/qrels.csv)
  python eval/evaluate.py run   --api http://localhost:8000 --k 100
      -> metrics table (overall + per query language), significance tests, LaTeX/CSV output
"""
import argparse, csv, json, pathlib, sys
import requests
from ranx import Qrels, Run, evaluate, compare

EVAL = pathlib.Path(__file__).parent
# name -> /search params. These are the ablation configurations for the article.
CONFIGS = {
    "BM25":          {"mode": "bm25",   "rerank": "false"},
    "Dense(BGE-M3)": {"mode": "dense",  "rerank": "false"},
    "Hybrid(RRF)":   {"mode": "hybrid", "rerank": "false"},
    "Hybrid+Rerank": {"mode": "hybrid", "rerank": "true"},
}
METRICS = ["recall@10", "recall@100", "mrr@10", "ndcg@10", "map@100"]

def read_queries():
    with open(EVAL / "queries.csv", encoding="utf-8") as f:
        return {r["qid"]: r for r in csv.DictReader(f)}

def search(api, text, k, params):
    r = requests.get(f"{api}/search", params={"q": text, "k": k, **params}, timeout=120)
    r.raise_for_status()
    return r.json()

def build_run(api, queries, k, params, name):
    d = {}
    for qid, q in queries.items():
        hits = search(api, q["text"], k, params)
        # use reciprocal rank as score so ordering is exact regardless of score scale
        d[qid] = {h["id"]: 1.0 / (rank + 1) for rank, h in enumerate(hits)}
    run = Run(d, name=name)
    return run

def cmd_pool(a):
    queries = read_queries()
    rows = []
    for qid, q in queries.items():
        seen = {}
        for name, params in CONFIGS.items():
            for h in search(a.api, q["text"], a.depth, params):
                seen.setdefault(h["id"], h)
        for doc_id, h in seen.items():
            rows.append({"qid": qid, "doc_id": doc_id, "rel": "", "query": q["text"],
                         "title": h["title"], "abstract": (h.get("abstract") or "")[:400],
                         "url": h.get("url") or ""})
    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} (query, doc) pairs to judge -> {a.out}")
    print("Scale: 0 not relevant, 1 marginal, 2 relevant, 3 highly relevant. Save judged file as eval/qrels.csv")

def load_qrels(qids):
    d = {}
    with open(EVAL / "qrels.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["qid"] in qids and r["rel"].strip() != "":
                d.setdefault(r["qid"], {})[r["doc_id"]] = int(r["rel"])
    # queries with no relevant doc (rel>0) cannot be scored; report and drop
    empty = [q for q, docs in d.items() if not any(v > 0 for v in docs.values())]
    for q in empty: del d[q]
    missing = set(qids) - set(d)
    if missing: print(f"warning: {len(missing)} queries without relevant judgments dropped: {sorted(missing)[:10]}", file=sys.stderr)
    return d

def subset(qrels_d, runs, qids):
    q = Qrels({k: v for k, v in qrels_d.items() if k in qids})
    rs = []
    for r in runs:
        nr = Run({k: v for k, v in r.to_dict().items() if k in qids}, name=r.name)
        rs.append(nr)
    return q, rs

def cmd_run(a):
    queries = read_queries()
    qrels_d = load_qrels(set(queries))
    queries = {k: v for k, v in queries.items() if k in qrels_d}
    runs = [build_run(a.api, queries, a.k, p, n) for n, p in CONFIGS.items()]
    out = EVAL / "results"; out.mkdir(exist_ok=True)

    groups = {"all": set(queries)}
    for lang in sorted({q["lang"] for q in queries.values()}):
        groups[lang] = {k for k, q in queries.items() if q["lang"] == lang}

    table = []
    for g, qids in groups.items():
        if not qids: continue
        qrels, rs = subset(qrels_d, runs, qids)
        for r in rs:
            scores = evaluate(qrels, r, METRICS)
            table.append({"group": g, "n_queries": len(qids), "system": r.name,
                          **{m: round(scores[m], 4) for m in METRICS}})
    with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)

    # significance (two-sided Student's t-test, Bonferroni-corrected) over all queries
    qrels, rs = subset(qrels_d, runs, groups["all"])
    report = compare(qrels, runs=rs, metrics=["ndcg@10", "recall@100", "mrr@10"],
                     stat_test="student", max_p=0.05, rounding_digits=3)
    (out / "significance.tex").write_text(report.to_latex(), encoding="utf-8")
    print(report)

    # language-pair view: query language x hit language is only possible if you add `doc_lang`; see README
    print("\nPer-group metrics:")
    for r in table:
        print(f"{r['group']:>4} n={r['n_queries']:<4} {r['system']:<14} " +
              " ".join(f"{m}={r[m]:.3f}" for m in METRICS))
    print(f"\nSaved: {out/'metrics.csv'}, {out/'significance.tex'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pool"); p.add_argument("--api", default="http://localhost:8000")
    p.add_argument("--depth", type=int, default=20); p.add_argument("--out", default=str(EVAL / "pool.csv"))
    p.set_defaults(f=cmd_pool)
    r = sp.add_parser("run"); r.add_argument("--api", default="http://localhost:8000")
    r.add_argument("--k", type=int, default=100); r.set_defaults(f=cmd_run)
    a = ap.parse_args(); a.f(a)
