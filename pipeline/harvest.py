"""Incremental OAI-PMH harvest -> data/<repo>/records.jsonl (Dublin Core)."""
import sys, json, time, yaml, pathlib
from sickle import Sickle
from sickle.oaiexceptions import NoRecordsMatch
from survey import find_oai, UA

FIELDS = ["title","creator","subject","description","publisher","contributor",
          "date","type","format","identifier","language","relation","rights"]

def harvest(repo, out_dir="data", delay=1.0):
    url, _ = find_oai(repo["base_url"])
    if not url: print("no OAI for", repo["id"]); return
    out = pathlib.Path(out_dir) / repo["id"]; out.mkdir(parents=True, exist_ok=True)
    stamp = out / "last_harvest.txt"
    kw = {"metadataPrefix": "oai_dc", "ignore_deleted": True}
    if stamp.exists(): kw["from"] = stamp.read_text().strip()
    s = Sickle(url, headers=UA, timeout=60)
    n = 0
    # first (full) harvest starts a fresh file, so an interrupted run never leaves duplicates
    mode = "a" if stamp.exists() else "w"
    try:
        with open(out / "records.jsonl", mode, encoding="utf-8") as f:
            for rec in s.ListRecords(**kw):
                d = {k: [v for v in rec.metadata.get(k, []) if v] for k in FIELDS}
                d["oai_id"] = rec.header.identifier; d["datestamp"] = rec.header.datestamp
                d["repo"] = repo["id"]
                f.write(json.dumps(d, ensure_ascii=False) + "\n"); n += 1
                if n % 100 == 0:          # one OAI page = 100 records: pause per page, not per record
                    time.sleep(delay)
                if n % 500 == 0:
                    print(f"  {repo['id']}: {n} records...", flush=True)
    except NoRecordsMatch:
        # incremental harvest with nothing new (or an empty repository): not an error
        print(repo["id"], "no new records" + (f" since {kw['from']}" if "from" in kw else ""))
        return
    stamp.write_text(time.strftime("%Y-%m-%d"))
    print(repo["id"], "harvested", n)

if __name__ == "__main__":
    for r in yaml.safe_load(open(sys.argv[1] if len(sys.argv) > 1 else "repos.yaml"))["repos"]:
        try:
            harvest(r)
        except KeyboardInterrupt:
            print("interrupted: re-run to restart this repository from scratch"); sys.exit(1)
        except Exception as e:  # one failing repository must not stop the others
            print(r["id"], "FAILED:", type(e).__name__, e, file=sys.stderr)
