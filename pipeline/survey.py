"""Check each repo: OAI-PMH Identify, set list, record count estimate, DSpace REST flavour."""
import sys, yaml, requests
from sickle import Sickle

OAI_PATHS = ["/oai/request", "/server/oai/request"]  # DSpace 6 / 7+
UA = {"User-Agent": "dz-rag-research/0.1 (contact: ahcene.babouri@gmail.com)"}

def find_oai(base):
    for p in OAI_PATHS:
        url = base.rstrip("/") + p
        try:
            s = Sickle(url, headers=UA, timeout=30)
            ident = s.Identify()
            return url, ident
        except Exception:
            continue
    return None, None

def main(cfg):
    for r in yaml.safe_load(open(cfg))["repos"]:
        url, ident = find_oai(r["base_url"])
        print(f"== {r['id']} {r['base_url']}")
        if not url:
            print("   OAI-PMH: NOT FOUND"); continue
        print(f"   OAI: {url}\n   name: {ident.repositoryName}  version: {ident.protocolVersion}")
        s = Sickle(url, headers=UA, timeout=60)
        try:
            n = sum(1 for _ in s.ListIdentifiers(metadataPrefix="oai_dc"))
            print(f"   records: {n}")
        except Exception as e:
            print("   ListIdentifiers failed:", e)
        for api in ("/server/api", "/rest"):
            try:
                c = requests.get(r["base_url"].rstrip("/") + api, headers=UA, timeout=20).status_code
                print(f"   REST {api}: HTTP {c}")
            except Exception as e:
                print(f"   REST {api}: error")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "repos.yaml")
