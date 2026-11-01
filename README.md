# dz-rag: RAG discovery layer over 5 Algerian DSpace repositories (open-source only)
1. Edit repos.yaml  2. pip install -r requirements.txt
3. python pipeline/survey.py   4. python pipeline/harvest.py
5. docker compose up -d        (OpenSearch + Ollama)
6. uvicorn api:app --app-dir pipeline --port 8000  -> open http://localhost:8000 (web UI)
Next: index.py (embed abstracts with BGE-M3 -> OpenSearch hybrid), api.py, eval/.
