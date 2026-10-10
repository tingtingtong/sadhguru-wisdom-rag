# Deployment notes

## Required secrets (Streamlit Cloud: App settings > Secrets; Docker/API: `.env`)

| Key | Notes |
|---|---|
| `GROQ_API_KEY` | LLM. Free tier is ~200k tokens/day for `openai/gpt-oss-120b` and 8k tokens/min for the fallback `gpt-oss-20b`: roughly 70-130 answers/day. Use a paid tier for real traffic. |
| `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION` | Collection must be `sadhguru_wisdom_v3` (clean index, 36,637 chunks). Older collections have boilerplate/duplicates and an incompatible sparse encoding. |
| `OPENAI_API_KEY` | Query + ingestion embeddings (`text-embedding-3-large`, 3072-d). |
| `OPENROUTER_API_KEY`, `RERANKER_PROVIDER=jev` | Jev: reranker, relevance gate, topic guardrail. Optional: without it the app falls back to FlashRank + embedding guardrail (lower quality). |
| `LOGFIRE_TOKEN` | Optional tracing. |

Never commit real values: `.env` and `secrets_for_cloud.toml` are gitignored and a unit test fails if secret-like strings reach a committable file. Docker builds exclude them via `.dockerignore`.

## Runtime behaviour worth knowing
- **Jev is an alpha API** (`/api/alpha/decisions` on OpenRouter). Every use has a fallback: rerank -> FlashRank, gate -> skipped, guardrail -> embedding similarity. `GET /health/ready` reports `jev: ok | unavailable | disabled`.
- **Qdrant over gRPC** (port 6334) by default for speed; it falls back to REST automatically if gRPC is blocked. Set `QDRANT_PREFER_GRPC=false` to force REST.
- **Latency**: ~3 s median per answered question (Qdrant cluster is in sa-east-1; each round trip is ~0.35 s from India).
- The API warms its clients at startup; the first request after boot is ~5 s instead of ~20 s.

## Safety layers (defence in depth)
1. **Topic + injection guardrail** (one Jev call, runs concurrently with retrieval): blocks out-of-scope questions and prompt-injection attempts (P >= 0.20; 20/21 attacks caught, 0/289 legitimate questions flagged).
2. **Prompt hardening**: the context and the user's question are declared untrusted data, chat-template tokens are stripped, and the model replies `NO_ANSWER_IN_CONTEXT` when it cannot answer.
3. **Relevance gate + refusal detection**: unanswerable questions get the standard "couldn't find" message and **no sources** (22/22 in the end-to-end check).
4. **Output validation**: answers that echo the context scaffold or the rules are replaced.
5. **Errors**: users see generic messages; provider details (org ids, 429 bodies) stay in logs. All-models-rate-limited returns HTTP 503 + `Retry-After`.

`python -m evaluation.injection_tests` runs 21 user attacks and 3 poisoned-context attacks (24/24 defended, measured on the smaller `gpt-oss-20b`).

## Re-checking quality after any change (index, embedding model, Jev version)
```
python -m evaluation.calibration --e2e         # thresholds + refuse/answer behaviour
python -m evaluation.page_questions run --providers jev flashrank
python -m evaluation.injection_tests           # prompt-injection suite
python -m evaluation.runner --skip-generation  # full report
```

## Refreshing the index
```
SCRAPE_DATA_DIR=data/raw_vN python scripts/scrape.py --mode sitemap --full --max-pages 100000 --concurrency 6
QDRANT_COLLECTION=sadhguru_wisdom_vN SCRAPE_DATA_DIR=data/raw_vN python scripts/ingest.py   # resumable
```
