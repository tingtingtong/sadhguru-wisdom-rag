import os
import sys

from pydantic_settings import BaseSettings

# On Streamlit Cloud, secrets are in st.secrets — inject them as env vars so pydantic-settings
# can pick them up. Only when running inside Streamlit (already imported): importing it here
# unconditionally cost the API server and CLI scripts seconds of startup.
if "streamlit" in sys.modules:
    try:
        import streamlit as st

        # load_if_toml_exists() avoids Streamlit's red "No secrets files found" banner on local runs
        if st.secrets.load_if_toml_exists():
            for key, val in st.secrets.items():
                if isinstance(val, str):
                    os.environ.setdefault(key.upper(), val)
    except Exception:
        pass


class Settings(BaseSettings):
    # LLM
    llm_provider: str = "groq"
    groq_api_key: str = ""
    groq_primary_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "openai/gpt-oss-20b"  # separate model => separate rate limits
    groq_temperature: float = 0.3
    groq_max_rate_limit_wait_s: float = 10.0  # longest 429 "try again in" hint worth waiting out per request

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "sadhguru_wisdom_v3"
    qdrant_grpc_ingest: bool = True   # bulk uploads over gRPC (4x faster than JSON REST)
    qdrant_prefer_grpc: bool = True   # queries over gRPC: parallel searches share one connection

    # Embeddings
    embedding_provider: str = "openai"  # "local" or "openai"
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072
    openai_api_key: str = ""

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    retrieval_top_k: int = 10
    rerank_top_k: int = 7
    final_top_k: int = 5
    rrf_k: int = 60
    mmr_lambda: float = 0.7

    # Reranker: "flashrank" | "jev" | "bge"
    reranker_provider: str = "flashrank"
    # Jev (TypeSafe System One model, hosted via OpenRouter)
    openrouter_api_key: str = ""
    jev_model: str = "typesafe/jev-1.13"
    jev_url: str = "https://openrouter.ai/api/alpha/decisions"
    jev_timeout: float = 15.0
    jev_batch_size: int = 20  # candidates per listwise call (context window is 32k tokens)
    # Relevance gate: return no context (-> "couldn't find this") when retrieval can't answer
    relevance_gate: bool = True
    jev_answerable_min: float = 0.18   # Jev noul P(answerable); 0 of 54 answerable questions score below 0.20
    # Score floor for flashrank/bge. DISABLED (0): measured on 64 answerable + 22 unanswerable questions it
    # wrongly blocked 4 good ones and caught 1 bad one (FlashRank scores absent topics ~0.99 too).
    relevance_min_score: float = 0.0
    # BGE cross-encoder (local; needs sentence-transformers)
    bge_model: str = "BAAI/bge-reranker-v2-m3"

    # Guardrails
    guardrail_similarity_threshold: float = 0.35
    # Scope check: Jev P(in scope). Over 6 calibration runs (86 labeled questions) in-scope never scored below 0.26 and
    # out-of-scope never above 0.24; 0.25 sits in the gap every time. Scores wobble ~0.03 between runs.
    guardrail_use_jev: bool = True
    guardrail_jev_min: float = 0.25
    # Block when Jev P(prompt injection) >= this. Calibrated: 20/21 attacks >= 0.2; max over 289 legitimate questions 0.06.
    guardrail_injection_max: float = 0.20
    # Fallback when Jev is unavailable: min cosine similarity to the nearest indexed chunk. Weak on the
    # full index (sports/cars/money questions find close chunks); calibrated for text-embedding-3-large.
    guardrail_corpus_threshold: float = 0.45

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    rate_limit: str = "30/minute"

    # Logfire
    logfire_token: str = ""

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
