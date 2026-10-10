"""Sadhguru Wisdom RAG — Interactive Dashboard with Chat, Evaluation & Settings."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import streamlit as st

CACHE_FILE = Path("data/query_cache.json")


def _load_disk_cache() -> dict:
    """Load query cache from disk."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_disk_cache(cache: dict) -> None:
    """Persist query cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, default=str), encoding="utf-8")

# ─── Page config ───
st.set_page_config(
    page_title="Sadhguru Wisdom RAG",
    page_icon="🙏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Lazy imports (cached so they only load once) ───
@st.cache_resource
def load_pipeline():
    """Load all pipeline components once."""
    from config.settings import settings
    from guardrails.input_validator import validate_input
    from guardrails.output_validator import validate_output
    from guardrails.topic_classifier import classify_query
    from retrieval.engine import retrieve
    from retrieval.relevance import is_answerable
    from generation.prompts import build_system_prompt, build_no_context_response, build_user_message, is_no_answer
    from generation.gateway import get_llm_provider
    from ingestion.indexer import get_qdrant

    from retrieval.warmup import warm_up

    warm_up()  # cached with the pipeline: pays client set-up once per server, not per user
    provider = get_llm_provider()
    client = get_qdrant()
    collection_info = None
    try:
        collection_info = client.get_collection(settings.qdrant_collection)
    except Exception:
        pass

    return {
        "settings": settings,
        "validate_input": validate_input,
        "validate_output": validate_output,
        "classify_query": classify_query,
        "retrieve": retrieve,
        "is_answerable": is_answerable,
        "build_system_prompt": build_system_prompt,
        "build_no_context_response": build_no_context_response,
        "is_no_answer": is_no_answer,
        "build_user_message": build_user_message,
        "provider": provider,
        "qdrant_client": client,
        "collection_info": collection_info,
    }


# ─── Session state defaults ───
MAX_QUERIES_PER_MINUTE = 5

DEFAULTS = {
    "messages": [],
    "thread_id": str(uuid.uuid4()),
    "query_cache": None,
    "eval_report": None,
    "query_timestamps": [],
    # RAG settings
    "retrieval_top_k": 10,
    "rerank_top_k": 7,
    "final_top_k": 5,
    "use_reranker": True,
    "reranker_provider": None,  # None = use settings.reranker_provider (.env / secrets)
    "use_mmr": True,
    "mmr_lambda": 0.7,
    "temperature": 0.3,
    "content_type_filter": "All",
    "show_debug": True,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
if st.session_state.query_cache is None:
    st.session_state.query_cache = _load_disk_cache()


# ─── Custom CSS ───
st.markdown("""
<style>
    /* Tighter layout */
    .block-container { padding-top: 1rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-card h3 { color: #94a3b8; font-size: 0.75rem; margin: 0; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card .value { color: #f1f5f9; font-size: 1.8rem; font-weight: 700; margin: 4px 0; }
    .metric-card .sub { color: #64748b; font-size: 0.7rem; }

    /* Stage timing pills */
    .stage-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 4px;
    }
    .stage-fast { background: #064e3b; color: #6ee7b7; }
    .stage-medium { background: #713f12; color: #fcd34d; }
    .stage-slow { background: #7f1d1d; color: #fca5a5; }

    /* Chunk card */
    .chunk-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
    }
    .chunk-card .chunk-title { color: #60a5fa; font-weight: 600; font-size: 0.85rem; }
    .chunk-card .chunk-score { color: #a78bfa; font-size: 0.75rem; float: right; }
    .chunk-card .chunk-text { color: #cbd5e1; font-size: 0.8rem; margin-top: 6px; }
    .chunk-card .chunk-meta { color: #64748b; font-size: 0.7rem; margin-top: 4px; }

    /* Cache badge */
    .cache-hit { background: #064e3b; color: #6ee7b7; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }
    .cache-miss { background: #1e293b; color: #94a3b8; padding: 2px 8px; border-radius: 12px; font-size: 0.7rem; font-weight: 600; }

    /* Citation inline */
    .citation { background: #312e81; color: #a5b4fc; padding: 1px 6px; border-radius: 4px; font-size: 0.75rem; cursor: help; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🙏 Sadhguru RAG")
    page = st.radio(
        "Navigate",
        ["💬 Chat", "📊 Evaluation", "⚙️ System"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── RAG Settings ──
    st.markdown("### RAG Settings")

    st.session_state.content_type_filter = st.selectbox(
        "Content Filter",
        ["All", "article", "quote", "poem", "video_transcript"],
        help="Filter retrieval by content type",
    )

    st.session_state.final_top_k = st.slider(
        "Results (final_top_k)", 1, 20, st.session_state.final_top_k,
        help="Number of chunks returned to LLM",
    )
    st.session_state.retrieval_top_k = st.slider(
        "Retrieval depth (top_k)", 5, 50, st.session_state.retrieval_top_k,
        help="Initial dense/sparse retrieval count",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.use_reranker = st.toggle(
            "Reranker", value=st.session_state.use_reranker,
            help="Rerank candidates after hybrid fusion",
        )
    with col2:
        st.session_state.use_mmr = st.toggle(
            "MMR", value=st.session_state.use_mmr,
            help="Maximal Marginal Relevance for diversity",
        )

    if st.session_state.reranker_provider is None:
        st.session_state.reranker_provider = load_pipeline()["settings"].reranker_provider
    if st.session_state.use_reranker:
        providers = ["flashrank", "jev", "bge"]
        st.session_state.reranker_provider = st.selectbox(
            "Reranker model", providers,
            index=providers.index(st.session_state.reranker_provider)
            if st.session_state.reranker_provider in providers else 0,
            help="flashrank: small local cross-encoder | jev: TypeSafe via OpenRouter "
                 "(needs OPENROUTER_API_KEY) | bge: bge-reranker-v2-m3 (needs sentence-transformers)",
        )

    if st.session_state.use_mmr:
        st.session_state.mmr_lambda = st.slider(
            "MMR Lambda", 0.0, 1.0, st.session_state.mmr_lambda, 0.05,
            help="0=diversity, 1=relevance",
        )

    st.session_state.temperature = st.slider(
        "LLM Temperature", 0.0, 1.0, st.session_state.temperature, 0.05,
        help="Lower=factual, higher=creative",
    )

    st.divider()

    # ── Debug toggle ──
    st.session_state.show_debug = st.toggle(
        "Show pipeline details", value=st.session_state.show_debug,
    )

    st.divider()

    # ── Conversation ──
    st.markdown("### Conversation")
    st.caption(f"Thread: `{st.session_state.thread_id[:8]}...`")
    st.caption(f"Cache entries: {len(st.session_state.query_cache)}")
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.session_state.query_cache = {}
        _save_disk_cache({})
        st.rerun()


# ═══════════════════════════════════════════════════════════════
#  QUERY PIPELINE (direct, no API dependency)
# ═══════════════════════════════════════════════════════════════
# Bump when prompts / pipeline behaviour change in ways the settings snapshot does not capture
CACHE_VERSION = 3


def get_cache_key(question: str, settings_dict: dict) -> str:
    """Hash question + settings for cache key."""
    raw = f"{question.strip().lower()}|{json.dumps(settings_dict, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _check_rate_limit() -> tuple[bool, int]:
    """Check per-session rate limit. Returns (allowed, seconds_until_reset)."""
    now = time.time()
    window = 60  # 1 minute
    # Remove timestamps older than the window
    st.session_state.query_timestamps = [
        t for t in st.session_state.query_timestamps if now - t < window
    ]
    if len(st.session_state.query_timestamps) >= MAX_QUERIES_PER_MINUTE:
        oldest = st.session_state.query_timestamps[0]
        wait = int(window - (now - oldest)) + 1
        return False, wait
    return True, 0


def run_query(question: str) -> dict:
    """Run full RAG pipeline with timing for each stage."""
    # Rate limit check (cache hits are free)
    allowed, wait = _check_rate_limit()

    pipe = load_pipeline()
    settings_obj = pipe["settings"]

    # Build settings snapshot for cache key
    settings_snap = {
        "top_k": st.session_state.retrieval_top_k,
        "reranker": st.session_state.use_reranker,
        "reranker_provider": st.session_state.reranker_provider,
        "mmr": st.session_state.use_mmr,
        "mmr_lambda": st.session_state.mmr_lambda,
        "final_k": st.session_state.final_top_k,
        "content_type": st.session_state.content_type_filter,
        "temperature": st.session_state.temperature,
        # An answer depends on the index, the models and the prompt: changing any of them must
        # invalidate cached answers (a stale cache served old-index answers after the v3 switch).
        "collection": settings_obj.qdrant_collection,
        "llm": settings_obj.groq_primary_model,
        "cache_version": CACHE_VERSION,
    }
    cache_key = get_cache_key(question, settings_snap)

    # Check cache (cache hits bypass rate limit)
    if cache_key in st.session_state.query_cache:
        cached = st.session_state.query_cache[cache_key].copy()
        cached["cache_hit"] = True
        return cached

    # Enforce rate limit only for cache misses (actual API calls)
    if not allowed:
        return {
            "answer": f"Rate limit reached ({MAX_QUERIES_PER_MINUTE} queries/minute). Please wait {wait} seconds.",
            "sources": [],
            "stages": [],
            "cache_hit": False,
            "is_on_topic": True,
            "total_ms": 0,
            "chunks_raw": [],
            "settings_used": settings_snap,
        }
    st.session_state.query_timestamps.append(time.time())

    result = {
        "answer": "",
        "sources": [],
        "stages": [],
        "cache_hit": False,
        "is_on_topic": True,
        "total_ms": 0,
        "chunks_raw": [],
        "settings_used": settings_snap,
    }
    pipeline_start = time.perf_counter()

    # Override settings temporarily
    original_top_k = settings_obj.retrieval_top_k
    original_rerank_k = settings_obj.rerank_top_k
    original_final_k = settings_obj.final_top_k
    original_mmr_lambda = settings_obj.mmr_lambda
    original_temp = settings_obj.groq_temperature
    original_reranker = settings_obj.reranker_provider

    settings_obj.reranker_provider = st.session_state.reranker_provider
    settings_obj.retrieval_top_k = st.session_state.retrieval_top_k
    settings_obj.rerank_top_k = st.session_state.rerank_top_k
    settings_obj.final_top_k = st.session_state.final_top_k
    settings_obj.mmr_lambda = st.session_state.mmr_lambda
    settings_obj.groq_temperature = st.session_state.temperature

    try:
        # Stage 1: Input Validation
        t0 = time.perf_counter()
        valid, error = pipe["validate_input"](question)
        dt = (time.perf_counter() - t0) * 1000
        result["stages"].append({"name": "Input Validation", "ms": round(dt, 1), "ok": valid})
        if not valid:
            result["answer"] = f"Invalid input: {error}"
            result["is_on_topic"] = False
            return result

        # Stages 2+3 overlap: retrieval starts speculatively while the topic check runs
        # (the Jev scope call is ~0.4 s); its result is discarded if the question is off-topic.
        content_type = None if st.session_state.content_type_filter == "All" else st.session_state.content_type_filter

        def _timed(fn, *args, **kwargs):
            t = time.perf_counter()
            return fn(*args, **kwargs), (time.perf_counter() - t) * 1000

        with ThreadPoolExecutor(max_workers=2) as pool:
            retrieval_future = pool.submit(
                _timed, pipe["retrieve"], question,
                content_type=content_type,
                use_reranker=st.session_state.use_reranker,
                use_mmr=st.session_state.use_mmr,
                use_relevance_gate=False,
            )
            (on_topic, deflection), dt = _timed(pipe["classify_query"], question)
            result["stages"].append({"name": "Topic Classifier", "ms": round(dt, 1), "ok": on_topic})
            if not on_topic:
                result["answer"] = deflection
                result["is_on_topic"] = False
                return result
            chunks, dt = retrieval_future.result()
        result["stages"].append({"name": "Retrieval", "ms": round(dt, 1), "ok": len(chunks) > 0, "count": len(chunks)})
        result["chunks_raw"] = chunks

        if not chunks:
            result["answer"] = pipe["build_no_context_response"]()
            return result

        # Stage 4: Prompt Building
        t0 = time.perf_counter()
        system_prompt = pipe["build_system_prompt"](chunks)
        dt = (time.perf_counter() - t0) * 1000
        result["stages"].append({"name": "Prompt Build", "ms": round(dt, 1), "ok": True})

        # Stage 5: LLM Generation, overlapped with the relevance gate (Jev answerability check)
        t0 = time.perf_counter()
        loop = asyncio.new_event_loop()

        async def _generate_and_gate():
            gate = loop.run_in_executor(None, pipe["is_answerable"], question, chunks)                 if st.session_state.use_reranker else None
            answer_ = await pipe["provider"].generate(system_prompt, pipe["build_user_message"](question))
            return answer_, (await gate if gate else True)

        answer, answerable = loop.run_until_complete(_generate_and_gate())
        loop.close()
        dt = (time.perf_counter() - t0) * 1000
        result["stages"].append({"name": "LLM Generation + Relevance Gate", "ms": round(dt, 1), "ok": answerable})
        if not answerable or pipe["is_no_answer"](answer):
            result["answer"] = pipe["build_no_context_response"]()
            result["chunks_raw"] = []  # nothing to cite: don't show sources next to a refusal
            return result

        # Stage 6: Output Validation
        t0 = time.perf_counter()
        answer = pipe["validate_output"](answer)
        dt = (time.perf_counter() - t0) * 1000
        result["stages"].append({"name": "Output Guard", "ms": round(dt, 1), "ok": True})

        result["answer"] = answer
        result["sources"] = [
            {
                "title": c.get("title", ""),
                "url": c.get("url", ""),
                "text": c.get("text", ""),
                "content_type": c.get("content_type", ""),
                "score": c.get("score") or c.get("rrf_score") or c.get("rerank_score"),
                "topics": c.get("topics", []),
            }
            for c in chunks
        ]

    except Exception as e:
        # Users get a generic message; details (which can include provider/org identifiers) go to the
        # logs and the debug timing panel only.
        from generation.errors import BUSY_MESSAGE, GENERIC_ERROR_MESSAGE, LLMBusyError

        result["answer"] = BUSY_MESSAGE if isinstance(e, LLMBusyError) else GENERIC_ERROR_MESSAGE
        print(f"Pipeline error: {e!r}", flush=True)
        result["stages"].append({"name": "Error", "ms": 0, "ok": False, "error": str(e)})
    finally:
        # Restore settings
        settings_obj.retrieval_top_k = original_top_k
        settings_obj.rerank_top_k = original_rerank_k
        settings_obj.final_top_k = original_final_k
        settings_obj.mmr_lambda = original_mmr_lambda
        settings_obj.groq_temperature = original_temp
        settings_obj.reranker_provider = original_reranker

    result["total_ms"] = round((time.perf_counter() - pipeline_start) * 1000, 1)

    # Cache result (memory + disk), but never a failure: it would be replayed as the answer
    if not any(stage.get("name") == "Error" for stage in result["stages"]):
        st.session_state.query_cache[cache_key] = result.copy()
        _save_disk_cache(st.session_state.query_cache)
    result["cache_hit"] = False
    return result


# ═══════════════════════════════════════════════════════════════
#  RENDER HELPERS
# ═══════════════════════════════════════════════════════════════
def render_stage_timing(stages: list[dict], total_ms: float, cache_hit: bool):
    """Render pipeline timing breakdown."""
    if cache_hit:
        st.markdown('<span class="cache-hit">⚡ CACHE HIT</span>', unsafe_allow_html=True)
        return

    st.markdown(f'<span class="cache-miss">CACHE MISS — {total_ms:.0f}ms total</span>', unsafe_allow_html=True)

    cols = st.columns(len(stages))
    for i, stage in enumerate(stages):
        ms = stage["ms"]
        if ms < 50:
            cls = "stage-fast"
        elif ms < 500:
            cls = "stage-medium"
        else:
            cls = "stage-slow"

        status = "✓" if stage.get("ok", True) else "✗"
        extra = f" ({stage['count']})" if "count" in stage else ""
        with cols[i]:
            st.markdown(
                f'<div class="stage-pill {cls}">{status} {stage["name"]}{extra}<br>{ms:.0f}ms</div>',
                unsafe_allow_html=True,
            )


def render_chunks(sources: list[dict]):
    """Render retrieved chunks with scores and metadata."""
    for i, src in enumerate(sources):
        score = src.get("score")
        score_str = f"{score:.4f}" if score else "—"
        topics = ", ".join(src.get("topics", [])[:3]) if src.get("topics") else ""
        url = src.get("url", "")
        title = src.get("title", f"Chunk {i+1}")
        text = src.get("text", "")[:400]
        ctype = src.get("content_type", "")

        st.markdown(f"""
<div class="chunk-card">
    <span class="chunk-title">[{i+1}] {title}</span>
    <span class="chunk-score">Score: {score_str}</span>
    <div class="chunk-text">{text}...</div>
    <div class="chunk-meta">
        Type: {ctype} {'| Topics: ' + topics if topics else ''} {'| ' + url if url else ''}
    </div>
</div>
""", unsafe_allow_html=True)


def render_answer_with_citations(answer: str, sources: list[dict]):
    """Render answer with clickable citation references."""
    st.markdown(answer)

    # Show citation legend
    cited_sources = []
    for i, src in enumerate(sources):
        title = src.get("title", "")
        if title and title.lower() in answer.lower():
            cited_sources.append((i + 1, title, src.get("url", "")))

    if cited_sources:
        st.caption("**Citations found in answer:**")
        for num, title, url in cited_sources:
            if url:
                st.caption(f"  [{num}] [{title}]({url})")
            else:
                st.caption(f"  [{num}] {title}")


# ═══════════════════════════════════════════════════════════════
#  PAGE: CHAT
# ═══════════════════════════════════════════════════════════════
def page_chat():
    st.header("💬 Ask Sadhguru's Wisdom")

    # Quick stats bar
    pipe = load_pipeline()
    info = pipe["collection_info"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        points = info.points_count if info else 0
        st.metric("Indexed Chunks", f"{points:,}")
    with c2:
        st.metric("Retrieval Top-K", st.session_state.final_top_k)
    with c3:
        mode = []
        if st.session_state.use_reranker:
            mode.append(f"Rerank({st.session_state.reranker_provider})")
        if st.session_state.use_mmr:
            mode.append("MMR")
        st.metric("Pipeline", "+".join(mode) if mode else "Dense+Sparse")
    with c4:
        st.metric("Cache Size", len(st.session_state.query_cache))

    st.divider()

    # Display conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_answer_with_citations(msg["content"], msg.get("sources", []))

                if st.session_state.show_debug and msg.get("stages"):
                    with st.expander("⏱️ Pipeline Timing"):
                        render_stage_timing(msg["stages"], msg.get("total_ms", 0), msg.get("cache_hit", False))

                if st.session_state.show_debug and msg.get("sources"):
                    with st.expander(f"📄 Retrieved Chunks ({len(msg['sources'])})"):
                        render_chunks(msg["sources"])

                if msg.get("settings_used"):
                    with st.expander("⚙️ Settings Used"):
                        st.json(msg["settings_used"])
            else:
                st.markdown(msg["content"])

    # Chat input
    if question := st.chat_input("What would you like to know about Sadhguru's teachings?"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Seeking wisdom..."):
                result = run_query(question)

            render_answer_with_citations(result["answer"], result.get("sources", []))

            if st.session_state.show_debug and result.get("stages"):
                with st.expander("⏱️ Pipeline Timing", expanded=True):
                    render_stage_timing(result["stages"], result.get("total_ms", 0), result.get("cache_hit", False))

            if st.session_state.show_debug and result.get("sources"):
                with st.expander(f"📄 Retrieved Chunks ({len(result['sources'])})", expanded=False):
                    render_chunks(result["sources"])

            if result.get("settings_used"):
                with st.expander("⚙️ Settings Used"):
                    st.json(result["settings_used"])

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "stages": result.get("stages", []),
                "total_ms": result.get("total_ms", 0),
                "cache_hit": result.get("cache_hit", False),
                "settings_used": result.get("settings_used"),
            })


# ═══════════════════════════════════════════════════════════════
#  PAGE: EVALUATION
# ═══════════════════════════════════════════════════════════════
def _load_json(path: str):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def render_latest_results(report: dict):
    """Retrieval accuracy, threshold calibration, injection tests and index health, from the result files."""
    import pandas as pd

    st.subheader("Latest results")
    pg = report.get("retrieval", {}).get("page_grounded", {})
    prov = _load_json("evaluation/page_questions_results.json")
    cal = report.get("calibration") or _load_json("evaluation/calibration_results.json") or []
    inj = _load_json("evaluation/injection_results.json")
    scan = _load_json("evaluation/boilerplate_scan_v3.json")

    # ── Headline cards ──
    cols = st.columns(4)
    if pg:
        cols[0].metric("Right page in top 5", f"{pg.get('hit@5', 0):.1%}", help=f"{pg.get('n')} questions written from random pages")
        cols[1].metric("Right page ranked #1", f"{pg.get('hit@1', 0):.1%}")
    e2e = next((c for c in cal if c.get("check") == "end-to-end abstention"), None)
    if e2e:
        cols[2].metric("Absent questions refused", f"{e2e['correctly_refused']}/{e2e['unanswerable_n']}",
                       help="In-scope questions the index cannot answer; the bot should say so")
    if inj:
        s = inj["summary"]
        cols[3].metric("Injection attacks defended", f"{s['passed']}/{s['total']}")

    # ── Retrieval: provider comparison ──
    st.markdown("**Retrieval accuracy: does the right source page come back?**")
    rows = []
    if prov:
        for r in prov:
            rows.append({"Reranker": r["provider"], "Relevance gate": "on" if r["gate"] else "off", "Questions": r["n"],
                         "hit@5": r.get("hit@5"), "hit@1": r.get("hit@1"), "MRR": r.get("mrr"),
                         "Wrongly blocked": r.get("gate_blocked"), "Mean latency (ms)": r.get("latency_ms_mean")})
    if pg:
        rows.append({"Reranker": f"{pg.get('provider', '?')} (current config)", "Relevance gate": "on", "Questions": pg.get("n"),
                     "hit@5": pg.get("hit@5"), "hit@1": pg.get("hit@1"), "MRR": pg.get("mrr"),
                     "Wrongly blocked": pg.get("gate_blocked"), "Mean latency (ms)": pg.get("latency_ms_mean")})
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Reranker")[["hit@5", "hit@1"]].groupby(level=0).mean())
        st.caption("Strict metric: another page may answer equally well, so these are lower bounds. "
                   "Older rows use 60 questions; the current-config row uses 181.")

    # ── Calibration ──
    if cal:
        st.markdown("**Thresholds: are the guardrail and refusal behaviour still calibrated?**")
        out = []
        for c in cal:
            if c["check"] == "end-to-end abstention":
                out.append({"Check": "Refuse what the index can't answer / answer what it can", "Status": c["status"],
                            "Detail": f"{c['correctly_refused']}/{c['unanswerable_n']} absent refused; "
                                      f"{c['wrongly_refused']}/{c['answerable_n']} answerable wrongly refused"})
            elif c["check"] == "guardrail scope":
                out.append({"Check": f"Topic guardrail (threshold {c['threshold']})", "Status": c["status"],
                            "Detail": f"{c['in_scope_n']} in-scope (min score {c['in_scope_min']}), "
                                      f"{c['out_of_scope_n']} out-of-scope (max {c['out_of_scope_max']}); errors: "
                                      f"{c['in_scope_rejected']} + {c['out_of_scope_accepted']}"})
            else:
                out.append({"Check": f"Relevance gate (threshold {c['threshold']}), first-line filter", "Status": c["status"],
                            "Detail": f"{c['answerable_rejected']}/{c['answerable_n']} answerable blocked; "
                                      f"{c['unanswerable_accepted']}/{c['unanswerable_n']} absent let through (the LLM refuses these)"})
        st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)

    # ── Injection ──
    if inj:
        st.markdown("**Prompt-injection suite**")
        s = inj["summary"]
        (st.success if s["failed"] == 0 else st.error)(f"{s['passed']} of {s['total']} attacks defended")
        with st.expander("Show every attack and the outcome"):
            idf = pd.DataFrame([{"Type": r["kind"], "Attack": r["attack"], "Result": "FAILED" if r["violations"] else "defended",
                                 "Outcome": r.get("outcome", ""), "Response": r["answer"][:100]} for r in inj["rows"]])
            st.dataframe(idf, use_container_width=True, hide_index=True)

    # ── Index health ──
    if scan:
        st.markdown("**Index health**")
        c = st.columns(3)
        c[0].metric("Chunks indexed", f"{scan['total_points']:,}")
        c[1].metric("Menu/boilerplate chunks", f"{scan['flagged_boilerplate']:,}")
        c[2].metric("Exact duplicates", f"{scan['exact_duplicate_extra_copies']:,}")
        st.caption("The original index was 68% site-menu boilerplate with 26,506 duplicate copies.")


def page_evaluation():
    st.header("📊 RAG Evaluation Dashboard")

    # Load existing report (newest first) and show it straight away
    report_path = next((Path(p) for p in ("evaluation/report_v3.json", "evaluation/report.json") if Path(p).exists()),
                       Path("evaluation/report.json"))
    if st.session_state.eval_report is None and report_path.exists():
        st.session_state.eval_report = json.loads(report_path.read_text(encoding="utf-8"))

    col_run, col_load = st.columns([1, 1])
    with col_run:
        if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
            with st.spinner("Running evaluation (this takes a few minutes)..."):
                try:
                    from evaluation.runner import run_full_evaluation
                    loop = asyncio.new_event_loop()
                    report = loop.run_until_complete(run_full_evaluation())
                    loop.close()
                    st.session_state.eval_report = report
                    st.success("Evaluation complete!")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")

    with col_load:
        if report_path.exists():
            if st.button("📂 Load Previous Report", use_container_width=True):
                with open(report_path, "r") as f:
                    st.session_state.eval_report = json.load(f)
                st.success("Report loaded!")

    if not st.session_state.eval_report:
        st.info("No evaluation report loaded. Run an evaluation or load a previous report.")
        return

    report = st.session_state.eval_report

    # ── Overall Score ──
    st.divider()
    overall = report.get("overall", {})
    score = overall.get("score", 0)
    grade = overall.get("grade", "?")
    cats = overall.get("category_scores", {})  # {name: float_score}
    cat_weights = overall.get("category_weights", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
<div class="metric-card">
    <h3>Overall Score</h3>
    <div class="value">{score:.1f}/100</div>
    <div class="sub">Grade: {grade}</div>
</div>""", unsafe_allow_html=True)
    with c2:
        meta = report.get("metadata", {})
        st.markdown(f"""
<div class="metric-card">
    <h3>Test Cases</h3>
    <div class="value">{meta.get('total_cases', 0)}</div>
    <div class="sub">{meta.get('timestamp', '')[:10]}</div>
</div>""", unsafe_allow_html=True)
    with c3:
        best_cat = max(cats, key=lambda k: cats[k]) if cats else "—"
        best_score = cats.get(best_cat, 0) if cats else 0
        st.markdown(f"""
<div class="metric-card">
    <h3>Strongest</h3>
    <div class="value">{best_score:.0f}</div>
    <div class="sub">{best_cat}</div>
</div>""", unsafe_allow_html=True)
    with c4:
        worst_cat = min(cats, key=lambda k: cats[k]) if cats else "—"
        worst_score = cats.get(worst_cat, 0) if cats else 0
        st.markdown(f"""
<div class="metric-card">
    <h3>Weakest</h3>
    <div class="value">{worst_score:.0f}</div>
    <div class="sub">{worst_cat}</div>
</div>""", unsafe_allow_html=True)

    # ── Category Breakdown ──
    st.divider()
    st.subheader("Category Scores")

    if cats:
        import pandas as pd
        cat_names = list(cats.keys())
        cat_scores = [float(cats[c]) for c in cat_names]

        df = pd.DataFrame({
            "Category": cat_names,
            "Score": cat_scores,
        })
        st.bar_chart(df.set_index("Category")["Score"], color="#6366f1")

    st.divider()
    render_latest_results(report)

    # ── Detailed Results per Category ──
    st.divider()
    st.subheader("Detailed Results")

    # Categories are top-level keys in report (exclude metadata/overall)
    skip_keys = {"metadata", "overall", "calibration"}
    category_keys = [k for k in report if k not in skip_keys]
    tabs = st.tabs(category_keys) if category_keys else []

    for tab, cat_name in zip(tabs, category_keys):
        cat_data = report[cat_name]
        if not isinstance(cat_data, dict):
            continue
        with tab:
            # Summary metrics — show all top-level non-list/dict values
            summary_items = {k: v for k, v in cat_data.items() if not isinstance(v, (list, dict))}
            if summary_items:
                metric_cols = st.columns(min(len(summary_items), 6))
                for i, (k, v) in enumerate(summary_items.items()):
                    if i < 6:
                        with metric_cols[i]:
                            if isinstance(v, float):
                                st.metric(k.replace("_", " ").title(), f"{v:.3f}")
                            else:
                                st.metric(k.replace("_", " ").title(), str(v))

            # Per-case results — find list values that contain dicts
            for key, val in cat_data.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    st.caption(f"{len(val)} test cases")
                    for case in val:
                        question = case.get("question", case.get("id", str(case)[:60]))
                        is_ok = case.get("correct", case.get("on_topic", case.get("is_on_topic")))
                        status = "✅" if is_ok else ("❌" if is_ok is not None else "⬜")

                        with st.expander(f"{status} {str(question)[:80]}"):
                            if case.get("answer") or case.get("answer_preview"):
                                st.markdown(f"**Answer:** {(case.get('answer') or case.get('answer_preview', ''))[:300]}")
                            if case.get("metrics"):
                                st.json(case["metrics"])
                            if case.get("context"):
                                st.caption("**Retrieved Context:**")
                                for ctx in case["context"]:
                                    if isinstance(ctx, dict):
                                        st.caption(f"  - {ctx.get('title', '')}: {ctx.get('text', '')[:200]}")
                            if case.get("latency_ms"):
                                st.caption(f"Latency: {case['latency_ms']:.0f}ms")
                    break  # only show first list of cases per category

            # Sub-dicts (like latency summary stages)
            for key, val in cat_data.items():
                if isinstance(val, dict) and val:
                    with st.expander(f"📋 {key}"):
                        st.json(val)

    # ── Faithfulness Analysis ──
    gen_data = report.get("generation", {})
    faith_cases = []
    for key, val in gen_data.items():
        if isinstance(val, list) and val and isinstance(val[0], dict):
            faith_cases = val
            break
    faith_data = faith_cases
    if faith_data:
        st.divider()
        st.subheader("Faithfulness Deep Dive")

        low_faith = [c for c in faith_data if c.get("metrics", {}).get("faithfulness", 1) < 0.5]
        high_hall = [c for c in faith_data if c.get("metrics", {}).get("hallucination_score", 0) > 0.3]

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Low Faithfulness Cases", len(low_faith))
        with c2:
            st.metric("High Hallucination Cases", len(high_hall))

        flagged = low_faith + [c for c in high_hall if c not in low_faith]
        for case in flagged:
            metrics = case.get("metrics", {})
            faith = metrics.get("faithfulness", 0)
            hall = metrics.get("hallucination_score", 0)
            color = "#ef4444" if faith < 0.3 else "#f59e0b"

            with st.expander(f"⚠️ {case.get('question', '')[:60]} (faith={faith:.2f}, hall={hall:.2f})"):
                col_a, col_c = st.columns(2)
                with col_a:
                    st.markdown("**Answer:**")
                    st.markdown(case.get("answer", "")[:500])
                with col_c:
                    st.markdown("**Context Used:**")
                    for ctx in case.get("context", []):
                        st.caption(f"- {ctx.get('title', '')}: {ctx.get('text', '')[:200]}")

    # ── Latency Analysis ──
    latency_data = report.get("latency", {})
    if latency_data:
        st.divider()
        st.subheader("Latency Profile")

        lat_summary = latency_data.get("stats", {})
        if lat_summary.get("stages"):
            stage_data = lat_summary["stages"]
            import pandas as pd
            lat_df = pd.DataFrame([
                {
                    "Stage": name,
                    "Mean (ms)": vals.get("mean_ms", 0),
                    "P50 (ms)": vals.get("p50_ms", 0),
                    "P95 (ms)": vals.get("p95_ms", 0),
                }
                for name, vals in stage_data.items()
            ])
            st.dataframe(lat_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
#  PAGE: SYSTEM
# ═══════════════════════════════════════════════════════════════
def page_system():
    st.header("⚙️ System Status & Configuration")

    pipe = load_pipeline()
    settings_obj = pipe["settings"]
    info = pipe["collection_info"]

    # ── Connection Status ──
    st.subheader("Connections")
    c1, c2, c3 = st.columns(3)

    with c1:
        if info:
            st.success(f"Qdrant Cloud: {info.points_count:,} points")
            st.caption(f"Collection: {settings_obj.qdrant_collection}")
            st.caption(f"Vectors: {settings_obj.embedding_dim}d")
        else:
            st.error("Qdrant: Not connected")

    with c2:
        try:
            from generation.gateway import get_llm_provider
            provider = get_llm_provider()
            st.success(f"LLM: {settings_obj.llm_provider}")
            st.caption(f"Model: {settings_obj.groq_primary_model}")
            st.caption(f"Temperature: {settings_obj.groq_temperature}")
        except Exception as e:
            st.error(f"LLM: {e}")

    with c3:
        st.success(f"Embeddings: {settings_obj.embedding_provider}")
        st.caption(f"Model: {settings_obj.embedding_model}")
        st.caption(f"Dim: {settings_obj.embedding_dim}")

    # ── Current Settings ──
    st.divider()
    st.subheader("All Settings")

    settings_dict = {
        "LLM Provider": settings_obj.llm_provider,
        "Groq Model": settings_obj.groq_primary_model,
        "Groq Fallback": settings_obj.groq_fallback_model,
        "Temperature": settings_obj.groq_temperature,
        "Embedding Provider": settings_obj.embedding_provider,
        "Embedding Model": settings_obj.embedding_model,
        "Embedding Dim": settings_obj.embedding_dim,
        "Qdrant Collection": settings_obj.qdrant_collection,
        "Chunk Size": settings_obj.chunk_size,
        "Chunk Overlap": settings_obj.chunk_overlap,
        "Retrieval Top-K": settings_obj.retrieval_top_k,
        "Rerank Top-K": settings_obj.rerank_top_k,
        "Final Top-K": settings_obj.final_top_k,
        "RRF K": settings_obj.rrf_k,
        "MMR Lambda": settings_obj.mmr_lambda,
        "Guardrail Threshold": settings_obj.guardrail_similarity_threshold,
        "Rate Limit": settings_obj.rate_limit,
    }
    st.json(settings_dict)

    # ── Collection Stats ──
    if info:
        st.divider()
        st.subheader("Collection Details")

        try:
            client = pipe["qdrant_client"]
            # Sample a few points to show
            sample = client.scroll(
                collection_name=settings_obj.qdrant_collection,
                limit=5,
                with_payload=True,
                with_vectors=False,
            )
            if sample and sample[0]:
                st.caption(f"Sample of {len(sample[0])} documents:")
                for pt in sample[0]:
                    title = pt.payload.get("title", "Untitled")
                    ctype = pt.payload.get("content_type", "")
                    text_preview = pt.payload.get("text", "")[:150]
                    st.markdown(f"- **{title}** ({ctype}): {text_preview}...")
        except Exception as e:
            st.warning(f"Could not sample collection: {e}")

    # ── Corpus Stats ──
    st.divider()
    st.subheader("Corpus Stats")
    corpus_path = Path("data/raw/corpus.json")
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Total Documents", len(corpus))
        with c2:
            types = {}
            for doc in corpus:
                t = doc.get("content_type", "unknown")
                types[t] = types.get(t, 0) + 1
            st.json(types)
        with c3:
            total_chars = sum(len(d.get("text", "")) for d in corpus)
            st.metric("Total Characters", f"{total_chars:,}")
    else:
        st.info("No corpus.json found. Run the scraper first.")

    # ── Sample Questions ──
    st.divider()
    st.subheader("Test Questions")
    st.caption("Quick test to verify the pipeline is working:")

    test_q = st.text_input("Test query", value="What is meditation according to Sadhguru?")
    if st.button("Test Pipeline"):
        with st.spinner("Running..."):
            result = run_query(test_q)
            st.write(f"**Answer:** {result['answer'][:300]}")
            st.write(f"**Stages:** {result['stages']}")
            st.write(f"**Total:** {result['total_ms']}ms")
            st.write(f"**Chunks:** {len(result.get('sources', []))}")
            st.write(f"**Cache hit:** {result.get('cache_hit', False)}")


# ═══════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════
if page == "💬 Chat":
    page_chat()
elif page == "📊 Evaluation":
    page_evaluation()
elif page == "⚙️ System":
    page_system()
