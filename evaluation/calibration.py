"""Re-check the Jev-based thresholds against labelled data and report margins.

  guardrail_jev_min   P(in scope)  - must sit between the off-topic max and the on-topic min
  jev_answerable_min  P(answerable) - must sit between the unanswerable max and the answerable min

Usage: python -m evaluation.calibration [--answerable-n 50]
Prints OK / WARN per threshold; WARN when the classes overlap or error rates exceed 5% (the margin is reported but is
dominated by Jev's ~0.03 run-to-run wobble). Re-run after changing the index, the embedding model, or the Jev model version.
"""
from __future__ import annotations

import argparse
import json
import os
import random

os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")


def _summ(name: str, pos: list[float], neg: list[float], threshold: float, pos_label: str, neg_label: str) -> dict:
    pos_min, neg_max = min(pos), max(neg)
    fn = sum(p < threshold for p in pos)   # positives rejected
    fp = sum(n >= threshold for n in neg)  # negatives accepted
    margin = min(pos_min - threshold, threshold - neg_max)
    separable = pos_min > neg_max
    status = "OK" if separable and fn / len(pos) <= 0.05 and fp / len(neg) <= 0.05 else "WARN"
    row = {
        "check": name, "threshold": threshold, "status": status,
        f"{pos_label}_n": len(pos), f"{pos_label}_min": round(pos_min, 3), f"{pos_label}_rejected": fn,
        f"{neg_label}_n": len(neg), f"{neg_label}_max": round(neg_max, 3), f"{neg_label}_accepted": fp,
        "margin": round(margin, 3), "cleanly_separable": separable,
        "suggested_threshold": round((pos_min + neg_max) / 2, 3) if separable else None,
    }
    print(json.dumps(row), flush=True)
    return row


def _sweep(pos: list[float], neg: list[float], thresholds=(0.1, 0.15, 0.2, 0.25, 0.3, 0.4)) -> None:
    print("  threshold sweep (positives wrongly rejected / negatives wrongly accepted):")
    for t in thresholds:
        print(f"    {t:.2f}: {sum(p < t for p in pos)}/{len(pos)}  |  {sum(n >= t for n in neg)}/{len(neg)}")


def _e2e_abstention(answerable_qs: list[str], unanswerable_qs: list[str]) -> dict:
    """The real acceptance test: refuse what the corpus can't answer, answer what it can."""
    import asyncio

    from generation.gateway import get_llm_provider
    from generation.prompts import build_system_prompt, build_user_message, is_no_answer
    from retrieval.engine import retrieve
    from retrieval.relevance import is_answerable

    provider = get_llm_provider()

    def refused(q: str) -> bool:
        docs = retrieve(q, use_relevance_gate=False)
        if not docs:
            return True
        answer = asyncio.run(provider.generate(build_system_prompt(docs), build_user_message(q)))
        return (not is_answerable(q, docs)) or is_no_answer(answer)

    wrongly_refused = [q for q in answerable_qs if refused(q)]
    correctly_refused = [q for q in unanswerable_qs if refused(q)]
    row = {
        "check": "end-to-end abstention",
        "answerable_n": len(answerable_qs), "wrongly_refused": len(wrongly_refused),
        "false_refusal_rate": round(len(wrongly_refused) / len(answerable_qs), 3),
        "unanswerable_n": len(unanswerable_qs), "correctly_refused": len(correctly_refused),
        "correct_refusal_rate": round(len(correctly_refused) / len(unanswerable_qs), 3),
    }
    row["status"] = "OK" if row["false_refusal_rate"] <= 0.05 and row["correct_refusal_rate"] >= 0.85 else "WARN"
    print(json.dumps(row), flush=True)
    for q in wrongly_refused[:5]:
        print("   wrongly refused:", q[:100])
    for q in [q for q in unanswerable_qs if q not in correctly_refused][:5]:
        print("   wrongly answered:", q[:100])
    return row


def run(answerable_n: int = 50, seed: int = 3, e2e: bool = False, e2e_n: int = 20, model: str | None = None) -> list[dict]:
    from config.settings import settings
    from guardrails.topic_classifier import _jev_in_scope
    from retrieval.engine import retrieve
    from retrieval.relevance import jev_answerability

    cal = json.load(open("evaluation/calibration_set.json", encoding="utf-8"))
    rows = []

    on = [_jev_in_scope(q) for q in cal["scope_on"]]
    off = [_jev_in_scope(q) for q in cal["scope_off"]]
    if None in on + off:
        raise SystemExit("Jev unavailable (set OPENROUTER_API_KEY): cannot calibrate")
    rows.append(_summ("guardrail scope", on, off, settings.guardrail_jev_min, "in_scope", "out_of_scope"))

    pages = json.load(open("evaluation/page_questions.json", encoding="utf-8"))
    sample = random.Random(seed).sample(pages, min(answerable_n, len(pages)))
    settings.reranker_provider = "jev"

    def answerability(q: str) -> float:
        docs = retrieve(q, use_relevance_gate=False)
        return jev_answerability(q, docs) if docs else 0.0

    answerable_qs = [p["question"] for p in sample] + cal.get("answerable_extra", [])
    ans = [answerability(q) for q in answerable_qs]
    unans = [answerability(q) for q in cal["unanswerable"]]
    rows.append(_summ("relevance gate", ans, unans, settings.jev_answerable_min, "answerable", "unanswerable"))
    _sweep(ans, unans)
    if e2e:
        if model:  # e.g. a smaller model when the primary's daily token quota is spent
            settings.groq_primary_model = model
        settings.groq_max_rate_limit_wait_s = 90.0  # eval can afford to wait out per-minute limits
        rows.append(_e2e_abstention(answerable_qs[:e2e_n], cal["unanswerable"]))

    with open("evaluation/calibration_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    # The relevance gate is a first-line filter: absent topics often retrieve partially related passages, so
    # it can't separate perfectly (and Jev wobbles run to run). It is reported but not decisive; the
    # guardrail and, when run, the end-to-end abstention check decide the verdict.
    decisive = [r for r in rows if r["check"] in ("guardrail scope", "end-to-end abstention")]
    verdict = all(r["status"] == "OK" for r in decisive)
    print("OVERALL:", "OK" if verdict else "WARN (see rows above)")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--answerable-n", type=int, default=50)
    ap.add_argument("--e2e", action="store_true", help="also run the LLM-based abstention check (slower)")
    ap.add_argument("--e2e-n", type=int, default=20, help="answerable questions to run end-to-end")
    ap.add_argument("--model", default=None, help="override the Groq model for the e2e check")
    a = ap.parse_args()
    run(a.answerable_n, e2e=a.e2e, e2e_n=a.e2e_n, model=a.model)
