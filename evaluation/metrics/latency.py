"""Latency and performance metrics for RAG pipeline stages."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class StageLatency:
    stage: str
    duration_ms: float
    success: bool = True
    error: str | None = None


@dataclass
class PipelineLatency:
    total_ms: float
    stages: list[StageLatency] = field(default_factory=list)
    query: str = ""

    @property
    def stage_breakdown(self) -> dict[str, float]:
        return {s.stage: s.duration_ms for s in self.stages}

    @property
    def slowest_stage(self) -> str:
        if not self.stages:
            return "none"
        return max(self.stages, key=lambda s: s.duration_ms).stage

    @property
    def all_succeeded(self) -> bool:
        return all(s.success for s in self.stages)


@contextmanager
def timed_stage(stage_name: str, results: list[StageLatency]):
    """Context manager to time a pipeline stage."""
    start = time.perf_counter()
    stage = StageLatency(stage=stage_name, duration_ms=0.0)
    try:
        yield stage
        stage.duration_ms = round((time.perf_counter() - start) * 1000, 2)
    except Exception as e:
        stage.duration_ms = round((time.perf_counter() - start) * 1000, 2)
        stage.success = False
        stage.error = str(e)
    finally:
        results.append(stage)


def compute_latency_stats(latencies: list[PipelineLatency]) -> dict:
    """Compute aggregate latency statistics across multiple queries."""
    if not latencies:
        return {}

    totals = [l.total_ms for l in latencies]

    # Per-stage aggregation
    stage_times: dict[str, list[float]] = {}
    for lat in latencies:
        for s in lat.stages:
            stage_times.setdefault(s.stage, []).append(s.duration_ms)

    stage_stats = {}
    for stage, times in stage_times.items():
        stage_stats[stage] = {
            "mean_ms": round(sum(times) / len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "p50_ms": round(sorted(times)[len(times) // 2], 2),
            "p95_ms": round(sorted(times)[int(len(times) * 0.95)], 2) if len(times) >= 2 else round(max(times), 2),
        }

    return {
        "total": {
            "mean_ms": round(sum(totals) / len(totals), 2),
            "min_ms": round(min(totals), 2),
            "max_ms": round(max(totals), 2),
            "p50_ms": round(sorted(totals)[len(totals) // 2], 2),
            "p95_ms": round(sorted(totals)[int(len(totals) * 0.95)], 2) if len(totals) >= 2 else round(max(totals), 2),
        },
        "stages": stage_stats,
        "sample_count": len(latencies),
        "success_rate": round(sum(1 for l in latencies if l.all_succeeded) / len(latencies), 4),
    }
