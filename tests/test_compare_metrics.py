from datetime import UTC, datetime

from depuzzle.cli import format_change
from depuzzle.metrics import Metrics
from depuzzle.models import (
    Device,
    ExecutionConfig,
    InferenceTrace,
    Lifecycle,
    RuntimeStats,
    TokenEvent,
)


def create_trace(model, latency, ttft, token_count, runtime_stats=None):
    now = datetime(2026, 7, 24, tzinfo=UTC)
    tokens = [
        TokenEvent(token=f"token-{i}", timestamp=now)
        for i in range(token_count)
    ]
    return InferenceTrace(
        model=model,
        prompt="test prompt",
        start_time=now,
        end_time=now,
        tokens=tokens,
        total_latency=latency,
        time_to_first_token=ttft,
        lifecycle=Lifecycle.HOT,
        execution=ExecutionConfig(device=Device.CPU),
        runtime_stats=runtime_stats,
    )


def test_compare_metrics():
    trace1 = create_trace("qwen2.5:3b", 5.0, 0.5, 100)
    trace2 = create_trace("llama3.2:3b", 6.0, 0.8, 100)
    summary1 = Metrics(trace1).summary()
    summary2 = Metrics(trace2).summary()

    assert summary1["model"] == "qwen2.5:3b"
    assert summary2["model"] == "llama3.2:3b"
    assert summary1["latency_seconds"] < summary2["latency_seconds"]
    assert summary1["ttft_seconds"] < summary2["ttft_seconds"]


def test_format_change_handles_missing_values():
    assert format_change("generation_tokens_per_second", 10, None) == "N/A"
    assert format_change("generation_tokens_per_second", None, 10) == "N/A"


def test_runtime_stats_summary_and_comparison_are_available():
    stats1 = RuntimeStats(
        load_duration=100_000_000,
        prompt_eval_duration=200_000_000,
        prompt_eval_count=20,
        eval_duration=400_000_000,
        eval_count=40,
    )
    stats2 = RuntimeStats(
        load_duration=200_000_000,
        prompt_eval_duration=400_000_000,
        prompt_eval_count=20,
        eval_duration=200_000_000,
        eval_count=40,
    )
    summary1 = Metrics(create_trace("a", 1, 0.1, 40, stats1)).summary()
    summary2 = Metrics(create_trace("b", 1, 0.1, 40, stats2)).summary()

    assert summary1["ollama"]["generation_tokens_per_second"] == 100
    assert summary2["ollama"]["generation_tokens_per_second"] == 200
    assert format_change(
        "generation_tokens_per_second",
        summary1["ollama"]["generation_tokens_per_second"],
        summary2["ollama"]["generation_tokens_per_second"],
    ) == "[green]+100.00% (better)[/green]"

    # Missing backend measurements must be displayed as unavailable, not zero.
    assert summary1["ollama"]["load_duration_seconds"] == 0.1
