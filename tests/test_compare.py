from datetime import datetime

from llmprof.models import (
    InferenceTrace,
    TokenEvent,
)

from llmprof.metrics import Metrics


def create_trace(
    model,
    latency,
    ttft,
    token_count,
):

    tokens = []

    for i in range(token_count):
        tokens.append(
            TokenEvent(
                token=f"token-{i}",
                timestamp=datetime.now(),
            )
        )

    return InferenceTrace(
        model=model,
        prompt="test prompt",
        start_time=datetime.now(),
        end_time=datetime.now(),
        tokens=tokens,
        total_latency=latency,
        time_to_first_token=ttft,
    )


def test_compare_metrics():

    trace1 = create_trace(
        model="qwen2.5:3b",
        latency=5.0,
        ttft=0.5,
        token_count=100,
    )

    trace2 = create_trace(
        model="llama3.2:3b",
        latency=6.0,
        ttft=0.8,
        token_count=100,
    )


    summary1 = Metrics(trace1).summary()
    summary2 = Metrics(trace2).summary()


    assert summary1["model"] == "qwen2.5:3b"
    assert summary2["model"] == "llama3.2:3b"

    assert summary1["latency_seconds"] < summary2["latency_seconds"]

    assert summary1["ttft_seconds"] < summary2["ttft_seconds"]