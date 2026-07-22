from datetime import datetime

from llmprof.models import (
    TokenEvent,
    InferenceTrace,
)

from llmprof.metrics import Metrics


def create_trace():

    return InferenceTrace(
        model="qwen2.5:3b",
        prompt="hello",
        start_time=datetime.now(),
        end_time=datetime.now(),
        tokens=[
            TokenEvent(
                token="hello",
                timestamp=datetime.now(),
            ),
            TokenEvent(
                token="world",
                timestamp=datetime.now(),
            ),
        ],
        total_latency=2.0,
        time_to_first_token=0.5,
    )


def test_token_count():

    trace = create_trace()

    metrics = Metrics(trace)

    assert metrics.token_count() == 2


def test_latency():

    trace = create_trace()

    metrics = Metrics(trace)

    assert metrics.latency_seconds() == 2.0


def test_tokens_per_second():

    trace = create_trace()

    metrics = Metrics(trace)

    assert metrics.tokens_per_second() == 1.0
