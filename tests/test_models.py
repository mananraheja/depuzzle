from datetime import datetime

from llmprof.models import (
    TokenEvent,
    InferenceTrace,
)


def test_token_event_creation():

    event = TokenEvent(
        token="hello",
        timestamp=datetime.now(),
    )

    assert event.token == "hello"
    assert event.timestamp is not None


def test_inference_trace_creation():

    trace = InferenceTrace(
        model="qwen2.5:3b",
        prompt="Explain CPU cache",
        start_time=datetime.now(),
        end_time=datetime.now(),
        tokens=[],
        total_latency=1.5,
        time_to_first_token=0.2,
    )

    assert trace.model == "qwen2.5:3b"
    assert trace.prompt == "Explain CPU cache"
    assert trace.total_latency == 1.5
    assert len(trace.tokens) == 0