from datetime import datetime

from llmprof.models import TokenEvent, InferenceTrace


def test_token_event_creation():

    timestamp = datetime.now()

    event = TokenEvent(
        token="hello",
        timestamp=timestamp,
    )

    assert event.token == "hello"
    assert event.timestamp == timestamp


def test_inference_trace_creation():

    now = datetime.now()

    trace = InferenceTrace(
        model="qwen2.5:3b",
        prompt="Explain CPU cache",
        start_time=now,
        end_time=now,
        tokens=[
            TokenEvent(
                token="Hello",
                timestamp=now,
            )
        ],
        total_latency=1.5,
        time_to_first_token=0.2,
    )

    assert trace.model == "qwen2.5:3b"
    assert trace.prompt == "Explain CPU cache"
    assert trace.total_latency == 1.5
    assert len(trace.tokens) == 1
    assert trace.time_to_first_token == 0.2


def test_empty_trace():

    now = datetime.now()

    trace = InferenceTrace(
        model="test-model",
        prompt="",
        start_time=now,
        end_time=now,
        tokens=[],
        total_latency=0.0,
        time_to_first_token=None,
    )

    assert len(trace.tokens) == 0
    assert trace.time_to_first_token is None
