from datetime import UTC, datetime, timedelta

import pytest

from llmprof.metrics import Metrics
from llmprof.models import InferenceTrace, TokenEvent


@pytest.fixture
def sample_trace():
    start = datetime.fromisoformat("2026-07-22T10:00:00")

    tokens = [
        TokenEvent(
            token="Hello",
            timestamp=start + timedelta(seconds=1),
        ),
        TokenEvent(
            token="world",
            timestamp=start + timedelta(seconds=2),
        ),
        TokenEvent(
            token="!",
            timestamp=start + timedelta(seconds=3),
        ),
    ]

    return InferenceTrace(
        model="test-model",
        prompt="Say hello",
        start_time=start,
        end_time=start + timedelta(seconds=10),
        tokens=tokens,
        total_latency=10.0,
        time_to_first_token=1.0,
    )


def test_token_count(sample_trace):
    metrics = Metrics(sample_trace)

    assert metrics.token_count() == 3


def test_latency_seconds(sample_trace):
    metrics = Metrics(sample_trace)

    assert metrics.latency_seconds() == 10.0


def test_ttft_seconds(sample_trace):
    metrics = Metrics(sample_trace)

    assert metrics.ttft_seconds() == 1.0


def test_tokens_per_second(sample_trace):
    metrics = Metrics(sample_trace)

    assert metrics.tokens_per_second() == pytest.approx(0.3)


def test_summary(sample_trace):
    metrics = Metrics(sample_trace)

    assert metrics.summary() == {
        "model": "test-model",
        "tokens": 3,
        "latency_seconds": 10.0,
        "ttft_seconds": 1.0,
        "tokens_per_second": pytest.approx(0.3),
    }


def test_tokens_per_second_zero_latency():
    trace = InferenceTrace(
        model="test-model",
        prompt="hello",
        start_time=datetime.now(UTC),
        end_time=datetime.now(UTC),
        tokens=[
            TokenEvent(
                token="hello",
                timestamp=datetime.now(UTC),
            )
        ],
        total_latency=0,
        time_to_first_token=0,
    )

    metrics = Metrics(trace)

    assert metrics.tokens_per_second() == 0
