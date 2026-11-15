import json
from datetime import UTC, datetime

import pytest

from depuzzle.exporters import save_trace
from depuzzle.loaders import load_trace
from depuzzle.models import (
    BackendInfo,
    Device,
    ExecutionConfig,
    InferenceTrace,
    Lifecycle,
    RuntimeStats,
    TokenEvent,
)


def test_load_valid_trace(tmp_path):
    trace_data = {
        "model": "qwen2.5:3b",
        "prompt": "Explain virtual memory.",
        "start_time": "2026-07-22T10:00:00",
        "end_time": "2026-07-22T10:00:05",
        "total_latency": 5.0,
        "time_to_first_token": 0.5,
        "tokens": [
            {"token": "Hello", "timestamp": "2026-07-22T10:00:01"},
            {"token": "world", "timestamp": "2026-07-22T10:00:02"},
        ],
    }

    file_path = tmp_path / "trace.json"
    file_path.write_text(json.dumps(trace_data), encoding="utf-8")

    trace = load_trace(file_path)

    assert trace.model == "qwen2.5:3b"
    assert len(trace.tokens) == 2
    assert trace.tokens[0].token == "Hello"
    assert trace.total_latency == 5.0
    assert trace.lifecycle is None
    assert trace.execution is None
    assert trace.runtime_stats is None


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_trace("does_not_exist.json")


def test_empty_trace_file(tmp_path):
    trace = tmp_path / "empty.json"
    trace.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_trace(trace)


def test_invalid_json(tmp_path):
    trace = tmp_path / "invalid.json"
    trace.write_text('{"latency": 123,,}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_trace(trace)


def test_missing_required_field(tmp_path):
    trace = tmp_path / "missing_field.json"
    data = {
        "prompt": "Hello",
        "start_time": "2026-07-22T10:00:00",
        "end_time": "2026-07-22T10:00:01",
        "tokens": [],
        "total_latency": 1.0,
        "time_to_first_token": 0.2,
    }
    trace.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid trace format"):
        load_trace(trace)


def test_wrong_field_type(tmp_path):
    trace = tmp_path / "wrong_type.json"
    data = {
        "model": 123,
        "prompt": "Hello",
        "start_time": "2026-07-22T10:00:00",
        "end_time": "2026-07-22T10:00:01",
        "tokens": [],
        "total_latency": 1.0,
        "time_to_first_token": 0.2,
    }
    trace.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid trace format"):
        load_trace(trace)


def test_invalid_timestamp(tmp_path):
    trace = tmp_path / "invalid_timestamp.json"
    data = {
        "model": "llama3",
        "prompt": "Hello",
        "start_time": "not-a-timestamp",
        "end_time": "2026-07-22T10:00:01",
        "tokens": [],
        "total_latency": 1.0,
        "time_to_first_token": 0.2,
    }
    trace.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid trace format"):
        load_trace(trace)


def test_round_trip_preserves_typed_runtime_data(tmp_path):
    now = datetime(2026, 11, 15, tzinfo=UTC)
    original = InferenceTrace(
        model="llama3.2:3b",
        prompt="Measure this",
        start_time=now,
        end_time=now,
        tokens=[TokenEvent(token="hello", timestamp=now)],
        total_latency=2.5,
        time_to_first_token=0.3,
        lifecycle=Lifecycle.WARMUP,
        execution=ExecutionConfig(device=Device.HYBRID, gpu_layers=12),
        runtime_stats=RuntimeStats(
            load_duration=100_000_000,
            prompt_eval_duration=200_000_000,
            prompt_eval_count=42,
            eval_duration=300_000_000,
            eval_count=17,
        ),
        backend_info=BackendInfo(
            backend="ollama",
            processor="72%/28% CPU/GPU",
            context_length=8192,
        ),
    )

    path = tmp_path / "round-trip.json"
    save_trace(original, str(path))
    loaded = load_trace(path)

    assert loaded.lifecycle == Lifecycle.WARMUP
    assert loaded.execution == ExecutionConfig(device=Device.HYBRID, gpu_layers=12)
    assert loaded.runtime_stats == original.runtime_stats
    assert loaded.backend_info == original.backend_info
    assert loaded.tokens == original.tokens
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == 1


def test_unknown_schema_version_is_rejected(tmp_path):
    path = tmp_path / "future-version.json"
    data = {
        "schema_version": 2,
        "model": "test",
        "prompt": "",
        "start_time": "2026-07-22T10:00:00",
        "end_time": "2026-07-22T10:00:01",
        "tokens": [],
        "total_latency": 1.0,
        "time_to_first_token": None,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported trace schema version"):
        load_trace(path)


def test_invalid_optional_configuration_is_rejected(tmp_path):
    path = tmp_path / "invalid-execution.json"
    data = {
        "model": "test",
        "prompt": "",
        "start_time": "2026-07-22T10:00:00",
        "end_time": "2026-07-22T10:00:01",
        "tokens": [],
        "total_latency": 1.0,
        "time_to_first_token": None,
        "execution": {"device": "accelerator"},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid execution"):
        load_trace(path)



def test_boolean_schema_version_is_rejected(tmp_path):
    path = tmp_path / "boolean-schema-version.json"
    data = {
        "schema_version": True,
        "model": "test",
        "prompt": "",
        "start_time": "2026-07-22T10:00:00",
        "end_time": "2026-07-22T10:00:01",
        "tokens": [],
        "total_latency": 1.0,
        "time_to_first_token": None,
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported trace schema version"):
        load_trace(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("load_duration", "100"),
        ("prompt_eval_duration", True),
        ("eval_duration", float("inf")),
        ("load_duration", -1),
        ("prompt_eval_duration", float("nan")),
    ],
)
def test_invalid_runtime_duration_is_rejected(tmp_path, field, value):
    path = tmp_path / "invalid-runtime-duration.json"
    data = {
        "model": "test",
        "prompt": "",
        "start_time": "2026-07-22T10:00:00",
        "end_time": "2026-07-22T10:00:01",
        "tokens": [],
        "total_latency": 1.0,
        "time_to_first_token": None,
        "runtime_stats": {field: value},
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=f"runtime_stats\\.{field}"):
        load_trace(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_eval_count", 1.5),
        ("eval_count", "17"),
        ("prompt_eval_count", True),
        ("eval_count", -1),
    ],
)
def test_invalid_runtime_count_is_rejected(tmp_path, field, value):
    path = tmp_path / "invalid-runtime-count.json"
    data = {
        "model": "test",
        "prompt": "",
        "start_time": "2026-07-22T10:00:00",
        "end_time": "2026-07-22T10:00:01",
        "tokens": [],
        "total_latency": 1.0,
        "time_to_first_token": None,
        "runtime_stats": {field: value},
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match=f"runtime_stats\\.{field}"):
        load_trace(path)
