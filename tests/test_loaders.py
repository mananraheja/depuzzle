import json
import pytest

from llmprof.loaders import load_trace


def test_load_trace(tmp_path):

    trace_data = {
        "model": "qwen2.5:3b",
        "prompt": "Explain virtual memory.",
        "start_time": "2026-07-22T10:00:00",
        "end_time": "2026-07-22T10:00:05",
        "total_latency": 5.0,
        "time_to_first_token": 0.5,
        "tokens": [
            {
                "token": "Hello",
                "timestamp": "2026-07-22T10:00:01",
            },
            {
                "token": "world",
                "timestamp": "2026-07-22T10:00:02",
            },
        ],
    }

    file_path = tmp_path / "trace.json"

    with open(file_path, "w") as file:
        json.dump(trace_data, file)

    trace = load_trace(
        str(file_path)
    )

    assert trace.model == "qwen2.5:3b"
    assert len(trace.tokens) == 2
    assert trace.tokens[0].token == "Hello"
    assert trace.total_latency == 5.0


def test_load_missing_trace():

    with pytest.raises(FileNotFoundError):
        load_trace("does_not_exist.json")