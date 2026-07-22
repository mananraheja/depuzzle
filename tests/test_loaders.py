import json
import pytest

from llmprof.loaders import load_trace


# def test_load_trace(tmp_path):

#     trace_data = {
#         "model": "qwen2.5:3b",
#         "prompt": "Explain virtual memory.",
#         "start_time": "2026-07-22T10:00:00",
#         "end_time": "2026-07-22T10:00:05",
#         "total_latency": 5.0,
#         "time_to_first_token": 0.5,
#         "tokens": [
#             {
#                 "token": "Hello",
#                 "timestamp": "2026-07-22T10:00:01",
#             },
#             {
#                 "token": "world",
#                 "timestamp": "2026-07-22T10:00:02",
#             },
#         ],
#     }

#     file_path = tmp_path / "trace.json"

#     with open(file_path, "w") as file:
#         json.dump(trace_data, file)

#     trace = load_trace(str(file_path))

#     assert trace.model == "qwen2.5:3b"
#     assert len(trace.tokens) == 2
#     assert trace.tokens[0].token == "Hello"
#     assert trace.total_latency == 5.0


# def test_load_missing_trace():

#     with pytest.raises(FileNotFoundError):
#         load_trace("does_not_exist.json")


# def test_invalid_json(tmp_path):
#     trace = tmp_path / "invalid.json"
#     trace.write_text('{"latency": 123,,}', encoding="utf-8")

#     with pytest.raises(ValueError, match="Invalid JSON"):
#         load_trace(trace)


# def test_empty_trace_file(tmp_path):
#     trace = tmp_path / "empty.json"
#     trace.write_text("", encoding="utf-8")

#     with pytest.raises(ValueError, match="empty"):
#         load_trace(trace)


def test_load_valid_trace(tmp_path):
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

    trace = load_trace(str(file_path))

    assert trace.model == "qwen2.5:3b"
    assert len(trace.tokens) == 2
    assert trace.tokens[0].token == "Hello"
    assert trace.total_latency == 5.0


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
        # "model" is missing
    }

    trace.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid trace format"):
        load_trace(trace)


def test_wrong_field_type(tmp_path):
    trace = tmp_path / "wrong_type.json"

    data = {
        "model": 123,  # should be string
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