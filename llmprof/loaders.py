import json
import os
from datetime import datetime
from pathlib import Path

from llmprof.models import (
    InferenceTrace,
    TokenEvent,
)


def validate_trace_schema(data: dict, path: str):
    required_fields = [
        "model",
        "prompt",
        "start_time",
        "end_time",
        "tokens",
        "total_latency",
        "time_to_first_token",
    ]

    for field in required_fields:
        if field not in data:
            raise ValueError(
                f"Invalid trace format: missing field '{field}' in {path}"
            )

    if not isinstance(data["model"], str):
        raise ValueError(
            f"Invalid trace format: model must be a string"
        )

    if not isinstance(data["prompt"], str):
        raise ValueError(
            f"Invalid trace format: prompt must be a string"
        )

    if not isinstance(data["tokens"], list):
        raise ValueError(
            f"Invalid trace format: tokens must be a list"
        )


def load_trace(path: str) -> InferenceTrace:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Trace file not found: {path}"
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"Trace file is empty: {path}"
        )

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON trace: {path}"
        ) from e
    
    try:
        start_time = datetime.fromisoformat(data["start_time"])
        end_time = datetime.fromisoformat(data["end_time"])

    except ValueError as e:
        raise ValueError(
            f"Invalid trace format: invalid timestamp"
        ) from e

    validate_trace_schema(data, str(path))

    tokens = [
        TokenEvent(
            token=token["token"],
            timestamp=datetime.fromisoformat(
                token["timestamp"]
            ),
        )
        for token in data["tokens"]
    ]

    return InferenceTrace(
        model=data["model"],
        prompt=data["prompt"],
        start_time=start_time,
        end_time=end_time,
        tokens=tokens,
        total_latency=data["total_latency"],
        time_to_first_token=data["time_to_first_token"],
    )