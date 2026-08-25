import json
from datetime import datetime
from pathlib import Path

from depuzzle.models import (
    BackendInfo,
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
            raise ValueError(f"Invalid trace format: missing field '{field}' in {path}")

    if not isinstance(data["model"], str):
        raise ValueError("Invalid trace format: model must be a string")

    if not isinstance(data["prompt"], str):
        raise ValueError("Invalid trace format: prompt must be a string")

    if not isinstance(data["tokens"], list):
        raise ValueError("Invalid trace format: tokens must be a list")


def load_trace(path: str | Path) -> InferenceTrace:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Trace file is empty: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON trace: {path}") from e

    try:
        start_time = datetime.fromisoformat(data["start_time"])
        end_time = datetime.fromisoformat(data["end_time"])

    except ValueError as e:
        raise ValueError("Invalid trace format: invalid timestamp") from e

    backend_info = None

    if data.get("backend_info"):
        backend_info = BackendInfo(
            backend=data["backend_info"]["backend"],
            processor=data["backend_info"]["processor"],
            context_length=data["backend_info"]["context_length"],
        )

    validate_trace_schema(data, str(path))

    tokens = [
        TokenEvent(
            token=token["token"],
            timestamp=datetime.fromisoformat(token["timestamp"]),
        )
        for token in data["tokens"]
    ]

    return InferenceTrace(
        model=data.get("model"),
        prompt=data.get("prompt"),
        start_time=start_time,
        end_time=end_time,
        tokens=tokens,
        total_latency=data.get("total_latency"),
        time_to_first_token=data.get("time_to_first_token"),
        lifecycle=data.get("lifecycle"),
        execution=data.get("execution"),
        backend_info=backend_info,
    )
