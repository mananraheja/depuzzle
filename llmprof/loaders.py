import json
import os
from datetime import datetime
from pathlib import Path

from llmprof.models import (
    InferenceTrace,
    TokenEvent,
)


def load_trace(path: str) -> InferenceTrace:

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")

    if Path(path).stat().st_size == 0:
        raise ValueError(f"Trace file is empty: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Trace file not found: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON trace: {path}") from e

    try:
        tokens = [
            TokenEvent(
                token=token["token"],
                timestamp=datetime.fromisoformat(token["timestamp"]),
            )
            for token in data["tokens"]
        ]

        return InferenceTrace(
            model=data["model"],
            prompt=data["prompt"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            tokens=tokens,
            total_latency=data["total_latency"],
            time_to_first_token=data["time_to_first_token"],
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid trace format: {path}") from e
