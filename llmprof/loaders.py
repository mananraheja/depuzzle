import json
import os
from datetime import datetime

from llmprof.models import (
    InferenceTrace,
    TokenEvent,
)


def load_trace(filename: str) -> InferenceTrace:

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"{filename} does not exist."
        )

    with open(filename, "r", encoding="utf-8") as file:
        data = json.load(file)

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
        start_time=datetime.fromisoformat(
            data["start_time"]
        ),
        end_time=datetime.fromisoformat(
            data["end_time"]
        ),
        tokens=tokens,
        total_latency=data["total_latency"],
        time_to_first_token=data["time_to_first_token"],
    )