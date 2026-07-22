import json
from dataclasses import asdict

from llmprof.models import InferenceTrace


def save_trace(
    trace: InferenceTrace,
    filename: str,
):
    data = asdict(trace)

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
            default=str,
        )