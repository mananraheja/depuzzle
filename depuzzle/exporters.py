import json
from dataclasses import asdict

from depuzzle.loaders import TRACE_SCHEMA_VERSION
from depuzzle.models import InferenceTrace


def save_trace(
    trace: InferenceTrace,
    filename: str,
):
    data = asdict(trace)
    data["schema_version"] = TRACE_SCHEMA_VERSION

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
