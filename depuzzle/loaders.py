import json
import math
from datetime import datetime
from pathlib import Path

from depuzzle.models import (
    BackendInfo,
    Device,
    ExecutionConfig,
    InferenceTrace,
    Lifecycle,
    RuntimeStats,
    TokenEvent,
)

TRACE_SCHEMA_VERSION = 1


def validate_trace_schema(data: dict, path: str):
    if not isinstance(data, dict):
        raise ValueError(f"Invalid trace format: expected a JSON object in {path}")

    schema_version = data.get("schema_version", TRACE_SCHEMA_VERSION)
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != TRACE_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported trace schema version {schema_version!r} in {path}"
        )

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

    for index, token in enumerate(data["tokens"]):
        if (
            not isinstance(token, dict)
            or "token" not in token
            or "timestamp" not in token
        ):
            raise ValueError(
                f"Invalid trace format: token at index {index} must contain "
                "'token' and 'timestamp'"
            )


def _load_optional_models(data: dict):
    lifecycle = None
    if data.get("lifecycle") is not None:
        try:
            lifecycle = Lifecycle(data["lifecycle"])
        except ValueError as exc:
            raise ValueError("Invalid trace format: invalid lifecycle") from exc

    execution = None
    if data.get("execution") is not None:
        execution_data = data["execution"]
        if not isinstance(execution_data, dict) or "device" not in execution_data:
            raise ValueError("Invalid trace format: execution must contain device")
        try:
            execution = ExecutionConfig(
                device=Device(execution_data["device"]),
                gpu_layers=execution_data.get("gpu_layers"),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Invalid trace format: invalid execution configuration"
            ) from exc

    runtime_stats = None
    if data.get("runtime_stats") is not None:
        stats_data = data["runtime_stats"]
        if not isinstance(stats_data, dict):
            raise ValueError("Invalid trace format: runtime_stats must be an object")
        duration_fields = (
            "load_duration",
            "prompt_eval_duration",
            "eval_duration",
        )
        count_fields = ("prompt_eval_count", "eval_count")
        for field in duration_fields:
            value = stats_data.get(field)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or (isinstance(value, float) and not math.isfinite(value))
                or value < 0
            ):
                raise ValueError(
                    f"Invalid trace format: runtime_stats.{field} "
                    "must be a finite non-negative number"
                )
        for field in count_fields:
            value = stats_data.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(
                    f"Invalid trace format: runtime_stats.{field} "
                    "must be a non-negative integer"
                )
        try:
            runtime_stats = RuntimeStats(**stats_data)
        except TypeError as exc:
            raise ValueError(
                "Invalid trace format: invalid runtime_stats fields"
            ) from exc

    backend_info = None
    if data.get("backend_info") is not None:
        info_data = data["backend_info"]
        if not isinstance(info_data, dict):
            raise ValueError("Invalid trace format: backend_info must be an object")
        try:
            backend_info = BackendInfo(
                backend=info_data["backend"],
                processor=info_data["processor"],
                context_length=info_data["context_length"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "Invalid trace format: invalid backend_info fields"
            ) from exc

    return lifecycle, execution, runtime_stats, backend_info


def load_trace(path: str | Path) -> InferenceTrace:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Trace file is empty: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON trace: {path}") from exc

    validate_trace_schema(data, str(path))

    try:
        start_time = datetime.fromisoformat(data["start_time"])
        end_time = datetime.fromisoformat(data["end_time"])
        tokens = [
            TokenEvent(
                token=token["token"],
                timestamp=datetime.fromisoformat(token["timestamp"]),
            )
            for token in data["tokens"]
        ]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Invalid trace format: invalid timestamp or token event"
        ) from exc

    lifecycle, execution, runtime_stats, backend_info = _load_optional_models(data)

    return InferenceTrace(
        model=data["model"],
        prompt=data["prompt"],
        start_time=start_time,
        end_time=end_time,
        tokens=tokens,
        total_latency=data["total_latency"],
        time_to_first_token=data["time_to_first_token"],
        lifecycle=lifecycle,
        execution=execution,
        runtime_stats=runtime_stats,
        backend_info=backend_info,
    )
