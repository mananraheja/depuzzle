from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class Lifecycle(StrEnum):
    COLD = "cold"
    WARMUP = "warmup"
    HOT = "hot"


class Device(StrEnum):
    CPU = "cpu"
    GPU = "gpu"
    HYBRID = "hybrid"


@dataclass
class ExecutionConfig:
    device: Device
    gpu_layers: int | None = None


@dataclass
class ProfileConfig:
    lifecycle: Lifecycle
    execution: ExecutionConfig


@dataclass
class TokenEvent:
    token: str
    timestamp: datetime


@dataclass
class BackendInfo:
    backend: str
    processor: str
    context_length: int


@dataclass
class RuntimeStats:
    load_duration: float | None = None
    prompt_eval_duration: float | None = None
    prompt_eval_count: int | None = None
    eval_duration: float | None = None
    eval_count: int | None = None


@dataclass
class InferenceTrace:
    model: str
    prompt: str

    start_time: datetime
    end_time: datetime

    tokens: list[TokenEvent]

    total_latency: float
    time_to_first_token: float | None

    lifecycle: Lifecycle
    execution: ExecutionConfig

    runtime_stats: RuntimeStats | None = None
    backend_info: BackendInfo | None = None


@dataclass
class InferenceResult:
    model: str
    prompt: str

    token_count: int

    total_latency: float
    time_to_first_token: float | None

    tokens_per_second: float
