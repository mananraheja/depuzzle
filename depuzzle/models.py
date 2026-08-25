from dataclasses import dataclass
from datetime import datetime

from enum import Enum


class Lifecycle(str, Enum):
    COLD = "cold"
    WARMUP = "warmup"
    HOT = "hot"


class Device(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    HYBRID = "hybrid"

@dataclass
class ExecutionConfig(Enum):
    device: Device
    gpu_layers: int | None = None


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

    backend_info: BackendInfo | None = None


@dataclass
class InferenceResult:
    model: str
    prompt: str

    token_count: int

    total_latency: float
    time_to_first_token: float | None

    tokens_per_second: float
