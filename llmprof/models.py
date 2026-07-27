from dataclasses import dataclass
from datetime import datetime


@dataclass
class TokenEvent:
    token: str
    timestamp: datetime


@dataclass
class BackendInfo:
    backend: str
    cpu_offload_percent: int
    gpu_offload_percent: int
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


@dataclass
class InferenceResult:
    model: str
    prompt: str

    token_count: int

    total_latency: float
    time_to_first_token: float | None

    tokens_per_second: float
