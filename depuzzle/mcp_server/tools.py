from depuzzle.backends.ollama import OllamaBackend
from depuzzle.metrics import Metrics
from depuzzle.models import (
    Device,
    ExecutionConfig,
    Lifecycle,
    ProfileConfig,
)
from depuzzle.profiler import Profiler


def profile_model(model: str, prompt: str):
    backend = OllamaBackend(model)

    config = ProfileConfig(
        lifecycle=Lifecycle.HOT,
        execution=ExecutionConfig(device=Device.CPU),
    )

    profiler = Profiler(backend, config)

    trace = profiler.run(prompt)

    metrics = Metrics(trace)

    return metrics.summary()
