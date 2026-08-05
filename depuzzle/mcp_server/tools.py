from depuzzle.backends.ollama import OllamaBackend
from depuzzle.metrics import Metrics
from depuzzle.profiler import Profiler


def profile_model(model: str, prompt: str):
    backend = OllamaBackend(model)
    profiler = Profiler(backend)

    trace = profiler.run(prompt)

    metrics = Metrics(trace)

    return metrics.summary()
