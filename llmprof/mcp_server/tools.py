from llmprof.backends.ollama import OllamaBackend
from llmprof.profiler import Profiler
from llmprof.metrics import Metrics


def profile_model(model: str, prompt: str):
    backend = OllamaBackend(model)
    profiler = Profiler(backend)

    trace = profiler.run(prompt)

    metrics = Metrics(trace)

    return metrics.summary()