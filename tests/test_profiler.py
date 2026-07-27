from llmprof.backends.base import BackendInfo
from llmprof.profiler import Profiler


class FakeBackend:

    model = "fake-model"

    def generate(self, prompt):

        yield "Hello"
        yield " world"

    def get_info(self):
        return BackendInfo(
            backend="fake",
            processor="72%/28% CPU/GPU",
            context_length=4096,
        )


def test_profiler_creates_trace():

    backend = FakeBackend()

    profiler = Profiler(backend)

    trace = profiler.run("Say hello")

    assert trace.model == "fake-model"

    assert trace.prompt == "Say hello"

    assert len(trace.tokens) == 2


def test_profiler_records_latency():

    backend = FakeBackend()

    profiler = Profiler(backend)

    trace = profiler.run("Test")

    assert trace.total_latency >= 0


def test_first_token_latency_exists():

    backend = FakeBackend()

    profiler = Profiler(backend)

    trace = profiler.run("Test")

    assert trace.time_to_first_token is not None
