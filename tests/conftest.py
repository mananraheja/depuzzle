import pytest

from depuzzle.models import (
    BackendInfo,
    Device,
    ExecutionConfig,
    Lifecycle,
    ProfileConfig,
)


@pytest.fixture
def fake_trace():

    class FakeTrace:
        model = "test-model"
        prompt = "say hello"

        tokens = ["hello", "world", "!"]

        total_latency = 1.5
        time_to_first_token = 0.2

        lifecycle = Lifecycle.HOT
        execution = ExecutionConfig(device=Device.CPU)

        backend_info = None

    return FakeTrace()


@pytest.fixture
def profile_config():
    return ProfileConfig(
        lifecycle=Lifecycle.HOT,
        execution=ExecutionConfig(
            device=Device.CPU,
        ),
    )


@pytest.fixture
def fake_profiler(monkeypatch, fake_trace, profile_config):

    class FakeProfiler:
        def __init__(self, backend, config):
            self.backend = backend
            self.config = config

        def run(self, prompt):
            return fake_trace

    monkeypatch.setattr(
        "depuzzle.mcp_server.tools.Profiler",
        FakeProfiler,
    )

    return FakeProfiler


@pytest.fixture
def fake_backend():

    class FakeBackend:
        def __init__(self, model="fake-model"):
            self.model = model
            self.last_runtime_stats = None
            self.prepare_calls = 0
            self.unload_calls = 0
            self.generate_calls = 0

        def generate(self, prompt, execution_config=None):
            self.generate_calls += 1
            yield "hello"
            yield "world"

        def get_info(self):
            return BackendInfo(
                backend="fake",
                processor="CPU",
                context_length=1024,
            )

        def prepare(self):
            self.prepare_calls += 1

        def unload(self):
            self.unload_calls += 1

    return FakeBackend()
