import pytest

from depuzzle.models import Device, ExecutionConfig, Lifecycle, ProfileConfig


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

        def generate(self, prompt):
            yield "hello"
            yield "world"

        def get_info(self):
            return {
                "backend": "fake",
                "processor": "CPU",
                "context_length": 1000,
            }

    return FakeBackend()
