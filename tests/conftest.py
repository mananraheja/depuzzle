import pytest


@pytest.fixture
def fake_trace():

    class FakeTrace:
        model = "test-model"
        prompt = "say hello"
        
        tokens = [
            "hello",
            "world",
            "!"
        ]

        total_latency = 1.5
        time_to_first_token = 0.2

        backend_info = None
    
    return FakeTrace()


@pytest.fixture
def fake_profiler(monkeypatch, fake_trace):
    
    class FakeProfiler:

        def __init__(self, backend):
            self.backend = backend
        
        def run(self, prompt):
            return fake_trace
    
    monkeypatch.setattr(
        "llmprof.mcp_server.tools.Profiler",
        FakeProfiler,
    )

    return FakeProfiler


@pytest.fixture
def fake_backend():

    class FakeBackend:

        def __init__(self, model="test-model"):
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