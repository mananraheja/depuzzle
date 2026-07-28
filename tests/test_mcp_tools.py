from llmprof.mcp_server.tools import profile_model


def test_profile_model_returns_summary(monkeypatch):

    class FakeTrace:

        model = "test-model"

        tokens = ["hello", "world", "!"]

        total_latency = 1.5
        time_to_first_token = 0.2

        backend_info = None

    class FakeProfiler:

        def __init__(self, backend):
            pass

        def run(self, prompt):
            return FakeTrace()

    monkeypatch.setattr(
        "llmprof.mcp_server.tools.Profiler",
        FakeProfiler,
    )

    result = profile_model(
        model="test-model",
        prompt="hello",
    )

    assert result["model"] == "test-model"
    assert result["tokens"] == 3
    assert result["latency_seconds"] == 1.5
