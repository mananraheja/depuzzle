from llmprof.mcp_server.tools import profile_model


def test_profile_model_returns_summary(fake_profiler):

    result = profile_model(
        model="test-model",
        prompt="hello",
    )

    assert result["model"] == "test-model"
    assert result["tokens"] == 3
    assert result["latency_seconds"] == 1.5
