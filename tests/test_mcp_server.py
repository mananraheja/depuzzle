import pytest

from llmprof.mcp_server.server import mcp


@pytest.mark.asyncio
async def test_profile_model_registered():

    tools = await mcp.list_tools()

    tool_names = [tool.name for tool in tools]

    assert "profile_model" in tool_names


@pytest.mark.asyncio
async def test_profile_model_call(monkeypatch):

    class FakeTrace:

        model = "test-model"

        tokens = [
            "hello",
            "world",
        ]

        total_latency = 1.0
        time_to_first_token = 0.1

        backend_info = None

    class FakeProfiler:

        def __init__(self, backend):
            pass

        def run(self, prompt):
            return FakeTrace()

    monkeypatch.setattr("llmprof.mcp_server.tools.Profiler", FakeProfiler)

    result = await mcp.call_tool(
        "profile_model",
        {
            "model": "test-model",
            "prompt": "hello",
        },
    )

    assert result is not None
