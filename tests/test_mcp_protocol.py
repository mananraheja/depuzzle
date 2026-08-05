import pytest

from depuzzle.mcp_server.server import mcp


@pytest.mark.asyncio
async def test_profile_model_call(fake_profiler):

    result = await mcp.call_tool(
        "profile_model",
        {
            "model": "test-model",
            "prompt": "hello",
        },
    )

    assert result is not None
