import pytest

from llmprof.mcp_server.server import mcp


@pytest.mark.asyncio
async def test_profile_model_registered():

    tools = await mcp.list_tools()

    tool_names = [tool.name for tool in tools]

    assert "profile_model" in tool_names
