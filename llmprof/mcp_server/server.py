from fastmcp import FastMCP
from .tools import profile_model


mcp = FastMCP("llmprof")


mcp.tool()(profile_model)


if __name__ == "__main__":
    mcp.run()