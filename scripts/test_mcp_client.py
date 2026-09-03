import asyncio

from fastmcp import Client

from depuzzle.mcp_server.server import mcp


async def main():

    async with Client(mcp) as client:
        print("Connected!")

        tools = await client.list_tools()

        print("Available tools:")

        for tool in tools:
            print("- ", tool.name)

        result = await client.call_tool(
            "profile_model",
            {
                "model": "llama3.2:3b",
                "prompt": "hello",
            },
        )

        print("\nResult: ")
        print(result.structured_content)


if __name__ == "__main__":
    asyncio.run(main())
