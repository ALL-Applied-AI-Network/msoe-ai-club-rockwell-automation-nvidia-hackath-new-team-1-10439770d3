import asyncio
import os
from pathlib import Path
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_stdio_server_lists_and_calls_tools(tmp_path):
    async def scenario():
        env = dict(os.environ)
        env["ATEAM_MVP_ROOT"] = str(Path(__file__).resolve().parents[1])
        env["ATEAM_ROBOT_DB"] = str(tmp_path / "mcp.sqlite3")
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "robot_control.mcp_server"],
            env=env,
        )
        async with stdio_client(parameters) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                names = {tool.name for tool in (await session.list_tools()).tools}
                assert names == {
                    "robot_status",
                    "draft_point_to_point_move",
                    "draft_safe_demo",
                    "cancel_unexecuted_plan",
                }
                result = await session.call_tool("robot_status", {})
                assert result.isError is False

    asyncio.run(scenario())

