"""Small YAAD MCP client for setup sessions before Codex is restarted.

The token is read only from YAAD_TOKEN and is never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


URL = "https://www.yaadops.com/api/mcp"


async def run(tool_name: str | None, arguments: dict) -> None:
    token = os.environ.get("YAAD_TOKEN")
    if not token or not token.startswith("yaad_pat_"):
        raise SystemExit("YAAD_TOKEN is missing or malformed")
    headers = {"Authorization": f"Bearer {token}"}
    async with streamablehttp_client(URL, headers=headers) as streams:
        read, write = streams[:2]
        async with ClientSession(read, write) as session:
            await session.initialize()
            if tool_name is None:
                tools = await session.list_tools()
                print(
                    json.dumps(
                        [
                            {
                                "name": tool.name,
                                "description": tool.description,
                                "inputSchema": tool.inputSchema,
                            }
                            for tool in tools.tools
                        ],
                        indent=2,
                    )
                )
                return
            result = await session.call_tool(tool_name, arguments)
            payload = {
                "isError": result.isError,
                "structuredContent": result.structuredContent,
                "content": [
                    block.text for block in result.content if getattr(block, "type", None) == "text"
                ],
            }
            print(json.dumps(payload, indent=2))
            if result.isError:
                raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", nargs="?", help="tool name; omit to list tools and schemas")
    parser.add_argument("--arguments", default="{}", help="JSON object")
    parser.add_argument("--arguments-file", help="path to a JSON object; takes precedence")
    args = parser.parse_args()
    if args.arguments_file:
        with open(args.arguments_file, encoding="utf-8") as handle:
            arguments = json.load(handle)
    else:
        arguments = json.loads(args.arguments)
    if not isinstance(arguments, dict):
        raise SystemExit("--arguments must decode to a JSON object")
    asyncio.run(run(args.tool, arguments))


if __name__ == "__main__":
    main()
