import asyncio
import subprocess
import time
from pathlib import Path

import mcp as default_mcp
from fastmcp import Client
from fastmcp.client.client import CallToolResult

client: Client = Client("http://localhost:8080/mcp")

# サブプロセスでMCPサーバーを起動
# サブプロセスのCWDには、このスクリプト自身の親ディレクトリを指定すること
parent_dir = Path(__file__).parent

# TODO: サブプロセスの管理・MCPサーバーのリソース解放処理などを追加する
# uv run simple_server.py
subprocess.Popen(
    [
        "uv",
        "run",
        "simple_server.py",
    ],
    cwd=parent_dir,
)

time.sleep(5)  # サーバーが起動するまで待機


async def main():
    async with client:
        # 疎通確認
        is_ok: bool = await client.ping()
        if not is_ok:
            print("サーバーとの接続に失敗しました")
            return

        # クライアントが使えるツール・リソース・プロンプトの一覧を取得

        print("=" * 30)
        tools: list[default_mcp.types.Tool] = await client.list_tools()
        for tool in tools:
            print("Tool:", tool.name, tool.description)

        print("=" * 30)
        resources: list[default_mcp.types.Resource] = await client.list_resources()
        for resource in resources:
            print("Resource:", resource.name, resource.description)

        print("=" * 30)
        prompts: list[default_mcp.types.Prompt] = await client.list_prompts()
        for prompt in prompts:
            print("Prompt:", prompt.name, prompt.description)

        print("=" * 30)
        result: CallToolResult = await client.call_tool("add", {"a": 1, "b": 2})
        print(result.content[0].text)  # 3


asyncio.run(main())
