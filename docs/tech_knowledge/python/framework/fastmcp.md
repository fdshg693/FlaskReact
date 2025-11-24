# llm/mcp_customで利用しているFASTMCPについて

## FASTMCP CLI
Pythonスクリプトとしてでなく、直接CLIからMCPサーバーを起動することが可能
```bash
# デフォルトでは、STDIOモードで起動
fastmcp run my_server.py:mcp
# 明示的にHTTPモードで起動
fastmcp run my_server.py:mcp --transport http --port 8000