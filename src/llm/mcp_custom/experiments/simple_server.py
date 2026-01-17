from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

simple_mcp = FastMCP(
    name="SimpleServer",
    instructions="A simple MCP server that adds two numbers.",
)


@simple_mcp.tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@simple_mcp.resource("data://config")
def get_config() -> dict:
    """Provides the application configuration."""
    return {"theme": "dark", "version": "1.0"}


@simple_mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int) -> dict:
    """Retrieves a user's profile by ID."""
    # The {user_id} in the URI is extracted and passed to this function
    return {"id": user_id, "name": f"User {user_id}", "status": "active"}


@simple_mcp.prompt
def analyze_data(data_points: list[float]) -> str:
    """Creates a prompt asking for analysis of numerical data."""
    formatted_data = ", ".join(str(point) for point in data_points)
    return f"Please analyze these data points: {formatted_data}"


@simple_mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    return PlainTextResponse("OK")


if __name__ == "__main__":
    # transport -> Literal["stdio", "http", "sse", "streamable-http"]

    # HTTP 8000番ポートでサーバーを起動
    simple_mcp.run(transport="http", port=8080)
    # STDIOでサーバーを起動
    # simple_mcp.run(transport="stdio")
