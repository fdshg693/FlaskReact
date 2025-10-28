from typing import List

from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from loguru import logger

from config import load_dotenv_workspace
from llm.tools.search.local_document_search import create_search_local_text_tool
from llm.models import LLMModel

load_dotenv_workspace()


def function_calling(tools: List[BaseTool], query: str, verbose: bool = True) -> str:
    """
    Demonstrates how to use function calling with an LLM.

    Args:
        tools: List of LangChain tools to bind to the LLM
        query: The user query to process
    verbose: If True, prints/logs step-by-step tool calling process

    Returns:
        The AI's response content as a string
    """
    agent = create_agent(
        model=LLMModel.GPT_4O_MINI,
        tools=tools,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
    )
    return result


if __name__ == "__main__":
    tools: List[BaseTool] = [create_search_local_text_tool]
    search_query: str = "株式会社ヘッドウォーターズの作ったアプリについて、ローカルにあるドキュメントを元に回答してください。"
    # Configure log format for clarity when run as a script
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{message}",
    )
    function_result: str = function_calling(tools, search_query, verbose=True)
    print("\n=== FINAL RESULT ===")
    print(function_result)
