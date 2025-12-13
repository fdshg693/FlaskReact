from typing import List

from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from langgraph.graph.state import CompiledStateGraph
from typing import Any

from config import load_dotenv_workspace
from llm.langchain_custom.tools.search.local_document_search import (
    create_search_local_text_tool,
)
from llm.langchain_custom.models import LLMModel

load_dotenv_workspace()


def function_calling(
    tools: List[BaseTool],
    query: str,
    llm_model: LLMModel = LLMModel.GPT_4O_MINI,
) -> str:
    """
    Demonstrates how to use function calling with an LLM.

    Args:
        tools (List[BaseTool]): List of tools to be used by the agent.
        query (str): The user query to be processed.
    Returns:
        str: The result from the agent after processing the query.
    """
    agent: CompiledStateGraph[Any] = create_agent(
        model=llm_model,
        tools=tools,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": query}]},
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    text_search_tool: BaseTool = create_search_local_text_tool()
    tools: List[BaseTool] = [text_search_tool]
    search_query: str = "名前順にテキストファイルを並べてください。"
    function_result = function_calling(tools, search_query)
    print("\n=== FINAL RESULT ===")
    print(function_result)
