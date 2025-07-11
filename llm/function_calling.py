from typing import List, Dict
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import BaseTool

# Handle both relative and absolute imports for flexibility
try:
    from .document_search_tools import (
        get_local_document_content,
        search_local_text_documents,
    )
except ImportError:
    from document_search_tools import (
        get_local_document_content,
        search_local_text_documents,
    )

load_dotenv()


def function_calling(tools: List[BaseTool], query: str) -> str:
    """
    Demonstrates how to use function calling with an LLM.

    Args:
        tools: List of LangChain tools to bind to the LLM
        query: The user query to process

    Returns:
        The AI's response content as a string
    """
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    llm_with_tools = llm.bind_tools(tools)

    messages: List[BaseMessage] = [HumanMessage(query)]
    tools_dict: Dict[str, BaseTool] = {tool.name.lower(): tool for tool in tools}

    while True:
        ai_msg: BaseMessage = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        # Check if ai_msg has tool_calls attribute and if it has any tool calls
        tool_calls = getattr(ai_msg, "tool_calls", None)
        if not tool_calls:
            break
        else:
            for tool_call in tool_calls:
                selected_tool: BaseTool = tools_dict[tool_call["name"].lower()]
                tool_output: str = selected_tool.invoke(tool_call["args"])
                messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    # Ensure we return a string
    content = ai_msg.content
    if isinstance(content, list):
        # If content is a list, join it into a string
        return " ".join(str(item) for item in content)
    return str(content) if content is not None else ""


if __name__ == "__main__":
    tools: List[BaseTool] = [get_local_document_content, search_local_text_documents]
    search_query: str = (
        "株式会社ヘッドウォーターズの作ったアプリについて教えてください。"
    )
    function_result: str = function_calling(tools, search_query)
    print(function_result)
