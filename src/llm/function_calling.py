from typing import List, Dict, Any
from loguru import logger
from langchain.chat_models import init_chat_model
from config import load_dotenv_workspace
from langchain_core.messages import HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import BaseTool

from llm.document_search_tools import (
    get_local_document_content,
    search_local_text_documents,
)

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
    if verbose:
        logger.info("Initializing chat model with tools ...")
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    llm_with_tools = llm.bind_tools(tools)

    messages: List[BaseMessage] = [HumanMessage(query)]
    tools_dict: Dict[str, BaseTool] = {tool.name.lower(): tool for tool in tools}

    step: int = 0
    while True:
        step += 1
        ai_msg: BaseMessage = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if verbose:
            logger.info(f"[STEP {step}] AI response received.")
            logger.debug(f"AI raw message: {ai_msg}")

        # Extract potential tool calls
        tool_calls: Any = getattr(ai_msg, "tool_calls", None)
        if not tool_calls:
            if verbose:
                logger.info("No further tool calls. Exiting loop.")
            break

        for idx, tool_call in enumerate(tool_calls, start=1):
            name: str = tool_call.get("name", "<unknown>")
            call_id: str = tool_call.get("id", f"step{step}_call{idx}")
            args: Dict[str, Any] = tool_call.get("args", {})
            selected_tool: BaseTool = tools_dict.get(name.lower())
            if selected_tool is None:
                err_msg = f"Tool '{name}' not found in provided tools. Skipping."
                if verbose:
                    logger.error(err_msg)
                continue
            if verbose:
                logger.info(
                    f"Invoking tool [{name}] (call id={call_id}) with args={args}"  # noqa: E501
                )
            try:
                tool_output: str = selected_tool.invoke(args)
            except Exception as exc:  # noqa: BLE001
                tool_output = f"<ERROR: {exc}>"
                if verbose:
                    logger.exception(
                        f"Tool '{name}' raised an exception: {exc}"  # noqa: E501
                    )
            if verbose:
                truncated = (
                    tool_output
                    if len(str(tool_output)) < 500
                    else str(tool_output)[:500] + "...<truncated>"
                )
                logger.info(
                    f"Tool [{name}] output (len={len(str(tool_output))}): {truncated}"
                )
            messages.append(ToolMessage(str(tool_output), tool_call_id=call_id))

    # Ensure we return a string
    content = ai_msg.content
    if verbose:
        logger.info("Final AI content prepared for return.")
    if isinstance(content, list):
        # If content is a list, join it into a string
        return " ".join(str(item) for item in content)
    return str(content) if content is not None else ""


if __name__ == "__main__":
    tools: List[BaseTool] = [get_local_document_content, search_local_text_documents]
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
