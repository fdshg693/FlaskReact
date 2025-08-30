from typing import List
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from document_search_tools import (
    search_headwaters_company_info,
    search_local_text_documents,
    get_local_document_content,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

tools: List[BaseTool] = [
    search_headwaters_company_info,
    search_local_text_documents,
    get_local_document_content,
]
model: BaseChatModel = init_chat_model("gpt-4", model_provider="openai")
agent_executor: Runnable = create_react_agent(model, tools)


def main() -> None:
    """Main function to execute the agent."""
    config: dict = {"configurable": {"thread_id": "abc123"}}
    input_message: HumanMessage = HumanMessage(
        content="株式会社ヘッドウォーターズの作ったアプリについて教えてください。"
    )
    for step in agent_executor.stream(
        {"messages": [input_message]}, config, stream_mode="values"
    ):
        step["messages"][-1].pretty_print()


if __name__ == "__main__":
    main()
