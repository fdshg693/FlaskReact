# LangGraph React Agent for Document Search and Company Information Retrieval
#
# Primary Purpose: Creates and executes a conversational AI agent that can search for information
# about Headwaters Company and local documents using LangGraph's React agent framework
#
# Key Components:
# - Agent executor built with create_react_agent using GPT-4 model
# - Three main tools: search_headwaters_company_info, search_local_text_documents, get_local_document_content
# - Main execution function that processes Japanese language queries about company applications
#
# Dependencies:
# - langgraph: For React agent creation and execution
# - langchain: For chat model initialization and core messaging
# - document_search_tools: Custom module containing search and document retrieval tools
#
# Usage Context: Designed for querying information about Headwaters Company's applications
# and searching through local document repositories in Japanese language context

from typing import List, Dict, Any
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
    response: Dict[str, Any] = agent_executor.invoke(
        {
            "messages": [
                HumanMessage(
                    content="株式会社ヘッドウォーターズの作ったアプリについて教えてください。"
                )
            ]
        }
    )

    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
