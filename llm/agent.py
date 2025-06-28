from typing import List, Dict, Any
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from functionTools import (
    searchHeadwatersCompany,
    searchLocalDocuments,
    getLocalDocuments,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

tools: List[BaseTool] = [
    searchHeadwatersCompany,
    searchLocalDocuments,
    getLocalDocuments,
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
