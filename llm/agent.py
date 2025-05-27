from langgraph.prebuilt import create_react_agent
from functionTools import (
    searchHeadwatersCompany,
    searchLocalDocuments,
    getLocalDocuments,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage

tools = [searchHeadwatersCompany, searchLocalDocuments, getLocalDocuments]
model = init_chat_model("gpt-4", model_provider="openai")
agent_executor = create_react_agent(model, tools)

response = agent_executor.invoke(
    {
        "messages": [
            HumanMessage(
                content="株式会社ヘッドウォーターズの作ったアプリについて教えてください。"
            )
        ]
    }
)

print(response["messages"][-1].content)
