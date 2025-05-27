from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage
from functionTools import getLocalDocuments

load_dotenv()


def functionCalling(tools, query):
    """
    Demonstrates how to use function calling with an LLM.
    """
    llm = init_chat_model("gpt-4o-mini", model_provider="openai")
    llm_with_tools = llm.bind_tools(tools)

    messages = [HumanMessage(query)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    toolsDict = {tool.name.lower(): tool for tool in tools}

    for tool_call in ai_msg.tool_calls:
        selected_tool = toolsDict[tool_call["name"].lower()]
        tool_output = selected_tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))

    final_answer = llm_with_tools.invoke(messages)
    return final_answer.content


if __name__ == "__main__":
    tools = [getLocalDocuments]
    query = "株式会社ヘッドウォーターズについての最新のドキュメントについて簡単に要約してください"
    result = functionCalling(tools, query)
    print(result)
