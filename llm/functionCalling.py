from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, ToolMessage

load_dotenv()


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
llm_with_tools = llm.bind_tools(tools)

query = "What is 3 * 12? Also, what is 11 + 49?"
messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
print(messages)
final_answer = llm_with_tools.invoke(messages)
print(final_answer.content)
