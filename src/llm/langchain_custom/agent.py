from typing import List, Generator

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain.agents import create_agent
from llm.langchain_custom.tools.search.local_document_search import (
    create_search_local_text_tool,
)

from llm.langchain_custom.models import LLMModel, AgentPrompt


def create_agent_executor(
    tools: List[BaseTool],
    llm_model: LLMModel = LLMModel.GPT_4O_MINI,
) -> Runnable:
    agent = create_agent(
        llm_model,
        tools=tools,
    )
    return agent


def agent_run(prompt: AgentPrompt, agent) -> Generator[AgentPrompt, None, None]:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
    )

    for step in result["messages"]:
        yield step.content
    return


if __name__ == "__main__":
    document_search_tool: BaseTool = create_search_local_text_tool()
    tools: List[BaseTool] = [document_search_tool]
    agent = create_agent_executor(tools)
    search_query: str = "名前順にテキストファイルを並べてください。"
    function_result: Generator[AgentPrompt, None, None] = agent_run(search_query, agent)
    for idx, res in enumerate(function_result):
        print(f"---- STEP {idx} ----")
        print(res)
