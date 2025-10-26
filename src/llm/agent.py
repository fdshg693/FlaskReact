from typing import List, Generator

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent

from llm.models import LLMModel, ModelProvider, AgentPrompt


def create_agent_executor(
    tools: List[BaseTool],
    llm_model: LLMModel = LLMModel.GPT_4O_MINI,
    model_provider: ModelProvider = ModelProvider.OPENAI,
) -> Runnable:
    """Create a ReAct agent executor with the given tools and model."""
    model: BaseChatModel = init_chat_model(llm_model, model_provider=model_provider)
    agent_executor: Runnable = create_react_agent(model, tools)
    return agent_executor


def agent_run(
    prompt: AgentPrompt,
    tools: List[BaseTool] = [],
    llm_model: LLMModel = LLMModel.GPT_4O_MINI,
    model_provider: ModelProvider = ModelProvider.OPENAI,
) -> Generator[AgentPrompt, None, None]:
    """Run a ReAct agent with the given tools and model."""
    model: BaseChatModel = init_chat_model(llm_model, model_provider=model_provider)
    agent_executor: Runnable = create_react_agent(model, tools)
    config: dict = {"configurable": {"thread_id": "abc123"}}

    for step in agent_executor.stream(
        {"messages": [prompt]}, config, stream_mode="values"
    ):
        yield step["messages"][-1]
