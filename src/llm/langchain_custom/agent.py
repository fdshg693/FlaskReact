"""
エージェントの作成と実行に関するモジュール（langchain組み込み機能のラッパー）
現在機能していないようなので、要修正
"""

from typing import Any, Iterator, List

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from llm.langchain_custom.models import LLMModel


def create_agent_executor(
    tools: List[BaseTool],
    llm_model: LLMModel = LLMModel.GPT_4O_MINI,
) -> CompiledStateGraph[Any]:
    """
    エージェントを作成する関数。
    作られたエージェントは、指定されたツールとLLMモデルを使用して動作します。
    Args:
        tools (List[BaseTool]): エージェントが使用するツールのリスト。
        llm_model (LLMModel): 使用するLLMモデル。デフォルトはGPT_4O_MINI。
    Returns:
        CompiledStateGraph[Any]: 作成されたエージェント。
    Example:
        agent = create_agent_executor(some_tools, some_llm_model)

        result = agent.invoke({"messages": [{"role": "user", "content": "質問内容"}]})
    """
    agent: CompiledStateGraph[Any] = create_agent(
        llm_model,
        tools=tools,
    )
    return agent


def agent_run(prompt: str, agent: CompiledStateGraph[Any]) -> Iterator[Any]:
    """
    エージェントを実行し、ステップごとに結果を生成する関数。
    Args:
        prompt (str): ユーザーからの入力プロンプト。
        agent (CompiledStateGraph[Any]): 実行するエージェント。
    Yields:
        Iterator[Any]: エージェントの各ステップの結果を順次生成。
    Example:
        results = agent_run("質問内容", agent)
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
    )

    for step in result["messages"]:
        yield step.content
    return


if __name__ == "__main__":
    from llm.langchain_custom.examples.sample_tools import add_numbers

    sample_tools: List[BaseTool] = [add_numbers]
    sample_agent: CompiledStateGraph[Any] = create_agent_executor(sample_tools)
    search_query: str = "ツールを使って2+3+6を計算してください"
    function_result: Iterator[Any] = agent_run(prompt=search_query, agent=sample_agent)
    for idx, res in enumerate(function_result):
        print(f"---- STEP {idx} ----")
        print(res)
        print("\n")
