"""
シンプルな、LLMモデル呼び出しの例。
"""

import os

from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import AIMessage

from llm.langchain_custom.models import LLMModel, ModelProvider

from config import load_dotenv_workspace

load_dotenv_workspace()

DEFAULT_PROVIDER: ModelProvider = ModelProvider.OPENAI
DEFAULT_MODEL: LLMModel = LLMModel.GPT_4O_MINI


def get_model(
    model_name: LLMModel = DEFAULT_MODEL,
    model_provider: ModelProvider = DEFAULT_PROVIDER,
) -> BaseChatModel:
    """
    指定されたモデル名とプロバイダーに基づいてチャットモデルを初期化して返す。
    Args:
        model_name (LLMModel): 使用するLLMモデルの名前。デフォルトはGPT_4O_MINI。
        model_provider (ModelProvider): 使用するモデルプロバイダー。デフォルトはOPENAI。
    Returns:
        BaseChatModel: 初期化されたチャットモデルのインスタンス。
    Raises:
        ValueError: OPENAI_API_KEYが設定されていない場合に発生。
    Example:
        model = get_model(LLMModel.GPT_4O_MINI, ModelProvider.OPENAI)
    """
    if model_provider == ModelProvider.OPENAI and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY is missing. Set it in your environment or in the project's .env file."
        )

    model: BaseChatModel = init_chat_model(model_name, model_provider=model_provider)
    assert isinstance(model, BaseChatModel)
    return model


def main(prompt: str) -> str:
    model = get_model()

    response: AIMessage = model.invoke(prompt)
    content: str | list[str | dict] = response.content

    if isinstance(content, str):
        return content
    return str(content)


if __name__ == "__main__":
    content = main("1+1")
    print(content)
