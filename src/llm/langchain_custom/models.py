"""LLM model and provider enums for type-safe configuration."""

from enum import StrEnum


class ModelProvider(StrEnum):
    """Supported LLM model providers."""

    OPENAI = "openai"


class LLMModel(StrEnum):
    """Supported OpenAI models."""

    GPT_4 = "gpt-4"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_5 = "gpt-5"
