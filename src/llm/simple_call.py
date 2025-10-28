from langchain.chat_models import init_chat_model

from llm.models import LLMModel, ModelProvider

model = init_chat_model(LLMModel.GPT_4O_MINI, model_provider=ModelProvider.OPENAI)

response = model.invoke("1+1")

print(response.pretty_print())
