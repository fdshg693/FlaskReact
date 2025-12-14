from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5-nano", input="tell me a joke about computers"
)
print(response.output_text)
