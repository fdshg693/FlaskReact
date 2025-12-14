from anthropic import Anthropic
from anthropic.types.beta.parsed_beta_message import ParsedBetaMessage
from pydantic import BaseModel


class SentenceInfo(BaseModel):
    lang: str
    difficulty: str


client = Anthropic()

# With .parse() - can pass Pydantic model directly
response: ParsedBetaMessage[SentenceInfo] = client.beta.messages.parse(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    betas=["structured-outputs-2025-11-13"],
    messages=[
        {
            "role": "user",
            "content": "Analyze the following sentence and provide its language and difficulty level: 私は、昨日図書館で本を読みました。",
        }
    ],
    output_format=SentenceInfo,
)

print(response.parsed_output)
