import anthropic
from anthropic import Anthropic
from anthropic.resources import Messages
from anthropic.types import Message, ContentBlock, TextBlock

client: Anthropic = anthropic.Anthropic()


def simple_example() -> None:
    messages_instance: Messages = client.messages

    message: Message = messages_instance.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": "1+2",
            }
        ],
    )
    content: list[ContentBlock] = message.content
    assert isinstance(content[0], TextBlock)
    text = content[0].text
    print(text)


def chat_with_history() -> None:
    message: Message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": "I am Tanaka,",
            },
            {
                "role": "assistant",
                "content": "Hello",
            },
            {
                "role": "user",
                "content": "What is my name?",
            },
        ],
    )
    content: list[ContentBlock] = message.content
    assert isinstance(content[0], TextBlock)
    text = content[0].text
    print(text)


def force_single_concise_answer() -> None:
    message: Message = anthropic.Anthropic().messages.create(
        model="claude-haiku-4-5",
        # max_tokensを1にすることで、アルファベット1文字だけの回答を強制する
        max_tokens=1,
        messages=[
            {
                "role": "user",
                "content": "9*9= (A) 88, (B) 81, (C) 79",
            },
            # (まで回答を事前に与えることで、A,B,Cのいずれか1文字だけを回答させる
            {"role": "assistant", "content": "The answer is ("},
        ],
    )
    print("入力トークン")
    print(message.usage.input_tokens)
    print("出力トークン")
    print(message.usage.output_tokens)
    print("回答")
    print(message.content[0].text)


if __name__ == "__main__":
    # simple_example()
    # chat_with_history()
    force_single_concise_answer()
