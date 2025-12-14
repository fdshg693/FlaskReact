import anthropic
from anthropic.types import ContentBlock, Message, TextBlock

client = anthropic.Anthropic()


def cite_text() -> None:
    response: Message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "太郎君は15歳で、花子さんは12歳です。",
                        },
                        "title": "My Document",
                        "context": "This is a trustworthy document.",
                        "citations": {"enabled": True},
                    },
                    {"type": "text", "text": "太郎君と花子さんの年齢は？"},
                ],
            }
        ],
    )
    content: list[ContentBlock] = response.content
    first_block = content[0]
    assert isinstance(first_block, TextBlock)
    print("回答:", first_block.text)
    print("引用情報:", first_block.citations)


if __name__ == "__main__":
    cite_text()
