import base64

import anthropic
import httpx
from anthropic.types import TextBlock

# 形式: https://raw.githubusercontent.com/{username}/{repo}/{branch}/{path/to/file.pdf}
pdf_url = (
    "https://raw.githubusercontent.com/fdshg693/FlaskReact/dev/data/llm/pdf/sample.pdf"
)
pdf_data = base64.standard_b64encode(httpx.get(pdf_url).content).decode("utf-8")


def pdf_read_url() -> None:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "url",
                            "url": pdf_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": "What are the key findings in this document?",
                    },
                ],
            }
        ],
    )
    first_block = message.content[0]
    assert isinstance(first_block, TextBlock)
    print(first_block.text)


def pdf_url_base64() -> None:
    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": "What are the key findings in this document?",
                    },
                ],
            }
        ],
    )
    first_block = message.content[0]
    assert isinstance(first_block, TextBlock)
    print(first_block.text)


if __name__ == "__main__":
    pdf_read_url()
    # pdf_url_base64()
