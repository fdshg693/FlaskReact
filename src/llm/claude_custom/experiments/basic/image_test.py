import base64

import anthropic
import httpx
from anthropic.types import TextBlock

from config import load_dotenv_workspace

image_url = "https://upload.wikimedia.org/wikipedia/commons/a/a7/Camponotus_flavomarginatus_ant.jpg"


def send_base64_image() -> None:
    """
    URLから画像を取得し、Base64エンコードしてClaudeに送信する例
    """
    load_dotenv_workspace()
    client = anthropic.Anthropic()

    image_media_type = "image/jpeg"
    # Wikimedia requires a User-Agent header
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = httpx.get(image_url, headers=headers)
    response.raise_for_status()
    image_data = base64.standard_b64encode(response.content).decode("utf-8")

    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": "What is in the above image?"},
                ],
            }
        ],
    )
    first_block = message.content[0]
    assert isinstance(first_block, TextBlock)
    print(first_block.text)


def send_url_image() -> None:
    """
    URLのみをClaudeに送信する例
    """
    load_dotenv_workspace()
    client = anthropic.Anthropic()

    message_from_url = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url,
                        },
                    },
                    {"type": "text", "text": "What is in the above image?"},
                ],
            }
        ],
    )
    first_block = message_from_url.content[0]
    assert isinstance(first_block, TextBlock)
    print(first_block.text)


if __name__ == "__main__":
    send_base64_image()
    # send_url_image()
