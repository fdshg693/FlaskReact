import base64
from pathlib import Path
from typing import Any, Dict, List, Union

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def analyze_image(image_data: str) -> str:
    """Analyze an image using OpenAI's GPT-4o model.

    Args:
        image_data: Base64 encoded image data

    Returns:
        A description of the image in Japanese
    """
    # Load environment variables from .env file
    env_path: Path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    # Pass to LLM
    llm = ChatOpenAI(model="gpt-4o")

    message: Dict[str, Union[str, List[Dict[str, str]]]] = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Describe this image in concise Japanese.",
            },
            {
                "type": "image",
                "source_type": "base64",
                "data": image_data,
                "mime_type": "image/png",
            },
        ],
    }
    response: Any = llm.invoke([message])
    return response.content


if __name__ == "__main__":
    # Fetch image data from local file
    img_path: Path = Path(__file__).parent.parent / "data" / "fish1.png"
    image_data: str = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    result: str = analyze_image(image_data)
    print(result)
    # Expected output: A description of the image in the data/fish1.png file
