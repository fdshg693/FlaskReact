from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from loguru import logger


def analyze_image(image_data: str) -> str:
    """Analyze an image using OpenAI's GPT-4o model.

    Args:
        image_data: Base64 encoded image data

    Returns:
        A description of the image in Japanese

    Raises:
        ValueError: If image_data is empty or invalid
        RuntimeError: If the API call fails
    """
    if not image_data:
        raise ValueError("image_data cannot be empty")

    logger.info("Starting image analysis with OpenAI GPT-4o")

    # Load environment variables from .env file
    env_path: Path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    try:
        # Pass to LLM
        llm = ChatOpenAI(model="gpt-4o")

        message: dict[str, Any] = {
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
        logger.info("Image analysis completed successfully")
        return response.content

    except Exception as e:
        logger.error(f"Failed to analyze image: {e}")
        raise RuntimeError(f"Image analysis failed: {e}") from e


if __name__ == "__main__":
    # Fetch image data from local file
    img_path: Path = Path(__file__).parent.parent / "data" / "fish1.png"

    if not img_path.exists():
        logger.error(f"Image file not found: {img_path}")
        exit(1)

    logger.info(f"Loading image from: {img_path}")
    image_data: str = base64.b64encode(img_path.read_bytes()).decode("utf-8")

    try:
        result: str = analyze_image(image_data)
        logger.info(f"Analysis result: {result}")
    except (ValueError, RuntimeError) as e:
        logger.error(f"Image analysis failed: {e}")
        exit(1)
    # Expected output: A description of the image in the data/fish1.png file
