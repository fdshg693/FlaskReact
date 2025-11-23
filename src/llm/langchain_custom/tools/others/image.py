from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from langchain_openai import ChatOpenAI
from loguru import logger

from config import PATHS, load_dotenv_workspace
from llm.langchain_custom.models import LLMModel

"""
対応している画像形式(おそらくOPENAI 4oの仕様による)
- PNG
- JPEG
- GIF
- WEBP
"""


def analyze_image_raw(image_data: str) -> str:
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
    load_dotenv_workspace()

    try:
        # Pass to LLM
        llm = ChatOpenAI(model=LLMModel.GPT_4O)

        message: dict[str, Any] = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Describe this image in concise Japanese.",
                },
                {
                    "type": "image",
                    "base64": image_data,
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


def analyze_image_from_url(image_url: str) -> str:
    """Analyze an image from a URL using OpenAI's GPT-4o model.

    Args:
        image_url: URL of the image to analyze
    Returns:
        A description of the image in Japanese
    """
    if not image_url:
        raise ValueError("image_url cannot be empty")

    logger.info(f"Fetching image from URL: {image_url}")

    try:
        llm = ChatOpenAI(model=LLMModel.GPT_4O)

        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the content of this image."},
                {
                    "type": "image",
                    "url": image_url,
                    "mime_type": "image/gif",
                },
            ],
        }

        response: Any = llm.invoke([message])
        logger.info("Image analysis completed successfully")
        return response.content

    except Exception as e:
        logger.error(f"Failed to fetch or analyze image from URL: {e}")
        raise RuntimeError(f"Image analysis from URL failed: {e}") from e


def test_analyze_image_raw():
    """Test function for analyze_image_raw."""
    img_path: Path = PATHS.llm_data / "image" / "fish1.png"

    if not img_path.exists():
        logger.error(f"Image file not found: {img_path}")
        return

    logger.info(f"Loading image from: {img_path}")
    image_data: str = base64.b64encode(img_path.read_bytes()).decode("utf-8")

    try:
        result: str = analyze_image_raw(image_data)
        logger.info(f"Analysis result: {result}")
    except (ValueError, RuntimeError) as e:
        logger.error(f"Image analysis failed: {e}")


def test_analyze_image_from_url():
    """Test function for analyze_image_from_url."""

    # sample salmon image from wikipedia
    image_url: str = "https://upload.wikimedia.org/wikipedia/commons/b/be/Ocean_stage_and_spawning_pink_salmon.gif"

    try:
        result: str = analyze_image_from_url(image_url)
        logger.info(f"Analysis result: {result}")
    except (ValueError, RuntimeError) as e:
        logger.error(f"Image analysis from URL failed: {e}")


if __name__ == "__main__":
    # test_analyze_image_raw()
    test_analyze_image_from_url()
