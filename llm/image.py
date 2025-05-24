import base64
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


def AnalyzeImage(image_data):
    load_dotenv("../.env")
    # Pass to LLM
    llm = init_chat_model("openai:gpt-4o")

    message = {
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
    response = llm.invoke([message])
    return response.content


if __name__ == "__main__":
    # Fetch image data from local file
    img_path = os.path.join(os.path.dirname(__file__), "../data/fish1.png")
    with open(img_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")
    result = AnalyzeImage(image_data)
    print(result)
    # Expected output: A description of the image in the data/fish1.png file
