"""
OpenAIがサポートしているモデルの一覧を取得して表示するスクリプト
"""

from openai import OpenAI

client = OpenAI()

model_list = client.models.list().data

for model in model_list:
    print(f"Model ID: {model.id}, Owned by: {model.owned_by}")
