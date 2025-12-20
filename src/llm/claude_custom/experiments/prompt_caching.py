"""
プロンプトキャッシングの動作確認用スクリプト。
プロンプトをキャッシングすることで、プロンプトが長文の時などに、より早い応答を得ることができます。
"""

import anthropic

client = anthropic.Anthropic()

message_params = {
    "model": "claude-haiku-4-5",
    "max_tokens": 1024,
    "system": [
        {
            "type": "text",
            "text": "あなたは有能な解説者です。",
        },
        {
            "type": "text",
            "text": """
            「サーモン」は一般的に、養殖された生食用のサケ科の魚を指し、日本の天然の鮭は「鮭」または「サケ」として流通します。
            サーモンは人工飼料で育てられるため、寄生虫（アニサキスなど）の心配がほぼなく、刺身や寿司ネタとして生食されます。
            一方で、天然の鮭は寄生虫のリスクがあるため、加熱調理が推奨されます。
        """,
            "cache_control": {"type": "ephemeral"},
        },
    ],
    "messages": [{"role": "user", "content": "何に関する説明ですか？"}],
}

print("最初のリクエストに必要なキャッシュトークン数:")
response = client.messages.create(**message_params)
print(response.usage.cache_creation_input_tokens)

print("キャッシュを利用した2回目のリクエストに必要なキャッシュトークン数:")
response = client.messages.create(**message_params)
print(response.usage.cache_creation_input_tokens)
