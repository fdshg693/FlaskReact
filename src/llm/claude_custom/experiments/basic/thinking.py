import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=16000,
    thinking={"type": "enabled", "budget_tokens": 10000},
    messages=[
        {
            "role": "user",
            "content": "美味しいものから食べるべき？それとも美味しいものは後に取っておくべき？理由も教えて。",
        }
    ],
)

for block in response.content:
    if block.type == "thinking":
        print(f"\n思考: {block.thinking}")
    elif block.type == "text":
        print(f"\n応答: {block.text}")
