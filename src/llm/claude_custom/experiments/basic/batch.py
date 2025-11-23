import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.messages import MessageBatch

client = anthropic.Anthropic()


def send_batch_requests() -> str:
    message_batch: MessageBatch = client.messages.batches.create(
        requests=[
            Request(
                custom_id="request_1",
                params=MessageCreateParamsNonStreaming(
                    model="claude-haiku-4-5",
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": "1+2",
                        }
                    ],
                ),
            ),
            Request(
                custom_id="request_2",
                params=MessageCreateParamsNonStreaming(
                    model="claude-haiku-4-5",
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": "3+4",
                        }
                    ],
                ),
            ),
        ]
    )

    # 送ってすぐなので、「processing_status='in_progress'」となっているはず
    print("バッチリクエスト全文:")
    print(message_batch)
    print("=" * 40)
    print(f"バッチID: {message_batch.id}")
    return message_batch.id


def poll_batch_end(batch_id: str) -> None:
    import time

    while True:
        message_batch: MessageBatch = client.messages.batches.retrieve(
            message_batch_id=batch_id
        )
        print(f"現在の処理状況: {message_batch.processing_status}")
        if message_batch.processing_status == "ended":
            break
        time.sleep(30)

    print(f"ID: {batch_id}の処理結果確認用URL:")
    # 直接UERLにアクセスするだけでは、認証情報がないためアクセスできない点に注意
    # 結果はJSONLファイルとして返される
    print(message_batch.results_url)


def list_batches() -> None:
    """今までのバッチリクエストの一覧を取得するサンプルコード。"""
    for message_batch in client.messages.batches.list(limit=20):
        print(message_batch)


if __name__ == "__main__":
    # batch_id = send_batch_requests()
    # poll_batch_end(batch_id)

    list_batches()
