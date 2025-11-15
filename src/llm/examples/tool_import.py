from llm import tools


def test_tools():
    # 全ての公開APIが自動的にインポート可能
    tools.add(1, 2)


if __name__ == "__main__":
    test_tools()
