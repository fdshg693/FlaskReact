"""
ウェブ検索を実行するツールモジュール
Tavilyは細かい調整や、WEB UIでの確認ができて、カスタマイズ性が高い
DuckDuckGoはシンプルで使いやすい、またAPIキー不要で利用可能
"""

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

####===================== Tavily =====================####


def tavily_search(search_query: str, max_results: int) -> str:
    """
    Tavilyを使用してウェブ検索を実行します。
    Args:
        search_query (str): 検索クエリ
        max_results (int): 取得する最大結果数
    Returns:
        str: 検索結果の文字列形式
    """
    # TavilySearchクラスはBaseToolを継承しているため、そのまま使用可能
    tavily_search_tool: TavilySearch = TavilySearch(max_results=max_results)
    search_results = tavily_search_tool.run(search_query)
    return str(search_results)


# 勉強のため、以下のように書いているが、ツールとして利用する場合は、シンプルにTavilySearchクラスを直接使えば良い
@tool
def tavily_search_tool(
    search_query: str,
    max_results: int = 5,
) -> str:
    """
    Tavilyを使用してウェブ検索を実行します。
    Args:
        search_query (str): 検索クエリ
        max_results (int): 取得する最大結果数
    Returns:
        str: 検索結果の文字列形式
    """
    return tavily_search(search_query, max_results)


####===================== DuckDuckGo =====================####
def duckduckgo_search(search_query: str) -> str:
    """
    DuckDuckGoを使用してウェブ検索を実行します。
    Args:
        search_query (str): 検索クエリ
    Returns:
        str: 検索結果の文字列形式
    """
    # DuckDuckGoSearchRunクラスはBaseToolを継承しているため、そのまま使用可能
    duckduckgo_search_tool: DuckDuckGoSearchRun = DuckDuckGoSearchRun()
    search_results = duckduckgo_search_tool.run(search_query)
    return str(search_results)


# 勉強のため、以下のように書いているが、ツールとして利用する場合は、シンプルにTavilySearchクラスを直接使えば良い
@tool
def duckduckgo_search_tool(
    search_query: str,
) -> str:
    """
    DuckDuckGoを使用してウェブ検索を実行します。
    Args:
        search_query (str): 検索クエリ
    Returns:
        str: 検索結果の文字列形式
    """
    return duckduckgo_search(search_query)


if __name__ == "__main__":
    ###===================== Tavily =====================###
    # 関数として利用する場合
    print(tavily_search(search_query="東京の今日の天気", max_results=2))

    # ツールとして利用する場合
    print(
        tavily_search_tool.run({"search_query": "東京の今日の天気", "max_results": 2})
    )

    ###===================== DuckDuckGo =====================###
    # 関数として利用する場合
    print(duckduckgo_search(search_query="東京の今日の天気"))

    # ツールとして利用する場合
    print(duckduckgo_search_tool.run({"search_query": "東京の今日の天気"}))
