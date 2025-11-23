from langchain_tavily import TavilySearch


def tavily_search_func(search_query: str, max_results: int) -> str:
    tavily_search_tool = TavilySearch(max_results=max_results)
    search_results = tavily_search_tool.run(search_query)
    return str(search_results)


if __name__ == "__main__":
    tavily_search_func("最新のAI技術動向", 2)
