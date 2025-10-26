from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults


@tool
def tavily_search_func(search_query: str) -> str:
    tavily_search_tool = TavilySearchResults(max_results=2)
    search_results = tavily_search_tool.run(search_query)
    return str(search_results)
