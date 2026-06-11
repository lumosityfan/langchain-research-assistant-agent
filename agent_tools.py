from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from typing import Optional

@tool
def search_web(query: str) -> str:
    """Search the web for information using DuckDuckGo."""
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return f"Search results for '{query}':\n{results}"
    except Exception as e:
        return f"Error searching web: {str(e)}"
    
@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations safely."""
    try:
        import math
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Result: {result}"
    except Exception as e:
        return f"Error in calulation: {str(e)}"
    
@tool
def fetch_url_content(url: str) -> str:
    """Fetch and return the text content from a URL (first 2000 characters)."""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'ResearchAgent/1.0'})
        response.raise_for_status()
        content = response.text[:2000]
        return f"Content from {url}:\n{content}..."
    except Exception as e:
        return f"Error fetching URL: {str(e)}"
    
@tool
def summarize_text(text: str, max_words: Optional[int] = 100) -> str:
    """Summarize long test into a concise format."""
    sentences = text.split('. ')
    summary = '. '.join(sentences[:3])
    return f"Summary: {summary[:max_words * 5]}..."

research_tools = [search_web, calculate, fetch_url_content, summarize_text]
print(f"Loaded {len(research_tools)} tools")