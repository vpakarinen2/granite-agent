import datetime

from log import agent_logger
from ddgs import DDGS


def get_current_datetime():
    """Returns a highly precise temporal string for model grounding."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y | %H:%M:%S")


def perform_web_search(query: str, max_results=3):
    """Executes a live web search using DDGS."""
    agent_logger.info(f"External Search Triggered: '{query}'")
    
    try:
        results = []
        with DDGS() as ddgs:
            ddgs_gen = ddgs.text(query, max_results=max_results)
            for r in ddgs_gen:
                results.append(f"SOURCE: {r['href']}\nCONTENT: {r['body']}")
        
        if not results:
            agent_logger.warning(f"No results found for query: {query}")
            return "No relevant web data found."
            
        return "\n\n".join(results)
        
    except Exception as e:
        agent_logger.error(f"DDGS Search Error: {str(e)}")
        return f"Error connecting to search engine: {str(e)}"
