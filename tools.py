import datetime

from scrape import scrape_url
from log import agent_logger
from ddgs import DDGS


def get_current_datetime():
    """Returns highly precise temporal string for model grounding."""
    now = datetime.datetime.now()
    return now.strftime("%A, %B %d, %Y | %H:%M:%S")


def perform_web_search(query: str, max_results=4):
    """Executes live web search and deep-reads the top result."""
    agent_logger.info(f"External Search Triggered: '{query}'")
    
    try:
        results = []
        with DDGS() as ddgs:
            ddgs_results = list(ddgs.text(query, max_results=max_results))
            
            for index, r in enumerate(ddgs_results):
                if index == 0:
                    full_text = scrape_url(r['href'], max_chars=2500)
                    results.append(f"--- TOP RESULT (FULL TEXT) ---\nSOURCE: {r['href']}\nCONTENT:\n{full_text}\n---------------------------")
                else:
                    results.append(f"SOURCE: {r['href']}\nSNIPPET: {r['body']}")
        
        if not results:
            agent_logger.warning(f"No results found for query: {query}")
            return "No relevant web data found."
            
        return "\n\n".join(results)
        
    except Exception as e:
        agent_logger.error(f"DDGS Search Error: {str(e)}")
        return f"Error connecting to search engine: {str(e)}"
