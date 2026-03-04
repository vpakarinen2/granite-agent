import re

from curl_cffi import requests
from log import agent_logger


def is_url_accessible(url, timeout=5):
    """Performs quick request to see if the URL is alive."""
    try:
        response = requests.head(url, impersonate="chrome120", timeout=timeout)
        if response.status_code < 400:
            return True
        agent_logger.warning(f"Validation Failed: {url} returned status {response.status_code}")
        return False
    except Exception as e:
        agent_logger.error(f"Validation Error for {url}: {str(e)}")
        return False


def is_content_valid(text):
    """Checks if the scraped text is actually useful."""
    if not text or len(text.strip()) < 100:
        return False
    
    error_patterns = [
        r"404 Not Found",
        r"Access Denied",
        r"Security Check",
        r"Cloudflare",
        r"enable JavaScript",
        r"Just a moment",
        r"Forbidden",
        r"Unexpected Error"
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
            
    return True
