import sys
from datetime import datetime
from Utilities.utilities import log
import requests
import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
import bs4

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_current_time():
    """Returns the current live date, time, day of week, and local timezone."""
    now = datetime.now().astimezone()
    tz_name = now.tzname() or "Local Time"
    offset = now.strftime("%z")
    formatted_offset = f"UTC{offset[:3]}:{offset[3:]}" if offset else ""
    return now.strftime(f"%A, %B %d, %Y at %I:%M:%S %p {tz_name} ({formatted_offset})")


def get_current_date():
    """Returns today's date, weekday, and year."""
    now = datetime.now().astimezone()
    return now.strftime("%A, %B %d, %Y")


def get_user_city():
    return "Bangalore"


def get_city_weather(city: str = ""):
    city_name = str(city).strip()
    # If the model passes a placeholder or asks about user's city directly
    if not city_name or city_name.lower() in ("get_user_city", "user_city", "my city", "my location", "here", "none"):
        city_name = get_user_city()

    try:
        url = f"https://wttr.in/{city_name}?format=3"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text.strip()
    except Exception as e:
        return f"Could not retrieve weather for {city_name}: {e}"


def web_search(query: str, max_results: int = 5):
    """
    Searches the live web for up-to-date news, recent events, facts, or definitions.
    Uses Google-level live web search index with Google News RSS and Wikipedia fallback.
    """
    clean_query = str(query).strip()
    if not clean_query:
        return "Please provide a valid search query."

    results = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    q_enc = urllib.parse.quote_plus(clean_query)

    # 1. Primary Live Web Search (Yahoo Web Index - semantic & rich snippets like Google)
    try:
        web_url = f"https://search.yahoo.com/search?p={q_enc}"
        r_web = requests.get(web_url, headers=headers, timeout=6)
        if r_web.status_code == 200:
            soup = bs4.BeautifulSoup(r_web.text, "html.parser")
            for div in soup.select("div.algo"):
                h3 = div.select_one("h3")
                comp = div.select_one(".compText") or div.select_one("p")
                if h3 and comp:
                    title = h3.get_text().strip()
                    desc = comp.get_text().strip()
                    desc = re.sub(r"[\u200b\u200e\u200f\ufffd]", "", desc)
                    if title and desc:
                        results.append(f"- {title}: {desc}")
                if len(results) >= 3:
                    break
    except Exception:
        pass

    # 2. Real-time News & Current Events (Google News RSS)
    try:
        news_url = f"https://news.google.com/rss/search?q={urllib.parse.quote(clean_query)}&hl=en-US&gl=US&ceid=US:en"
        r_news = requests.get(news_url, headers=headers, timeout=5)
        if r_news.status_code == 200:
            root = ET.fromstring(r_news.content)
            for item in root.findall(".//item")[:2]:
                title = item.find("title").text if item.find("title") is not None else ""
                pub = item.find("pubDate").text if item.find("pubDate") is not None else ""
                src = item.find("source").text if item.find("source") is not None else ""
                date_clean = pub[:16] if pub else ""
                if title:
                    results.append(f"[{date_clean}] {title} (Source: {src})")
    except Exception:
        pass

    # 3. Secondary Web Search (Bing Web)
    if len(results) < 3:
        try:
            bing_url = f"https://www.bing.com/search?q={q_enc}"
            r_bing = requests.get(bing_url, headers=headers, timeout=5)
            if r_bing.status_code == 200:
                soup = bs4.BeautifulSoup(r_bing.text, "html.parser")
                for li in soup.select("li.b_algo"):
                    h2 = li.select_one("h2")
                    p = li.select_one(".b_caption p") or li.select_one("p")
                    if h2 and p:
                        title = h2.get_text().strip()
                        snippet = p.get_text().strip()
                        snippet = re.sub(r"[\u200b\u200e\u200f\ufffd]", "", snippet)
                        if title and snippet:
                            results.append(f"- {title}: {snippet}")
                    if len(results) >= max_results:
                        break
        except Exception:
            pass

    # 4. Fallback: Wikipedia Search API
    if not results:
        try:
            wiki_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action": "query",
                "list": "search",
                "srsearch": clean_query,
                "format": "json",
                "utf8": 1,
            }
            resp = requests.get(wiki_url, params=params, headers={"User-Agent": "SimpleAIAgent/1.0"}, timeout=4)
            if resp.status_code == 200:
                items = resp.json().get("query", {}).get("search", [])
                if items:
                    for item in items[:max_results]:
                        title = item.get("title", "")
                        raw_snippet = item.get("snippet", "")
                        clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet)
                        results.append(f"- {title}: {clean_snippet}")
        except Exception:
            pass

    if results:
        seen = set()
        final_list = []
        for r in results:
            clean_r = r.strip()
            if clean_r and clean_r not in seen:
                seen.add(clean_r)
                final_list.append(clean_r)
        return "Live Web Search Results:\n" + "\n".join(final_list[:max_results])
    return f"No recent web results found for '{clean_query}'."


# TOOLS list
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns current live date, time, weekday, and local timezone.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Returns today's date, weekday, and year.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "Returns current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_user_city",
            "description": "Returns the city where the user lives.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a live web and Google News search just like Google. Use this whenever the user asks for recent data, latest news, current events, scores, prices, updates, or anything requiring current web info.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query or keywords to search on Google/web",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time": get_current_time,
    "get_current_date": get_current_date,
    "get_city_weather": get_city_weather,
    "get_user_city": get_user_city,
    "web_search": web_search,
}


def run_tool(tool_call):
    func_name = tool_call.function.name
    func_args = {}

    try:
        func_args = json.loads(tool_call.function.arguments or "{}")
        log(f"🔧 {func_name}({func_args})")
        fn = FUNCTION_NAME_TO_FUNCTION_MAP[func_name]
        result = fn(**func_args)
    except Exception as e:
        result = f"Error while running {func_name} : {e}"

    log(f"   ↳ {result}")
    return str(result)
