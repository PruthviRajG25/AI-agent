import sys
from datetime import datetime
from Utilities.utilities import log
import requests
import json
import re

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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


def web_search(query: str):
    """
    Searches the web for up-to-date information, news, facts, or definitions.
    Uses DuckDuckGo Instant Answers and Wikipedia search as reliable fallback.
    """
    clean_query = str(query).strip()
    if not clean_query:
        return "Please provide a valid search query."

    # 1. Try DuckDuckGo Instant Answer API
    try:
        url = "https://api.duckduckgo.com/"
        params = {"q": clean_query, "format": "json", "no_html": 1, "skip_disambig": 1}
        resp = requests.get(url, params=params, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get("AbstractText", "").strip()
            if abstract:
                return f"DuckDuckGo Summary: {abstract}"
            related = [t["Text"] for t in data.get("RelatedTopics", []) if "Text" in t]
            if related:
                return "Search results:\n" + "\n".join(f"- {t}" for t in related[:3])
    except Exception:
        pass

    # 2. Fallback to Wikipedia Search API
    try:
        wiki_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": clean_query,
            "format": "json",
            "utf8": 1,
        }
        headers = {"User-Agent": "SimpleAIAgent/1.0"}
        resp = requests.get(wiki_url, params=params, headers=headers, timeout=6)
        if resp.status_code == 200:
            items = resp.json().get("query", {}).get("search", [])
            if items:
                lines = []
                for item in items[:3]:
                    title = item.get("title", "")
                    raw_snippet = item.get("snippet", "")
                    clean_snippet = re.sub(r"<[^>]+>", "", raw_snippet)
                    lines.append(f"- {title}: {clean_snippet}")
                return "Wikipedia results:\n" + "\n".join(lines)
    except Exception:
        pass

    return f"No search results found for '{clean_query}'."


# TOOLS = [{}, {}, {}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time as a formatted string.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_city_weather",
            "description": "Returns the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City to get the weather for.",
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
            "description": "Returns the city where the user lives. Use this when the user says 'my city', 'my Home', 'where I live' or 'my location' without naming the city.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web for up-to-date information, news, facts, definitions, or general knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query or keywords to look up on the web.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time": get_current_time,
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
