from datetime import datetime
from decimal import Decimal, InvalidOperation
from xml.etree import ElementTree
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from Utilities.utilities import log
from rag import search
import requests
import json


def get_current_time(timezone_name="local"):
    """Return the current date and time for a local or IANA timezone."""
    if timezone_name == "local":
        current_time = datetime.now()
        return current_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        try:
            current_time = datetime.now(ZoneInfo(timezone_name))
        except ZoneInfoNotFoundError as error:
            raise ValueError(f"Unknown timezone: {timezone_name}") from error
    return current_time.strftime("%Y-%m-%d %H:%M:%S %Z")


def get_current_date(timezone_name="local"):
    """Return today's date for a local or IANA timezone."""
    return get_current_time(timezone_name).split(" ")[0]


def get_city_weather(city):
    response = requests.get(
        "https://wttr.in/",
        params={"format": "3", "q": city},
        timeout=10,
    )
    response.raise_for_status()
    return response.text


def get_user_city():
    return "Bangalore"


def _google_news_search(query, num_results):
    response = requests.get(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()
    feed = ElementTree.fromstring(response.content)
    results = []
    for item in feed.findall("./channel/item")[:num_results]:
        title = item.findtext("title", default="").strip()
        url = item.findtext("link", default="").strip()
        published = item.findtext("pubDate", default="").strip()
        results.append(f"- {title}\n  {url}\n  Published: {published}")
    return results


def web_search(query, num_results=5, search_type="current"):
    """Return current Google News or general Google web results."""
    if not query.strip():
        raise ValueError("Search query cannot be empty.")
    if not 1 <= num_results <= 10:
        raise ValueError("num_results must be between 1 and 10.")
    if search_type not in {"current", "web"}:
        raise ValueError("search_type must be current or web.")

    formatted_results = []
    if search_type == "current":
        formatted_results = _google_news_search(query, num_results)

    if not formatted_results:
        from googlesearch import search

        results = search(
            query,
            num_results=num_results,
            advanced=True,
            unique=True,
            sleep_interval=1,
            timeout=10,
        )
        for result in results:
            description = getattr(result, "description", "").strip()
            title = getattr(result, "title", "").strip() or result.url
            formatted_results.append(
                f"- {title}\n  {result.url}\n  {description}"
            )

    if not formatted_results:
        formatted_results = _google_news_search(query, num_results)
    if not formatted_results:
        return "No Google search results found."
    return "Google search results:\n" + "\n".join(formatted_results)


def convert_currency(amount, from_currency, to_currency):
    """Convert an amount using the latest Frankfurter exchange rate."""
    try:
        value = Decimal(str(amount))
    except (InvalidOperation, ValueError) as error:
        raise ValueError("amount must be a valid number.") from error
    if value < 0:
        raise ValueError("amount must not be negative.")

    base = from_currency.strip().upper()
    quote = to_currency.strip().upper()
    if len(base) != 3 or len(quote) != 3:
        raise ValueError("Currencies must be three-letter ISO codes.")
    if base == quote:
        return f"{value:.2f} {base} = {value:.2f} {quote}"

    response = requests.get(
        f"https://api.frankfurter.dev/v2/rate/{base}/{quote}",
        timeout=10,
    )
    response.raise_for_status()
    rate = Decimal(str(response.json()["rate"]))
    converted = value * rate
    return f"{value:.2f} {base} = {converted:.2f} {quote} (rate: {rate})"


def search_knowledge_base(query):
    results = search(query, top_k=2)

    log(f"📚 Knowledge base search: {query}")
    for r in results:
        log(f"   {r['score']:.2f} - {r['source']} #{r['chunk_id']}")

    blocks = []
    for r in results:
        sourceFileName = r['source']
        chunkId        = r['chunk_id']
        chunkText      = r['text']

        header = f"[{sourceFileName} #{chunkId}]"
        block  = f"{header}\n{chunkText}"
        blocks.append(block)

    return "\n\n".join(blocks)


#TOOLS = [{}, {}, {}]
TOOLS = [
    {
        "type" : "function",
        "function" : {
            "name" : "get_current_time",
            "description" : "Returns the current date and time, optionally for an IANA timezone such as Asia/Kolkata.",
            "parameters" : {
                "type": "object",
                "properties": {
                    "timezone_name": {
                        "type": "string",
                        "description": "IANA timezone, or local for the computer timezone.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Returns today's date, optionally for an IANA timezone such as America/New_York.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone_name": {
                        "type": "string",
                        "description": "IANA timezone, or local for the computer timezone.",
                    }
                },
            },
        },
    },
    {
        "type" : "function",
        "function" : {
            "name" : "get_city_weather",
            "description" : "Returns the current weather for a given city.",
            "parameters" : {
                "type" : "object",
                "properties" : {
                    "city" : {
                        "type" : "string",
                        "description" : "City to get the weather for.",
                    }
                },
                "required": ["city"]
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches Google for current information, news, sources, and recent events. Do not use for Zorvex company policies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise web search query.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return, from 1 to 10.",
                    },
                    "search_type": {
                        "type": "string",
                        "enum": ["current", "web"],
                        "description": "Use current for recent news and web for general Google results.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Converts money using the latest exchange rate from Frankfurter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to convert."},
                    "from_currency": {"type": "string", "description": "Three-letter source currency code."},
                    "to_currency": {"type": "string", "description": "Three-letter target currency code."},
                },
                "required": ["amount", "from_currency", "to_currency"],
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
            "name": "search_knowledge_base",
            "description" : "Searches the Zorvex Technologies company knowledge base: leave, medical, HR, hiring, tech, incident "
                            "reports and travel expense policies. Use it for any question about company policies or documents. "
                            "Returns the most relevant passages with their source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question, rewritten as a short, self-contained search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


FUNCTION_NAME_TO_FUNCTION_MAP = {
    "get_current_time" : get_current_time,
    "get_current_date": get_current_date,
    "get_city_weather" : get_city_weather,
    "get_user_city" : get_user_city,
    "web_search": web_search,
    "convert_currency": convert_currency,
    "search_knowledge_base" : search_knowledge_base,
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