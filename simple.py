# standard library imports
import os

# Third-party imports
from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace

# Local modules
from tools import TOOLS, run_tool
from Utilities.utilities import log



load_dotenv() #by default looks for .env file

client = OpenAI(
    base_url=os.getenv("BASE_URL", "http://127.0.0.1:11434/v1"),
    api_key=os.getenv("API_KEY", "dummy"),
)

messages = []

# Safety limit: the max number of model rounds for ONE user question
MAX_STEPS = 5

messages.append(
    {
        "role": "system",
        "content": (
            "You are a helpful, humble assistant for Zorvex Technologies employees. "
            "Answer general questions (programming, science, advice) from your own knowledge. "
            "Use web_search with search_type=current for recent or current information, news, sources, and events; use search_type=web for general web results. "
            "Use convert_currency for exchange-rate conversions, and get_current_date or get_current_time for date/time questions. "
            "Use get_city_weather and get_user_city for weather questions. "
            "Use search_knowledge_base ONLY for questions about Zorvex policies, benefits, leave, medical cover, HR, hiring, tech processes, incident reports or travel expenses. "
            "For those company questions, call search_knowledge_base first, answer ONLY from its returned text, and cite the source like (leave_policy.txt #3). "
            "If the returned text does not contain a company answer, say exactly: I could not find this in the knowledge base. "
            "For every other question, answer helpfully from your knowledge or the appropriate live tool; do not search the knowledge base and do not use the knowledge-base fallback. "
            "Keep answers under 100 words."
        ),
    }
)

def stream_chat(history):
    stream = client.chat.completions.create(
            model=os.getenv("MODEL", "llama3.2:latest"),
            messages=history,
            tools=TOOLS,
            stream=True,
        )

    full_response = ""
    calls = {}
    label_printed = False

    #Streaming started
    try:
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            #1. Content Streaming : Do we have message (content) from the LLM
            if delta.content:
                if not label_printed:
                    print("🤖 Assistant: ", end="", flush=True)
                    label_printed = True
                print(delta.content, end="", flush=True)
                full_response += delta.content

            #2. Tool Calls
            for tc in (delta.tool_calls or []):
                obj = calls.setdefault(
                    tc.index,
                    SimpleNamespace(id=None, function=SimpleNamespace(name="", arguments="")),
                )
                if tc.id:
                    obj.id = tc.id
                if tc.function and tc.function.name:
                    obj.function.name += tc.function.name
                if tc.function and tc.function.arguments:
                    obj.function.arguments += tc.function.arguments

    except KeyboardInterrupt:
        raise
    
    if label_printed:
        print()  # Move to the next line after the assistant's response

    calls_list = []
    for key in sorted(calls):
        calls_list.append(calls[key])
    
    return full_response, calls_list


def build_tool_calls_payload(tool_calls):
    """
    Converts our collected tool calls into the dict format
    that the API expects inside an assistant message.
    """

    payload = []
    for tool_call in tool_calls:
        tool_call_dict = {
            "id": tool_call.id,
            "type": "function",
            "function": {
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            },
        }
        payload.append(tool_call_dict)

    return payload
    

def main():
    try:
        # Chat loop starts from here
        while True:

            user_input = input("\nPro: ")

            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                break

            messages.append(
                {
                    "role": "user",
                    "content": user_input
                }
            )

            # Chained Tool Calling
            for step in range(MAX_STEPS):
                full_response, tool_calls = stream_chat(messages)

                if not tool_calls:
                    break

                log(f"Step {step + 1}: model requested {len(tool_calls)} tool calls")

                messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "tool_calls": build_tool_calls_payload(tool_calls)
                })

                for tool_call in tool_calls:
                    result = run_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id" : tool_call.id,
                        "content" : str(result),
                    })

            else:
                # Ran out of steps: The model never gave the final answer
                fallback = "Sorry, I could not finish this within the step limit."
                messages.append({
                    "role": "assistant",
                    "content": fallback
                })
                print("🤖 Assistant: " + fallback)
                continue

            messages.append({
                "role": "assistant",
                "content": full_response
            })
    except (KeyboardInterrupt, EOFError):
        print("\nStopped by user. Exiting safely.")


if __name__ == "__main__":
    main()