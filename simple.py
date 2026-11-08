# standard library imports
import os
import sys
import threading
import time

# Ensure proper UTF-8 output on Windows terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Third-party imports
from openai import OpenAI
from dotenv import load_dotenv
from types import SimpleNamespace

# Local modules
from tools import TOOLS, run_tool
from Utilities.utilities import log


load_dotenv()  # by default looks for .env file

client = OpenAI(
    base_url=os.getenv("BASE_URL", "http://127.0.0.1:11434/v1"),
    api_key=os.getenv("API_KEY", "dummy"),
)


class Spinner:
    """Lightweight animated terminal spinner."""

    def __init__(self, message="🤔 Thinking..."):
        self.message = message
        self.running = False
        self.thread = None

    def _spin(self):
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        idx = 0
        while self.running:
            sys.stdout.write(f"\r\033[90m{chars[idx % len(chars)]} {self.message}\033[0m")
            sys.stdout.flush()
            idx += 1
            time.sleep(0.08)
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=0.3)
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()


messages = []

# Safety limit: the max number of model rounds for ONE user question
MAX_STEPS = 5

messages.append(
    {
        "role": "system",
        "content": (
            "You are a helpful, humble assistant. "
            "Answer general questions (programming, science, advice) from your own knowledge. "
            "Use tools when needed: get_current_time for time/date, get_city_weather for weather, "
            "get_user_city for user's city/location, and web_search to search the web for up-to-date facts or queries. "
            "Keep answers concise and under 150 words."
        ),
    }
)


def stream_chat(history):
    spinner = Spinner("🤔 Thinking...")
    spinner.start()

    try:
        stream = client.chat.completions.create(
            model=os.getenv("MODEL", "llama3.2:latest"),
            messages=history,
            tools=TOOLS,
            stream=True,
        )
    except Exception as e:
        spinner.stop()
        print(f"\n❌ API Error: {e}")
        return f"Error: {e}", []

    full_response = ""
    calls = {}
    label_printed = False

    # Streaming started
    try:
        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # 1. Content Streaming : Do we have message (content) from the LLM
            if delta.content:
                if spinner.running:
                    spinner.stop()
                if not label_printed:
                    print("🤖 Assistant: ", end="", flush=True)
                    label_printed = True
                print(delta.content, end="", flush=True)
                full_response += delta.content

            # 2. Tool Calls
            for tc in (delta.tool_calls or []):
                if spinner.running:
                    spinner.stop()
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
        print("\n {Stopped By You}")
        calls = {}
    finally:
        spinner.stop()

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
    print("=" * 50)
    print("🤖 AI Agent (Streaming & Tools Enabled) is ready!")
    print("Type your message below. Type 'exit' to quit.")
    print("=" * 50)

    # Chat loop starts from here
    while True:
        try:
            user_input = input("\n YOU: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
            print("👋 Goodbye!")
            break

        messages.append(
            {
                "role": "user",
                "content": user_input,
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
                "tool_calls": build_tool_calls_payload(tool_calls),
            })

            for tool_call in tool_calls:
                result = run_tool(tool_call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

        else:
            # Ran out of steps: The model never gave the final answer
            fallback = "Sorry, I could not finish this within the step limit."
            messages.append({
                "role": "assistant",
                "content": fallback,
            })
            print("🤖 Assistant: " + fallback)
            continue

        messages.append({
            "role": "assistant",
            "content": full_response,
        })


if __name__ == "__main__":
    main()
