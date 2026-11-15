"""Smoke tests for the current assistant modules.

These tests avoid network calls so they can run without Ollama being active.
"""

from chunk_search import chunk_document
from simple import build_tool_calls_payload
from tools import get_current_time, get_user_city, run_tool


def run_tests():
    print("=" * 60)
    print("RUNNING AI AGENT SMOKE TESTS")
    print("=" * 60)

    document = {
        "source": "test.txt",
        "text": "First section.\n\nSecond section.\n\n",
    }
    chunks = chunk_document(document)
    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == 0
    assert chunks[1]["text"] == "Second section."
    print(">>> Chunking passed")

    assert get_user_city() == "Bangalore"
    assert len(get_current_time()) == 19
    print(">>> Built-in tools passed")

    tool_call = type(
        "ToolCall",
        (),
        {
            "id": "test-call",
            "function": type(
                "Function",
                (),
                {"name": "get_user_city", "arguments": "{}"},
            )(),
        },
    )()
    assert run_tool(tool_call) == "Bangalore"
    print(">>> Tool dispatch passed")

    payload = build_tool_calls_payload([tool_call])
    assert payload[0]["function"]["name"] == "get_user_city"
    print(">>> Tool-call payload passed")

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
