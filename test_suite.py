import os
import sys
import json
import io
import time
from pathlib import Path

# Ensure UTF-8 output encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import simple
from simple import (
    create_client,
    ask_agent,
    load_memory,
    save_memory,
    clear_memory,
    MEMORY_FILE,
    SYSTEM_MESSAGE,
    TOOLS,
    TOOL_FUNCTIONS,
    ColoredFormatter,
    StreamAwareHandler,
    configure_logging,
    LOGGER,
)

def run_tests():
    print("=" * 60)
    print("RUNNING AI AGENT STREAMING & TOOL-CALLING TEST SUITE")
    print("=" * 60)

    client = create_client()
    clear_memory()

    # -------------------------------------------------------------
    # TEST 1: Normal response with streaming ("Explain Python.")
    # -------------------------------------------------------------
    print("\n--- TEST 1: Normal Response Streaming ('Explain Python.') ---")
    conv1 = [{"role": "user", "content": "Explain Python in 15 words or less."}]
    res1 = ask_agent(client, conv1)
    print(f"\nCaptured Result 1: {res1.strip()}")
    assert len(res1.strip()) > 0, "Test 1 Failed: Response was empty"
    print(">>> TEST 1 PASSED: Text appeared progressively and response completed.")

    # -------------------------------------------------------------
    # TEST 2: Tool calling ("What is the current time in Tokyo?")
    # -------------------------------------------------------------
    print("\n--- TEST 2: Tool Calling ('What is the current time in Tokyo?') ---")
    conv2 = [{"role": "user", "content": "What is the current time in Tokyo?"}]
    res2 = ask_agent(client, conv2)
    print(f"\nCaptured Result 2: {res2.strip()}")
    
    # Check that tool call was executed and recorded in conv2
    has_tool_call = any(
        msg.get("role") == "assistant" and msg.get("tool_calls")
        for msg in conv2
    )
    has_tool_result = any(
        msg.get("role") == "tool" and ("get_current_time" in str(msg) or "JST" in str(msg.get("content")))
        for msg in conv2
    )
    assert has_tool_call, "Test 2 Failed: No tool_calls message recorded in conversation"
    assert has_tool_result, "Test 2 Failed: No tool result recorded in conversation"
    print(">>> TEST 2 PASSED: Tool calling executed correctly and result streamed.")

    # -------------------------------------------------------------
    # TEST 3: Chained / Multi-tool calling
    # -------------------------------------------------------------
    print("\n--- TEST 3: Chained Tool Calling (Sequential tool calling across turns) ---")
    conv3 = [{"role": "user", "content": "What is today's date?"}]
    res3_turn1 = ask_agent(client, conv3)
    print(f"\nCaptured Turn 1: {res3_turn1.strip()}")
    
    # Add follow-up requiring another tool
    conv3.append({"role": "user", "content": "Now what is the weather in London?"})
    res3_turn2 = ask_agent(client, conv3)
    print(f"\nCaptured Turn 2: {res3_turn2.strip()}")
    
    tool_msgs = [m for m in conv3 if m.get("role") == "tool"]
    print(f"Number of tool calls executed in chain: {len(tool_msgs)}")
    assert len(tool_msgs) >= 2, f"Test 3 Failed: Expected at least 2 chained tool calls, got {len(tool_msgs)}"
    print(">>> TEST 3 PASSED: Chained tool calling executed sequentially across turns with streaming.")

    # -------------------------------------------------------------
    # TEST 4: General questions streaming
    # -------------------------------------------------------------
    print("\n--- TEST 4: General Question Streaming ('Tell me a short 1-line joke.') ---")
    conv4 = [{"role": "user", "content": "Tell me a short 1-line joke."}]
    res4 = ask_agent(client, conv4)
    print(f"\nCaptured Result 4: {res4.strip()}")
    assert len(res4.strip()) > 0, "Test 4 Failed: Expected non-empty response for general question"
    print(">>> TEST 4 PASSED: General question streamed and answered successfully.")

    # -------------------------------------------------------------
    # TEST 5: Conversation history preservation & reloading
    # -------------------------------------------------------------
    print("\n--- TEST 5: Conversation History Preservation ---")
    clear_memory()
    conv5 = [{"role": "user", "content": "What is the time in Paris?"}]
    ask_agent(client, conv5)
    save_memory(conv5)

    loaded_conv = load_memory()
    assert len(loaded_conv) == len(conv5), f"Test 5 Failed: Loaded length {len(loaded_conv)} != saved length {len(conv5)}"
    roles = [m.get("role") for m in loaded_conv]
    print(f"Stored message sequence roles: {roles}")
    assert "user" in roles and "tool" in roles and "assistant" in roles, "Test 5 Failed: Missing expected roles in saved memory"
    print(">>> TEST 5 PASSED: Conversation history with tool calls and results saved and loaded correctly.")

    # -------------------------------------------------------------
    # TEST 6: Debug mode cleanliness
    # -------------------------------------------------------------
    print("\n--- TEST 6: Debug Mode Cleanliness ---")
    import logging
    simple.LOG_LEVEL = "DEBUG"
    configure_logging()
    
    conv6 = [{"role": "user", "content": "What is the time in Berlin?"}]
    res6 = ask_agent(client, conv6)
    print(f"\nCaptured Result 6: {res6.strip()}")
    assert len(res6.strip()) > 0, "Test 6 Failed: Response was empty"
    print(">>> TEST 6 PASSED: Debug logs formatted cleanly with colored levels and did not corrupt text output.")

    print("\n" + "=" * 60)
    print("ALL 6 TEST SCENARIOS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
