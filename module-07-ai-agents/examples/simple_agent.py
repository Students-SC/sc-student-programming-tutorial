"""
simple_agent.py -- A minimal AI agent that can run shell commands

This demonstrates the core agent loop:
  1. Send a prompt to the LLM
  2. If the LLM wants to run a command, execute it
  3. Feed the result back to the LLM
  4. Repeat until the LLM is done

The agent uses an OpenAI-compatible API (e.g., vLLM server).

Usage:
  export AGENT_API_URL="http://<server>:<port>/v1"
  python3 simple_agent.py "What GPU is on this node?"
"""

import json
import os
import subprocess
import sys
import requests

API_URL = os.environ.get("AGENT_API_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("AGENT_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
MAX_TURNS = 10

SYSTEM_PROMPT = """\
You are a helpful HPC assistant. You can run shell commands to help the user.

When you need to run a command, respond with a JSON block like this:
```tool_call
{"tool": "run_command", "command": "<your command here>"}
```

After seeing the command output, continue reasoning and either run more commands
or give your final answer. When you're done, just respond with plain text
(no tool_call block).

Important rules:
- Only run safe, read-only commands unless the user asks you to compile or write files
- Keep commands short and focused
- Explain what you're doing and why
"""


def run_command(command, timeout=30):
    """Execute a shell command and return its output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]: " + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[command timed out after {timeout}s]"
    except Exception as e:
        return f"[error: {e}]"


def call_llm(messages):
    """Send messages to the LLM API and return the response text."""
    response = requests.post(
        f"{API_URL}/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.3,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def parse_tool_call(text):
    """Check if the LLM response contains a tool call. Returns (tool, command) or None."""
    if "```tool_call" in text:
        try:
            start = text.index("```tool_call") + len("```tool_call")
            end = text.index("```", start)
            payload = json.loads(text[start:end].strip())
            if payload.get("tool") == "run_command":
                return payload["command"]
        except (ValueError, json.JSONDecodeError, KeyError):
            pass
    return None


def run_agent(user_task):
    """Run the agent loop for a given user task."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_task},
    ]

    print(f"User: {user_task}\n")

    for turn in range(MAX_TURNS):
        print(f"--- Agent turn {turn + 1} ---")

        response_text = call_llm(messages)
        print(f"Agent: {response_text}\n")

        messages.append({"role": "assistant", "content": response_text})

        command = parse_tool_call(response_text)
        if command is None:
            print("(Agent finished -- no more tool calls)")
            break

        print(f"  [Running: {command}]")
        output = run_command(command)
        print(f"  [Output: {output[:500]}]\n")

        messages.append({
            "role": "user",
            "content": f"Command output:\n```\n{output}\n```",
        })

    else:
        print(f"(Reached maximum {MAX_TURNS} turns)")


def main():
    if len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
    else:
        task = "What GPU is available on this node? Show me its specs."

    print("=" * 60)
    print("  Simple AI Agent Demo")
    print(f"  API: {API_URL}")
    print(f"  Model: {MODEL_NAME}")
    print("=" * 60)
    print()

    run_agent(task)


if __name__ == "__main__":
    main()
