"""
build_agent.py -- Build your own AI agent

EXERCISE: Fill in the 3 TODOs to create a working agent that can run shell
commands on the cluster.

Usage:
  export AGENT_API_URL="http://<server>:<port>/v1"
  python3 build_agent.py "Write a hello world C program, compile it, and run it"
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
You are a helpful HPC programming assistant running on a compute node with
an AMD MI210 GPU. You can run shell commands to help the user.

When you need to run a command, respond with a JSON block like this:
```tool_call
{"tool": "run_command", "command": "<your command here>"}
```

After seeing the command output, continue reasoning and either run more commands
or give your final answer. When you're done, just respond with plain text.

Available tools: gcc, g++, mpicc, hipcc, python3, rocm-smi, ls, cat, etc.
"""


def run_command(command, timeout=30):
    """Execute a shell command and return its output.

    TODO 1: Implement this function.

    Steps:
      1. Use subprocess.run() to execute the command:
         - shell=True (so we can use pipes, redirects, etc.)
         - capture_output=True (capture stdout and stderr)
         - text=True (return strings, not bytes)
         - timeout=timeout (don't hang forever)
      2. Combine stdout and stderr into a single string
      3. If the return code is non-zero, append it to the output
      4. Return the output string

    Handle exceptions:
      - subprocess.TimeoutExpired: return a timeout message
      - Any other exception: return the error message

    YOUR CODE HERE:
    """
    return "(run_command not implemented yet)"


def call_llm(messages):
    """Send a conversation to the LLM API and return the response text.

    TODO 2: Implement this function.

    Steps:
      1. Make a POST request to f"{API_URL}/chat/completions" with JSON body:
         {
           "model": MODEL_NAME,
           "messages": messages,
           "max_tokens": 1024,
           "temperature": 0.3,
         }
      2. Parse the JSON response
      3. Return the content of the first choice's message

    Hint: Use requests.post(url, json={...}, timeout=60)
          The response structure is: response.json()["choices"][0]["message"]["content"]

    YOUR CODE HERE:
    """
    return "(call_llm not implemented yet)"


def parse_tool_call(text):
    """Check if the LLM response contains a tool call.

    TODO 3: Implement this function.

    The LLM signals a tool call by including a fenced block like:
      ```tool_call
      {"tool": "run_command", "command": "ls -la"}
      ```

    Steps:
      1. Check if "```tool_call" appears in the text
      2. Extract the JSON between ```tool_call and the closing ```
      3. Parse it as JSON
      4. If the "tool" field is "run_command", return the "command" field
      5. If anything fails (no match, bad JSON, etc.), return None

    YOUR CODE HERE:
    """
    return None


def run_agent(user_task):
    """Run the agent loop."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_task},
    ]

    print(f"User: {user_task}\n")

    for turn in range(MAX_TURNS):
        print(f"--- Agent turn {turn + 1} ---")

        # Call the LLM
        response_text = call_llm(messages)
        print(f"Agent: {response_text}\n")

        messages.append({"role": "assistant", "content": response_text})

        # Check for a tool call
        command = parse_tool_call(response_text)
        if command is None:
            print("(Agent finished -- no more tool calls)")
            break

        # Execute the command
        print(f"  [Running: {command}]")
        output = run_command(command)
        print(f"  [Output: {output[:500]}]\n")

        # Feed the result back to the LLM
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
        task = ("Write a simple C program that prints 'Hello from the agent!', "
                "compile it with gcc, and run it.")

    print("=" * 60)
    print("  Build-Your-Own AI Agent")
    print(f"  API: {API_URL}")
    print(f"  Model: {MODEL_NAME}")
    print("=" * 60)
    print()

    run_agent(task)


if __name__ == "__main__":
    main()
