#!/usr/bin/env python3
"""
ask_model.py -- Send a single chat-completion request to the local vLLM server.

Auto-discovers the server URL from the shared file written by start_server.sh.
Streams the response and reports time-to-first-token + tokens/sec, which is
useful for getting a feel for different model sizes/configurations.

Examples:
    # Simplest -- pass prompt as an argument
    python3 ask_model.py "Write a HIP kernel that adds two vectors."

    # Pipe a prompt from stdin (good for long prompts or pasted code)
    cat snippet.c | python3 ask_model.py "Explain what this code does."

    # Read entire prompt from stdin
    echo "What is MPI_Allreduce?" | python3 ask_model.py

    # Tune sampling
    python3 ask_model.py "Write a quicksort in C." --temperature 0.0 --max-tokens 1024

    # Override defaults
    python3 ask_model.py "..." --system "You are a terse expert."
    python3 ask_model.py "..." --url http://10.0.100.182:8321/v1
    python3 ask_model.py "..." --no-stream

    # Just list available models on the server
    python3 ask_model.py --list-models

URL discovery order:
    1. --url command-line flag
    2. AGENT_API_URL environment variable
    3. --url-file (default: $WORK/sc26_agent_server_url)

Requires only the Python standard library -- no venv activation needed.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

_WORK = os.environ.get("WORK")
DEFAULT_URL_FILE = f"{_WORK}/sc26_agent_server_url" if _WORK else None
DEFAULT_SYSTEM = "You are a helpful, concise coding assistant."


def discover_url(cli_url, url_file):
    if cli_url:
        return cli_url.rstrip("/")
    env_url = os.environ.get("AGENT_API_URL")
    if env_url:
        return env_url.rstrip("/")
    if url_file and os.path.isfile(url_file):
        with open(url_file) as f:
            url = f.read().strip()
        if url:
            return url.rstrip("/")
    tried = "--url, $AGENT_API_URL"
    if url_file:
        tried += f", {url_file}"
    else:
        tried += " (and $WORK is not set, so no default URL file)"
    sys.exit(
        "ERROR: Could not determine server URL.\n"
        f"  Tried: {tried}\n"
        "  Is the vLLM server running? Check with: squeue -u $USER --name=vllm-server"
    )


def http_json(url, payload=None, timeout=300, stream=False):
    """POST JSON (or GET if payload is None) and return the response object."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummy",  # vLLM ignores this but some clients require it.
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers,
                                 method="POST" if data else "GET")
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.URLError as e:
        sys.exit(f"ERROR: Could not reach {url}\n  {e}")


def list_models(base_url):
    with http_json(f"{base_url}/models", timeout=10) as resp:
        data = json.loads(resp.read())
    return [m["id"] for m in data.get("data", [])]


def stream_chat(base_url, model, messages, temperature, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    start = time.time()
    first_token = None
    final_usage = None

    with http_json(f"{base_url}/chat/completions", payload) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            body = line[len("data: "):]
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                final_usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first_token is None:
                        first_token = time.time()
                    sys.stdout.write(content)
                    sys.stdout.flush()

    elapsed = time.time() - start
    ttft = (first_token - start) if first_token else 0.0
    return elapsed, ttft, final_usage


def nonstream_chat(base_url, model, messages, temperature, max_tokens):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    start = time.time()
    with http_json(f"{base_url}/chat/completions", payload) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - start
    text = data["choices"][0]["message"]["content"]
    sys.stdout.write(text)
    sys.stdout.flush()
    return elapsed, data.get("usage")


def main():
    p = argparse.ArgumentParser(
        description="Send a single chat-completion request to a local vLLM "
                    "(OpenAI-compatible) server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See the docstring at the top of this file for examples.",
    )
    p.add_argument("prompt", nargs="?",
                   help="The user prompt. If omitted, read from stdin.")
    p.add_argument("--model", default=None,
                   help="Model name (default: first model the server reports).")
    p.add_argument("--system", default=DEFAULT_SYSTEM,
                   help="System prompt (default: a generic coding assistant prompt).")
    p.add_argument("--max-tokens", type=int, default=512,
                   help="Maximum tokens to generate (default: 512).")
    p.add_argument("--temperature", type=float, default=0.2,
                   help="Sampling temperature (default: 0.2; use 0.0 for deterministic).")
    p.add_argument("--url", default=None,
                   help="Server base URL. Overrides --url-file and $AGENT_API_URL.")
    p.add_argument("--url-file", default=DEFAULT_URL_FILE,
                   help="File containing the server URL "
                        "(default: $WORK/sc26_agent_server_url).")
    p.add_argument("--no-stream", action="store_true",
                   help="Wait for the full response instead of streaming.")
    p.add_argument("--list-models", action="store_true",
                   help="List available models on the server and exit.")
    args = p.parse_args()

    base_url = discover_url(args.url, args.url_file)

    if args.list_models:
        for name in list_models(base_url):
            print(name)
        return

    if args.prompt is not None:
        prompt = args.prompt
        if not sys.stdin.isatty():
            extra = sys.stdin.read().strip()
            if extra:
                prompt = f"{prompt}\n\n{extra}"
    elif not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    else:
        sys.exit("ERROR: No prompt. Pass it as an argument or pipe via stdin.")

    if not prompt:
        sys.exit("ERROR: Empty prompt.")

    model = args.model or list_models(base_url)[0]

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": prompt},
    ]

    print(f"--- {model} @ {base_url}", file=sys.stderr)
    print(f"--- temperature={args.temperature}  max_tokens={args.max_tokens}\n",
          file=sys.stderr)

    if args.no_stream:
        elapsed, usage = nonstream_chat(base_url, model, messages,
                                        args.temperature, args.max_tokens)
        completion = (usage or {}).get("completion_tokens")
        rate = (completion / elapsed) if completion and elapsed > 0 else None
    else:
        elapsed, ttft, usage = stream_chat(base_url, model, messages,
                                           args.temperature, args.max_tokens)
        completion = (usage or {}).get("completion_tokens")
        rate = (completion / elapsed) if completion and elapsed > 0 else None

    print("\n", file=sys.stderr)
    print(f"--- elapsed: {elapsed:.2f}s", file=sys.stderr, end="")
    if not args.no_stream:
        print(f"  |  time-to-first-token: {ttft:.2f}s", file=sys.stderr, end="")
    if completion is not None:
        print(f"  |  completion tokens: {completion}", file=sys.stderr, end="")
    if rate is not None:
        print(f"  |  {rate:.1f} tok/s", file=sys.stderr, end="")
    print("", file=sys.stderr)


if __name__ == "__main__":
    main()
