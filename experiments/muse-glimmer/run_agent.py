"""Run one bounded agent trial using llama.cpp's container-backed tools."""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def request(path, payload=None):
    req = urllib.request.Request(
        "http://127.0.0.1:8080" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-tool-cwd": "/workspace",
            "x-tool-runtime": "docker-container:muse-glimmer-playground",
        },
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()
    # Start the server with --tools-runtime docker-container:muse-glimmer-playground.
    # Tool execution must stay inside that disposable container.
    allowed = {"read_file", "write_file", "exec_shell_command"}
    tools = [t["definition"] for t in request("/tools") if t["tool"] in allowed]
    if len(tools) != len(allowed):
        raise ValueError("Start llama-server with all three required tools")
    messages = [
        {
            "role": "user",
            "content": "Work in /workspace. Inspect chunks.py and test_chunks.py, "
            "run the tests, and fix chunks.py so all tests pass. Do not modify "
            "the tests or add dependencies. Run the tests again after your fix "
            "and report the result. Use the available tools to do the work.",
        }
    ]
    trace = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for turn in range(args.max_turns):
        payload = {
            "model": "muse-glimmer-30B",
            "messages": messages,
            "tools": tools,
            "stream": False,
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "seed": 42,
            "chat_template_kwargs": {"reasoning_strength": "low"},
        }
        started = time.monotonic()
        response = request("/v1/chat/completions", payload)
        entry = {"request": payload, "response": response, "tools": []}
        entry["elapsed_seconds"] = round(time.monotonic() - started, 3)
        trace.append(entry)
        args.output.write_text(json.dumps(trace, indent=2) + "\n")
        choice = response["choices"][0]
        message = choice["message"]
        messages.append(
            {k: v for k, v in message.items() if k in {"role", "content", "tool_calls"}}
        )
        calls = message.get("tool_calls") or []
        print(
            f"Turn {turn + 1}: {choice['finish_reason']}, {len(calls)} tools",
            flush=True,
        )
        if not calls:
            print(message.get("content") or "No final answer", flush=True)
            return
        for call in calls:
            function = call["function"]
            if function["name"] not in allowed:
                raise ValueError("Unexpected tool")
            result = request(
                "/tools",
                {"tool": function["name"], "params": json.loads(function["arguments"])},
            )
            entry["tools"].append({"call": call, "result": result})
            args.output.write_text(json.dumps(trace, indent=2) + "\n")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(result),
                }
            )
    print(
        f"Stopped at the {args.max_turns}-turn limit; check the trace and tests",
        flush=True,
    )


if __name__ == "__main__":
    main()
