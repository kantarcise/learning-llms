"""Save identical prompts and raw responses from a local model API."""

import argparse
import json
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api", choices=("llama", "ollama"), default="llama")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", help="Run only the named task")
    parser.add_argument(
        "--context",
        type=int,
        default=8192,
        help="Ollama only; set llama context at launch",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--no-think", action="store_true", help="Ollama only")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--template-kwargs", default='{"reasoning_strength":"low"}')
    parser.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high", "xhigh")
    )
    args = parser.parse_args()
    if urlparse(args.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        parser.error("Use a loopback endpoint for this local experiment")
    if args.no_think and args.api != "ollama":
        parser.error("Muse's template does not support disabling thinking")
    prompts = json.loads(Path(__file__).with_name("prompts.json").read_text())
    if args.task and args.task not in {task["id"] for task in prompts}:
        parser.error("Unknown task")
    args.output.mkdir(parents=True, exist_ok=True)
    for task in prompts:
        if args.task and args.task != task["id"]:
            continue
        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": task["prompt"]}],
            "stream": False,
        }
        if args.api == "ollama":
            path = "/api/chat"
            payload.update(
                think=not args.no_think,
                keep_alive=0,
                options={
                    "num_ctx": args.context,
                    "num_predict": args.max_tokens,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "top_k": args.top_k,
                    "seed": 42,
                },
            )
        else:
            path = "/v1/chat/completions"
            payload.update(
                max_tokens=args.max_tokens,
                temperature=1.0,
                top_p=0.95,
                top_k=args.top_k,
                seed=42,
                chat_template_kwargs=json.loads(args.template_kwargs),
            )
            if args.reasoning_effort:
                payload["reasoning_effort"] = args.reasoning_effort
        request = urllib.request.Request(
            args.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        record = {"task": task, "request": payload, "api": args.api}
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                record["response"] = json.load(response)
        except (OSError, ValueError) as error:
            record["error"] = str(error)
        record["elapsed_seconds"] = round(time.monotonic() - started, 3)
        destination = args.output / (task["id"] + ".json")
        destination.write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"{task['id']}: {record['elapsed_seconds']}s -> {destination}", flush=True
        )


if __name__ == "__main__":
    main()
