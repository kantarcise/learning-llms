"""Score saved Chapter 7 responses with a local Ollama judge (§7.8)."""

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter
from urllib.request import Request, urlopen

Record = dict[str, str]


def query_model(
    prompt: str,
    model: str = "llama3:8b",
    temperature: float = 0.0,
    seed: int | None = 123,
    base_url: str = "http://localhost:11434",
) -> str:
    options: dict[str, float | int] = {"temperature": temperature}
    if seed is not None:
        options["seed"] = seed
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": options,
    }
    request = Request(
        f"{base_url}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        return json.load(response)["message"]["content"].strip()


def judge_info(
    model: str, base_url: str = "http://localhost:11434"
) -> dict[str, object]:
    version_req = Request(f"{base_url}/api/version")
    with urlopen(version_req, timeout=30) as response:
        ollama_version = json.load(response).get("version")
    tags_req = Request(f"{base_url}/api/tags")
    model_meta: dict[str, object] = {}
    with urlopen(tags_req, timeout=30) as response:
        for item in json.load(response).get("models", []):
            if item.get("name") == model:
                model_meta = {
                    "name": item.get("name"),
                    "digest": item.get("digest"),
                    "size": item.get("size"),
                    "details": item.get("details"),
                }
                break
    return {"ollama_version": ollama_version, "model": model_meta}


def judge_prompt(entry: Record) -> str:
    return (
        f"Instruction: {entry['instruction']}\nInput: {entry['input']}\n"
        f"Reference answer: {entry['output']}\nCandidate answer: {entry['model_response']}\n"
        "Rate how well the candidate fulfills the instruction from 0 to 100. "
        "Allow correct answers with different wording."
    )


def score_once(
    entries: list[Record],
    model: str,
    temperature: float,
    seed: int | None,
) -> dict[str, object]:
    start = perf_counter()
    scores: list[int | None] = []
    invalid: list[dict[str, object]] = []
    for index, entry in enumerate(entries):
        reply = query_model(
            judge_prompt(entry) + " Reply with one integer only.",
            model=model,
            temperature=temperature,
            seed=seed,
        )
        if reply.isascii() and reply.isdecimal() and 0 <= int(reply) <= 100:
            scores.append(int(reply))
        else:
            scores.append(None)
            invalid.append({"index": index, "raw_reply": reply})
    elapsed = perf_counter() - start
    valid = [s for s in scores if s is not None]
    return {
        "temperature": temperature,
        "seed": seed,
        "seconds": elapsed,
        "scored_count": len(valid),
        "total_count": len(entries),
        "mean": mean(valid) if valid else None,
        "min": min(valid) if valid else None,
        "max": max(valid) if valid else None,
        "scores": scores,
        "invalid": invalid,
    }


def main(
    responses_path: Path,
    output_path: Path,
    model: str,
    repeats: int,
    temperature: float,
    seeds: list[int] | None,
    explain_count: int,
) -> None:
    entries: list[Record] = json.loads(responses_path.read_text(encoding="utf-8"))
    explanations: list[dict[str, str]] = []
    for entry in entries[:explain_count]:
        explanation = query_model(
            judge_prompt(entry) + " Explain your score.",
            model=model,
            temperature=0.0,
            seed=123,
        )
        explanations.append(
            {
                "instruction": entry["instruction"],
                "input": entry["input"],
                "reference": entry["output"],
                "model_response": entry["model_response"],
                "explanation": explanation,
            }
        )

    run_seeds: list[int | None]
    if seeds:
        run_seeds = list(seeds)
    else:
        run_seeds = [123] * repeats

    runs = [
        score_once(entries, model=model, temperature=temperature, seed=seed)
        for seed in run_seeds
    ]
    means = [r["mean"] for r in runs if isinstance(r["mean"], float)]
    result = {
        "responses_file": str(responses_path),
        "judge": judge_info(model),
        "prompt_suffix_score": " Reply with one integer only.",
        "prompt_suffix_explain": " Explain your score.",
        "explanations": explanations,
        "runs": runs,
        "summary": {
            "run_means": means,
            "mean_of_means": mean(means) if means else None,
            "min_mean": min(means) if means else None,
            "max_mean": max(means) if means else None,
            "range": (max(means) - min(means)) if means else None,
        },
    }
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"{responses_path.name}: {len(runs)} run(s), means={[round(m, 2) for m in means]}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("responses", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="llama3:8b")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--explain-count", type=int, default=0)
    args = parser.parse_args()
    main(
        args.responses,
        args.output,
        args.model,
        args.repeats,
        args.temperature,
        args.seeds,
        args.explain_count,
    )
