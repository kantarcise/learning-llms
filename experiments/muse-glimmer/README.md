# Experiment — Muse Glimmer for everyday engineering

Related: [issue #3](https://github.com/kantarcise/learning-llms/issues/3).

## What this trial showed

Muse Glimmer is usable for interactive coding chat on this RTX 3090. In this trial it passed two small executable coding tasks, explained a KV-cache calculation correctly, and used tools to fix a tiny Python module until seven test methods passed. The user also tried the chat interface and reported that it felt fast.

Qwen3.8-27B reused the weights already downloaded by LM Studio; Gemma4:26b and Qwen3.5:9b reused existing Ollama installations. **Only Muse was downloaded for this experiment.** The exact Gemma-4-31B and Qwen3.6-27B opponents mentioned in the issue were not downloaded or evaluated.

Muse and Qwen3.8 both gave useful answers with mistakes in code review and system design. This is an exploratory side experiment, separate from chapter progress, with one sample per task/configuration and no overall benchmark score. Read this file for the findings; the other files preserve evidence and reproduction tools.

## Question

Can Muse Glimmer produce useful code, spot correctness problems, and help reason about serving systems on this Linux machine? Start with chat, then try an agent that can edit files and run tests. A handful of tasks can guide more tinkering; it cannot establish general superiority over another model.

## Prediction before running

No user prediction recorded. Working hypothesis for the trial: text-only Q4_K_M should fit the available 24 GiB GPU at modest context. Coding quality remains unverified.

## Setup and reproduction

Observed on 2026-10-12:

- Ubuntu 22.04.4 LTS, kernel `6.8.0-138-generic`, x86_64.
- NVIDIA RTX 3090, 24,576 MiB VRAM; approximately 770 MiB already used by desktop applications before inference.
- NVIDIA driver `560.35.03`; driver reports CUDA compatibility 12.6. Installed CUDA toolkit runtime is 12.3.
- Python: command-line `python3` 3.11.11 for the standard-library experiment scripts; repository environment 3.12.10 for the locked documentation build; disposable agent container 3.9.19. Model inference itself runs in llama.cpp/Ollama, not Python.
- System RAM: 31 GiB, approximately 25 GiB available; no swap.
- Disk: approximately 63 GiB free before downloading.
- Ollama `0.20.0`. Installed comparisons: `gemma4:26b` (25.8B, Q4_K_M) and `qwen3.5:9b` (9.7B, Q4_K_M). These differ from the models in the motivating post. `gemma4:latest` and `gemma4:e4b` are also present and share one model digest.
- A later inspection also found LM Studio's existing `lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf`, 16,810,714,336 bytes. Its internal metadata identifies Qwen3.8-27B, architecture `qwen35`, file type 15 (Q4_K_M), with thinking-control support. Metadata is saved in `results/qwen3.8-local-artifact.json`. No Qwen download is required; reuse this file with llama.cpp. The user chose installed models for this first trial and asked about Qwen3.8, so include this stronger existing opponent as well. The earlier Qwen3.5-9B runs remain diagnostic baselines.

The sandbox cannot access the GPU; host-level `nvidia-smi` succeeds. Do not diagnose a broken driver from the sandbox failure.

Selected Muse artifact:

- Repository: [`meta-models/Muse-Glimmer-30B-GGUF`](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF).
- Revision: `70bf1b61ac09f91b24d39038091b41c582bc5d7a`.
- File: `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`, 16,756,683,904 bytes.
- Expected SHA-256 from Hugging Face LFS metadata: `4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e`.
- Text only. Vision and DFlash are separate, later trials.
- Runtime selected for compilation: llama.cpp `v0.6.0`, checked-out commit `d81235049384534c167caea52b85a694f6103d14`. This checkout registers `MUSE_GLIMMER`.
- Verified weights and working binaries now live under `/home/sezai/.local/share/muse-glimmer/`, outside Git. The source/build tree and working logs remain under `/tmp/muse-glimmer-runtime/`. `serve.sh` supplies the library path needed by the relocated binaries, and library resolution was checked against the persistent directory.

The [model card](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF#things-that-will-otherwise-cost-you-an-hour) requires Jinja templating and explains that thinking cannot be disabled. Use low reasoning effort for the first interactive trial. Do not add a stop string for `<|eom|>`.

Build command (uses an already installed CUDA toolkit):

```bash
cmake -S /tmp/muse-glimmer-runtime/llama.cpp \
  -B /tmp/muse-glimmer-runtime/llama.cpp/build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF
cmake --build /tmp/muse-glimmer-runtime/llama.cpp/build -j 6 --target llama-server
```

Verified text-only launch configuration is in [serve.sh](serve.sh) (the live trial additionally enables the container-backed tools below):

```bash
bash experiments/muse-glimmer/serve.sh
```

Once healthy, open `http://127.0.0.1:8080` for chat. Each request gets one 8,192-token context; output and reasoning need room inside that context.

Run the saved prompts from the repository root:

```bash
python3 experiments/muse-glimmer/run.py --model muse-glimmer-30B \
  --output experiments/muse-glimmer/results/muse-q4km
python3 experiments/muse-glimmer/run.py --api ollama \
  --base-url http://127.0.0.1:11434 --model qwen3.5:9b \
  --output experiments/muse-glimmer/results/qwen3.5-9b
```

Run models sequentially so they do not compete for VRAM. The Ollama runner unloads after each task (`keep_alive=0`), so wall time includes cold starts. The runner saves the exact request, raw response, errors, and elapsed time. It does not measure time to first token or peak memory. Record model digests and runtime startup logs separately. A fixed seed across different runtimes does not make their sampling identical. Reasoning settings are model specific.

Qwen3.8 reuses the existing LM Studio GGUF with the same llama.cpp build, one 8,192-token slot, text only, and no speculative decoding. Its [official model card](https://huggingface.co/Qwen/Qwen3.8-27B) documents low reasoning effort and thinking-mode sampling with top-k 20. After stopping Muse, launch it on a separate localhost port:

```bash
env LD_LIBRARY_PATH="$HOME/.local/share/muse-glimmer/runtime" \
  "$HOME/.local/share/muse-glimmer/runtime/llama-server" \
  -m "$HOME/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf" \
  --alias qwen3.8-27B --host 127.0.0.1 --port 8081 \
  -ngl 99 -c 8192 -np 1 --jinja \
  --chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"low"}' \
  --temp 1.0 --top-p 0.95 --top-k 20

python3 experiments/muse-glimmer/run.py --model qwen3.8-27B \
  --base-url http://127.0.0.1:8081 --top-k 20 --reasoning-effort low \
  --template-kwargs '{"enable_thinking":true,"reasoning_effort":"low"}' \
  --output experiments/muse-glimmer/results/qwen3.8-27b
```

Use a fresh output directory for repeated trials to preserve earlier answers. The existing Qwen artifact's SHA-256 is `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520`; its original download revision is unknown. Startup logs show its extra MTP tensors being ignored in this ordinary decoding run.

## Measurements

`prompts.json` contains five original prompts and explicit review criteria. Two require executable Python answers. Review generated source before running it, and run it in a sandbox with a timeout:

```bash
timeout 10s python3 experiments/muse-glimmer/check_code.py bug-fix /path/to/reviewed-answer.py
timeout 10s python3 experiments/muse-glimmer/check_code.py constrained-code /path/to/reviewed-answer.py
```

These checks cover representative edge cases, not exhaustive correctness. Review the linear-time constraint and source-only output separately. For the queue worker and serving design, assess concrete failure sequences, feasible fixes, and uncertainty. For the KV-cache explanation, the expected byte counts are in the prompt's review field.

Treat a token-limit finish with no final answer as an incomplete response. Preserve it and repeat with a larger budget or a different recorded reasoning setting. Do not silently extract proposed code from a reasoning trace and count it as success.

## Monitoring and performance

The saved [GPU samples](results/gpu-samples.csv) contain 897 readings over approximately 15 minutes, targeting one reading per second. Columns have no header in the original capture: timestamp, total device memory used in MiB, device memory capacity in MiB, and GPU utilization percent. The command was equivalent to:

```bash
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits -l 1
```

| Observation | Recorded value | Interpretation |
| --- | ---: | --- |
| First / last sample | `2026/10/12 22:12:09.195` / `2026/10/12 22:27:08.297` | Host timestamps as captured; timezone not recorded in CSV |
| Device capacity | 24,576 MiB | RTX 3090 |
| Minimum / maximum device memory used | 740 / 17,188 MiB | Whole-device observations including desktop processes |
| Maximum sampled GPU utilization | 100% | A sample peak, not sustained utilization |
| Muse generation rate, five chat prompts | 40.98–41.86 tokens/s | Server decode timing, including reasoning tokens |
| Qwen3.8 generation rate, five chat prompts | 38.38–40.10 tokens/s | Same runtime; different tokenization, output lengths and sampling |
| Muse initial load / Qwen3.8 load | approximately 179 / 66 seconds | Different cache state; not a controlled loading benchmark |

The CSV spans activity and idle periods during the trial; it is not annotated with per-request boundaries. Therefore 17,188 MiB is the maximum **observed across the capture**, not a per-model peak allocation. Sampling can miss brief spikes. RAM, power, temperature, per-process VRAM, and time to first token were not captured as time series. No performance monitoring of the user's subsequent chat is included.

Request wall times below come from the runner's monotonic clock. llama.cpp responses also include prompt/decode timing and token counts; Ollama responses include load and evaluation durations. Gemma was unloaded after each request, so its wall times include approximately 47–62 seconds of loading. Its five generation durations were approximately 18–35 seconds. Comparing those wall times directly with the kept-loaded Muse and Qwen3.8 would conflate loading and inference.

Muse often generated fewer tokens than Qwen3.8: for the bug fix, 305 versus 639, including reasoning. The shorter total response time therefore reflects output length as well as decode rate. These are single-request observations, not concurrency, throughput-under-load, or percentile-latency measurements.

## Results versus prediction

Muse's download completed and its SHA-256 matched the published LFS hash. The CUDA build loaded the model in approximately 179 seconds from disk, with one 8,192-token slot. The smoke test returned `391` for `17 * 23` in 3.853 seconds after loading. Runtime timing reported 26.90 generated tokens/second for this tiny response; this is not a general throughput result. Total GPU memory after loading was approximately 16,794 MiB, including the desktop. See `results/muse-smoke.json`.

Muse completed all five engineering prompts at low reasoning effort, with the model kept loaded:

| Task | Wall time | Reviewed result |
| --- | ---: | --- |
| Bug fix | 7.473 s | Passed the executable checks |
| Constrained function | 6.965 s | Passed the executable checks; expected linear algorithm |
| Queue-worker review | 17.971 s | Useful conflict-safe repair, but an incorrect failure sequence in the explanation |
| Serving design | 13.987 s | Useful metrics and cancellation plan; proposed bounds need refinement |
| KV-cache explanation | 19.955 s | Correct worked shapes and byte counts |

The queue-worker response uses `ON CONFLICT(job_id) DO NOTHING`, commit-before-ack, rollback, and visible failures. However, its duplicate sequence says a crash before commit produces a durable duplicate; that is wrong for a normal transaction, and the original code already acknowledges first. A valid redelivery window in the repaired design is commit succeeding before acknowledgement succeeds. Its serving answer also mixes token count with estimated KV cost without defining units and suggests an arbitrary queue multiplier. Useful answers still require review.

Four initial Qwen3.5-9B thinking-mode requests reached the 4,096-token limit without final answers; raw responses are preserved under `results/qwen3.5-9b/`. The serving-design response completed. This is an observed configuration failure, not a conclusion about all Qwen coding ability.

Repeating Qwen's bug-fix prompt with a 16,384-token context and 8,192-token output budget completed after 6,083 generated tokens and 80.879 seconds including loading. Its final source passed the chunking checks, including generators and invalid sizes. The original failed run took 72.708 seconds with the smaller budget. See `results/qwen3.5-9b-retry/`; the testable source is copied verbatim from the final response, not from its reasoning trace.

Qualitative review of Qwen's serving answer: admission control, cancellation, and phase-specific metrics are useful starting points. The suggested 80–85% memory threshold and cancellation at twice median decode time are arbitrary, and the response does not establish a token reservation policy or clearly separate queue wait, time to first token, and inter-token latency. Do not implement these thresholds without workload measurements.

Gemma's two final coding answers passed the same edge-case checks. Its queue-worker review identified premature acknowledgement and swallowed exceptions, but its proposed unique constraint does not fully handle redelivery: a duplicate insert raises, so the message can repeatedly fail without acknowledgement. The fix needs to recognize an already committed job safely (for example, conflict handling in the same transaction) and acknowledge it. This is a useful example of a plausible review with an incomplete repair.

| Gemma task | Wall time including load |
| --- | ---: |
| Bug fix | 74.949 s |
| Constrained function | 90.189 s |
| Queue-worker review | 66.881 s |
| Serving design | 78.528 s |
| KV-cache explanation | 84.889 s |

Gemma completed all five requests within the original budget. Its KV-cache answer has the expected shapes and byte counts. Its serving answer usefully separates queue wait, TTFT, output-token latency, and KV occupancy. However, ordinary continuous batching alone does not guarantee that a long prefill cannot stall decoding; that requires appropriate scheduling, often chunked prefill. Total VRAM admission also needs room for weights and temporary buffers, not just cached tokens.

Qwen without thinking completed all five prompts in approximately 5–10 seconds per request in this session. Its constrained function passed, but its bug fix failed: it consumes the entire iterator into one list and never applies `chunk_size`; it also accepts booleans. Its KV-cache explanation incorrectly claims O(1) inference and gives 51,456 instead of 51,712 bytes after the append. Its serving answer suggests killing an execution thread to free GPU memory, which is not a reliable cancellation/cleanup strategy. Its queue review identifies the duplicate risk, but broadly treating every `IntegrityError` as success can conceal unrelated integrity failures. These outputs are under `results/qwen3.5-9b-no-think/`.

Timing differences here include model loading, filesystem cache state, reasoning effort, and different runtimes. They are observations about these requests, not an isolated speed comparison. Sampled outputs should not be generalized to every response from these models.

Qwen3.8-27B completed all five prompts at low reasoning effort on the same llama.cpp build, with the model kept loaded:

| Task | Wall time | Reviewed result |
| --- | ---: | --- |
| Bug fix | 16.988 s | Passed the executable checks |
| Constrained function | 12.503 s | Passed the executable checks; expected linear algorithm |
| Queue-worker review | 30.872 s | Conflict-safe insert, but no rollback on database failure and assumptions about the queue API |
| Serving design | 32.599 s | Token reservation includes output budget; queue cap and likely bottleneck are still guesses |
| KV-cache explanation | 29.209 s | Correct worked shapes and byte counts |

The Qwen3.8 queue answer overstates that a few tests prove a no-loss guarantee. Its repair needs transaction cleanup appropriate to the database, and the suggested `nack` interface is an assumption. Its serving answer usefully includes a decode reservation, but a fixed queue cap and a guessed prefill bottleneck need measurements. These are useful starting points, not production-ready designs.

Practical verdict: Muse is worth more tinkering on this machine. It supplied correct small coding answers, a correct worked explanation, and a verified tool-driven fix. In these single samples it completed chat requests sooner than Qwen3.8 at their respective low reasoning settings. Both produced review/design answers needing corrections. This is not evidence that Muse is generally better than Qwen or Gemma; Qwen3.8 and Gemma were not tested as agents here. DFlash, vision, long-context retrieval, and repository-scale engineering remain untested.

## Trade-offs and limitations

This is a small, single-machine exploration with different model sizes and runtimes. It does not establish benchmark rankings. Chat answers do not demonstrate reliable autonomous repository work. Agent evaluation must separately test file discovery, edits, tool arguments, test execution, and recovery after a failed test.

## Agent trial: inspect, edit, and verify

The question was whether Muse could inspect a failing test and produce a focused patch in a disposable workspace without repeated human correction. This trial used a tiny fixture rather than a repository checkout.

The `agent-fixture/` directory supplies a deliberately broken module and seven unittest methods. Baseline execution in the disposable container produced three failures and eight errors (subtests account for the extra errors). This is test-fixture validation, not a model result.

For the agent trial, a container named `muse-glimmer-playground` uses the existing `python:3.9-slim` image, no network, no host mounts, a 512 MiB RAM limit, one CPU, and 64 processes. Only the fixture is copied into `/workspace`. This Python version differs from the repository's Python 3.12 environment; the fixture uses compatible standard-library code.

Image ID: `sha256:7bfcddd27f2b23ee199f3e4a938eaa80ef01c609d6d1c160787e764229ec3cb9`.

Start llama-server with these additional arguments to route its tools into that container:

```bash
--tools read_file,write_file,exec_shell_command \
--tools-runtime docker-container:muse-glimmer-playground
```

`run_agent.py` uses the server's tool definitions, permits eight model turns by default (`--max-turns` changes the bound), and saves requests, responses, and actual tool results. Run it only with the container-backed server configuration above:

```bash
python3 experiments/muse-glimmer/run_agent.py \
  --output experiments/muse-glimmer/results/muse-agent.json
```

Independently rerun the tests and inspect the module and unchanged test file before judging success. A final message claiming success is not sufficient. The fixture is deliberately tiny and does not test repository-scale planning.

Observed agent result: the first run exhausted eight turns because the copied module was owned by UID 1000 and container root lacked capabilities to overwrite it. This was a setup error; preserve `results/muse-agent.json` as the failed trace. Recreating that same file as container root fixed ownership without changing the starting code. A fresh retry with a twelve-turn allowance completed in seven turns: it recovered from trying to read the workspace directory as a file, discovered the files, ran the failing tests, wrote a focused fix, reran tests, and reported success. An independent unittest run passed all seven methods, and a byte comparison confirmed the test file was unchanged. See `results/muse-agent-retry.json`, `results/muse-agent-chunks.py`, and `results/agent-verified-tests.txt`.


To reproduce the successful agent trial, create a fresh container with a different name or remove the previous disposable container first. The following commands assume the name is available and the existing image is installed. Copy files as container root so the limited-capability process can overwrite the module:

```bash
docker run -d --name muse-glimmer-playground --network none \
  --memory 512m --cpus 1 --pids-limit 64 --cap-drop ALL \
  --security-opt no-new-privileges --workdir /workspace \
  python:3.9-slim sleep infinity
docker exec -i muse-glimmer-playground sh -c 'cat > chunks.py' \
  < experiments/muse-glimmer/agent-fixture/chunks.py
docker exec -i muse-glimmer-playground sh -c 'cat > test_chunks.py' \
  < experiments/muse-glimmer/agent-fixture/test_chunks.py
bash experiments/muse-glimmer/serve.sh \
  --tools read_file,write_file,exec_shell_command \
  --tools-runtime docker-container:muse-glimmer-playground
```

Run the agent from a second terminal with `--max-turns 12` and a fresh output path. The successful saved trace sums to 36.338 seconds of model API calls; this excludes separate tool execution time. Independently check the outcome:

```bash
docker exec muse-glimmer-playground python -m unittest -v
docker cp muse-glimmer-playground:/workspace/test_chunks.py /tmp/muse-test-chunks.py
cmp experiments/muse-glimmer/agent-fixture/test_chunks.py /tmp/muse-test-chunks.py
```

## Evidence and remaining work

| Files | Purpose |
| --- | --- |
| [prompts.json](prompts.json) | Exact original tasks and review criteria |
| [run.py](run.py), [check_code.py](check_code.py) | Local API runner and shared executable checks |
| [serve.sh](serve.sh), [run_agent.py](run_agent.py), [agent-fixture](agent-fixture/) | Launcher and bounded agent reproduction |
| [Muse outputs](results/muse-q4km/), [Qwen3.8 outputs](results/qwen3.8-27b/), [Gemma outputs](results/gemma4-26b/) | Exact requests, responses, timings, and reviewed source copies |
| [Qwen3.5 initial](results/qwen3.5-9b/), [larger-budget retry](results/qwen3.5-9b-retry/), [thinking disabled](results/qwen3.5-9b-no-think/) | Preserve incomplete and failed answers as well as successes |
| [Muse server log](results/muse-server.log), [Qwen3.8 server log](results/qwen3.8-server.log), [GPU samples](results/gpu-samples.csv) | Startup/runtime observations and device monitoring |
| [Qwen artifact metadata](results/qwen3.8-local-artifact.json), [smoke test](results/muse-smoke.json) | Artifact identity and initial API check |
| [Initial agent trace](results/muse-agent.json), [successful retry](results/muse-agent-retry.json) | Setup failure and complete retry, including tool results |
| [Baseline tests](results/agent-baseline.txt), [independent final tests](results/agent-verified-tests.txt), [final module](results/muse-agent-chunks.py) | Before/after evidence |

Raw model outputs include reasoning and are kept verbatim, including incorrect answers; they are evidence rather than endorsed advice. Model weights and runtime binaries are outside Git. No new production dependencies were added.

The next useful experiment is a small real repository issue with hidden checks, repeated runs, and the same agent harness for each model. DFlash, vision, long-context retrieval, concurrency, and repository-scale engineering remain untested. Issue #3 should remain open for that follow-up; this report records the completed first trial.
