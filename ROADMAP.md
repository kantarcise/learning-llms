# Six-month roadmap

Weekly budget: two hours reading, two hours implementation, one hour coding practice, and one hour explanation or review. Dates begin when study starts; the months are checkpoints rather than deadlines.

| Month | Focus | Evidence of understanding |
| --- | --- | --- |
| 1 | PyTorch, tokenization, attention | Implement causal attention and explain each tensor's shape. |
| 2 | Model architecture, generation, KV caching | Generate text and compare cached versus uncached execution. |
| 3 | Remote inference serving and benchmarking | Stream a response from the RTX 3090 to the Mac and measure which workloads meet a client latency target. |
| 4 | Document-inference orchestration | Build a modest asynchronous pipeline with bounded queues, page progress, and ordered results. |
| 5 | Reliability and performance | Recover from a worker failure and investigate one measured bottleneck. |
| 6 | Interview rehearsal | Explain the project, solve timed problems, and handle design follow-ups. |

For month six, shift to one hour reading, one hour project work, and four hours interview preparation each week.

Keep the main reading focused on Raschka. Use selected Machine Learning Systems material for hardware acceleration, benchmarking, serving, and inference at scale. Extensive fine-tuning, custom CUDA kernels, and multi-GPU deployment can wait unless a target role calls for them.

As serving and performance work deepens across Months 3–6, revisit modern attention and memory optimizations alongside our measurements: **FlashAttention / PyTorch SDPA** (tiled SRAM attention and BF16/FP16 preconditions), **GQA (Grouped-Query Attention)**, **PagedAttention** (virtual KV-cache blocks in vLLM), **FlashDecoding / FlashAttention-3**, **MLA (Multi-head Latent Attention)**, and **PyTorch `FlexAttention`**.

## Month 3 lab — call the RTX 3090 from the Mac

The [month-two experiment](docs/preparation/cosmopedia.md#what-happens-as-requests-arrive-faster) measured when tokens were ready on the GPU. This lab follows the same request all the way back to the person waiting for the answer. Use the existing Linux RTX 3090 as the remote compute host and the Mac as the client. Track the workstation inventory, measurements, and eventual keep-or-upgrade decision in [issue #65](https://github.com/kantarcise/learning-llms/issues/65).

1. **Serve a model on Linux.** Run a persistent inference engine such as vLLM or llama.cpp with a small instruction-tuned model that fits the 3090. Record the engine version, model revision, launch command, and how to reach its HTTP endpoint over the LAN or an SSH tunnel.
2. **Send one real request from the Mac.** Give the endpoint a short instruction with an easy-to-check answer. Read the streamed response as it arrives. Trace the path from the HTTP request through the server queue, prompt processing, and token generation to the text displayed on the Mac.
3. **Measure the user's wait.** Choose a p95 time-to-first-token target before testing. On the Mac, time request submission to the first nonempty answer text and to completion. Start with one caller, then vary concurrent callers and prompt length separately while keeping the output budget fixed. Record completed requests per second, errors, and the highest sustained load tested that meets the target. Use server timings, where available, to investigate where requests spent their time.
4. **Hand it to a second caller.** Save the server launch command, a copyable client request, and a small results table. Someone else should be able to reproduce the request and see which workload the deployment handled. Reuse that endpoint for the month-four document pipeline.

## Monthly review

- What can I now explain without notes?
- Which implementation or measurement supports that explanation?
- What still confuses me?
- What should I reduce or revisit next month?
