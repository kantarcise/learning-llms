# Six-month roadmap

Weekly budget: two hours reading, two hours implementation, one hour coding practice, and one hour explanation or review. Dates begin when study starts; the months are checkpoints rather than deadlines.

| Month | Focus | Evidence of understanding |
| --- | --- | --- |
| 1 | PyTorch, tokenization, attention | Implement causal attention and explain each tensor's shape. |
| 2 | Model architecture, generation, KV caching | Generate text and compare cached versus uncached execution. |
| 3 | Serving and benchmarking | Serve a small model; measure concurrency and sequence-length effects. |
| 4 | Document-inference orchestration | Build a modest asynchronous pipeline with bounded queues, page progress, and ordered results. |
| 5 | Reliability and performance | Recover from a worker failure and investigate one measured bottleneck. |
| 6 | Interview rehearsal | Explain the project, solve timed problems, and handle design follow-ups. |

For month six, shift to one hour reading, one hour project work, and four hours interview preparation each week.

Keep the main reading focused on Raschka. Use selected Machine Learning Systems material for hardware acceleration, benchmarking, serving, and inference at scale. Extensive fine-tuning, custom CUDA kernels, and multi-GPU deployment can wait unless a target role calls for them.

## Monthly review

- What can I now explain without notes?
- Which implementation or measurement supports that explanation?
- What still confuses me?
- What should I reduce or revisit next month?
