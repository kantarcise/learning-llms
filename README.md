# learning-llms

Learning large language models by reading, implementing, measuring, and writing.

This repository follows my study of Sebastian Raschka's *Build a Large Language Model (From Scratch)*, then connects those foundations to inference engineering. The pace is six hours a week for six months. Notes and explanations will grow as I learn; an empty template does not count as completed learning.

## Book companion

The main artifact is a chapter-by-chapter reading companion, following the approach in my DSA and data-engineering notes. Pages use the book as their organizing structure and grow with explanations, examples, diagrams, and useful tangents.

- [Companion home and chapter navigation](docs/index.md)
- `mkdocs.yml` defines a Material for MkDocs site, consistent with my existing learning sites. Cloudflare Pages build settings and the Git publishing workflow are documented in [Deployment](DEPLOYMENT.md). Account connection and domain setup remain to be done.

## Start here

- [Step Zero — before the book arrives](reading/00-step-zero.md)
- [Reading index](reading/README.md)
- [Six-month roadmap](ROADMAP.md)
- [Experiments](experiments/README.md)
- [Coding and design practice](practice/README.md)

## How I study

1. Read with full attention.
2. Close the source and explain the idea in my own words.
3. Implement a small example or predict an experiment's outcome.
4. Run it, inspect the result, and document what changed my understanding.
5. Revisit unresolved questions.

Use the [chapter-note template](templates/chapter-note.md) and [experiment template](templates/experiment.md) as needed. Link notes to the code they explain and link experiments to their results.

## Sources

- [Book, exercises, and learning resources](https://sebastianraschka.com/llms-from-scratch/)
- [Author's companion code](https://github.com/rasbt/LLMs-from-scratch)
- [Machine Learning Systems](https://mlsysbook.ai/) — selected reading on hardware, benchmarking, and serving.

Write original explanations and link to source material. Attribute any adapted code and retain its applicable license notices. Keep book PDFs, model weights, credentials, and private work data outside the repository.

## Environment

Python environments and dependencies use uv, with the documentation dependency declared in `pyproject.toml` and resolved in `uv.lock`. Run `uv sync --locked` to set up the environment and `uv run --locked python -m mkdocs serve` to preview the companion. Record tested versions and run commands beside runnable exercises. Use Ruff for Python linting and formatting. CPU execution is sufficient for the initial tensor exercises; real GPU measurements come later.
