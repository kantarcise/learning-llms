# Project context and working agreements

## Purpose

Sezai is a senior software engineer managing a data platform, preparing over six months for inference-platform and model-serving engineering roles at leading AI companies. Study budget: six hours per week. He has bought Sebastian Raschka's Build a Large Language Model (From Scratch); at project setup he is waiting for it to arrive.

The main deliverable is a readable, chapter-by-chapter book companion, published from day one. Writing and documenting are central to how he learns. Keep the book's chapters and sections at the center; the interview roadmap supports that work.

## Reference projects

These are the user's examples of the desired structure and reading experience:

- https://github.com/kantarcise/learningdsainpython
- https://learningdsainpython.kantarcise.com/DS%26A-PythonPrimer/
- https://github.com/kantarcise/learningFundamentalsOfDataEngineering
- https://learningdataengineering.kantarcise.com/

Their pattern is Material for MkDocs, book-aligned navigation, detailed section headings, conversational explanations, code examples, figures, links, and useful tangents. Let sections develop naturally instead of forcing every idea into a rigid worksheet.

## How to collaborate

- The user reads and brings notes, passages, or questions. Help unpack concepts, verify examples, make diagrams when useful, and edit explanations into readable pages.
- Preserve the user's voice. Support his understanding and original writing; do not invent personal insights, completed exercises, reading progress, or benchmark results.
- Link chapter explanations to runnable code as it develops. Record input/output shapes and hand-worked examples where helpful.
- Attribute adapted material and link to sources. Do not commit book PDFs, model weights, credentials, or private employer data.
- Prefer small, focused, human-readable changes. Follow existing conventions.
- Ask before adding new production dependencies. Use Ruff for Python linting and formatting.
- Ask one targeted question when intent is materially ambiguous; otherwise complete authorized work.
- Before declaring completion, verify the requested behavior and summarize changes and checks, including any unverified parts.

## Repository map

- docs/index.md: companion home.
- docs/chapters/: seven book chapters, currently scaffolds.
- docs/appendices/a-pytorch.md: PyTorch companion notes.
- docs/00-step-zero.md: preparation before the book arrives.
- templates/: chapter-note and experiment starters.
- experiments/ and practice/: implementation, measurements, and interview practice as they happen.
- ROADMAP.md: six-month plan at six hours a week.
- DEPLOYMENT.md: Cloudflare Pages connection and publishing instructions.
- mkdocs.yml: site navigation and Material theme.

## Deployment and handoff status

As of 2026-10-03:

- Local Git repository: /home/sezai/repositories/learning-llms, branch main.
- Public GitHub repository created: https://github.com/kantarcise/learning-llms. The companion is published on main.
- Cloudflare is the user's chosen hosting provider. Do not substitute Sites hosting.
- Build command: bash build.sh. Output directory: site. Production branch: main. Python version: 3.12.10.
- Python project management uses uv. pyproject.toml declares dependencies, uv.lock locks them, and build.sh runs uv run --locked python -m mkdocs build --strict.
- GitHub Actions configuration checks documentation builds on pushes to main and pull requests.
- Local Markdown links and navigation targets were checked, and build.sh passed bash syntax validation.
- Strict documentation build passed using uv, Python 3.12.10, and pinned Material 9.6.14. GitHub Actions also checks the build.
- GitHub CLI authentication was refreshed and verified for kantarcise on 2026-10-03.
- Cloudflare account connection, Git integration, custom domain, and live deployment have not been configured. No API tokens are required in repository files for the intended Git integration.

Next task: connect the public GitHub repository to Cloudflare Pages using DEPLOYMENT.md. Keep chapter scaffold status honest and update this status section as work proceeds.
