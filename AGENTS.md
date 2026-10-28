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
- Prefer CLIs over browser automation whenever the operation is supported. Use the authenticated GitHub CLI (`gh`) and Cloudflare Wrangler on this machine for repository and deployment work. If authentication expires or required permissions are missing, ask the user to reauthenticate or grant the needed access rather than switching to the browser. Use the browser only when the task requires it or CLI capabilities are insufficient, and explain why.
- Type-hint new and modified Python function/method parameters and return values, including constructors and tests (self/cls need no annotation). Prefer clear standard types and torch.Tensor; document tensor shapes, dtypes, and axis meanings in docstrings. Annotate stored collections when inference is unclear. Do not add typing dependencies or obscure the teaching code with unnecessary type machinery. Type hints do not validate tensor shapes at runtime.
- Preserve recorded model outputs and deliberately faulty benchmark fixtures verbatim; apply the typing standard to maintained source, not historical experiment evidence. Existing unrelated source can migrate incrementally. The character-log experiment enforces annotation presence with Ruff in CI; this is not full static type checking.
- Ask before adding new production dependencies. Use Ruff for Python linting and formatting.
- Ask one targeted question when intent is materially ambiguous; otherwise complete authorized work.
- Before merging any PR to main, run every command in DEPLOYMENT.md under "Required local checks before merging" on this machine against the final branch content. Fix failures before merging; do not treat absent PR checks as success. Rerun after further edits or incorporating changes from main, and record the tested commit and results in the PR. Local checks are mandatory even for documentation-only changes.
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
- User confirmed working Cloudflare build command: mkdocs build. Output directory: site. Production branch: main. Python version: 3.12.10.
- Python project management uses uv. pyproject.toml declares dependencies, uv.lock locks them, and builds run uv run --locked python -m mkdocs build --strict directly.
- Current policy: GitHub Actions runs documentation and experiment checks only on pushes to main; the full local suite is required before merging (see DEPLOYMENT.md).
- Local Markdown links and navigation targets were checked. No build wrapper script is needed.
- Strict documentation build passed using uv, Python 3.12.10, and pinned Material 9.6.14. GitHub Actions also checks the build.
- GitHub CLI authentication was refreshed and verified for kantarcise on 2026-10-03.
- Cloudflare Pages is live at https://learningllms.kantarcise.com/ with a user-configured custom domain. HTTPS home page was independently fetched and contained the site title, chapter navigation, and search markup. Successful build dependency-install settings have not been inspected.

Keep chapter scaffold status honest. Local work and CI use uv; Cloudflare uses the working MkDocs build configuration.
