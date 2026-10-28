# Deploy on Cloudflare Pages

This repository builds a static Material for MkDocs site. The initial chapter pages can ship immediately and grow with each reading session.

## Connect the repository

After the repository is on GitHub, create a Cloudflare **Pages** project with Git integration and select this repository. Use these settings:

| Setting | Value |
| --- | --- |
| Production branch | main |
| Framework preset | None |
| Root directory | Repository root (leave blank) |
| Build command | `mkdocs build` |
| Build output directory | site |
| Python | 3.12.10, specified by .python-version |

The user confirmed successful deployment with `mkdocs build`. The dependency installation settings of that successful build have not been independently inspected. If automatic `pip install .` fails, use `SKIP_DEPENDENCY_INSTALL=1` and the explicit command `python -m pip install -r requirements.txt && python -m mkdocs build`.

Automatic preview builds are disabled. In Pages → learning-llms → Settings → Builds & deployments, keep production deployments enabled for `main` and set preview branch deployments to **None**. This Cloudflare setting is independent of the GitHub Actions workflow. If your dashboard requires an explicit runtime override, set PYTHON_VERSION to 3.12.10 for production.

Cloudflare handles repository access through its Git integration. The site build does not need a Cloudflare API token or GitHub token in repository files.

## Local build

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

~~~bash
uv sync --locked
uv run --locked python -m mkdocs build --strict
~~~

For local editing:

~~~bash
uv run --locked python -m mkdocs serve
~~~

The strict build fails on documentation warnings, including unresolved relative links. Generated HTML lives in site/ and is excluded from Git.

## Required local checks before merging

Run all of these from the repository root on the final PR content, including for documentation-only PRs. These are the same commands as `.github/workflows/docs.yml`; keep this list synchronized when checks change. No training run or dataset download is required.

~~~bash
uv run --locked python -m mkdocs build --strict
uv run --locked --group experiments ruff check --select ANN .
uv run --locked --group experiments ruff check experiments/character-logs
uv run --locked --group experiments ruff format --check experiments/character-logs
uv run --locked --group experiments python -m unittest discover -s experiments/character-logs -p 'test_*.py' -v
uv run --locked --group experiments --group cosmopedia ruff check experiments/cosmopedia
uv run --locked --group experiments --group cosmopedia ruff format --check experiments/cosmopedia
uv run --locked --group experiments --group cosmopedia python -m unittest discover -s experiments/cosmopedia -p 'test_*.py' -v
~~~

Every command must succeed. Fix failures before merging. Record the tested commit and results in the PR, and rerun after further edits or incorporating changes from `main`. Review changed pages locally for readability and figure layout as well: a successful build does not check visual quality.

## Publishing workflow

1. Edit on a branch and open a PR. Branch pushes do not run GitHub Actions or automatic Cloudflare preview builds.
2. Bring the branch up to date with `main` and run the complete local checks above on its final content.
3. Record the tested commit and check results in the PR before merging.
4. Merge to `main`. GitHub Actions checks the result on Ubuntu; Cloudflare independently builds and publishes production.
5. Check both remote results and the deployed home page, changed pages, navigation, search, and code blocks. If a regression reaches `main`, fix it or revert the offending change and verify the resulting deployment.

The local requirement is a working agreement in AGENTS.md, not a GitHub-enforced merge gate. Local success cannot guarantee success on the Linux runner or Cloudflare. Actions does not block Cloudflare publication, and a failing check does not automatically revert a deployment. We accept that tradeoff for this learning site while retaining checks on every push to `main`.

## Domain

The companion is live at https://learningllms.kantarcise.com/. The user configured the custom domain in Cloudflare Pages. `site_url` in `mkdocs.yml` uses this address for canonical URLs and the sitemap.

## References

- [Cloudflare's MkDocs guide](https://developers.cloudflare.com/pages/framework-guides/deploy-an-mkdocs-site/)
- [Branch build controls](https://developers.cloudflare.com/pages/configuration/branch-build-controls/)
- [Build configuration](https://developers.cloudflare.com/pages/configuration/build-configuration/)
- [Python version configuration](https://developers.cloudflare.com/pages/configuration/build-image/)

Local work and GitHub Actions use uv. Cloudflare uses pip and MkDocs directly. requirements.txt is exported from uv.lock; after dependency changes, regenerate it with `uv export --locked --no-dev --no-hashes --no-emit-project --format requirements-txt --output-file requirements.txt`.
