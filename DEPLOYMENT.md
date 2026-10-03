# Deploy on Cloudflare Pages

This repository builds a static Material for MkDocs site. The initial chapter pages can ship immediately and grow with each reading session.

## Connect the repository

After the repository is on GitHub, create a Cloudflare **Pages** project with Git integration and select this repository. Use these settings:

| Setting | Value |
| --- | --- |
| Production branch | main |
| Framework preset | None |
| Root directory | Repository root (leave blank) |
| Build command | `python -m pip install uv==0.12.13 && uv run --locked python -m mkdocs build --strict` |
| Build output directory | site |
| Python | 3.12.10, specified by .python-version |

Apply the same settings to preview builds. If your dashboard requires an explicit runtime override, set PYTHON_VERSION to 3.12.10 for production and preview.

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

## Publishing workflow

1. Edit a chapter Markdown page under docs/.
2. Run the strict build locally, or let GitHub Actions check it.
3. Push a branch to get a Cloudflare preview (when previews are enabled).
4. Merge to main to trigger the production build.
5. Check the deployed home page, chapter navigation, search, and code blocks.

## Domain

The first deployment can use the pages.dev address Cloudflare assigns. Add your chosen custom domain through the Pages project's Custom domains interface afterward. No domain or DNS change has been configured here. Once the final address is chosen, add site_url to mkdocs.yml for canonical URLs and the sitemap.

## References

- [Cloudflare's MkDocs guide](https://developers.cloudflare.com/pages/framework-guides/deploy-an-mkdocs-site/)
- [Build configuration](https://developers.cloudflare.com/pages/configuration/build-configuration/)
- [Python version configuration](https://developers.cloudflare.com/pages/configuration/build-image/)

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. The Cloudflare build command installs pinned uv before building. Locally and in GitHub Actions, uv is already installed.
