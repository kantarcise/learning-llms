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

The companion is live at https://learningllms.kantarcise.com/. The user configured the custom domain in Cloudflare Pages. `site_url` in `mkdocs.yml` uses this address for canonical URLs and the sitemap.

## References

- [Cloudflare's MkDocs guide](https://developers.cloudflare.com/pages/framework-guides/deploy-an-mkdocs-site/)
- [Build configuration](https://developers.cloudflare.com/pages/configuration/build-configuration/)
- [Python version configuration](https://developers.cloudflare.com/pages/configuration/build-image/)

Local work and GitHub Actions use uv. Cloudflare uses pip and MkDocs directly. requirements.txt is exported from uv.lock; after dependency changes, regenerate it with `uv export --locked --no-dev --no-hashes --no-emit-project --format requirements-txt --output-file requirements.txt`.
