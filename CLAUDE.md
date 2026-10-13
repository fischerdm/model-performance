# CLAUDE.md

Worked examples of model-performance metrics, rendered as a Quarto website and
published to https://fischerdm.github.io/model-performance/. See README.md for
the layout.

## Commands

```bash
uv sync                  # install the environment
uv run pytest            # test the metric implementations
uv run quarto preview    # render and serve the site locally
uv run quarto render     # full render (re-executes only changed pages)
```

## Conventions

- **Metrics** go in `src/modelperf/metrics.py`, implemented from their
  definitions (not wrapped from scikit-learn), and each one gets a test in
  `tests/test_metrics.py` checking it against scikit-learn or a known value.
- **Plots** go in `src/modelperf/plots.py`; pages call `use_style()` and the
  plotting functions rather than styling charts inline.
- **Pages** (`docs/<section>/<topic>.qmd`) only call the `modelperf` package.
  Each metric page has a `description` in its front matter, a hidden setup
  cell (`#| include: false`), and opens with a bold **Question answered:** line.
  New pages must be added to the sidebar in `_quarto.yml` and linked from
  `index.qmd`.
- Examples use freMTPL2 (`HasClaim`, `PurePremium`) with Null / GLM / GBM
  models; California Housing only where clean error-metric behaviour is needed.

## Publishing

The site is published by `.github/workflows/publish.yml` on every push to
`main`. CI does not execute Python: it uses the results frozen in `_freeze/`.
After changing a page, render locally and **commit `_freeze/`** together with
the page, otherwise the published page is stale or the build fails.

## Workflow

One branch and one PR per topic (`feature/<topic>`), merged with merge commits.
Update the README layout and `index.qmd` when a topic adds a page.
