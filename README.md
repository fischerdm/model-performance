# model-performance

Documented, worked examples of measuring model performance: metrics for binary
and continuous targets, model comparison and drift detection. Every method is
applied to real models (GLM vs gradient boosting) on the French motor insurance
dataset freMTPL2, and rendered as a [Quarto](https://quarto.org) website.

**Read the site:** https://fischerdm.github.io/model-performance/

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Quarto](https://quarto.org/docs/get-started/)
(`brew install --cask quarto`).

```bash
uv sync                  # create .venv with all dependencies
uv run pytest            # test the metric implementations
uv run quarto preview    # render the site and open it in the browser
```

The first render downloads the data to `data/raw/` and fits all models
(about a minute); fitted models are cached in `artifacts/`.

## Layout

```
_quarto.yml              site configuration and page order
index.qmd                home page
docs/
  foundations/           data and models
  binary/                confusion matrix, ROC/AUC, precision-recall, calibration
  continuous/            error metrics, deviance, lift charts and Lorenz curves
  comparison/            paired bootstrap, DeLong and Diebold–Mariano tests
  monitoring/            drift: PSI, domain classifier, CBPE
src/modelperf/
  data.py                loading and splitting the datasets
  models.py              Null / GLM / GBM per target, cached
  drift.py               simulated production batches with drift
  metrics.py             metrics implemented from their definitions
  plots.py               chart style and plotting functions
tests/                   metrics checked against scikit-learn
```
