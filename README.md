# VEC CollapseCheck

**Detect obvious population collapse before submitting a generative VEC prediction.**

A generative submission can have the correct shape and gene panel yet still be nearly one cell copied thousands of times. CollapseCheck inspects the predicted cell population itself and reports conservative diversity diagnostics.

It does **not** compare against hidden data and does not claim that a diverse prediction is biologically correct.

## Checks

- exact duplicate-cell fraction after scorer-style float32 casting;
- median and near-zero per-gene variance;
- effective matrix rank;
- sampled pairwise cell-distance distribution;
- number of unique sampled expression rows.

## Install and run

```bash
pip install -e .
vec-collapsecheck prediction.h5ad
vec-collapsecheck prediction.h5ad --json collapse.json
```

Exit codes:

- `0`: no obvious collapse;
- `3`: low-diversity warning;
- `4`: severe collapse;
- `2`: unreadable/invalid input.

The thresholds intentionally target obvious pipeline failures rather than subtle biological judgments.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check src tests
```

The test suite includes end-to-end synthetic H5AD CLI tests for a repeated-cell collapse and a diverse population.
