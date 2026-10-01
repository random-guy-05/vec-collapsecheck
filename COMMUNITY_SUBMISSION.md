# Community Contribution submission text

## Title
VEC CollapseCheck — pre-submission population-collapse detector

## Description
VEC CollapseCheck detects a class of generative-model failures that ordinary structural validation cannot: a prediction with the correct dimensions and legal values but almost no cell-to-cell diversity. It audits exact duplicate cells, per-gene variance, effective matrix rank, unique sampled rows and pairwise cell distances, then returns conservative severe-collapse / low-diversity / no-obvious-collapse classifications with JSON and CI-friendly exit codes. The tool uses only the submitted prediction, makes no hidden-data claims, and includes end-to-end synthetic H5AD tests for collapsed and diverse populations.
