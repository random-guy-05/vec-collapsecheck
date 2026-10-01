import numpy as np

from vec_collapsecheck.core import analyze_matrix


def test_repeated_average_cell_is_severe():
    base = np.arange(20, dtype=float)
    matrix = np.repeat(base[None, :], 200, axis=0)
    report = analyze_matrix(matrix)
    assert report.classification == "SEVERE_COLLAPSE"
    assert report.duplicate_fraction > 0.99


def test_diverse_population_is_not_flagged():
    rng = np.random.default_rng(2)
    matrix = np.log1p(rng.poisson(3, size=(300, 80))).astype(float)
    report = analyze_matrix(matrix)
    assert report.classification == "NO_OBVIOUS_COLLAPSE"
    assert report.effective_rank > 5
