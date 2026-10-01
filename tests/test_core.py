import numpy as np
import pytest
from scipy import sparse

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


def test_sparse_input_matches_dense_classification():
    rng = np.random.default_rng(7)
    matrix = np.log1p(rng.poisson(2, size=(250, 60))).astype(np.float32)
    dense_report = analyze_matrix(matrix, seed=3)
    sparse_report = analyze_matrix(sparse.csr_matrix(matrix), seed=3)
    assert sparse_report.classification == dense_report.classification
    assert sparse_report.effective_rank == dense_report.effective_rank
    np.testing.assert_allclose(
        sparse_report.duplicate_fraction,
        dense_report.duplicate_fraction,
    )


def test_sampling_limits_are_validated():
    matrix = np.ones((10, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="max_cells"):
        analyze_matrix(matrix, max_cells=1)
    with pytest.raises(ValueError, match="max_genes_for_rank"):
        analyze_matrix(matrix, max_genes_for_rank=0)
