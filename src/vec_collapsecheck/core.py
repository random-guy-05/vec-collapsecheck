from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class CollapseReport:
    classification: str
    sampled_cells: int
    sampled_genes: int
    duplicate_fraction: float
    unique_row_fraction: float
    median_gene_variance: float
    near_zero_variance_gene_fraction: float
    effective_rank: int
    median_pairwise_distance: float
    reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _dense_rows(matrix, rows: np.ndarray) -> np.ndarray:
    part = matrix[rows]
    if sparse.issparse(part):
        part = part.toarray()
    return np.asarray(part, dtype=np.float32)


def _row_hashes(array: np.ndarray) -> list[bytes]:
    output: list[bytes] = []
    for row in np.ascontiguousarray(array):
        output.append(hashlib.blake2b(row.tobytes(), digest_size=16).digest())
    return output


def analyze_matrix(
    matrix,
    *,
    max_cells: int = 1024,
    max_genes_for_rank: int = 512,
    seed: int = 0,
) -> CollapseReport:
    n_cells, n_genes = map(int, matrix.shape)
    if n_cells < 2 or n_genes < 1:
        raise ValueError("expression matrix must contain at least 2 cells and 1 gene")
    if max_cells < 2:
        raise ValueError("max_cells must be at least 2")
    if max_genes_for_rank < 1:
        raise ValueError("max_genes_for_rank must be at least 1")

    rng = np.random.default_rng(seed)
    rows = (
        np.arange(n_cells)
        if n_cells <= max_cells
        else np.sort(rng.choice(n_cells, max_cells, replace=False))
    )
    expression = _dense_rows(matrix, rows)
    if not np.isfinite(expression).all():
        raise ValueError("expression matrix contains non-finite values")

    hashes = _row_hashes(expression)
    unique_fraction = len(set(hashes)) / len(hashes)
    duplicate_fraction = 1.0 - unique_fraction

    variances = np.var(expression.astype(np.float64), axis=0)
    median_variance = float(np.median(variances))
    variance_scale = max(float(np.median(np.mean(expression, axis=0) ** 2)), 1.0)
    near_zero = float(np.mean(variances <= 1e-8 * variance_scale))

    gene_indices = np.arange(n_genes)
    if n_genes > max_genes_for_rank:
        gene_indices = np.sort(
            rng.choice(n_genes, max_genes_for_rank, replace=False)
        )
    centered = expression[:, gene_indices].astype(np.float64)
    centered -= centered.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values.size and singular_values[0] > 0:
        effective_rank = int(
            np.sum(singular_values > singular_values[0] * 1e-6)
        )
    else:
        effective_rank = 0

    if len(expression) > 256:
        pair_rows = np.sort(rng.choice(len(expression), 256, replace=False))
        distance_sample = expression[pair_rows].astype(np.float64)
    else:
        distance_sample = expression.astype(np.float64)
    pair_count = min(4096, len(distance_sample) * 8)
    first = rng.integers(0, len(distance_sample), size=pair_count)
    second = rng.integers(0, len(distance_sample), size=pair_count)
    keep = first != second
    distances = np.linalg.norm(
        distance_sample[first[keep]] - distance_sample[second[keep]],
        axis=1,
    )
    median_distance = float(np.median(distances)) if len(distances) else 0.0

    reasons: list[str] = []
    severe = False
    if duplicate_fraction >= 0.90:
        reasons.append("at least 90% of sampled cells are exact duplicates")
        severe = True
    if effective_rank <= 1:
        reasons.append("centered expression has effective rank <= 1")
        severe = True
    if near_zero >= 0.98:
        reasons.append("at least 98% of genes have near-zero cell-to-cell variance")
        severe = True

    warning = False
    if duplicate_fraction >= 0.10:
        reasons.append("at least 10% of sampled cells are exact duplicates")
        warning = True
    if effective_rank < min(5, len(rows) - 1, len(gene_indices)):
        reasons.append("expression diversity is very low-rank")
        warning = True
    if median_distance <= 1e-6:
        reasons.append("sampled pairwise cell distances are essentially zero")
        warning = True

    classification = (
        "SEVERE_COLLAPSE"
        if severe
        else "LOW_DIVERSITY_WARNING"
        if warning
        else "NO_OBVIOUS_COLLAPSE"
    )
    return CollapseReport(
        classification=classification,
        sampled_cells=len(rows),
        sampled_genes=n_genes,
        duplicate_fraction=duplicate_fraction,
        unique_row_fraction=unique_fraction,
        median_gene_variance=median_variance,
        near_zero_variance_gene_fraction=near_zero,
        effective_rank=effective_rank,
        median_pairwise_distance=median_distance,
        reasons=reasons,
    )
