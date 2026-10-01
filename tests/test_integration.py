import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd


def _write(path, matrix):
    ad.AnnData(
        X=np.asarray(matrix, dtype=np.float32),
        var=pd.DataFrame(index=[f"g{i}" for i in range(matrix.shape[1])]),
    ).write_h5ad(path)


def test_cli_distinguishes_collapsed_and_diverse_h5ad(tmp_path):
    collapsed = tmp_path / "collapsed.h5ad"
    diverse = tmp_path / "diverse.h5ad"
    row = np.linspace(0, 2, 40)
    _write(collapsed, np.repeat(row[None, :], 150, axis=0))
    rng = np.random.default_rng(5)
    _write(diverse, np.log1p(rng.poisson(3, size=(150, 40))))

    collapsed_run = subprocess.run(
        [sys.executable, "-m", "vec_collapsecheck.cli", str(collapsed)],
        capture_output=True,
        text=True,
        check=False,
    )
    diverse_run = subprocess.run(
        [sys.executable, "-m", "vec_collapsecheck.cli", str(diverse)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert collapsed_run.returncode == 4
    assert "SEVERE_COLLAPSE" in collapsed_run.stdout
    assert diverse_run.returncode == 0, diverse_run.stdout + diverse_run.stderr
    assert "NO_OBVIOUS_COLLAPSE" in diverse_run.stdout
