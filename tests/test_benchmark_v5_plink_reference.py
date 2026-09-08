from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2, f

from homoeogwas.interact import _batch_nested_f
from scripts.benchmarks.v201.comparators import _nested_snp_product_design

PLINK = Path("/home/yys05/.local/share/mamba/envs/baseline_ext/bin/plink")
PLINK_SHA256 = "4674107b608b0125f37f6c3d7ef47759dd0ffdb5b639ebc5b1dadb6530ecf01a"


def _alleles(dosage: int) -> str:
    return {0: "G G", 1: "A G", 2: "A A"}[int(dosage)]


def test_raw_nested_f_matches_pinned_plink19_interaction_statistic(tmp_path):
    if not PLINK.is_file():
        pytest.skip("pinned PLINK 1.9 numeric-reference binary is unavailable")
    assert hashlib.sha256(PLINK.read_bytes()).hexdigest() == PLINK_SHA256

    rng = np.random.default_rng(92_070_001)
    n = 96
    left = rng.binomial(2, 0.32, n)
    right = rng.binomial(2, 0.41, n)
    phenotype = 0.25 * left - 0.15 * right + 0.7 * left * right
    phenotype = phenotype + rng.normal(0.0, 1.0, n)
    prefix = tmp_path / "plink-toy"
    prefix.with_suffix(".map").write_text(
        "1 m_left 0 101\n2 m_right 0 202\n", encoding="utf-8",
    )
    prefix.with_suffix(".ped").write_text(
        "".join(
            f"F{i} I{i} 0 0 0 {phenotype[i]:.17g} "
            f"{_alleles(left[i])} {_alleles(right[i])}\n"
            for i in range(n)
        ),
        encoding="utf-8",
    )
    out = tmp_path / "plink-result"
    completed = subprocess.run(
        [
            str(PLINK), "--file", str(prefix), "--allow-no-sex",
            "--epistasis", "--epi1", "1", "--threads", "1",
            "--out", str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [
        line.split()
        for line in out.with_suffix(".epi.qt").read_text(encoding="utf-8").splitlines()[1:]
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0][1] == "m_left"
    assert rows[0][3] == "m_right"
    plink_stat = float(rows[0][-2])
    plink_p = float(rows[0][-1])

    identity = np.eye(n)
    design = _nested_snp_product_design(
        identity,
        np.ones((n, 1)),
        left.astype(float),
        right.astype(float),
    )
    assert design is not None
    internal_p = float(
        _batch_nested_f(phenotype[:, None], design[0], design[1])[0]
    )
    # PLINK 1.9 and HomoeoGWAS fit the same one-degree-of-freedom
    # interaction statistic.  Their reported probabilities intentionally use
    # different reference distributions: PLINK's --epistasis output is the
    # asymptotic chi-square tail, while _batch_nested_f retains the finite-
    # sample residual degrees of freedom.  Bind both facts so the bridge
    # cannot silently turn into a false claim of identical p-values.
    internal_stat = float(f.isf(internal_p, 1, n - 4))
    assert internal_stat == pytest.approx(plink_stat, rel=3e-6)
    assert plink_p == pytest.approx(float(chi2.sf(internal_stat, 1)), rel=5e-4)
    assert internal_p == pytest.approx(float(f.sf(internal_stat, 1, n - 4)))
    assert internal_p > plink_p
