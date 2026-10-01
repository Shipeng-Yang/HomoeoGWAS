"""A shared phenotype-independent panel context must not change any per-response output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import homoeogwas.omnib_family as F
from homoeogwas.group_family import MasterGroupFamily
from homoeogwas.interact import SubgenomeData
from homoeogwas.resampling_checkpoint import CheckpointStore

B = 19
SETTINGS = dict(
    cap=150, n_pc=3, feature_seed=4401, grm_method="grm_from_X",
    maf_min=0.01, burden_maf=0.01, min_snp=3,
)


def panel_fixture(seed: int = 7301):
    rng = np.random.default_rng(seed)
    n, groups, snps_per_gene = 48, 5, 5
    subdata = {}
    for sub in ("A", "B", "D"):
        X = rng.integers(0, 3, size=(n, groups * snps_per_gene)).astype(float)
        gene_snp = {
            f"g{index}": np.arange(index * snps_per_gene, (index + 1) * snps_per_gene)
            for index in range(groups)
        }
        subdata[sub] = SubgenomeData(
            X=X, gene_snp=gene_snp,
            samples=[f"s{index}" for index in range(n)], chunk=None)
    genes = [(f"g{index}",) * 3 for index in range(groups - 1)]
    genes.append((f"g{groups - 1}", f"g{groups - 1}", "missing"))
    family = MasterGroupFamily(
        subgenomes=("A", "B", "D"),
        group_ids=tuple(f"group_{index}" for index in range(groups)),
        genes=tuple(genes),
    )
    responses = [rng.normal(size=n) for _ in range(2)]
    return subdata, family, responses, np.arange(n)


def run(subdata, family, y, sample_idx, root, *, seed, panel_context=None):
    return F.run_group_scan_omnib(
        subdata, family, y, sample_idx,
        hypothesis_unit="group", family_scope="primary_only",
        transform="INT", bootstrap_B=B, bootstrap_seed=seed, n_jobs=2,
        checkpoint_dir=root, checkpoint_block_size=5, alpha=0.05,
        inferential=True, panel_context=panel_context, **SETTINGS,
    )


def canonical(value):
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, np.ndarray):
        return canonical(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


def identity(result, root: Path, seed: int) -> dict:
    diagnostics = dict(result.model_diagnostics)
    diagnostics.pop("parallel_execution", None)
    checkpoint = diagnostics["resampling_checkpoint"]
    store = CheckpointStore(
        root, checkpoint["manifest_id"], B, checkpoint["block_size"], base_seed=seed)
    null_p = store.concatenate(require_complete=True)
    return {
        "diagnostics": json.dumps(canonical(diagnostics), sort_keys=True, default=str),
        "top": json.dumps(canonical(result.top), sort_keys=True, default=str),
        "null_p": hashlib.sha256(np.ascontiguousarray(null_p).tobytes()).hexdigest(),
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.iterdir())
            if path.name == "manifest.json" or path.suffix == ".npz"
        },
    }


def test_shared_context_is_byte_identical_across_order(tmp_path):
    subdata, family, responses, sample_idx = panel_fixture()
    seeds = (2026, 2027)
    reference = [
        identity(run(subdata, family, y, sample_idx, tmp_path / f"ref{i}", seed=seed),
                 tmp_path / f"ref{i}", seed)
        for i, (y, seed) in enumerate(zip(responses, seeds))
    ]
    assert reference[0]["diagnostics"] != reference[1]["diagnostics"]
    assert reference[0]["null_p"] != reference[1]["null_p"]
    context = F.build_omnib_panel_context(subdata, family, sample_idx, **SETTINGS)
    frozen = {
        "kernels": {sub: hashlib.sha256(context.kernels[sub].tobytes()).hexdigest()
                    for sub in context.kernels},
        "features": F._feature_cache_sha256(context.feature_identity),
        "mask": context.component_estimable.copy(),
        "dfn": context.component_dfn.copy(),
    }
    for step, i in enumerate((0, 1, 0)):
        root = tmp_path / f"ctx{step}"
        result = run(subdata, family, responses[i], sample_idx, root,
                     seed=seeds[i], panel_context=context)
        assert identity(result, root, seeds[i]) == reference[i]
    assert context.subgenome_manifest_identity is not None
    assert F._feature_cache_sha256(context.feature_identity) == frozen["features"]
    assert {sub: hashlib.sha256(context.kernels[sub].tobytes()).hexdigest()
            for sub in context.kernels} == frozen["kernels"]
    np.testing.assert_array_equal(context.component_estimable, frozen["mask"])
    np.testing.assert_array_equal(context.component_dfn, frozen["dfn"])


def test_context_rejects_other_inputs(tmp_path):
    subdata, family, responses, sample_idx = panel_fixture()
    context = F.build_omnib_panel_context(subdata, family, sample_idx, **SETTINGS)
    with pytest.raises(ValueError, match="other sample set"):
        run(subdata, family, responses[0][:-1], sample_idx[:-1], tmp_path / "a",
            seed=1, panel_context=context)
    other = F.build_omnib_panel_context(
        subdata, family, sample_idx, **(SETTINGS | {"cap": 100}))
    with pytest.raises(ValueError, match="other settings"):
        run(subdata, family, responses[0], sample_idx, tmp_path / "b",
            seed=1, panel_context=other)
    copied = {sub: data for sub, data in subdata.items()}
    with pytest.raises(ValueError, match="other genotype"):
        run(copied, family, responses[0], sample_idx, tmp_path / "c",
            seed=1, panel_context=context)
    with pytest.raises(ValueError, match="checkpointed calibration"):
        run(subdata, family, responses[0], sample_idx, None,
            seed=1, panel_context=context)
