"""Bitwise identity of the pooled checkpoint bootstrap against 9c14d6f."""

from __future__ import annotations

import hashlib
import json
import multiprocessing.context
import platform
from pathlib import Path

import numpy as np
import pytest

import homoeogwas.interact as I
import homoeogwas.omnib_family as F
from homoeogwas.group_family import MasterGroupFamily
from homoeogwas.interact import SubgenomeData
from homoeogwas.resampling_checkpoint import CheckpointStore

REFERENCE_PATH = Path(__file__).parent / "data" / "omnib_pool_reference_9c14d6f.json"
SCENARIOS = {
    "joint_bs5": {"hypothesis_unit": "group", "family_scope": "joint", "block_size": 5},
    "edge_bs25": {"hypothesis_unit": "edge", "family_scope": "primary_only", "block_size": 25},
}
BOOTSTRAP_B = 19


def pool_family_fixture(seed: int = 7301):
    rng = np.random.default_rng(seed)
    n, groups, snps_per_gene = 48, 5, 5
    subdata = {}
    for sub in ("A", "B", "D"):
        X = rng.integers(0, 3, size=(n, groups * snps_per_gene)).astype(float)
        gene_snp = {
            f"g{index}": np.arange(
                index * snps_per_gene, (index + 1) * snps_per_gene)
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
    return subdata, family, rng.normal(size=n), np.arange(n)


def run_pool_scenario(root, scenario: str, *, n_jobs: int):
    subdata, family, y, sample_idx = pool_family_fixture()
    config = SCENARIOS[scenario]
    return F.run_group_scan_omnib(
        subdata, family, y, sample_idx,
        hypothesis_unit=config["hypothesis_unit"],
        family_scope=config["family_scope"],
        cap=150, n_pc=3, transform="INT", bootstrap_B=BOOTSTRAP_B,
        bootstrap_seed=2026, feature_seed=4401, n_jobs=n_jobs,
        grm_method="grm_from_X", maf_min=0.01, burden_maf=0.01, min_snp=3,
        checkpoint_dir=root, checkpoint_block_size=config["block_size"],
        alpha=0.05, inferential=True,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_identity(result, root: Path) -> dict:
    diagnostics = result.model_diagnostics
    fwer = diagnostics["bootstrap_fwer"]
    checkpoint = diagnostics["resampling_checkpoint"]
    store = CheckpointStore(
        root, checkpoint["manifest_id"], BOOTSTRAP_B,
        checkpoint["block_size"], base_seed=2026)
    null_p = store.concatenate(require_complete=True)
    safe = np.where(np.isfinite(null_p), null_p, np.inf).min(axis=0)
    return {
        "G": int(result.G),
        "n_valid": int(result.n_valid),
        "manifest_id": checkpoint["manifest_id"],
        "completed_ranges": checkpoint["completed_ranges"],
        "primary_null_p_sha256": checkpoint["primary_null_p_sha256"],
        "family_order_sha256": fwer["family_order_sha256"],
        "hypothesis_ids": fwer["hypothesis_ids"],
        "observed_p": fwer["observed_p"],
        "adjusted_p": fwer["adjusted_p"],
        "empirical_p": fwer["empirical_p"],
        "threshold": fwer["threshold"],
        "rejected_hypothesis_ids": fwer["rejected_hypothesis_ids"],
        "null_min": [float(value) for value in safe],
        "top": [[hit["hypothesis_id"], hit["p_interaction"]] for hit in result.top],
        "feature_cache_sha256": diagnostics["feature_provenance"][
            "feature_cache_sha256"],
        "fixed_mask_sha256": diagnostics["prepared_design"]["fixed_mask_sha256"],
        "null_fit_sha256": diagnostics["prepared_design"]["null_fit_sha256"],
        "prepared_design_sha256": diagnostics["prepared_design"]["sha256"],
        "checkpoint_files": {
            path.name: _sha256(path)
            for path in sorted(root.iterdir())
            if path.name == "manifest.json" or path.suffix == ".npz"
        },
    }


def runtime_fingerprint() -> dict:
    import scipy
    from threadpoolctl import threadpool_info

    blas = sorted(
        (
            {
                "internal_api": library.get("internal_api"),
                "version": library.get("version"),
                "architecture": library.get("architecture"),
            }
            for library in threadpool_info()
            if library.get("user_api") == "blas"
        ),
        key=json.dumps,
    )
    return {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "machine": platform.machine(),
        "blas": blas,
    }


def _reference():
    payload = json.loads(REFERENCE_PATH.read_text())
    if payload["runtime"] != runtime_fingerprint():
        pytest.skip(
            "reference fixture is bound to another numeric runtime: "
            f"{payload['runtime']} != {runtime_fingerprint()}")
    return payload


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("n_jobs", [1, 4])
def test_pooled_scan_reproduces_9c14d6f_reference(tmp_path, scenario, n_jobs):
    reference = _reference()
    root = tmp_path / scenario
    result = run_pool_scenario(root, scenario, n_jobs=n_jobs)
    assert scan_identity(result, root) == reference["scenarios"][scenario]


def test_pooled_scan_is_bitwise_invariant_to_worker_count(tmp_path):
    identities = {}
    for n_jobs in (1, 4, 32):
        root = tmp_path / f"jobs_{n_jobs}"
        result = run_pool_scenario(root, "joint_bs5", n_jobs=n_jobs)
        identities[n_jobs] = scan_identity(result, root)
    assert identities[1] == identities[4] == identities[32]


def test_pooled_scan_resume_after_interruption_is_bitwise_identical(
        monkeypatch, tmp_path):
    original = CheckpointStore.write_block
    calls = {"count": 0}

    def interrupt_after_two(self, start, stop, matrix):
        path = original(self, start, stop, matrix)
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated interruption")
        return path

    monkeypatch.setattr(CheckpointStore, "write_block", interrupt_after_two)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_pool_scenario(tmp_path / "resume", "joint_bs5", n_jobs=4)
    monkeypatch.setattr(CheckpointStore, "write_block", original)
    assert len(list((tmp_path / "resume").glob("block_*.npz"))) == 2

    resumed = run_pool_scenario(tmp_path / "resume", "joint_bs5", n_jobs=2)
    full = run_pool_scenario(tmp_path / "full", "joint_bs5", n_jobs=1)
    assert scan_identity(resumed, tmp_path / "resume") == scan_identity(
        full, tmp_path / "full")


def _count_fork_pools(monkeypatch):
    original = multiprocessing.context.ForkContext.Pool
    created = []

    def counting(self, *args, **kwargs):
        created.append(kwargs.get("processes", args[0] if args else None))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(multiprocessing.context.ForkContext, "Pool", counting)
    return created


def test_checkpoint_scan_forks_at_most_two_pools_per_response(
        monkeypatch, tmp_path):
    created = _count_fork_pools(monkeypatch)
    result = run_pool_scenario(tmp_path / "pooled", "joint_bs5", n_jobs=4)
    assert len(result.model_diagnostics["resampling_checkpoint"][
        "completed_ranges"]) == 4
    assert created == [4, 4]
    execution = result.model_diagnostics["parallel_execution"]
    assert execution["pool_reuse"] is True
    assert execution["pool_forks_per_response"] == 2
    assert execution["bootstrap_blocks_scored"] == 4
    assert execution["bootstrap_tasks_per_block"] == 4
    assert execution["requested_jobs"] == 4
    assert execution["effective_jobs"] == 4
    assert execution["backend"] == "fork_shared_memory"
    assert execution["process_model"] == "processes"
    assert execution["inner_threads"] == 1
    assert len(execution["worker_pids"]) == 4


def test_serial_checkpoint_scan_forks_no_pool(monkeypatch, tmp_path):
    created = _count_fork_pools(monkeypatch)
    result = run_pool_scenario(tmp_path / "serial", "joint_bs5", n_jobs=1)
    assert created == []
    execution = result.model_diagnostics["parallel_execution"]
    assert execution["pool_reuse"] is True
    assert execution["pool_forks_per_response"] == 0
    assert execution["backend"] == "serial"
    assert execution["effective_jobs"] == 1


def test_single_eigh_draws_match_per_block_generation():
    subdata, family, y, sample_idx = pool_family_fixture()
    scores, _expanded = F.score_omnib_family(
        subdata, family, y, sample_idx, bootstrap_B=0, feature_seed=4401,
        n_jobs=1, grm_method="grm_from_X", maf_min=0.01, min_snp=3)
    null_fit = (
        scores.W, scores.null_covariance, scores.null_beta,
        scores.covariance_components)
    once, _, _ = I.null_replicates_by_index(
        scores.null_kernels, scores.y, scores.null_design,
        indices=range(BOOTSTRAP_B), base_seed=2026, null_fit=null_fit)
    blockwise = []
    for start in range(0, BOOTSTRAP_B, 5):
        chunk, _, _ = I.null_replicates_by_index(
            scores.null_kernels, scores.y, scores.null_design,
            indices=range(start, min(start + 5, BOOTSTRAP_B)),
            base_seed=2026, null_fit=null_fit)
        blockwise.extend(chunk)
    assert len(once) == len(blockwise) == BOOTSTRAP_B
    for left, right in zip(once, blockwise, strict=True):
        assert left.tobytes() == right.tobytes()

    values, vectors = np.linalg.eigh(
        0.5 * (scores.null_covariance + scores.null_covariance.T))
    again, again_vectors = np.linalg.eigh(
        0.5 * (scores.null_covariance + scores.null_covariance.T))
    assert values.tobytes() == again.tobytes()
    assert vectors.tobytes() == again_vectors.tobytes()


def test_edge_task_ranges_cover_every_edge_with_about_n_jobs_tasks():
    ranges = F._edge_task_ranges(240, 128)
    assert len(ranges) == 128
    assert ranges[0] == (0, 2)
    assert ranges[-1] == (239, 240)
    assert all(hi > lo for lo, hi in ranges)
    assert [lo for lo, _hi in ranges[1:]] == [hi for _lo, hi in ranges[:-1]]
    assert F._edge_task_ranges(3, 8) == [(0, 1), (1, 2), (2, 3)]
    assert F._edge_task_ranges(7, 1) == [(0, 7)]
    assert F._edge_task_ranges(0, 4) == []
