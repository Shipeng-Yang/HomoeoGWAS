"""Opt-in hash-keyed GRM cache: bitwise equivalence and fail-closed reads."""
from __future__ import annotations

import copy
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import homoeogwas.interact as I
import homoeogwas.omnib_family as F
from homoeogwas.grm_cache import (
    GRMCache,
    GRMCacheError,
    grm_cache_key,
    runtime_fingerprint,
)
from homoeogwas.group_family import MasterGroupFamily
from homoeogwas.interact import SubgenomeData, _build_grm, _load_subgenome
from homoeogwas.io import plink_bim_sha256

SUBS = ("A", "D")
N_SAMPLES = 40
GENES = 3
SNPS_PER_GENE = 8


def _write_panel(root, *, seed=77):
    from bed_reader import to_bed

    rng = np.random.default_rng(seed)
    samples = [f"s{i:03d}" for i in range(N_SAMPLES)]
    m = GENES * SNPS_PER_GENE
    genotype, mapping = {}, {}
    for sub in SUBS:
        prefix = root / f"geno_{sub}"
        prefix.parent.mkdir(parents=True, exist_ok=True)
        freq = rng.uniform(0.15, 0.5, size=m)
        values = rng.binomial(2, freq, size=(N_SAMPLES, m)).astype(np.float32)
        values[rng.random(size=values.shape) < 0.02] = np.nan
        to_bed(str(prefix) + ".bed", values, properties={
            "fid": ["0"] * N_SAMPLES, "iid": samples,
            "sid": [f"{sub}-{index}" for index in range(m)],
            "chromosome": [sub] * m, "bp_position": list(range(1, m + 1)),
            "allele_1": ["A"] * m, "allele_2": ["C"] * m,
        }, count_A1=True)
        npz = root / f"map_{sub}.npz"
        np.savez(
            npz,
            gene_ids=np.asarray([f"g{g}" for g in range(GENES)], dtype=object),
            snp_idx=np.asarray([
                np.arange(g * SNPS_PER_GENE, (g + 1) * SNPS_PER_GENE)
                for g in range(GENES)
            ], dtype=object),
            bim_sha256=np.asarray(plink_bim_sha256(prefix)),
            n_variants=np.asarray(m), subgenome=np.asarray(sub),
        )
        genotype[sub], mapping[sub] = str(prefix), str(npz)
    groups = root / "groups.tsv"
    groups.write_text(
        "group_id\tgene_A\tgene_D\n"
        + "".join(f"grp{g}\tg{g}\tg{g}\n" for g in range(GENES)),
        encoding="utf-8",
    )
    phenotype = root / "phenotype.tsv"
    phenotype.write_text(
        "sample\ttrait\n" + "".join(
            f"{sample}\t{rng.normal():.10g}\n"
            for index, sample in enumerate(samples) if index % 10 != 3
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(
        samples=samples, genotype=genotype, mapping=mapping,
        groups=groups, phenotype=phenotype,
    )


@pytest.fixture
def panel(tmp_path):
    return _write_panel(tmp_path / "panel")


def _subdata(panel):
    return {
        sub: _load_subgenome(panel.genotype[sub], panel.mapping[sub])
        for sub in SUBS
    }


def _sample_idx(panel):
    return np.asarray(
        [index for index in range(N_SAMPLES) if index % 10 != 3], int)


def _bits(K):
    return np.ascontiguousarray(K).view(np.uint64)


def _strip_cache(provenance):
    return {key: value for key, value in provenance.items() if key != "cache"}


def test_build_grm_is_bitwise_deterministic_without_cache():
    rng = np.random.default_rng(5)
    X = rng.integers(0, 3, size=(30, 40)).astype(float)
    X[rng.random(size=X.shape) < 0.05] = np.nan
    sd = SubgenomeData(X=X, gene_snp={}, samples=[f"s{i}" for i in range(30)])
    idx = np.arange(2, 28)
    first, first_prov = _build_grm(
        sd, idx, "grm_from_X", 0.05, return_provenance=True)
    second, second_prov = _build_grm(
        sd, idx, "grm_from_X", 0.05, return_provenance=True)
    assert np.array_equal(_bits(first), _bits(second))
    assert first_prov == second_prov


def test_loader_binds_file_backed_genotype_source(panel):
    sd = _load_subgenome(panel.genotype["A"], panel.mapping["A"])
    assert sd.genotype_source == {
        "plink_prefix": panel.genotype["A"],
        "bim_sha256": plink_bim_sha256(panel.genotype["A"]),
        "n_variants": GENES * SNPS_PER_GENE,
    }


def test_cache_miss_then_hit_returns_identical_bytes_and_provenance(
        panel, tmp_path):
    subdata = _subdata(panel)
    idx = _sample_idx(panel)
    cache = GRMCache(tmp_path / "grm-cache")
    fresh, fresh_prov = _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True)

    miss, miss_prov = _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True,
        grm_cache=cache, subgenome="A")
    assert miss_prov["cache"]["enabled"] is True
    assert miss_prov["cache"]["hit"] is False
    key = miss_prov["cache"]["key"]
    npy_path, json_path = cache.entry_paths(key)
    assert npy_path.exists() and json_path.exists()
    assert np.array_equal(_bits(miss), _bits(fresh))
    assert _strip_cache(miss_prov) == fresh_prov

    hit, hit_prov = _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True,
        grm_cache=cache, subgenome="A")
    assert hit_prov["cache"] == miss_prov["cache"] | {"hit": True}
    assert np.array_equal(_bits(hit), _bits(fresh))
    assert hit.dtype == np.float64 and hit.flags.c_contiguous
    assert _strip_cache(hit_prov) == fresh_prov

    record = json.loads(json_path.read_text(encoding="utf-8"))
    assert record["key"] == key == grm_cache_key(record["key_inputs"])
    assert record["npy_sha256"] == hit_prov["cache"]["npy_sha256"]
    assert record["grm_provenance"] == fresh_prov
    assert record["runtime"] == runtime_fingerprint()
    assert set(record["runtime"]["blas_thread_env"]) == {
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
    assert record["matrix"] == F._array_identity(fresh)
    inputs = record["key_inputs"]
    assert inputs["subgenome"] == "A"
    assert inputs["method"] == "grm_from_X"
    assert inputs["maf_min"] == 0.01
    assert inputs["samples"]["ids"] == [panel.samples[i] for i in idx]
    assert inputs["samples"]["row_indices"] == idx.tolist()
    assert set(inputs["genotype"]) == {"bed_sha256", "bim_sha256", "n_variants"}
    assert inputs["mask_policy"]["filter_policy"] == "legacy_maf_only"
    assert "path" not in json.dumps(inputs).lower()


def test_cache_key_depends_on_bytes_not_paths(panel, tmp_path):
    subdata = _subdata(panel)
    idx = _sample_idx(panel)
    cache = GRMCache(tmp_path / "grm-cache")
    _, prov = _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True,
        grm_cache=cache, subgenome="A")
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    for suffix in (".bed", ".bim", ".fam"):
        shutil.copy(panel.genotype["A"] + suffix, moved / ("copy" + suffix))
    shutil.copy(panel.mapping["A"], moved / "copy.npz")
    relocated = _load_subgenome(str(moved / "copy"), str(moved / "copy.npz"))
    _, relocated_prov = _build_grm(
        relocated, idx, "grm_from_X", 0.01, return_provenance=True,
        grm_cache=cache, subgenome="A")
    assert relocated_prov["cache"]["key"] == prov["cache"]["key"]
    assert relocated_prov["cache"]["hit"] is True

    variants = {
        "subgenome": dict(subgenome="D"),
        "maf_min": dict(maf_min=0.02),
        "samples": dict(sample_idx=idx[:-1]),
        "order": dict(sample_idx=idx[::-1]),
        "genotype": dict(sd=subdata["D"]),
    }
    keys = {"base": prov["cache"]["key"]}
    for label, override in variants.items():
        sd = override.get("sd", subdata["A"])
        _, other = _build_grm(
            sd, override.get("sample_idx", idx), "grm_from_X",
            override.get("maf_min", 0.01), return_provenance=True,
            grm_cache=cache, subgenome=override.get("subgenome", "A"))
        keys[label] = other["cache"]["key"]
    assert len(set(keys.values())) == len(keys)


def test_explicit_mask_policy_is_part_of_the_key(panel, tmp_path):
    subdata = _subdata(panel)
    idx = _sample_idx(panel)
    cache = GRMCache(tmp_path / "grm-cache")
    mask, qc = I.build_retained_variant_mask(
        subdata["A"].X, sample_idx=idx, call_rate_min=0.9, maf_min=0.01,
        mac_min=5)
    _, legacy = _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True,
        grm_cache=cache, subgenome="A")
    fresh, fresh_prov = _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True,
        retained_variant_mask=mask)
    thresholds = {"call_rate_min": 0.9, "maf_min": 0.01, "mac_min": 5}
    cached, cached_prov = _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True,
        retained_variant_mask=mask, grm_cache=cache, subgenome="A",
        mask_policy={"thresholds": thresholds})
    assert cached_prov["cache"]["hit"] is False
    assert cached_prov["cache"]["key"] != legacy["cache"]["key"]
    assert np.array_equal(_bits(cached), _bits(fresh))
    assert _strip_cache(cached_prov) == fresh_prov
    record = json.loads(
        cache.entry_paths(cached_prov["cache"]["key"])[1].read_text())
    assert record["key_inputs"]["mask_policy"] == {
        "filter_policy": "explicit_retained_variant_mask",
        "retained_variant_mask_sha256": qc["retained_variant_mask_sha256"],
        "retained_variant_count": qc["n_variants_retained"],
        "thresholds": thresholds,
    }
    again, again_prov = _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True,
        retained_variant_mask=mask, grm_cache=cache, subgenome="A",
        mask_policy={"thresholds": thresholds})
    assert again_prov["cache"]["hit"] is True
    assert np.array_equal(_bits(again), _bits(fresh))


def _prime(panel, tmp_path):
    subdata = _subdata(panel)
    idx = _sample_idx(panel)
    cache = GRMCache(tmp_path / "grm-cache", incomplete_grace_seconds=0.0)
    _, prov = _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True,
        grm_cache=cache, subgenome="A")
    npy_path, json_path = cache.entry_paths(prov["cache"]["key"])
    return subdata, idx, cache, npy_path, json_path


def _rerun(subdata, idx, cache):
    return _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True,
        grm_cache=cache, subgenome="A")


def test_tampered_matrix_bytes_fail_closed(panel, tmp_path):
    subdata, idx, cache, npy_path, _ = _prime(panel, tmp_path)
    K = np.load(npy_path)
    K[0, 1] = np.nextafter(K[0, 1], np.inf)
    np.save(npy_path, K)
    with pytest.raises(GRMCacheError, match="sha256"):
        _rerun(subdata, idx, cache)


@pytest.mark.parametrize("remove", ["npy", "json"])
def test_incomplete_entry_fails_closed(panel, tmp_path, remove):
    subdata, idx, cache, npy_path, json_path = _prime(panel, tmp_path)
    (npy_path if remove == "npy" else json_path).unlink()
    with pytest.raises(GRMCacheError, match="incomplete"):
        _rerun(subdata, idx, cache)


def test_runtime_fingerprint_mismatch_fails_closed(panel, tmp_path):
    subdata, idx, cache, _, json_path = _prime(panel, tmp_path)
    record = json.loads(json_path.read_text())
    record["runtime"]["numpy"] = "0.0.0"
    json_path.write_text(json.dumps(record))
    with pytest.raises(GRMCacheError, match="runtime"):
        _rerun(subdata, idx, cache)


def test_thread_policy_change_fails_closed(panel, tmp_path, monkeypatch):
    subdata, idx, cache, _, _ = _prime(panel, tmp_path)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    with pytest.raises(GRMCacheError, match="runtime"):
        _rerun(subdata, idx, cache)


def test_key_input_mismatch_fails_closed(panel, tmp_path):
    subdata, idx, cache, _, json_path = _prime(panel, tmp_path)
    record = json.loads(json_path.read_text())
    record["key_inputs"]["samples"]["ids"][0] = "impostor"
    json_path.write_text(json.dumps(record))
    with pytest.raises(GRMCacheError, match="identity"):
        _rerun(subdata, idx, cache)


def test_provenance_mask_mismatch_fails_closed(panel, tmp_path):
    subdata, idx, cache, _, json_path = _prime(panel, tmp_path)
    record = json.loads(json_path.read_text())
    record["grm_provenance"]["retained_variant_mask"][0] ^= 1
    json_path.write_text(json.dumps(record))
    with pytest.raises(GRMCacheError, match="provenance"):
        _rerun(subdata, idx, cache)


def test_stale_mapping_fingerprint_fails_closed(panel, tmp_path):
    subdata, idx, cache, _, _ = _prime(panel, tmp_path)
    bim = panel.genotype["A"] + ".bim"
    lines = open(bim, encoding="utf-8").read().splitlines()
    fields = lines[0].split()
    fields[1] = "renamed"
    lines[0] = "\t".join(fields)
    with open(bim, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with pytest.raises(GRMCacheError, match="BIM"):
        _rerun(subdata, idx, cache)


def test_cache_requires_file_backed_identity_and_grm_from_x(panel, tmp_path):
    cache = GRMCache(tmp_path / "grm-cache")
    synthetic = SubgenomeData(
        X=np.ones((6, 4)), gene_snp={}, samples=[f"s{i}" for i in range(6)])
    with pytest.raises(GRMCacheError, match="genotype"):
        _build_grm(synthetic, np.arange(6), "grm_from_X", 0.0,
                   return_provenance=True, grm_cache=cache, subgenome="A")
    subdata = _subdata(panel)
    with pytest.raises(GRMCacheError, match="grm_from_X"):
        _build_grm(subdata["A"], _sample_idx(panel), "compute_grm_maf", 0.01,
                   return_provenance=True, grm_cache=cache, subgenome="A")
    with pytest.raises(GRMCacheError, match="subgenome"):
        _build_grm(subdata["A"], _sample_idx(panel), "grm_from_X", 0.01,
                   return_provenance=True, grm_cache=cache)


def test_store_never_overwrites_and_rejects_divergent_bytes(panel, tmp_path):
    subdata, idx, cache, npy_path, json_path = _prime(panel, tmp_path)
    before = (npy_path.stat().st_mtime_ns, npy_path.stat().st_ino,
              npy_path.read_bytes(), json_path.read_bytes())
    K, prov = _rerun(subdata, idx, cache)
    record = json.loads(json_path.read_text())
    stored = cache.store(
        record["key"], record["key_inputs"], K, _strip_cache(prov))
    assert stored["hit"] is True
    assert (npy_path.stat().st_mtime_ns, npy_path.stat().st_ino,
            npy_path.read_bytes(), json_path.read_bytes()) == before
    divergent = K.copy()
    divergent[0, 0] = np.nextafter(divergent[0, 0], np.inf)
    with pytest.raises(GRMCacheError, match="differs"):
        cache.store(
            record["key"], record["key_inputs"], divergent,
            _strip_cache(prov))
    assert (npy_path.stat().st_mtime_ns, npy_path.stat().st_ino,
            npy_path.read_bytes(), json_path.read_bytes()) == before
    assert not [p for p in cache.root.iterdir() if p.name.startswith(".")]


def test_cache_dir_must_be_absolute_and_creatable(tmp_path):
    with pytest.raises(GRMCacheError, match="absolute"):
        GRMCache("relative/cache")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(GRMCacheError, match="directory"):
        GRMCache(blocker / "cache")
    nested = tmp_path / "deep" / "er" / "cache"
    assert GRMCache(nested).root == nested
    assert nested.is_dir()


def _family():
    return MasterGroupFamily(
        subgenomes=SUBS,
        group_ids=tuple(f"grp{g}" for g in range(GENES)),
        genes=tuple((f"g{g}", f"g{g}") for g in range(GENES)),
    )


def _scan(panel, tmp_path, label, *, grm_cache=None):
    subdata = _subdata(panel)
    idx = _sample_idx(panel)
    rng = np.random.default_rng(11)
    y = rng.normal(size=idx.size)
    kwargs = dict(
        hypothesis_unit="edge", feature_seed=4103, bootstrap_B=3,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X",
        maf_min=0.01, min_snp=3, checkpoint_dir=tmp_path / f"cp_{label}",
        checkpoint_block_size=2, evidence_role="legacy",
    )
    if grm_cache is not None:
        kwargs["grm_cache"] = grm_cache
    result = F.run_group_scan_omnib(subdata, _family(), y, idx, **kwargs)
    manifest = json.loads(
        (tmp_path / f"cp_{label}" / "manifest.json").read_text())
    return I._json_safe(result.__dict__), manifest


def _pop_cache_fields(payload):
    payload = copy.deepcopy(payload)
    subgenomes = payload["model_diagnostics"]["grm_provenance"]["subgenomes"]
    return payload, {sub: subgenomes[sub].pop("cache", None) for sub in subgenomes}


def test_group_scan_result_and_manifest_identical_with_and_without_cache(
        panel, tmp_path):
    plain, plain_manifest = _scan(panel, tmp_path, "plain")
    cache = GRMCache(tmp_path / "grm-cache")
    cold, cold_manifest = _scan(panel, tmp_path, "cold", grm_cache=cache)
    warm, warm_manifest = _scan(panel, tmp_path, "warm", grm_cache=cache)

    plain_body, plain_cache = _pop_cache_fields(plain)
    cold_body, cold_cache = _pop_cache_fields(cold)
    warm_body, warm_cache = _pop_cache_fields(warm)
    assert plain_cache == {sub: None for sub in SUBS}
    assert {sub: cold_cache[sub]["hit"] for sub in SUBS} == {
        sub: False for sub in SUBS}
    assert {sub: warm_cache[sub]["hit"] for sub in SUBS} == {
        sub: True for sub in SUBS}
    assert {sub: warm_cache[sub]["key"] for sub in SUBS} == {
        sub: cold_cache[sub]["key"] for sub in SUBS}
    assert len(set(cold_cache[sub]["key"] for sub in SUBS)) == len(SUBS)
    assert json.dumps(cold_body, sort_keys=True) == json.dumps(
        plain_body, sort_keys=True)
    assert json.dumps(warm_body, sort_keys=True) == json.dumps(
        plain_body, sort_keys=True)
    assert cold_manifest == plain_manifest
    assert warm_manifest == plain_manifest
    for manifest in (cold_manifest, warm_manifest):
        grm = manifest["manifest"]["grm"]["subgenomes"]
        assert set(grm) == set(SUBS)
        assert all("cache" not in grm[sub] for sub in SUBS)
    kernels = plain_manifest["manifest"]["prepared_design"]
    assert kernels["null_fit_sha256"] == warm["model_diagnostics"][
        "prepared_design"]["null_fit_sha256"]
    for sub in SUBS:
        record = json.loads(cache.entry_paths(warm_cache[sub]["key"])[1].read_text())
        assert record["matrix"]["sha256"] == F._array_identity(
            np.load(cache.entry_paths(warm_cache[sub]["key"])[0]))["sha256"]


def test_benchmark_mask_records_bind_thresholds_into_the_key(panel, tmp_path):
    subdata = _subdata(panel)
    idx = _sample_idx(panel)
    ic = {
        "benchmark_identity": {
            "panel_id": "FIXTURE", "sample_context": "full",
            "feature_seed": 4103,
        },
        "subgenomes": list(SUBS),
        "genotype": dict(panel.genotype),
    }
    records = I._build_benchmark_mask_records(ic, subdata, idx)
    y = np.random.default_rng(11).normal(size=idx.size)
    kwargs = dict(
        hypothesis_unit="edge", feature_seed=4103, bootstrap_B=3,
        bootstrap_seed=2026, n_jobs=1, grm_method="grm_from_X",
        maf_min=0.01, min_snp=3, inferential=False,
        retained_variant_masks=records, evidence_role="benchmark_qa",
    )
    plain = I._json_safe(F.run_group_scan_omnib(
        subdata, _family(), y, idx, **kwargs).__dict__)
    cache = GRMCache(tmp_path / "grm-cache")
    cold = I._json_safe(F.run_group_scan_omnib(
        subdata, _family(), y, idx, grm_cache=cache, **kwargs).__dict__)
    warm = I._json_safe(F.run_group_scan_omnib(
        subdata, _family(), y, idx, grm_cache=cache, **kwargs).__dict__)
    plain_body, _ = _pop_cache_fields(plain)
    cold_body, cold_cache = _pop_cache_fields(cold)
    warm_body, warm_cache = _pop_cache_fields(warm)
    assert json.dumps(cold_body, sort_keys=True) == json.dumps(
        plain_body, sort_keys=True)
    assert json.dumps(warm_body, sort_keys=True) == json.dumps(
        plain_body, sort_keys=True)
    assert all(warm_cache[sub]["hit"] for sub in SUBS)
    for sub in SUBS:
        record = json.loads(cache.entry_paths(cold_cache[sub]["key"])[1].read_text())
        assert record["key_inputs"]["mask_policy"] == {
            "filter_policy": "explicit_retained_variant_mask",
            "retained_variant_mask_sha256": records[sub]["sha256"],
            "retained_variant_count": records[sub]["retained_variant_count"],
            "thresholds": {"call_rate_min": 0.9, "maf_min": 0.01, "mac_min": 5},
        }
        assert record["grm_provenance"]["filter_policy"] == (
            "explicit_retained_variant_mask")


def _cli_config(panel, out_dir, cache_dir=None):
    grm = {"method": "grm_from_X", "maf_min": 0.01, "scope": "all_subgenomes"}
    if cache_dir is not None:
        grm["cache_dir"] = str(cache_dir)
    return {
        "interact": {
            "mode": "group", "subgenomes": list(SUBS),
            "groups": str(panel.groups), "statistic": "omniB",
            "hypothesis_unit": "edge", "subset_order": 2,
            "family_scope": "primary_only", "primary_transform": "INT",
            "primary_multiplicity": "bootstrap_minp",
            "genotype": dict(panel.genotype),
            "snp_to_gene": dict(panel.mapping),
            "phenotype": str(panel.phenotype), "sample_col": "sample",
            "trait": "trait",
            "burden": {"cap": 150, "min_snp": 3, "maf_min": 0.01, "n_pc": 3,
                       "feature_seed": 4103},
            "grm": grm,
            "calibration": {"method": "bootstrap", "B": 19, "seed": 2026,
                            "qa_only": True},
        },
        "outputs": {"out_dir": str(out_dir), "full_ranking": True,
                    "plots": False},
    }


def _run_cli(panel, tmp_path, label, cache_dir=None):
    out_dir = tmp_path / f"out_{label}"
    cfg = _cli_config(panel, out_dir, cache_dir)
    config_path = tmp_path / f"config_{label}.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    I.validate_interact_config(cfg)
    assert I.cmd_interact(SimpleNamespace(
        config=str(config_path), out_dir=None, n_jobs=1)) == 0
    payload = json.loads((out_dir / "interact_trait.json").read_text())
    for volatile in ("config_path", "config_sha256"):
        payload["provenance"].pop(volatile)
    ranking = (out_dir / "interact_trait_ranking_group_INT.tsv").read_bytes()
    return payload, ranking


def test_cli_result_identical_with_and_without_cache(panel, tmp_path):
    cache_dir = tmp_path / "grm-cache"
    plain, plain_ranking = _run_cli(panel, tmp_path, "plain")
    cold, cold_ranking = _run_cli(panel, tmp_path, "cold", cache_dir)
    warm, warm_ranking = _run_cli(panel, tmp_path, "warm", cache_dir)
    assert cold_ranking == plain_ranking == warm_ranking
    plain_body, plain_cache = _pop_cache_fields(plain["results"]["INT"])
    cold_body, cold_cache = _pop_cache_fields(cold["results"]["INT"])
    warm_body, warm_cache = _pop_cache_fields(warm["results"]["INT"])
    plain["results"]["INT"] = plain_body
    cold["results"]["INT"] = cold_body
    warm["results"]["INT"] = warm_body
    assert plain_cache == {sub: None for sub in SUBS}
    assert all(cold_cache[sub]["hit"] is False for sub in SUBS)
    assert all(warm_cache[sub]["hit"] is True for sub in SUBS)
    assert json.dumps(cold, sort_keys=True) == json.dumps(plain, sort_keys=True)
    assert json.dumps(warm, sort_keys=True) == json.dumps(plain, sort_keys=True)
    assert len(list(cache_dir.glob("*.npy"))) == len(SUBS)


def test_cli_reports_cache_failure_without_traceback(panel, tmp_path, capsys):
    cache_dir = tmp_path / "grm-cache"
    _run_cli(panel, tmp_path, "cold", cache_dir)
    for npy in cache_dir.glob("*.npy"):
        npy.unlink()
    out_dir = tmp_path / "out_broken"
    cfg = _cli_config(panel, out_dir, cache_dir)
    config_path = tmp_path / "config_broken.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert I.cmd_interact(SimpleNamespace(
        config=str(config_path), out_dir=None, n_jobs=1)) == 1
    assert "GRM cache" in capsys.readouterr().out
    assert not (out_dir / "interact_trait.json").exists()


@pytest.mark.parametrize("cache_dir, message", [
    ("relative/cache", "absolute"),
    (7, "string"),
    ("", "string"),
])
def test_validate_rejects_invalid_cache_dir(panel, tmp_path, cache_dir, message):
    cfg = _cli_config(panel, tmp_path / "out")
    cfg["interact"]["grm"]["cache_dir"] = cache_dir
    with pytest.raises(SystemExit, match=message):
        I.validate_interact_config(cfg)


def test_validate_rejects_cache_dir_under_a_file(panel, tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    cfg = _cli_config(panel, tmp_path / "out", blocker / "cache")
    with pytest.raises(SystemExit, match="directory"):
        I.validate_interact_config(cfg)


def test_validate_accepts_absent_or_creatable_cache_dir(panel, tmp_path):
    I.validate_interact_config(_cli_config(panel, tmp_path / "out"))
    nested = tmp_path / "not" / "yet" / "there"
    I.validate_interact_config(_cli_config(panel, tmp_path / "out", nested))
    assert not nested.exists()
    assert I.preflight_interact(_cli_config(panel, tmp_path / "out", nested)) == []


def test_validate_rejects_cache_dir_outside_canonical_group_omnib(tmp_path):
    cfg = {
        "interact": {
            "mode": "pairwise", "subgenomes": ["A", "B"], "statistic": "burden",
            "genotype": {"A": "a", "B": "b"},
            "snp_to_gene": {"A": "na", "B": "nb"},
            "pairs": "pairs.tsv", "phenotype": "ph.tsv",
            "sample_col": "sample", "trait": "trait",
            "grm": {"cache_dir": str(tmp_path / "cache")},
        },
    }
    with pytest.raises(SystemExit, match="canonical group omniB"):
        I.validate_interact_config(cfg)


def test_cache_dir_is_not_part_of_the_key(panel, tmp_path):
    subdata = _subdata(panel)
    idx = _sample_idx(panel)
    first = GRMCache(tmp_path / "cache-one")
    second = GRMCache(tmp_path / "cache-two")
    _, one = _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True,
        grm_cache=first, subgenome="A")
    _, two = _build_grm(
        subdata["A"], idx, "grm_from_X", 0.01, return_provenance=True,
        grm_cache=second, subgenome="A")
    assert one["cache"]["key"] == two["cache"]["key"]
    assert one["cache"]["npy_sha256"] == two["cache"]["npy_sha256"]
    assert os.path.basename(first.entry_paths(one["cache"]["key"])[0]) == (
        one["cache"]["key"] + ".npy")
