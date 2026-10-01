"""Batch-executed interact cases must equal one CLI invocation per case."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

import homoeogwas.interact as I
from homoeogwas import interact_batch as batch
from tests.test_grm_cache import SUBS, _write_panel


def _phenotypes(panel, root: Path, count: int) -> list[Path]:
    rng = np.random.default_rng(404)
    paths = []
    for index in range(count):
        path = root / f"pheno_{index}.tsv"
        path.write_text("sample\ttrait\n" + "".join(
            f"{sample}\t{rng.normal():.10g}\n"
            for position, sample in enumerate(panel.samples) if position % 10 != 3
        ))
        paths.append(path)
    return paths


def _config(panel, phenotype: Path, out_dir: Path, seed: int, *, benchmark: bool) -> dict:
    interact = {
        "mode": "group", "subgenomes": list(SUBS),
        "groups": str(panel.groups), "statistic": "omniB",
        "hypothesis_unit": "group", "subset_order": 2,
        "family_scope": "primary_only", "primary_transform": "INT",
        "primary_multiplicity": "bootstrap_minp",
        "genotype": dict(panel.genotype), "snp_to_gene": dict(panel.mapping),
        "phenotype": str(phenotype), "sample_col": "sample", "trait": "trait",
        "burden": {"cap": 150, "min_snp": 3, "maf_min": 0.01, "n_pc": 3,
                   "feature_seed": 4103},
        "grm": {"method": "grm_from_X", "maf_min": 0.01, "scope": "all_subgenomes"},
        "calibration": {"method": "bootstrap", "B": 19, "seed": seed, "qa_only": True,
                        "checkpoint": {"enabled": True, "block_size": 5,
                                       "root": str(out_dir / "checkpoints")}},
    }
    if benchmark:
        interact["benchmark_identity"] = {
            "panel_id": "SYNTH.PANEL", "sample_context": "full", "feature_seed": 4103}
    return {"interact": interact,
            "outputs": {"out_dir": str(out_dir), "full_ranking": True, "plots": False}}


def _identity(out_dir: Path) -> dict:
    payload = json.loads((out_dir / "interact_trait.json").read_text())
    for volatile in ("config_path", "config_sha256"):
        payload["provenance"].pop(volatile, None)
    payload["provenance"].pop("parallel_execution", None)
    payload["results"]["INT"]["model_diagnostics"].pop("parallel_execution", None)
    text = json.dumps(payload, sort_keys=True).replace(str(out_dir), "<OUT>")
    files = {}
    for path in sorted((out_dir / "checkpoints").rglob("*")):
        if path.is_file():
            files[str(path.relative_to(out_dir))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return {
        "payload": text,
        "ranking": (out_dir / "interact_trait_ranking_group_INT.tsv").read_bytes(),
        "checkpoint_files": files,
    }


SOURCE = {"git_commit": "1" * 40, "git_tree": "2" * 40,
          "package_source_sha256": "3" * 64, "source_clean": True}
POLICY = {name: "1" for name in (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
RUNTIME = {"homoeogwas": "0.test", "python": "3.test", "numpy": "2.test",
           "scipy": "1.test", "blas_thread_policy": POLICY}


@pytest.fixture(autouse=True)
def formal_runtime(monkeypatch):
    from homoeogwas import formal_provenance as P

    monkeypatch.setattr(P, "capture_source_identity", lambda: SOURCE)
    monkeypatch.setattr(P, "runtime_fingerprint", lambda _policy: RUNTIME)
    for name in POLICY:
        monkeypatch.setenv(name, "1")


def _write_config(path: Path, cfg: dict) -> Path:
    from homoeogwas.formal_provenance import PRE_RUN_MANIFEST_SCHEMA

    manifest = path.with_suffix(".pre_run_manifest.json")
    cfg = dict(cfg) | {"provenance": {"pre_run_manifest": str(manifest),
                                      "blas_thread_policy": POLICY}}
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    manifest.write_text(json.dumps({
        "schema": PRE_RUN_MANIFEST_SCHEMA,
        "config": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "source_identity": SOURCE,
        "runtime_fingerprint": RUNTIME,
    }, sort_keys=True) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("benchmark", [False, True])
def test_batch_equals_single_cli_in_any_order(tmp_path, benchmark, monkeypatch):
    import shutil

    import homoeogwas.omnib_family as F

    panel = _write_panel(tmp_path / "panel")
    phenos = _phenotypes(panel, tmp_path, 2)
    seeds = (2026, 2027)
    order = (0, 1, 0)
    cases, reference = [], []
    for step, index in enumerate(order):
        out_dir = tmp_path / f"case{step}"
        config = _write_config(
            tmp_path / f"case{step}.yaml",
            _config(panel, phenos[index], out_dir, seeds[index], benchmark=benchmark))
        assert I.cmd_interact(SimpleNamespace(
            config=str(config), out_dir=None, n_jobs=2)) == 0
        reference.append(_identity(out_dir))
        shutil.rmtree(out_dir)
        cases.append({
            "case_id": f"case{step}", "config": str(config),
            "execution_record": str(tmp_path / "records" / f"case{step}.json"),
            "log": str(tmp_path / "logs" / f"case{step}.log"),
        })
    assert reference[0]["ranking"] != reference[1]["ranking"]
    assert reference[0]["ranking"] == reference[2]["ranking"]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    calls = {"load": 0, "context": 0, "mask": 0}

    def counted(name, function):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return function(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(I, "_load_subgenome", counted("load", I._load_subgenome))
    monkeypatch.setattr(I, "_build_benchmark_mask_records",
                        counted("mask", I._build_benchmark_mask_records))
    monkeypatch.setattr(F, "build_omnib_panel_context",
                        counted("context", F.build_omnib_panel_context))
    assert batch.run_batch(cases_path, n_jobs=2, batch_id="b1") == 0
    assert calls == {"load": len(SUBS), "context": 1, "mask": int(benchmark)}
    for step in range(len(order)):
        assert _identity(tmp_path / f"case{step}") == reference[step]
        record = json.loads(Path(cases[step]["execution_record"]).read_text())
        assert record["exit_code"] == 0 and record["batch_id"] == "b1"
        assert record["config_sha256"] == record["config_sha256_after"]


def test_batch_resume_skips_recorded_cases_and_isolates_failures(tmp_path):
    panel = _write_panel(tmp_path / "panel")
    (pheno,) = _phenotypes(panel, tmp_path, 1)
    good = _write_config(tmp_path / "good.yaml",
                         _config(panel, pheno, tmp_path / "good", 2026, benchmark=False))
    bad_cfg = _config(panel, pheno, tmp_path / "bad", 2026, benchmark=False)
    bad_cfg["interact"]["phenotype"] = str(tmp_path / "missing.tsv")
    bad = _write_config(tmp_path / "bad.yaml", bad_cfg)
    cases = [
        {"case_id": name, "config": str(path),
         "execution_record": str(tmp_path / "records" / f"{name}.json"),
         "log": str(tmp_path / "logs" / f"{name}.log")}
        for name, path in (("bad", bad), ("good", good))
    ]
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(cases))
    assert batch.run_batch(cases_path, n_jobs=1, batch_id="r1") == 1
    records = {c["case_id"]: json.loads(Path(c["execution_record"]).read_text())
               for c in cases}
    assert records["bad"]["exit_code"] != 0
    assert records["good"]["exit_code"] == 0
    before = Path(cases[1]["execution_record"]).read_bytes()
    assert batch.run_batch(cases_path, n_jobs=1, batch_id="r2") == 1
    assert Path(cases[1]["execution_record"]).read_bytes() == before
    assert "missing.tsv" in Path(records["bad"]["log"]).read_text()
    assert Path(records["good"]["log"]).name == "good.log.attempt1"


def _batch(tmp_path, configs, name):
    cases = [{"case_id": f"{name}{k}", "config": str(path),
              "execution_record": str(tmp_path / name / "records" / f"{k}.json"),
              "log": str(tmp_path / name / "logs" / f"{k}.log")}
             for k, path in enumerate(configs)]
    cases_path = tmp_path / f"{name}.json"
    cases_path.write_text(json.dumps(cases))
    return cases_path


def _single_then_clear(configs):
    import shutil

    reference = []
    for path in configs:
        cfg = yaml.safe_load(Path(path).read_text())
        out_dir = Path(cfg["outputs"]["out_dir"])
        assert I.cmd_interact(SimpleNamespace(config=str(path), out_dir=None, n_jobs=1)) == 0
        reference.append(_identity(out_dir))
        shutil.rmtree(out_dir)
    return reference


def test_batch_rebuilds_context_when_sample_set_or_groups_change(tmp_path, monkeypatch):
    import homoeogwas.omnib_family as F

    panel = _write_panel(tmp_path / "panel")
    full, = _phenotypes(panel, tmp_path, 1)
    partial = tmp_path / "pheno_partial.tsv"
    lines = full.read_text().splitlines()
    partial.write_text("\n".join(lines[:1] + lines[1:-4]) + "\n")
    groups2 = tmp_path / "groups2.tsv"
    groups2.write_text(panel.groups.read_text().replace("grp0\t", "grpX\t"))
    configs = []
    for k, (pheno, groups) in enumerate(
            ((full, panel.groups), (partial, panel.groups), (full, groups2), (full, panel.groups))):
        cfg = _config(panel, pheno, tmp_path / f"out{k}", 2026 + k, benchmark=True)
        cfg["interact"]["groups"] = str(groups)
        configs.append(_write_config(tmp_path / f"cfg{k}.yaml", cfg))
    reference = _single_then_clear(configs)
    calls = {"context": 0}
    original = F.build_omnib_panel_context

    def counted(*args, **kwargs):
        calls["context"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(F, "build_omnib_panel_context", counted)
    assert batch.run_batch(_batch(tmp_path, configs, "b"), n_jobs=1) == 0
    assert calls["context"] == 4
    for k in range(4):
        assert _identity(tmp_path / f"out{k}") == reference[k]


def test_batch_with_grm_cache_does_not_share_context(tmp_path, monkeypatch):
    panel = _write_panel(tmp_path / "panel")
    phenos = _phenotypes(panel, tmp_path, 2)
    configs = []
    for k, pheno in enumerate(phenos):
        cfg = _config(panel, pheno, tmp_path / f"out{k}", 2026 + k, benchmark=False)
        cfg["interact"]["grm"]["cache_dir"] = str(tmp_path / "grm-cache")
        configs.append(_write_config(tmp_path / f"cfg{k}.yaml", cfg))
    reference = _single_then_clear(configs)
    contexts = []
    original = I.run_group_scan_omnib

    def spy(*args, **kwargs):
        contexts.append(kwargs.get("panel_context"))
        return original(*args, **kwargs)

    monkeypatch.setattr(I, "run_group_scan_omnib", spy)
    assert batch.run_batch(_batch(tmp_path, configs, "c"), n_jobs=1) == 0
    assert contexts == [None, None]
    for k in range(2):
        payload = json.loads((tmp_path / f"out{k}" / "interact_trait.json").read_text())
        grm = payload["results"]["INT"]["model_diagnostics"]["grm_provenance"]["subgenomes"]
        assert all(grm[sub]["cache"]["hit"] is True for sub in SUBS)
        observed, expected = _identity(tmp_path / f"out{k}"), reference[k]
        assert observed["ranking"] == expected["ranking"]
        assert observed["checkpoint_files"] == expected["checkpoint_files"]
        assert _without_cache(observed["payload"]) == _without_cache(expected["payload"])


def _without_cache(text):
    payload = json.loads(text)
    for sub in payload["results"]["INT"]["model_diagnostics"]["grm_provenance"]["subgenomes"].values():
        sub.pop("cache", None)
    for sub in (payload["provenance"].get("grm_provenance") or {}).get("subgenomes", {}).values():
        sub.pop("cache", None)
    return json.dumps(payload, sort_keys=True)
