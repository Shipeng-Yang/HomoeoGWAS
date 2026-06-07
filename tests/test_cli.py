"""Unit tests for cli.py — Phase 2 EXIT ``homoeogwas fit``.

Tiny synthetic panels only — no Horvath/wheat data, no GPU, no external
binaries. Exercises config parsing, the flag budget, the end-to-end fit
pipeline (in-memory + streaming + J=3 Hadamard) and error handling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homoeogwas import cli  # noqa: E402

# ---------------------------------------------------------------------
# synthetic panel
# ---------------------------------------------------------------------

def _write_bed(prefix: Path, dosage, samples, chrom, pos):
    from bed_reader import to_bed
    prefix.parent.mkdir(parents=True, exist_ok=True)
    m = dosage.shape[1]
    to_bed(str(prefix.with_suffix(".bed")), dosage,
           properties={
               "fid": ["0"] * len(samples),
               "iid": list(samples),
               "sid": [f"{prefix.name}_{i}" for i in range(m)],
               "chromosome": [str(c) for c in chrom],
               "bp_position": [int(p) for p in pos],
               "allele_1": ["A"] * m, "allele_2": ["B"] * m,
           }, count_A1=True)


def _make_panel(tmp: Path, subgenomes, n=60, m=2000, seed=0):
    """Write per-subgenome BEDs + a phenotype TSV; return (geno_dir, pheno)."""
    rng = np.random.default_rng(seed)
    samples = [f"S{i:03d}" for i in range(n)]
    geno_dir = tmp / "panel"
    dos_by_sg = {}
    for _si, sg in enumerate(subgenomes):
        dos = rng.binomial(2, 0.3, size=(n, m)).astype(np.float32)
        # two chromosomes per subgenome
        chrom = [f"{sg}{1 + (j // (m // 2))}" for j in range(m)]
        pos = [(j % (m // 2)) * 100_000 + 1 for j in range(m)]
        _write_bed(geno_dir / sg / "all", dos, samples, chrom, pos)
        dos_by_sg[sg] = dos
    # phenotype with mild genetic signal from subgenome-0 markers
    g = dos_by_sg[subgenomes[0]][:, :3].astype(np.float64)
    g = (g - g.mean(0)) / (g.std(0) + 1e-9)
    y = 0.8 * g.sum(1) + rng.standard_normal(n)
    pheno = tmp / "pheno.tsv"
    pd.DataFrame({"sample": samples, "trait": y}).to_csv(
        pheno, sep="\t", index=False)
    return geno_dir, pheno


def _write_config(tmp: Path, geno_dir: Path, pheno: Path, subgenomes,
                  *, mode="memory", hadamard=False, out_dir=None,
                  trait="trait", chunk_size=10):
    cfg = {
        "fit_version": 1,
        "panel": {"name": "synthetic", "subgenomes": list(subgenomes)},
        "phenotype": {"path": str(pheno), "sample_col": "sample",
                      "trait": trait},
        "genotype": {
            "scan_bed_prefix_template": str(geno_dir / "{subgenome}" / "all"),
            "grm": {"source": "bed",
                    "bed_prefix_template": str(geno_dir / "{subgenome}" / "all"),
                    "maf_min": 0.01}},
        "kernels": {"normalize": "trace", "include_hadamard": hadamard,
                    "hadamard_name": "hom"},
        "reml": {"n_starts": 3, "seed": 1},
        "scan": {"mode": mode, "backend": "cpu", "batch_size": 1000,
                 "chunk_size": chunk_size, "maf_min": 0.01,
                 "call_rate_min": 0.5},
        "plots": {"enabled": True},
        "outputs": {"out_dir": str(out_dir or (tmp / "out")),
                    "prefix": trait},
    }
    path = tmp / "fit.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


# ---------------------------------------------------------------------
# parser / flag budget
# ---------------------------------------------------------------------

def test_fit_flag_budget():
    parser = cli.build_parser()
    sub = [a for a in parser._actions
           if a.__class__.__name__ == "_SubParsersAction"][0]
    fit = sub.choices["fit"]
    dests = {a.dest for a in fit._actions if a.dest != "help"}
    assert dests == {"config", "out_dir", "backend", "dry_run", "force"}
    assert len(dests) <= 5


def test_parser_fit_requires_config():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["fit"])               # --config is required


def test_version_flag():
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as e:
        parser.parse_args(["--version"])
    assert e.value.code == 0


# ---------------------------------------------------------------------
# config loading / validation
# ---------------------------------------------------------------------

def test_load_config_and_validate(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"])
    cfg_path = _write_config(tmp_path, geno_dir, pheno, ["A", "C"])
    cfg = cli.load_config(cfg_path)
    cli.validate_config(cfg)                     # no raise
    assert cfg["panel"]["subgenomes"] == ["A", "C"]


def test_panel_manifest_merge(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"])
    manifest = tmp_path / "manifest.yaml"
    with open(manifest, "w") as fh:
        yaml.safe_dump({"panel": "syn_panel", "subgenomes": ["A", "C"]}, fh)
    cfg = {
        "fit_version": 1, "panel_manifest": "manifest.yaml",
        "phenotype": {"path": str(pheno), "sample_col": "sample",
                      "trait": "trait"},
        "genotype": {"scan_bed_prefix_template":
                     str(geno_dir / "{subgenome}" / "all"),
                     "grm": {"source": "bed", "bed_prefix_template":
                             str(geno_dir / "{subgenome}" / "all")}},
        "outputs": {"out_dir": str(tmp_path / "out")},
    }
    cfg_path = tmp_path / "fit.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh)
    resolved = cli.load_config(cfg_path)
    # subgenomes + panel.name inherited from the manifest
    assert resolved["panel"]["subgenomes"] == ["A", "C"]
    assert resolved["panel"]["name"] == "syn_panel"


def test_validate_rejects_bad_config(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"])
    base = cli.load_config(_write_config(tmp_path, geno_dir, pheno, ["A", "C"]))
    bad = json.loads(json.dumps(base))
    bad["panel"]["subgenomes"] = []
    with pytest.raises(SystemExit):
        cli.validate_config(bad)
    bad2 = json.loads(json.dumps(base))
    bad2["scan"]["mode"] = "turbo"
    with pytest.raises(SystemExit):
        cli.validate_config(bad2)
    bad3 = json.loads(json.dumps(base))
    bad3["kernels"]["include_hadamard"] = True
    bad3["kernels"]["hadamard_name"] = "A"        # collides with subgenome
    with pytest.raises(SystemExit):
        cli.validate_config(bad3)


# ---------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------

def test_dry_run_no_scan(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"])
    cfg_path = _write_config(tmp_path, geno_dir, pheno, ["A", "C"])
    out = tmp_path / "out"
    rc = cli.main(["fit", "--config", str(cfg_path), "--dry-run"])
    assert rc == 0
    assert not (out / "summary_trait.json").exists()   # nothing scanned


def test_dry_run_reports_missing_path(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"])
    cfg_path = _write_config(tmp_path, geno_dir, pheno, ["A", "C"])
    (geno_dir / "C" / "all.bed").unlink()              # break a path
    rc = cli.main(["fit", "--config", str(cfg_path), "--dry-run"])
    assert rc == 1                                     # preflight problem


# ---------------------------------------------------------------------
# end-to-end fit
# ---------------------------------------------------------------------

def _assert_fit_outputs(out: Path, prefix: str):
    summary_path = out / f"summary_{prefix}.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["acceptance_all_passed"] is True
    assert (out / f"lambda_gc_{prefix}.tsv").exists()
    assert (out / f"qq_{prefix}.png").exists()
    assert (out / f"manhattan_{prefix}.png").exists()
    assert (out / "resolved_config.yaml").exists()
    assert np.isfinite(summary["lambda_gc"]["all"])
    return summary


def test_fit_pipeline_in_memory(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"], n=80, m=2000)
    out = tmp_path / "out"
    cfg_path = _write_config(tmp_path, geno_dir, pheno, ["A", "C"],
                             mode="memory", out_dir=out)
    rc = cli.main(["fit", "--config", str(cfg_path), "--backend", "cpu"])
    assert rc == 0
    summary = _assert_fit_outputs(out, "trait")
    assert summary["scan"]["resolved"] == "memory"
    ss = pd.read_csv(out / "sumstats_trait.tsv", sep="\t")
    assert len(ss) > 0
    assert np.all((ss["p"] >= 0) & (ss["p"] <= 1))
    assert np.all(ss["chi2"] >= 0)


def test_fit_pipeline_streaming(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"], n=80, m=2000)
    out = tmp_path / "out"
    cfg_path = _write_config(tmp_path, geno_dir, pheno, ["A", "C"],
                             mode="stream", out_dir=out, chunk_size=7)
    rc = cli.main(["fit", "--config", str(cfg_path), "--backend", "cpu"])
    assert rc == 0
    summary = _assert_fit_outputs(out, "trait")
    assert summary["scan"]["resolved"] == "stream"
    for sg in ("A", "C"):
        assert (out / f"sumstats_trait_{sg}.tsv.gz").exists()


def test_fit_pipeline_j3_hadamard(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "B", "D"], n=90, m=2000)
    out = tmp_path / "out"
    cfg_path = _write_config(tmp_path, geno_dir, pheno, ["A", "B", "D"],
                             mode="memory", hadamard=True, out_dir=out)
    rc = cli.main(["fit", "--config", str(cfg_path), "--backend", "cpu"])
    assert rc == 0
    summary = _assert_fit_outputs(out, "trait")
    assert summary["kernel_names"] == ["A", "B", "D", "hom"]
    assert summary["include_hadamard"] is True


def test_fit_force_overwrite(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"], n=70, m=2000)
    out = tmp_path / "out"
    cfg_path = _write_config(tmp_path, geno_dir, pheno, ["A", "C"],
                             out_dir=out)
    assert cli.main(["fit", "--config", str(cfg_path), "--backend", "cpu"]) == 0
    # second run without --force must refuse
    with pytest.raises(SystemExit):
        cli.main(["fit", "--config", str(cfg_path), "--backend", "cpu"])
    # with --force it succeeds
    assert cli.main(["fit", "--config", str(cfg_path), "--backend", "cpu",
                     "--force"]) == 0


# ---------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------

def test_fit_missing_trait(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"])
    cfg_path = _write_config(tmp_path, geno_dir, pheno, ["A", "C"],
                             trait="trait")
    cfg = yaml.safe_load(open(cfg_path))
    cfg["phenotype"]["trait"] = "nonexistent_trait"
    cfg["outputs"]["prefix"] = "nonexistent_trait"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh)
    with pytest.raises(SystemExit):
        cli.main(["fit", "--config", str(cfg_path), "--backend", "cpu"])


def test_fit_missing_config():
    with pytest.raises(SystemExit):
        cli.main(["fit", "--config", "/no/such/config.yaml"])


# ---------------------------------------------------------------------
# LOCO pipeline (Phase 3 — M3.1)
# ---------------------------------------------------------------------


def _write_loco_config(tmp: Path, geno_dir: Path, pheno: Path, subgenomes,
                       *, mode="memory", hadamard=False, out_dir=None,
                       trait="trait", chunk_size=10,
                       min_denom_frac=1.0e-6):
    cfg = {
        "fit_version": 1,
        "panel": {"name": "synthetic_loco", "subgenomes": list(subgenomes)},
        "phenotype": {"path": str(pheno), "sample_col": "sample",
                      "trait": trait},
        "genotype": {
            "scan_bed_prefix_template": str(geno_dir / "{subgenome}" / "all"),
            "grm": {"source": "bed",
                    "bed_prefix_template": str(geno_dir / "{subgenome}" / "all"),
                    "maf_min": 0.01}},
        "kernels": {"normalize": "trace", "include_hadamard": hadamard,
                    "hadamard_name": "hom"},
        "reml": {"n_starts": 3, "seed": 1},
        "scan": {"mode": mode, "backend": "cpu", "batch_size": 1000,
                 "chunk_size": chunk_size, "maf_min": 0.01,
                 "call_rate_min": 0.5,
                 "loco": {"enabled": True, "fallback": "error",
                          "min_denominator_fraction": min_denom_frac}},
        "plots": {"enabled": True},
        "outputs": {"out_dir": str(out_dir or (tmp / "out")),
                    "prefix": trait},
    }
    path = tmp / "fit_loco.yaml"
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def test_fit_pipeline_loco_in_memory(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"], n=80, m=2000)
    out = tmp_path / "out_loco"
    cfg_path = _write_loco_config(tmp_path, geno_dir, pheno, ["A", "C"],
                                  mode="memory", out_dir=out)
    rc = cli.main(["fit", "--config", str(cfg_path), "--backend", "cpu"])
    assert rc == 0
    summary = _assert_fit_outputs(out, "trait")
    assert summary["loco"] is not None
    assert summary["loco"]["enabled"] is True
    assert summary["loco"]["chrom_count"] >= 2
    # filename gets the _loco suffix in run_scan
    assert (out / "sumstats_trait_loco.tsv").exists()
    # acceptance includes LOCO gates (name reflects what's
    # actually checked — V_(-c) Cholesky jitter, not K PSD)
    acc_names = {c["check"] for c in summary["acceptance"]}
    assert "loco_all_chrom_V_pd" in acc_names
    assert "loco_chrom_to_subgenome_unique" in acc_names
    assert "loco_all_chrom_retained_above_warn" in acc_names
    # provenance uses retained/removed (not "excluded")
    one_chrom = next(iter(summary["loco"]["loco_chrom_provenance"].values()))
    assert "denom_retained" in one_chrom
    assert "denom_removed" in one_chrom
    assert "n_variants_retained" in one_chrom
    assert "n_variants_removed" in one_chrom
    assert "retained_fraction_low_risk" in one_chrom


def test_fit_pipeline_loco_streaming(tmp_path):
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"], n=80, m=2000)
    out = tmp_path / "out_loco_stream"
    cfg_path = _write_loco_config(tmp_path, geno_dir, pheno, ["A", "C"],
                                  mode="stream", out_dir=out, chunk_size=7)
    rc = cli.main(["fit", "--config", str(cfg_path), "--backend", "cpu"])
    assert rc == 0
    summary = _assert_fit_outputs(out, "trait")
    assert summary["loco"]["enabled"] is True
    for sg in ("A", "C"):
        assert (out / f"sumstats_trait_loco_{sg}.tsv.gz").exists()


def test_fit_validate_loco_rejects_npz(tmp_path):
    """LOCO requires bed source — npz cache lacks per-chrom partials."""
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"], n=40, m=500)
    cfg_path = _write_loco_config(tmp_path, geno_dir, pheno, ["A", "C"])
    cfg = yaml.safe_load(open(cfg_path))
    cfg["genotype"]["grm"]["source"] = "npz"
    cfg["genotype"]["grm"]["npz_path"] = str(tmp_path / "fake.npz")
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh)
    with pytest.raises(SystemExit, match="requires genotype.grm.source=bed"):
        cli.main(["fit", "--config", str(cfg_path), "--backend", "cpu"])


def test_fit_validate_loco_rejects_fallback_emmax(tmp_path):
    """fallback=emmax is reserved but unimplemented in Phase 3 M3.1."""
    geno_dir, pheno = _make_panel(tmp_path, ["A", "C"], n=40, m=500)
    cfg_path = _write_loco_config(tmp_path, geno_dir, pheno, ["A", "C"])
    cfg = yaml.safe_load(open(cfg_path))
    cfg["scan"]["loco"]["fallback"] = "emmax"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh)
    with pytest.raises(SystemExit, match="fallback=emmax"):
        cli.main(["fit", "--config", str(cfg_path), "--backend", "cpu"])


def test_fit_loco_flag_budget_unchanged(tmp_path):
    """LOCO is wired through YAML, the 5-flag CLI surface stays unchanged."""
    parser = cli.build_parser()
    fit_subparser = parser._subparsers._group_actions[0].choices["fit"]  # type: ignore
    optional_flags = [a for a in fit_subparser._actions
                      if a.option_strings and a.dest != "help"]
    # config, out-dir, backend, dry-run, force = 5 flags
    assert len(optional_flags) == 5
