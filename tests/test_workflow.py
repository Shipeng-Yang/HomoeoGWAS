"""Tests for the breeder-level workflow engine + MCP server wiring."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from homoeogwas import workflow

HAVE_MCP = importlib.util.find_spec("mcp") is not None


def _touch_bed(prefix):
    for ext in (".bed", ".bim", ".fam"):
        (prefix.parent).mkdir(parents=True, exist_ok=True)
        Path(str(prefix) + ext).write_text("x")


def test_infer_interaction_mode():
    assert workflow.infer_interaction_mode(["A", "B"]) == "pairwise"
    assert workflow.infer_interaction_mode(["A", "B", "C"]) == "triad"
    assert workflow.infer_interaction_mode(["A", "B", "C", "D"]) == "group"


def test_mcp_prep_homoeologs_four_copy_command_targets_group_wide_table():
    from homoeogwas.mcp_server import _prep_homoeologs_command

    mode, args = _prep_homoeologs_command(
        ["A", "B", "C", "D"], "genes_{S}.tsv", "quartets.tsv",
        from_table="curated.tsv", table_format="wide")
    assert mode == "group"
    assert args[:3] == ["prep-homoeologs", "--mode", "group"]
    assert args[args.index("--subgenomes") + 1] == "A,B,C,D"
    assert args[args.index("--table-format") + 1] == "wide"


def test_build_fit_config_shape():
    cfg = workflow.build_fit_config(
        subgenomes=["A", "B", "D"], phenotype="p.tsv", sample_col="IID",
        trait="yield", bed_template="g/{subgenome}/all", out_dir="out",
        include_hadamard=True, loco=True)
    assert cfg["panel"]["subgenomes"] == ["A", "B", "D"]
    assert cfg["phenotype"] == {"path": "p.tsv", "sample_col": "IID",
                                "trait": "yield"}
    assert cfg["genotype"]["scan_bed_prefix_template"] == "g/{subgenome}/all"
    assert cfg["kernels"]["include_hadamard"] is True
    assert cfg["scan"]["loco"] == {"enabled": True, "fallback": "error"}
    assert cfg["outputs"] == {"out_dir": "out", "prefix": "yield"}


def test_build_interact_config_modes():
    common = dict(subgenomes=["A", "B", "C"],
                  bed_prefixes={"A": "a", "B": "b", "C": "c"},
                  snp_to_gene={"A": "na", "B": "nb", "C": "nc"},
                  phenotype="p", sample_col="IID", trait="t", out_dir="o")
    cfg = workflow.build_interact_config(triads="tr.tsv", **common)
    assert cfg["interact"]["mode"] == "group"
    assert cfg["interact"]["groups"] == "tr.tsv"
    assert cfg["interact"]["hypothesis_unit"] == "group"
    assert cfg["interact"]["subset_order"] == 2
    assert cfg["interact"]["statistic"] == "omniB"
    assert cfg["interact"]["calibration"]["method"] == "bootstrap"
    assert cfg["interact"]["grm"] == {
        "method": "grm_from_X", "maf_min": 0.01, "scope": "all_subgenomes"}
    assert cfg["interact"]["primary_multiplicity"] == "bootstrap_minp"
    assert cfg["interact"]["primary_transform"] == "INT"
    assert cfg["interact"]["burden"] == {
        "cap": 150, "min_snp": 3, "maf_min": 0.01, "n_pc": 3}
    triad3 = workflow.build_interact_config(
        triads="tr.tsv", statistic="triad3", **common)
    assert triad3["interact"]["statistic"] == "triad3"
    assert triad3["interact"]["calibration"]["B"] == 2000
    assert triad3["interact"]["primary_multiplicity"] == "bootstrap_minp"
    with pytest.raises(ValueError, match="triads TSV"):
        workflow.build_interact_config(**common)   # triad without triads
    # pairwise
    pw = workflow.build_interact_config(
        subgenomes=["A", "B"], bed_prefixes={"A": "a", "B": "b"},
        snp_to_gene={"A": "na", "B": "nb"}, phenotype="p", sample_col="IID",
        trait="t", out_dir="o", pairs="pr.tsv")
    assert pw["interact"]["mode"] == "group"
    assert pw["interact"]["groups"] == "pr.tsv"
    assert pw["interact"]["hypothesis_unit"] == "edge"
    with pytest.raises(ValueError, match="requires exactly three"):
        workflow.build_interact_config(
            subgenomes=["A", "B"], bed_prefixes={"A": "a", "B": "b"},
            snp_to_gene={"A": "na", "B": "nb"}, phenotype="p",
            sample_col="IID", trait="t", out_dir="o", pairs="pr.tsv",
            statistic="triad3")


def test_build_interact_config_accepts_explicit_canonical_group_options():
    cfg = workflow.build_interact_config(
        subgenomes=["A", "B", "D"],
        bed_prefixes={"A": "a", "B": "b", "D": "d"},
        snp_to_gene={"A": "na", "B": "nb", "D": "nd"},
        phenotype="p", sample_col="IID", trait="t", out_dir="o",
        groups="groups.tsv", hypothesis_unit="edge", subset_order=2,
        family_scope="primary_only")
    assert cfg["interact"]["mode"] == "group"
    assert cfg["interact"]["groups"] == "groups.tsv"
    assert cfg["interact"]["hypothesis_unit"] == "edge"


def test_build_interact_config_routes_four_copies_to_pair_edge_group_omnib():
    subs = ["A", "B", "C", "D"]
    cfg = workflow.build_interact_config(
        subgenomes=subs,
        bed_prefixes={sub: sub.lower() for sub in subs},
        snp_to_gene={sub: f"n{sub.lower()}" for sub in subs},
        phenotype="p", sample_col="IID", trait="t", out_dir="o",
        groups="quartets.tsv",
    )
    assert cfg["interact"] | {
        "mode": "group",
        "groups": "quartets.tsv",
        "hypothesis_unit": "group",
        "subset_order": 2,
        "family_scope": "primary_only",
    } == cfg["interact"]
    assert cfg["outputs"]["full_ranking"] is True


def test_check_sample_ids(tmp_path):
    intp = tmp_path / "int.tsv"
    pd.DataFrame({"IID": [1, 2, 3], "y": [0.1, 0.2, 0.3]}).to_csv(
        intp, sep="\t", index=False)
    r = workflow.check_sample_ids(str(intp), "IID")
    assert r["ok"] and r["integer_like"] is True
    strp = tmp_path / "str.tsv"
    pd.DataFrame({"IID": ["s1", "s2"], "y": [0.1, 0.2]}).to_csv(
        strp, sep="\t", index=False)
    assert workflow.check_sample_ids(str(strp), "IID")["integer_like"] is False
    assert workflow.check_sample_ids(str(strp), "nope")["ok"] is False


def test_materialize_bed_layout_symlinks_and_reports_missing(tmp_path):
    _touch_bed(tmp_path / "one" / "g")          # A present
    # B intentionally absent
    tmpl, missing = workflow._materialize_bed_layout(
        {"A": str(tmp_path / "one" / "g"), "B": str(tmp_path / "missing")},
        ["A", "B"], tmp_path / "work")
    assert tmpl.endswith("geno/{subgenome}/all")
    assert (tmp_path / "work" / "geno" / "A" / "all.bed").exists()  # symlinked
    assert missing == ["B"]                      # missing flagged, not guessed


def test_materialize_bed_layout_refreshes_stale_symlinks(tmp_path):
    first = tmp_path / "first" / "panel"
    second = tmp_path / "second" / "panel"
    _touch_bed(first)
    _touch_bed(second)
    work = tmp_path / "work"
    workflow._materialize_bed_layout({"A": str(first)}, ["A"], work)
    workflow._materialize_bed_layout({"A": str(second)}, ["A"], work)
    link = work / "geno" / "A" / "all.bed"
    assert link.resolve() == Path(str(second) + ".bed").resolve()


def test_run_gwas_dry_run_generates_config(tmp_path):
    pheno = tmp_path / "p.tsv"
    pd.DataFrame({"IID": ["s1", "s2"], "yield": [1.0, 2.0]}).to_csv(
        pheno, sep="\t", index=False)
    for s in ("A", "B"):
        _touch_bed(tmp_path / f"sub_{s}")
    res = workflow.run_gwas(
        phenotype=str(pheno), sample_col="IID", trait="yield",
        subgenomes=["A", "B"],
        bed_prefixes={"A": str(tmp_path / "sub_A"), "B": str(tmp_path / "sub_B")},
        out_dir=str(tmp_path / "out"), dry_run=True)
    assert res["dry_run"] is True
    cfg = res["config"]
    assert cfg.endswith("fit.generated.yaml")
    # validate + fit + plot planned
    cmds = [s["command"] for s in res["steps"]]
    assert any("fit" in c for c in cmds) and any("validate" in c for c in cmds)
    # generated config is loadable and has the trait
    import yaml
    loaded = yaml.safe_load(open(cfg))
    assert loaded["phenotype"]["trait"] == "yield"


def test_run_gwas_stringifies_integer_sample_ids(tmp_path):
    pheno = tmp_path / "p.tsv"
    pd.DataFrame({"IID": [1, 2, 3], "yield": [1.0, 2.0, 3.0]}).to_csv(
        pheno, sep="\t", index=False)
    for s in ("A", "B"):
        _touch_bed(tmp_path / f"sub_{s}")
    res = workflow.run_gwas(
        phenotype=str(pheno), sample_col="IID", trait="yield",
        subgenomes=["A", "B"],
        bed_prefixes={"A": str(tmp_path / "sub_A"), "B": str(tmp_path / "sub_B")},
        out_dir=str(tmp_path / "out"), dry_run=True)
    assert res["ok"] is True
    assert any("read the sample column as strings" in warning
               for warning in res["warnings"])
    # missing PLINK files also block, not guess
    res2 = workflow.run_gwas(
        phenotype=str(pheno), sample_col="IID", trait="yield",
        subgenomes=["A"], bed_prefixes={"A": str(tmp_path / "nope")},
        out_dir=str(tmp_path / "o2"), dry_run=True)
    assert res2["ok"] is False and "missing" in res2["reason"]


def test_run_gwas_stops_after_failed_validation(tmp_path, monkeypatch):
    pheno = tmp_path / "p.tsv"
    pd.DataFrame({"IID": ["s1", "s2"], "yield": [1.0, 2.0]}).to_csv(
        pheno, sep="\t", index=False)
    for s in ("A", "B"):
        _touch_bed(tmp_path / f"sub_{s}")
    calls = []

    def fake_run(args, *, dry_run=False):
        calls.append(list(args))
        return {"command": ["homoeogwas", *args], "returncode": 1}

    monkeypatch.setattr(workflow, "run_cli", fake_run)
    res = workflow.run_gwas(
        phenotype=str(pheno), sample_col="IID", trait="yield",
        subgenomes=["A", "B"],
        bed_prefixes={"A": str(tmp_path / "sub_A"), "B": str(tmp_path / "sub_B")},
        out_dir=str(tmp_path / "out"))
    assert res["ok"] is False
    assert len(calls) == 1 and calls[0][0] == "validate"


def test_run_interaction_dry_run(tmp_path):
    res = workflow.run_interaction(
        phenotype="p", sample_col="IID", trait="t", subgenomes=["A", "B", "C"],
        bed_prefixes={"A": "a", "B": "b", "C": "c"},
        snp_to_gene={"A": "na", "B": "nb", "C": "nc"},
        out_dir=str(tmp_path), groups="tr.tsv", hypothesis_unit="edge",
        dry_run=True)
    assert res["mode"] == "group"
    assert res["hypothesis_unit"] == "edge"
    assert res["config"].endswith("interact.generated.group.omnib.yaml")
    assert [step["command"][3] for step in res["steps"]] == [
        "validate", "interact", "audit"]


@pytest.mark.parametrize(
    "subgenomes,expected", [(["A", "D"], "edge"), (["A", "B", "D"], "group")])
def test_run_interaction_defaults_primary_unit_by_copy_count(
        tmp_path, subgenomes, expected):
    res = workflow.run_interaction(
        phenotype="p", sample_col="IID", trait="t", subgenomes=subgenomes,
        bed_prefixes={s: s.lower() for s in subgenomes},
        snp_to_gene={s: f"n{s.lower()}" for s in subgenomes},
        out_dir=str(tmp_path / expected), groups="groups.tsv", dry_run=True)
    assert res["hypothesis_unit"] == expected


def test_run_interaction_audits_then_summarizes(tmp_path, monkeypatch):
    pheno = tmp_path / "p.tsv"
    pd.DataFrame({"IID": ["s1", "s2"], "t": [1.0, 2.0]}).to_csv(
        pheno, sep="\t", index=False)
    for s in ("A", "D"):
        _touch_bed(tmp_path / s)
        (tmp_path / f"{s}.npz").write_text("mapping")
    groups = tmp_path / "groups.tsv"
    groups.write_text("group_id\tgene_A\tgene_D\ng1\ta1\td1\n")
    calls = []

    def fake_run(args, *, dry_run=False):
        calls.append(list(args))
        return {"command": ["homoeogwas", *args], "returncode": 0}

    monkeypatch.setattr(workflow, "run_cli", fake_run)
    monkeypatch.setattr(
        workflow, "summarize_interaction",
        lambda out_dir, trait: {"ok": True, "trait": trait, "primary_unit": "edge"})
    res = workflow.run_interaction(
        phenotype=str(pheno), sample_col="IID", trait="t",
        subgenomes=["A", "D"],
        bed_prefixes={s: str(tmp_path / s) for s in ("A", "D")},
        snp_to_gene={s: str(tmp_path / f"{s}.npz") for s in ("A", "D")},
        out_dir=str(tmp_path / "out"), groups=str(groups), n_jobs=4)
    assert [call[0] for call in calls] == ["validate", "interact", "audit"]
    assert res["summary"]["primary_unit"] == "edge"


def test_summarize_interaction_reports_authoritative_family_and_paths(tmp_path):
    group_hash = "a" * 64
    edge_hash = "b" * 64
    hit = {
        "hypothesis_id": "group:g1",
        "group_id": "g1",
        "p_interaction": 0.001,
        "p_adjusted_bootstrap_minp": 0.02,
        "driving_component": "pc1",
    }
    family = {
        "n_groups_raw": 10,
        "n_unique_edges": 30,
        "group_family_sha256": group_hash,
        "edge_family_sha256": edge_hash,
    }
    payload = {
        "trait": "height",
        "provenance": {
            "primary_transform": "INT",
            "hypothesis_unit": "group",
            "family_scope": "primary_only",
            **family,
        },
        "results": {"INT": {
            "n": 120,
            "G": 10,
            "n_valid": 9,
            "lambda_gc_obs": 1.03,
            "top": [{"group_id": "g2", "p_interaction": 0.03,
                     "driving_component": "minor_burden"}],
            "model_diagnostics": {
                "family_provenance": family,
                "bootstrap_fwer": {
                    "declared_hypothesis_unit": "group",
                    "family_scope": "primary_only",
                    "n_rejected": 1,
                    "empirical_p": 0.02,
                    "B": 2000,
                    "sig": [hit],
                },
            },
        }},
    }
    (tmp_path / "interact_height.json").write_text(json.dumps(payload))
    ranking = tmp_path / "interact_height_ranking_group_INT.tsv"
    ranking.write_text("hypothesis_id\tp_interaction\n")
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()
    (audit_dir / "homoeogwas_audit.json").write_text(json.dumps({
        "overall_status": "INTERNAL_DISCOVERY_REPLICATION_REQUIRED"}))

    summary = workflow.summarize_interaction(str(tmp_path), "height")
    assert summary["n_groups_raw"] == 10
    assert summary["n_unique_edges"] == 30
    assert summary["group_family_sha256"] == group_hash
    assert summary["edge_family_sha256"] == edge_hash
    assert summary["significant"] == [hit]
    assert summary["significant"][0]["p_adjusted_bootstrap_minp"] == 0.02
    assert summary["lambda_gc"] == 1.03
    assert summary["ranking_tsv"] == str(ranking)
    assert summary["audit_json"] == str(audit_dir / "homoeogwas_audit.json")
    assert summary["audit_status"] == "INTERNAL_DISCOVERY_REPLICATION_REQUIRED"
    assert summary["top_descriptive"][0]["group_id"] == "g2"
    assert summary["top_descriptive_role"] == "descriptive_raw_p_ranking"


def test_get_guidance_returns_spec():
    g = workflow.get_guidance("interaction")
    assert "triad" in g["hint"]
    # AGENTS.md ships in the repo root
    assert g["spec"] and "Agent Workflow Specification" in g["spec"]


@pytest.mark.skipif(not HAVE_MCP, reason="mcp not installed")
def test_mcp_server_builds_and_exposes_tools():
    from homoeogwas import mcp_server
    server = mcp_server.build_server()
    assert server is not None
    # the FastMCP server should expose our breeder-level tools
    import asyncio
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"run_gwas", "run_interaction", "get_guidance", "audit_results",
            "summarize_results"} <= names
    interaction = next(t for t in tools if t.name == "run_interaction")
    assert interaction.inputSchema["properties"]["null_variance"]["default"] == (
        "smooth_pc4")


def test_build_interact_config_defaults_to_smooth_pc4_with_checkpoint(tmp_path):
    cfg = workflow.build_interact_config(
        subgenomes=["A", "D"], bed_prefixes={"A": "a", "D": "d"},
        snp_to_gene={"A": "a.npz", "D": "d.npz"}, phenotype="p.tsv",
        sample_col="sample", trait="t", out_dir=str(tmp_path / "o"), groups="g.tsv")
    cal = cfg["interact"]["calibration"]
    assert cal["null_variance"] == "smooth_pc4"
    assert cal["checkpoint"] == {"enabled": True, "block_size": 25,
                                 "root": str(tmp_path / "o" / "checkpoints")}
    homo = workflow.build_interact_config(
        subgenomes=["A", "D"], bed_prefixes={"A": "a", "D": "d"},
        snp_to_gene={"A": "a.npz", "D": "d.npz"}, phenotype="p.tsv",
        sample_col="sample", trait="t", out_dir="o", groups="g.tsv",
        null_variance="homoscedastic")
    assert homo["interact"]["calibration"]["null_variance"] == "homoscedastic"
    assert "checkpoint" not in homo["interact"]["calibration"]
    with pytest.raises(ValueError):
        workflow.build_interact_config(
            subgenomes=["A", "D"], bed_prefixes={"A": "a", "D": "d"},
            snp_to_gene={"A": "a.npz", "D": "d.npz"}, phenotype="p.tsv",
            sample_col="sample", trait="t", out_dir="o", groups="g.tsv",
            null_variance="other")


def test_generated_smooth_pc4_interact_config_passes_validation(tmp_path):
    from homoeogwas.interact import validate_interact_config
    cfg = workflow.build_interact_config(
        subgenomes=["A", "D"], bed_prefixes={"A": "a", "D": "d"},
        snp_to_gene={"A": "a.npz", "D": "d.npz"}, phenotype="p.tsv",
        sample_col="sample", trait="t", out_dir=str(tmp_path / "o"), groups="g.tsv")
    validate_interact_config(cfg)


def test_build_fit_config_scan_jobs_uses_streaming():
    cfg = workflow.build_fit_config(
        subgenomes=["A"], phenotype="p", sample_col="s", trait="t",
        bed_template="b/{subgenome}/all", out_dir="o", scan_jobs=16)
    assert cfg["scan"]["n_jobs"] == 16 and cfg["scan"]["mode"] == "stream"
    serial = workflow.build_fit_config(
        subgenomes=["A"], phenotype="p", sample_col="s", trait="t",
        bed_template="b/{subgenome}/all", out_dir="o")
    assert "n_jobs" not in serial["scan"] and serial["scan"]["mode"] == "memory"


def test_audit_and_summarize_results(tmp_path, monkeypatch):
    out = tmp_path / "run"
    (out / "audit").mkdir(parents=True)
    (out / "audit" / "homoeogwas_audit.json").write_text(json.dumps({
        "overall_status": "AUDIT_COMPLETE",
        "records": [{"source": "x.json", "command": "fit", "trait": "t",
                     "status": "AUDIT_COMPLETE", "discovery_count": 0,
                     "lambda_gc": 1.01, "flags": []}]}))
    monkeypatch.setattr(workflow, "run_cli", lambda args, dry_run=False: {
        "command": args, "returncode": 0})
    res = workflow.audit_results(out_dir=str(out))
    assert res["ok"] and res["overall_status"] == "AUDIT_COMPLETE"
    assert res["records"][0]["lambda_gc"] == 1.01
    assert workflow.audit_results(out_dir=str(tmp_path / "absent"))["ok"] is False
    assert workflow.summarize_results(out_dir=str(out), trait="t")["ok"] is False
    (out / "summary_t.json").write_text(json.dumps({
        "trait": "t", "n_analysis": 10, "subgenomes": ["A"],
        "reml": {"pve": {"A": 0.3, "e": 0.7}}, "lambda_gc": {"all": 1.0},
        "outputs": {"sumstats": []}}))
    s = workflow.summarize_results(out_dir=str(out), trait="t")
    assert s["ok"] and s["kind"] == "fit" and s["pve"]["A"] == 0.3
