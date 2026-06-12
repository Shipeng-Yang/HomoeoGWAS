"""Tests for the breeder-level workflow engine + MCP server wiring."""
from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from homoeogwas import workflow

HAVE_MCP = importlib.util.find_spec("mcp") is not None


def _touch_bed(prefix):
    for ext in (".bed", ".bim", ".fam"):
        (prefix.parent).mkdir(parents=True, exist_ok=True)
        (prefix.with_suffix(ext)).write_text("x")


def test_infer_interaction_mode():
    assert workflow.infer_interaction_mode(["A", "B"]) == "pairwise"
    assert workflow.infer_interaction_mode(["A", "B", "C"]) == "triad"
    with pytest.raises(ValueError, match="2-/3-subgenome subsets"):
        workflow.infer_interaction_mode(["A", "B", "C", "D"])


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
    assert cfg["interact"]["mode"] == "triad"
    assert cfg["interact"]["triads"] == "tr.tsv"
    with pytest.raises(ValueError, match="triads TSV"):
        workflow.build_interact_config(**common)   # triad without triads
    # pairwise
    pw = workflow.build_interact_config(
        subgenomes=["A", "B"], bed_prefixes={"A": "a", "B": "b"},
        snp_to_gene={"A": "na", "B": "nb"}, phenotype="p", sample_col="IID",
        trait="t", out_dir="o", pairs="pr.tsv")
    assert pw["interact"]["mode"] == "pairwise"
    assert pw["interact"]["pairs"] == "pr.tsv"


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


def test_run_gwas_blocks_integer_sample_ids(tmp_path):
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
    assert res["ok"] is False and res.get("blocked") == "integer_sample_ids"
    # missing PLINK files also block, not guess
    res2 = workflow.run_gwas(
        phenotype=str(pheno), sample_col="IID", trait="yield",
        subgenomes=["A"], bed_prefixes={"A": str(tmp_path / "nope")},
        out_dir=str(tmp_path / "o2"), allow_integer_ids=True, dry_run=True)
    assert res2["ok"] is False and "missing" in res2["reason"]


def test_run_interaction_dry_run(tmp_path):
    res = workflow.run_interaction(
        phenotype="p", sample_col="IID", trait="t", subgenomes=["A", "B", "C"],
        bed_prefixes={"A": "a", "B": "b", "C": "c"},
        snp_to_gene={"A": "na", "B": "nb", "C": "nc"},
        out_dir=str(tmp_path), triads="tr.tsv", dry_run=True)
    assert res["mode"] == "triad"
    assert "interact.generated.triad" in res["config"]


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
    assert {"run_gwas", "run_interaction", "get_guidance"} <= names
