from __future__ import annotations

import pandas as pd
import pytest
import yaml

from homoeogwas.evidence import (
    EvidenceError,
    assign_evidence_tiers,
    join_candidate_evidence,
    load_evidence_manifest,
)


def _write_manifest(tmp_path, sources):
    path = tmp_path / "evidence.yaml"
    path.write_text(yaml.safe_dump({
        "evidence_version": 1,
        "sources": sources,
    }, sort_keys=False))
    return path


def test_evidence_join_keeps_candidates_missing_from_sources(tmp_path):
    annotation = tmp_path / "annotation.tsv"
    pd.DataFrame({
        "gene": ["g1"], "description": ["transporter"],
    }).to_csv(annotation, sep="\t", index=False)
    manifest = load_evidence_manifest(_write_manifest(tmp_path, [{
        "name": "local_annotation", "kind": "annotation",
        "path": "annotation.tsv", "gene_col": "gene",
        "citation": "local-build-v1",
    }]))

    joined, provenance = join_candidate_evidence(
        pd.DataFrame({"gene_id": ["g1", "g2"]}), manifest)

    assert joined["gene_id"].drop_duplicates().tolist() == ["g1", "g2"]
    assert bool(joined.loc[joined.gene_id == "g1", "matched"].iloc[0])
    assert not bool(joined.loc[joined.gene_id == "g2", "matched"].iloc[0])
    assert len(provenance["sources"][0]["sha256"]) == 64
    assert provenance["sources"][0]["row_count"] == 1


def test_tier_precedence_is_functional_qtl_expression_annotation():
    formal = pd.DataFrame({
        "hypothesis_id": ["h-functional", "h-qtl", "h-expression", "h-annotation", "h-none"],
        "gene_A": ["f", "q", "e", "a", "n"],
        "gene_C": ["f2", "q2", "e2", "a2", "n2"],
    })
    evidence = pd.DataFrame([
        {"gene_id": "f", "source": "knockout", "kind": "functional", "matched": True, "supported": True, "citation": "doi:f"},
        {"gene_id": "q", "source": "qtl", "kind": "qtl", "matched": True, "supported": True, "citation": "doi:q"},
        {"gene_id": "e", "source": "order", "kind": "expression", "matched": True, "supported": True, "citation": "doi:e"},
        {"gene_id": "a", "source": "tair", "kind": "annotation", "matched": True, "supported": False, "citation": "db:a"},
    ])

    tiers = assign_evidence_tiers(formal, evidence).set_index("hypothesis_id")

    assert tiers.loc["h-functional", "evidence_tier"] == "direct functional evidence"
    assert tiers.loc["h-qtl", "evidence_tier"] == "QTL or literature support"
    assert tiers.loc["h-expression", "evidence_tier"] == "matched-tissue expression support"
    assert tiers.loc["h-annotation", "evidence_tier"] == "orthology/domain annotation only"
    assert tiers.loc["h-none", "evidence_tier"] == "no linked external evidence"


def test_support_column_uses_declared_truthy_values(tmp_path):
    expression = tmp_path / "expression.csv"
    pd.DataFrame({
        "gene": ["g1", "g2", "g3"],
        "expressed": ["YES", "0", "supported"],
    }).to_csv(expression, index=False)
    manifest = load_evidence_manifest(_write_manifest(tmp_path, [{
        "name": "expr", "kind": "expression", "path": "expression.csv",
        "gene_col": "gene", "support_col": "expressed",
    }]))

    joined, _ = join_candidate_evidence(
        pd.DataFrame({"gene_id": ["g1", "g2", "g3"]}), manifest)

    assert joined.set_index("gene_id")["supported"].to_dict() == {
        "g1": True, "g2": False, "g3": True}


def test_manifest_rejects_unknown_kind_and_missing_columns(tmp_path):
    table = tmp_path / "table.tsv"
    table.write_text("wrong\nvalue\n")
    with pytest.raises(EvidenceError, match="unsupported kind"):
        load_evidence_manifest(_write_manifest(tmp_path, [{
            "name": "bad", "kind": "mechanism", "path": "table.tsv",
            "gene_col": "gene",
        }]))
    manifest = load_evidence_manifest(_write_manifest(tmp_path, [{
        "name": "bad-column", "kind": "annotation", "path": "table.tsv",
        "gene_col": "gene",
    }]))
    with pytest.raises(EvidenceError, match="gene column"):
        join_candidate_evidence(pd.DataFrame({"gene_id": ["g1"]}), manifest)
