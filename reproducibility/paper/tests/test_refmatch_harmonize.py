"""Unit tests for the FASTA REF-match harmonisation (m3_3_v2_prepare_candidates).

Covers the four resolution classes:
  - direct match (A1_IS_REF / A2_IS_REF)
  - reverse-complement match for strand-flipped array alleles (A*_IS_REF_RC)
  - palindromic A/T or C/G SNPs (resolve via direct priority, flagged)
  - true mismatch (neither strand) -> REF_MISMATCH, dropped from scoring
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pysam
import pytest

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "scripts" / "phase3" / "m3_3_v2_prepare_candidates.py"

_spec = importlib.util.spec_from_file_location("m3_3_prep", _SCRIPT)
m3_3_prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3_3_prep)
harmonize_with_fasta = m3_3_prep.harmonize_with_fasta

# FASTA contig "ctg1" with a known + strand sequence (positions 1-based):
#   1:A 2:C 3:G 4:T 5:A 6:T 7:G 8:C 9:A 10:T
_SEQ = "ACGTATGCAT"


@pytest.fixture()
def fasta(tmp_path):
    p = tmp_path / "tiny.fa"
    p.write_text(">ctg1\n" + _SEQ + "\n")
    pysam.faidx(str(p))
    return p


def _run(fasta_path, rows):
    """rows: list of (snp_id, pos, a1, a2). chrom fixed = ctg1, identity map."""
    cand = pd.DataFrame(
        [{"snp_id": sid, "chrom": "ctg1", "pos": pos} for sid, pos, _, _ in rows]
    )
    allele_map = {sid: ("ctg1", a1, a2, pos) for sid, pos, a1, a2 in rows}
    out = harmonize_with_fasta(cand, fasta_path, allele_map, chrom_map={})
    return out.set_index("snp_id")


def test_direct_match(fasta):
    # pos2 = C; alleles C/T -> A1_IS_REF (ref=C, alt=T)
    out = _run(fasta, [("s", 2, "C", "T")])
    assert out.loc["s", "ref_match_status"] == "A1_IS_REF"
    assert out.loc["s", "ref_fasta"] == "C"
    assert out.loc["s", "alt_fasta"] == "T"
    assert not out.loc["s", "is_palindromic"]


def test_direct_match_a2(fasta):
    # pos3 = G; alleles T/G -> A2_IS_REF (ref=G, alt=T)
    out = _run(fasta, [("s", 3, "T", "G")])
    assert out.loc["s", "ref_match_status"] == "A2_IS_REF"
    assert out.loc["s", "ref_fasta"] == "G"
    assert out.loc["s", "alt_fasta"] == "T"


def test_reverse_complement(fasta):
    # pos3 = G; array alleles C/A on the opposite strand.
    # comp(G)=C == a1 -> A1_IS_REF_RC; + strand ref=G, alt=comp(A)=T
    out = _run(fasta, [("s", 3, "C", "A")])
    assert out.loc["s", "ref_match_status"] == "A1_IS_REF_RC"
    assert out.loc["s", "ref_fasta"] == "G"
    assert out.loc["s", "alt_fasta"] == "T"
    assert not out.loc["s", "is_palindromic"]


def test_reverse_complement_a2(fasta):
    # pos3 = G; alleles A/C -> comp(G)=C == a2 -> A2_IS_REF_RC; alt=comp(A)=T
    out = _run(fasta, [("s", 3, "A", "C")])
    assert out.loc["s", "ref_match_status"] == "A2_IS_REF_RC"
    assert out.loc["s", "ref_fasta"] == "G"
    assert out.loc["s", "alt_fasta"] == "T"


def test_palindromic_resolves_direct(fasta):
    # pos1 = A; palindromic A/T SNP resolves via direct priority, flagged.
    out = _run(fasta, [("s", 1, "A", "T")])
    assert out.loc["s", "ref_match_status"] == "A1_IS_REF"
    assert out.loc["s", "ref_fasta"] == "A"
    assert bool(out.loc["s", "is_palindromic"]) is True


def test_palindromic_cg(fasta):
    # pos3 = G; palindromic C/G, direct a2==G -> A2_IS_REF, flagged palindrome.
    out = _run(fasta, [("s", 3, "C", "G")])
    assert out.loc["s", "ref_match_status"] == "A2_IS_REF"
    assert bool(out.loc["s", "is_palindromic"]) is True


def test_true_mismatch(fasta):
    # pos1 = A; alleles C/G -> direct no; comp(A)=T not in {C,G} -> REF_MISMATCH
    out = _run(fasta, [("s", 1, "C", "G")])
    assert out.loc["s", "ref_match_status"] == "REF_MISMATCH"
    assert out.loc["s", "alt_fasta"] is None


def test_rc_does_not_flip_beta_columns(fasta):
    # Harmonisation must not touch any non-allele columns; carry a beta through.
    cand = pd.DataFrame([{"snp_id": "s", "chrom": "ctg1", "pos": 3, "beta": -1.23}])
    allele_map = {"s": ("ctg1", "C", "A", 3)}  # RC case
    out = harmonize_with_fasta(cand, fasta, allele_map, chrom_map={})
    assert out.loc[0, "ref_match_status"] == "A1_IS_REF_RC"
    assert out.loc[0, "beta"] == -1.23  # untouched (allele-frame harmonisation only)
