"""Tests for the `interact` input builders (prep-snps / prep-homoeologs)."""
from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from homoeogwas import prep

SUBS = ["A", "B", "C"]


def _write_bed(prefix, n_samples, chrom, pos):
    from bed_reader import to_bed
    rng = np.random.default_rng(0)
    m = len(pos)
    dosage = rng.integers(0, 3, size=(n_samples, m)).astype(np.float32)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    to_bed(str(prefix.with_suffix(".bed")), dosage,
           properties={
               "fid": ["0"] * n_samples,
               "iid": [f"s{i}" for i in range(n_samples)],
               "sid": [f"{chrom[j]}:{pos[j]}" for j in range(m)],
               "chromosome": [str(c) for c in chrom],
               "bp_position": [int(p) for p in pos],
               "allele_1": ["A"] * m, "allele_2": ["B"] * m,
           }, count_A1=True)


def _fixture(tmp_path):
    """Synthetic 3-subgenome dataset: one chrom each, 2 genes, 4 SNPs."""
    # genes: g<S>1 at 100-200, g<S>2 at 300-400 on chrom chr<S>
    gff = tmp_path / "genes.gff"
    lines = []
    for s in SUBS:
        c = f"chr{s}"
        lines.append(f"{c}\tsrc\tgene\t100\t200\t.\t+\t.\tID=g{s}1")
        lines.append(f"{c}\tsrc\tgene\t300\t400\t.\t-\t.\tID=g{s}2")
    gff.write_text("\n".join(lines) + "\n")
    # SNPs per subgenome: 150 (in g1), 205 (5bp past g1 end), 350 (in g2), 900 (intergenic)
    pos = [150, 205, 350, 900]
    for s in SUBS:
        _write_bed(tmp_path / f"sub_{s}", 12, [f"chr{s}"] * 4, pos)
    sgmap = tmp_path / "sgmap.tsv"
    pd.DataFrame({"chrom": [f"chr{s}" for s in SUBS], "subgenome": SUBS,
                  "base_group": ["bc1"] * 3}).to_csv(sgmap, sep="\t", index=False)
    bed_by_sub = {s: str(tmp_path / f"sub_{s}") for s in SUBS}
    return gff, sgmap, bed_by_sub


def test_prep_snps_assigns_0based_bim_indices(tmp_path):
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    out = tmp_path / "prep"
    sg = prep.load_subgenome_map(str(sgmap))
    summ = prep.build_snp_to_gene(str(gff), bed_by_sub, sg, flank_bp=0,
                                  min_snp=1, out_dir=str(out))
    z = np.load(out / "snp_to_gene_A.npz", allow_pickle=True)
    gene_snp = {g: list(z["snp_idx"][i]) for i, g in enumerate(z["gene_ids"])}
    # gA1 gets SNP row 0 (pos 150); gA2 gets SNP row 2 (pos 350); 205/900 dropped
    assert gene_snp == {"gA1": [0], "gA2": [2]}
    assert summ["subgenomes"]["A"]["n_snp_in_genes"] == 2
    assert (out / "genes_A.tsv").exists()


def test_prep_snps_flank_pulls_in_nearby_snp(tmp_path):
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    out = tmp_path / "prep"
    sg = prep.load_subgenome_map(str(sgmap))
    prep.build_snp_to_gene(str(gff), bed_by_sub, sg, flank_bp=10, min_snp=1,
                           out_dir=str(out))
    z = np.load(out / "snp_to_gene_A.npz", allow_pickle=True)
    gene_snp = {g: list(z["snp_idx"][i]) for i, g in enumerate(z["gene_ids"])}
    assert gene_snp["gA1"] == [0, 1]   # pos 205 now within 200+10 flank


def test_npz_roundtrips_through_interact_loader(tmp_path):
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    out = tmp_path / "prep"
    sg = prep.load_subgenome_map(str(sgmap))
    prep.build_snp_to_gene(str(gff), bed_by_sub, sg, out_dir=str(out))
    from homoeogwas.interact import _load_subgenome
    sd = _load_subgenome(bed_by_sub["A"], str(out / "snp_to_gene_A.npz"))
    assert set(sd.gene_snp) == {"gA1", "gA2"}
    assert list(sd.gene_snp["gA2"]) == [2]
    # indices must address real dosage columns
    assert sd.X.shape[1] == 4 and sd.X[:, 2].shape[0] == sd.X.shape[0]


def _genes_universe(tmp_path):
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    out = tmp_path / "prep"
    sg = prep.load_subgenome_map(str(sgmap))
    prep.build_snp_to_gene(str(gff), bed_by_sub, sg, out_dir=str(out))
    return prep._load_gene_universe(str(out / "genes_{S}.tsv"), SUBS)


def test_homoeologs_from_table_long(tmp_path):
    uni = _genes_universe(tmp_path)
    tbl = tmp_path / "ortho.tsv"
    pd.DataFrame({
        "gene": ["gA1", "gB1", "gC1", "gA2", "gB2", "gC2", "junkX"],
        "group": ["og1", "og1", "og1", "og2", "og2", "og2", "og9"],
    }).to_csv(tbl, sep="\t", index=False)
    df = prep.homoeologs_from_table(str(tbl), "long", SUBS, uni)
    assert list(df.columns) == ["gene_A", "gene_B", "gene_C"]
    assert len(df) == 2
    row = df[df["gene_A"] == "gA1"].iloc[0]
    assert (row["gene_B"], row["gene_C"]) == ("gB1", "gC1")


def test_homoeologs_from_table_wide(tmp_path):
    uni = _genes_universe(tmp_path)
    tbl = tmp_path / "wide.tsv"
    pd.DataFrame({"group": ["og1", "og2"], "A": ["gA1", "gA2"],
                  "B": ["gB1", "gB2"], "C": ["gC1", "gC2"]}).to_csv(
        tbl, sep="\t", index=False)
    df = prep.homoeologs_from_table(str(tbl), "wide", SUBS, uni)
    assert len(df) == 2 and set(df["gene_C"]) == {"gC1", "gC2"}


def test_pairs_table_compatible_with_interact_loader(tmp_path):
    uni = _genes_universe(tmp_path)
    tbl = tmp_path / "ortho.tsv"
    pd.DataFrame({"gene": ["gA1", "gB1", "gC1"], "group": ["og1"] * 3}).to_csv(
        tbl, sep="\t", index=False)
    df = prep.homoeologs_from_table(str(tbl), "long", SUBS, uni)
    out = tmp_path / "triads.tsv"
    df.to_csv(out, sep="\t", index=False)
    from homoeogwas.interact import _load_pairs
    triads = _load_pairs(str(out), SUBS)
    assert triads == [("gA1", "gB1", "gC1")]


@pytest.mark.skipif(shutil.which("diamond") is None,
                    reason="diamond not installed")
def test_homoeologs_diamond_rbh(tmp_path):
    uni = _genes_universe(tmp_path)
    # g1 family shares one sequence; g2 family another; families are dissimilar
    seq1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    seq2 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFELRRQACGRSDPTTNGNGWDSYKNVWNTGPCKACGVPYDESS"
    faas = {}
    for s in SUBS:
        p = tmp_path / f"prot_{s}.faa"
        p.write_text(f">g{s}1\n{seq1}\n>g{s}2\n{seq2}\n")
        faas[s] = str(p)
    df = prep.homoeologs_diamond(faas, SUBS, uni, threads=2, mode="triad")
    assert list(df.columns) == ["gene_A", "gene_B", "gene_C"]
    triset = {tuple(r) for r in df.itertuples(index=False, name=None)}
    assert ("gA1", "gB1", "gC1") in triset
    assert ("gA2", "gB2", "gC2") in triset


@pytest.mark.skipif(shutil.which("diamond") is None,
                    reason="diamond not installed")
def test_homoeologs_diamond_base_group_filter(tmp_path):
    uni = _genes_universe(tmp_path)
    seq1 = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    seq2 = "MGSSHHHHHHSSGLVPRGSHMASMTGGQQMGRGSEFELRRQACGRSDPTTNGNGWDSYKNVWNTGPCKACGVPYDESS"
    faas = {}
    for s in SUBS:
        p = tmp_path / f"prot_{s}.faa"
        p.write_text(f">g{s}1\n{seq1}\n>g{s}2\n{seq2}\n")
        faas[s] = str(p)
    # put g1 family in base group bg1 and g2 family in bg2 -> all triads share group
    bg = {f"g{s}1": "bg1" for s in SUBS} | {f"g{s}2": "bg2" for s in SUBS}
    df = prep.homoeologs_diamond(faas, SUBS, uni, threads=2, mode="triad",
                                 gene_base_group=bg)
    assert len(df) == 2
    # now break the group of gC1 -> the (A1,B1,C1) triad must be dropped
    bg2 = dict(bg, gC1="other")
    df2 = prep.homoeologs_diamond(faas, SUBS, uni, threads=2, mode="triad",
                                  gene_base_group=bg2)
    assert ("gA1", "gB1", "gC1") not in {tuple(r) for r in
                                         df2.itertuples(index=False, name=None)}


def test_diamond_missing_errors(tmp_path):
    uni = _genes_universe(tmp_path)
    with pytest.raises(SystemExit):
        prep.homoeologs_diamond({s: "x.faa" for s in SUBS}, SUBS, uni,
                                diamond="/no/such/diamond")


def test_cli_prep_snps_dispatch(tmp_path):
    from homoeogwas.cli import main
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    out = tmp_path / "cli_prep"
    rc = main(["prep-snps", "--gff", str(gff), "--subgenome-map", str(sgmap),
               "--bed", f"A={bed_by_sub['A']}", "--bed", f"B={bed_by_sub['B']}",
               "--bed", f"C={bed_by_sub['C']}", "--out-dir", str(out)])
    assert rc == 0
    assert (out / "snp_to_gene_B.npz").exists()
