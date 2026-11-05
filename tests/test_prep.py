"""Tests for the `interact` input builders (prep-snps / prep-homoeologs)."""
from __future__ import annotations

import shutil
from pathlib import Path

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
    to_bed(str(prefix) + ".bed", dosage,
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
    assert len(str(z["bim_sha256"].item())) == 64
    assert int(z["n_variants"].item()) == 4
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


def test_interact_loader_rejects_npz_from_different_bim(tmp_path):
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    out = tmp_path / "prep"
    sg = prep.load_subgenome_map(str(sgmap))
    prep.build_snp_to_gene(str(gff), bed_by_sub, sg, out_dir=str(out))
    bim = Path(bed_by_sub["A"] + ".bim")
    fields = bim.read_text().splitlines()[0].split()
    fields[1] = "changed_variant_id"
    lines = bim.read_text().splitlines()
    lines[0] = "\t".join(fields)
    bim.write_text("\n".join(lines) + "\n")
    from homoeogwas.interact import _load_subgenome
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _load_subgenome(bed_by_sub["A"], str(out / "snp_to_gene_A.npz"))


def test_interaction_config_and_preflight_accept_csv_and_verified_npz(tmp_path):
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    out = tmp_path / "prep"
    sg = prep.load_subgenome_map(str(sgmap))
    prep.build_snp_to_gene(str(gff), bed_by_sub, sg, out_dir=str(out))
    pairs = tmp_path / "pairs.tsv"
    pd.DataFrame({"gene_A": ["gA1"], "gene_B": ["gB1"]}).to_csv(
        pairs, sep="\t", index=False)
    phenotype = tmp_path / "phenotype.csv"
    pd.DataFrame({
        "sample": [f"s{i}" for i in range(12)],
        "trait": np.arange(12, dtype=float),
    }).to_csv(phenotype, index=False)
    cfg = {
        "interact": {
            "mode": "pairwise", "subgenomes": ["A", "B"],
            "genotype": {s: bed_by_sub[s] for s in ("A", "B")},
            "snp_to_gene": {
                s: str(out / f"snp_to_gene_{s}.npz") for s in ("A", "B")},
            "pairs": str(pairs), "phenotype": str(phenotype),
            "sample_col": "sample", "trait": "trait",
            "burden": {"cap": 150, "min_snp": 1},
        },
        "outputs": {"out_dir": str(tmp_path / "interact")},
    }
    from homoeogwas.interact import preflight_interact, validate_interact_config
    validate_interact_config(cfg)
    assert preflight_interact(cfg) == []


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


def test_prep_homoeologs_group_cli_writes_four_copy_canonical_table(tmp_path):
    subs = ["A", "B", "C", "D"]
    for sub in subs:
        pd.DataFrame({
            "gene_id": [f"g{sub}1", f"g{sub}2"],
            "chrom": [f"chr{sub}", f"chr{sub}"],
            "n_snp": [3, 3],
            "callable": [1, 1],
        }).to_csv(tmp_path / f"genes_{sub}.tsv", sep="\t", index=False)
    table = tmp_path / "groups.tsv"
    pd.DataFrame({
        "group": ["og1", "og2"],
        **{sub: [f"g{sub}1", f"g{sub}2"] for sub in subs},
    }).to_csv(table, sep="\t", index=False)
    output = tmp_path / "quartets.tsv"

    from homoeogwas.cli import main
    assert main([
        "prep-homoeologs", "--mode", "group",
        "--subgenomes", ",".join(subs),
        "--genes", str(tmp_path / "genes_{S}.tsv"),
        "--out", str(output), "--from-table", str(table),
        "--table-format", "wide",
    ]) == 0
    result = pd.read_csv(output, sep="\t", dtype=str)
    assert list(result.columns) == [
        "group_id", "gene_A", "gene_B", "gene_C", "gene_D"]
    assert result["group_id"].tolist() == ["og1", "og2"]


def test_group_diamond_requires_consistent_all_pair_clique(tmp_path, monkeypatch):
    subs = ["A", "B", "C", "D"]
    proteins = {}
    universe = {}
    for sub in subs:
        path = tmp_path / f"{sub}.faa"
        path.write_text(f">g{sub}1\nMA\n>g{sub}2\nMT\n")
        proteins[sub] = str(path)
        universe[sub] = {f"g{sub}1": 3, f"g{sub}2": 3}

    def fake_which(_):
        return "/fake/diamond"

    def fake_rbh(faa_a, faa_b, diamond, threads, tmp, timeout=None):
        a = Path(faa_a).stem
        b = Path(faa_b).stem
        mapping = {f"g{a}1": f"g{b}1", f"g{a}2": f"g{b}2"}
        if (a, b) == ("C", "D"):
            mapping[f"g{a}2"] = f"g{b}1"  # break only the second 4-clique
        return mapping

    monkeypatch.setattr(prep.shutil, "which", fake_which)
    monkeypatch.setattr(prep, "_diamond_rbh", fake_rbh)
    result = prep.homoeologs_diamond(
        proteins, subs, universe, diamond="/fake/diamond", mode="group")
    assert list(result.columns) == [
        "group_id", "gene_A", "gene_B", "gene_C", "gene_D"]
    assert result[[f"gene_{s}" for s in subs]].to_dict("records") == [{
        "gene_A": "gA1", "gene_B": "gB1", "gene_C": "gC1", "gene_D": "gD1"}]


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


def _subst_fixture(tmp_path):
    """A/D fixture engineered to trigger the OLD paralog-substitution bug.

    chrD has a SNP-poor TRUE homoeolog (gD1t, 0 SNPs) and a SNP-rich paralog
    (gD1p, 1 SNP) in the same ortholog group as gA1; and a clean 1:1 group
    (gA2/gD2) whose D copy gD2 is SNP-poor (0 SNPs). With the decoupled
    definition both groups must be DROPPED (recorded), never substituted.
    """
    gff = tmp_path / "g.gff"
    gff.write_text(
        "chrA\ts\tgene\t100\t200\t.\t+\t.\tID=gA1\n"
        "chrA\ts\tgene\t300\t400\t.\t+\t.\tID=gA2\n"
        "chrA\ts\tgene\t500\t600\t.\t+\t.\tID=gA3\n"
        "chrD\ts\tgene\t100\t200\t.\t+\t.\tID=gD1t\n"   # true homoeolog, 0 SNP
        "chrD\ts\tgene\t300\t400\t.\t+\t.\tID=gD1p\n"   # paralog, has SNP
        "chrD\ts\tgene\t500\t600\t.\t+\t.\tID=gD2\n"    # 0 SNP
        "chrD\ts\tgene\t700\t800\t.\t+\t.\tID=gD3\n")
    _write_bed(tmp_path / "sub_A", 12, ["chrA"] * 3, [150, 350, 550])
    _write_bed(tmp_path / "sub_D", 12, ["chrD"] * 2, [350, 750])  # in gD1p, gD3
    sgmap = tmp_path / "sg.tsv"
    pd.DataFrame({"chrom": ["chrA", "chrD"], "subgenome": ["A", "D"]}).to_csv(
        sgmap, sep="\t", index=False)
    out = tmp_path / "prep"
    sg = prep.load_subgenome_map(str(sgmap))
    prep.build_snp_to_gene(str(gff), {"A": str(tmp_path / "sub_A"),
                                      "D": str(tmp_path / "sub_D")}, sg,
                           min_snp=1, out_dir=str(out))
    uni = prep._load_gene_universe(str(out / "genes_{S}.tsv"), ["A", "D"])
    return uni


def test_genes_universe_is_full_not_callable_filtered(tmp_path):
    """genes_<S>.tsv must list EVERY gene (incl. 0-SNP), so homology is defined
    on the full genome — the decoupling that prevents paralog substitution."""
    uni = _subst_fixture(tmp_path)
    # 0-SNP true homoeologs are present in the universe (not filtered out)
    assert uni["D"]["gD1t"] == 0 and uni["D"]["gD2"] == 0
    assert uni["D"]["gD1p"] == 1 and uni["D"]["gD3"] == 1


def test_uncallable_true_homoeolog_dropped_not_substituted(tmp_path):
    uni = _subst_fixture(tmp_path)
    tbl = tmp_path / "ortho.tsv"
    pd.DataFrame({
        "gene": ["gA1", "gD1t", "gD1p", "gA2", "gD2", "gA3", "gD3"],
        "group": ["og1", "og1", "og1", "og2", "og2", "og3", "og3"],
    }).to_csv(tbl, sep="\t", index=False)
    df = prep.homoeologs_from_table(str(tbl), "long", ["A", "D"], uni)
    # og1 is 1:many ambiguous -> dropped; og2's D copy is uncallable -> dropped;
    # the SNP-rich paralog gD1p must NEVER be substituted in.
    assert "gD1p" not in set(df.get("gene_D", []))
    # only the clean, fully-callable og3 survives
    assert set(zip(df["gene_A"], df["gene_D"], strict=True)) == {("gA3", "gD3")}
    audit = df.attrs["audit"].set_index("group")
    assert audit.loc["og1", "status"] == "ambiguous_1tomany"
    assert audit.loc["og2", "status"] == "dropped_callability"
    assert audit.loc["og3", "status"] == "tested"
    assert df.attrs["n_dropped_callability"] == 1
    assert df.attrs["n_dropped_ambiguous"] == 1


def test_gate_follows_npz_callable_not_just_nsnp(tmp_path):
    """If prep-snps --min-snp > 1, a gene with 1 SNP is NOT in the NPZ. The pair
    gate must follow that NPZ membership (callable_map), so prep never reports a
    pair as testable that `interact` would then silently drop for lack of burden.
    """
    gff = tmp_path / "g.gff"
    gff.write_text(
        "chrA\ts\tgene\t100\t200\t.\t+\t.\tID=gA1\n"
        "chrD\ts\tgene\t100\t200\t.\t+\t.\tID=gD1\n")
    _write_bed(tmp_path / "sub_A", 12, ["chrA"], [150])   # gA1: 1 SNP
    _write_bed(tmp_path / "sub_D", 12, ["chrD"], [150])   # gD1: 1 SNP
    sgmap = tmp_path / "sg.tsv"
    pd.DataFrame({"chrom": ["chrA", "chrD"], "subgenome": ["A", "D"]}).to_csv(
        sgmap, sep="\t", index=False)
    out = tmp_path / "prep"
    sg = prep.load_subgenome_map(str(sgmap))
    prep.build_snp_to_gene(str(gff), {"A": str(tmp_path / "sub_A"),
                                      "D": str(tmp_path / "sub_D")}, sg,
                           min_snp=2, out_dir=str(out))   # 1-SNP genes uncallable
    uni = prep._load_gene_universe(str(out / "genes_{S}.tsv"), ["A", "D"])
    cmap = prep._load_callable(str(out / "genes_{S}.tsv"), ["A", "D"])
    assert uni["A"]["gA1"] == 1 and cmap["A"] == set()    # present but uncallable
    tbl = tmp_path / "t.tsv"
    pd.DataFrame({"gene": ["gA1", "gD1"], "group": ["og1", "og1"]}).to_csv(
        tbl, sep="\t", index=False)
    df = prep.homoeologs_from_table(str(tbl), "long", ["A", "D"], uni,
                                    min_snp_pair=1, callable_map=cmap)
    # n_snp>=1 would WRONGLY pass; the callable gate (NPZ membership) drops it
    assert len(df) == 0
    assert df.attrs["audit"].iloc[0]["status"] == "dropped_callability"


def test_cli_prep_snps_dispatch(tmp_path):
    from homoeogwas.cli import main
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    out = tmp_path / "cli_prep"
    rc = main(["prep-snps", "--gff", str(gff), "--subgenome-map", str(sgmap),
               "--bed", f"A={bed_by_sub['A']}", "--bed", f"B={bed_by_sub['B']}",
               "--bed", f"C={bed_by_sub['C']}", "--out-dir", str(out)])
    assert rc == 0
    assert (out / "snp_to_gene_B.npz").exists()


# --- DIAMOND input checks, timeout with process-group kill, zero-pair refusal, version ---

_FAKE_DIAMOND = r'''#!/bin/bash
# fake diamond for tests: makedb stores the input path; blastp maps query gX<n> -> subject gY<n>
cmd=$1; shift
[ "$cmd" = version ] && { echo "diamond version 2.2.0"; exit 0; }
while [ $# -gt 0 ]; do case "$1" in --in) IN=$2; shift;; -d) DB=$2; shift;; -q) Q=$2; shift;; -o) O=$2; shift;; esac; shift; done
if [ "$cmd" = makedb ]; then
  [ -n "$FAKE_HANG" ] && { bash -c "exec -a fakehang_$FAKE_HANG sleep 600" & wait; }
  echo "$IN" > "$DB"; exit 0
fi
if [ "$cmd" = blastp ]; then
  : > "$O"; [ -n "$FAKE_EMPTY" ] && exit 0
  S=$(cat "$DB")
  for q in $(grep '^>' "$Q" | cut -c2- | cut -d' ' -f1); do
    for s in $(grep '^>' "$S" | cut -c2- | cut -d' ' -f1); do
      [ "${q:2}" = "${s:2}" ] && printf '%s\t%s\t100\n' "$q" "$s" >> "$O"
    done
  done
fi
'''


def _fake_diamond(tmp_path):
    p = tmp_path / "diamond"
    p.write_text(_FAKE_DIAMOND)
    p.chmod(0o755)
    return str(p)


def _faas(tmp_path, body=None, header=lambda s, i: f"g{s}{i}"):
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQV"
    out = {}
    for s in SUBS:
        p = tmp_path / f"prot_{s}.faa"
        p.write_text("".join(f">{header(s, i)}\n{body or seq}\n" for i in (1, 2)))
        out[s] = str(p)
    return out


def test_diamond_rbh_records_version_with_fake_binary(tmp_path):
    uni = _genes_universe(tmp_path)
    df = prep.homoeologs_diamond(_faas(tmp_path), SUBS, uni, diamond=_fake_diamond(tmp_path),
                                 threads=1, mode="triad")
    assert ("gA1", "gB1", "gC1") in {tuple(r) for r in df.itertuples(index=False, name=None)}
    assert df.attrs["diamond_version"] == "diamond version 2.2.0"


def test_protein_fasta_with_stop_characters_refused_before_diamond(tmp_path):
    uni = _genes_universe(tmp_path)
    with pytest.raises(SystemExit, match="characters DIAMOND does not accept"):
        prep.homoeologs_diamond(_faas(tmp_path, body="MKTAY*IAKQ.RQ"), SUBS, uni,
                                diamond=_fake_diamond(tmp_path), mode="triad")


def test_protein_fasta_headers_must_match_gene_ids(tmp_path):
    uni = _genes_universe(tmp_path)
    faas = _faas(tmp_path, header=lambda s, i: f"TraesCS{s}{i}.1")
    with pytest.raises(SystemExit, match="match the gene ids"):
        prep.homoeologs_diamond(faas, SUBS, uni, diamond=_fake_diamond(tmp_path), mode="triad")


def test_diamond_timeout_kills_process_group(tmp_path, monkeypatch):
    import subprocess as sp
    uni = _genes_universe(tmp_path)
    token = f"t{tmp_path.name}"
    monkeypatch.setenv("FAKE_HANG", token)
    with pytest.raises(SystemExit, match="timed out after 2 s"):
        prep.homoeologs_diamond(_faas(tmp_path), SUBS, uni, diamond=_fake_diamond(tmp_path),
                                mode="triad", timeout=2)
    left = sp.run(["pgrep", "-f", f"fakehang_{token}"], capture_output=True, text=True)
    assert left.stdout.strip() == ""


def test_zero_reciprocal_pairs_refused(tmp_path, monkeypatch):
    uni = _genes_universe(tmp_path)
    monkeypatch.setenv("FAKE_EMPTY", "1")
    with pytest.raises(SystemExit, match="no reciprocal best hits"):
        prep.homoeologs_diamond(_faas(tmp_path), SUBS, uni, diamond=_fake_diamond(tmp_path),
                                mode="triad")


# --- refuse zero genes / zero SNPs in genes and list the chromosome sets ---

def test_prep_snps_refuses_gff_chromosomes_absent_from_bim(tmp_path):
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    gff.write_text(gff.read_text().replace("chrA\t", "NC_0001.1\t").replace("chrB\t", "NC_0002.1\t")
                   .replace("chrC\t", "NC_0003.1\t"))
    sg = prep.load_subgenome_map(str(sgmap))
    sg["chrom"] = ["NC_0001.1", "NC_0002.1", "NC_0003.1"]               # map follows the GFF names
    out = tmp_path / "prep"
    with pytest.raises(SystemExit, match="no SNP of the .bim lies on a gene chromosome") as e:
        prep.build_snp_to_gene(str(gff), bed_by_sub, sg, out_dir=str(out))
    assert "NC_0001.1" in str(e.value) and "chrA" in str(e.value)
    assert not list(out.glob("snp_to_gene_*.npz"))


def test_prep_snps_refuses_zero_genes_for_a_subgenome(tmp_path):
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    gff.write_text(gff.read_text().replace("chrC\t", "Chr3\t"))       # GFF name not in the map
    sg = prep.load_subgenome_map(str(sgmap))
    out = tmp_path / "prep"
    with pytest.raises(SystemExit, match="0 genes for subgenome C") as e:
        prep.build_snp_to_gene(str(gff), bed_by_sub, sg, out_dir=str(out))
    assert "Chr3" in str(e.value)
    assert not list(out.glob("snp_to_gene_*.npz"))


# --- input guards: edge cases ---

def test_terminal_stop_codon_is_accepted_but_internal_stop_refused(tmp_path):
    uni = _genes_universe(tmp_path)
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQV"
    df = prep.homoeologs_diamond(_faas(tmp_path, body=seq + "*"), SUBS, uni,
                                 diamond=_fake_diamond(tmp_path), mode="triad")
    assert len(df) == 2
    with pytest.raises(SystemExit, match="characters DIAMOND does not accept"):
        prep.homoeologs_diamond(_faas(tmp_path, body="MKTAY*IAKQ*"), SUBS, uni,
                                diamond=_fake_diamond(tmp_path), mode="triad")


def test_run_bounded_kills_group_when_parent_is_interrupted(tmp_path, monkeypatch):
    import subprocess as sp
    token = f"int{tmp_path.name}"
    script = tmp_path / "hang.sh"
    script.write_text(f'#!/bin/bash\nbash -c "exec -a hold_{token} sleep 600" & wait\n')
    script.chmod(0o755)
    real_wait = sp.Popen.wait
    calls = []

    def interrupted(self, timeout=None):     # Ctrl-C arrives during the first (bounded) wait only
        calls.append(timeout)
        if len(calls) == 1:
            raise KeyboardInterrupt
        return real_wait(self, timeout=timeout)
    monkeypatch.setattr(sp.Popen, "wait", interrupted, raising=True)
    with pytest.raises(KeyboardInterrupt):
        prep._run_bounded([str(script)], 30, "makedb")
    monkeypatch.setattr(sp.Popen, "wait", real_wait, raising=True)
    import time
    time.sleep(0.5)
    ps = sp.run(["ps", "-eo", "pid,stat,args"], capture_output=True, text=True).stdout
    left = [line for line in ps.splitlines() if f"hold_{token}" in line and "Z" not in line.split()[1]]
    assert left == []


def test_prep_snps_refuses_zero_snps_inside_genes(tmp_path):
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    # same chromosome names, but genes moved far away from every SNP
    gff.write_text(gff.read_text().replace("\t100\t200\t", "\t100000\t100100\t").replace("\t300\t400\t", "\t200000\t200100\t"))
    sg = prep.load_subgenome_map(str(sgmap))
    out = tmp_path / "prep"
    with pytest.raises(SystemExit, match="0 SNPs inside genes"):
        prep.build_snp_to_gene(str(gff), bed_by_sub, sg, out_dir=str(out))
    assert not list(out.glob("snp_to_gene_*.npz")) and not list(out.glob("genes_*.tsv"))
