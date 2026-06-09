"""Build the two input files that ``homoeogwas interact`` needs.

``interact`` tests homoeolog-pair / triad burden-product interactions, but it
consumes two preprocessed files that nothing else in the package produced, so a
user running on their own polyploid had no supported way to make them. These
two subcommands close that gap:

* ``homoeogwas prep-snps`` — from a GFF + per-subgenome PLINK BEDs + a
  subgenome map, write one ``snp_to_gene_<S>.npz`` per subgenome (the
  ``gene_ids`` + ``snp_idx`` arrays the interact engine loads) plus a readable
  ``genes_<S>.tsv``. Pure Python, deterministic, no external tools.
* ``homoeogwas prep-homoeologs`` — assemble the ``gene_<S>`` triad/pair TSV that
  pairs homoeologous genes across subgenomes, either by re-using a user's own
  orthology table (``--from-table``, the publication-grade path) or by a
  convenience DIAMOND reciprocal-best-hit run (``--method diamond-rbh``).

The single shared description of "which chromosome belongs to which subgenome"
is one **subgenome-map TSV** (columns ``chrom``, ``subgenome`` and optional
``base_group``); both subcommands read it, and ``genes_<S>.tsv`` from
``prep-snps`` is the authoritative gene-id universe that ``prep-homoeologs``
validates against. See ``docs/interact_inputs.md`` for the full format spec.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# shared: subgenome map + GFF + bim
# ----------------------------------------------------------------------


def load_subgenome_map(path: str, chrom_col: str = "chrom",
                       subgenome_col: str = "subgenome",
                       base_group_col: str | None = None) -> pd.DataFrame:
    """Load the subgenome-map TSV → frame with columns chrom/subgenome[/base_group].

    The ``chrom`` value must be the chromosome name exactly as it appears in
    BOTH the GFF and the PLINK ``.bim`` (rename beforehand if they differ).
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    for col in (chrom_col, subgenome_col):
        if col not in df.columns:
            raise SystemExit(
                f"ERR: subgenome map {path} missing column {col!r}; "
                f"has {list(df.columns)}")
    out = pd.DataFrame({"chrom": df[chrom_col].astype(str),
                        "subgenome": df[subgenome_col].astype(str)})
    if base_group_col:
        if base_group_col not in df.columns:
            raise SystemExit(f"ERR: subgenome map missing base-group column "
                             f"{base_group_col!r}; has {list(df.columns)}")
        out["base_group"] = df[base_group_col].astype(str)
    if out["chrom"].duplicated().any():
        dup = out.loc[out["chrom"].duplicated(), "chrom"].tolist()[:5]
        raise SystemExit(f"ERR: subgenome map has duplicate chrom rows: {dup}")
    return out


def _parse_attr(attrs: str, key: str) -> str | None:
    """Pull ``key`` out of a GFF column-9 attribute string (GFF3 or GTF style)."""
    for sep in (";",):
        for field in attrs.split(sep):
            field = field.strip()
            if not field:
                continue
            if "=" in field:                       # GFF3: ID=geneX
                k, _, v = field.partition("=")
                if k.strip() == key:
                    return v.strip().strip('"')
            else:                                   # GTF: gene_id "geneX"
                parts = field.split(None, 1)
                if len(parts) == 2 and parts[0] == key:
                    return parts[1].strip().strip('"')
    return None


def parse_gff_genes(gff_path: str, feature: str = "gene", id_attr: str = "ID",
                    keep_chroms: set[str] | None = None) -> list[dict]:
    """Stream a GFF/GTF → list of gene dicts {chrom,start,end,strand,gene_id}.

    Only ``feature`` rows on ``keep_chroms`` (if given) are returned. Genes
    without the id attribute are skipped (counted by the caller via length).
    """
    genes: list[dict] = []
    op = open
    if str(gff_path).endswith(".gz"):
        import gzip
        op = gzip.open
    with op(gff_path, "rt") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != feature:
                continue
            chrom = f[0]
            if keep_chroms is not None and chrom not in keep_chroms:
                continue
            gid = _parse_attr(f[8], id_attr)
            if gid is None:
                continue
            genes.append({"chrom": chrom, "start": int(f[3]), "end": int(f[4]),
                          "strand": f[6], "gene_id": gid})
    return genes


def _read_bim(bed_prefix: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (chrom, pos) arrays in ``.bim`` row order = BED column order."""
    bim = Path(str(bed_prefix) + ".bim")
    if not bim.exists():
        raise SystemExit(f"ERR: missing {bim}")
    df = pd.read_csv(bim, sep=r"\s+", header=None,
                     usecols=[0, 3], names=["chrom", "pos"], dtype={0: str})
    return df["chrom"].to_numpy(), df["pos"].to_numpy(dtype=np.int64)


# ----------------------------------------------------------------------
# prep-snps
# ----------------------------------------------------------------------


def build_snp_to_gene(gff: str, bed_by_sub: dict[str, str],
                      sgmap: pd.DataFrame, *, feature: str = "gene",
                      id_attr: str = "ID", flank_bp: int = 0,
                      min_snp: int = 1, out_dir: str) -> dict:
    """Write ``snp_to_gene_<S>.npz`` + ``genes_<S>.tsv`` for each subgenome.

    Each SNP is assigned, in ``.bim`` row order, to every gene whose body
    (± ``flank_bp``) contains it; a SNP in no gene is dropped from the NPZ but
    counted. ``snp_idx[i]`` holds the 0-based ``.bim`` row indices for gene i,
    which equal the BED dosage-column indices the interact engine indexes into.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chrom_to_sub = dict(zip(sgmap["chrom"], sgmap["subgenome"], strict=False))
    genes_all = parse_gff_genes(gff, feature=feature, id_attr=id_attr,
                                keep_chroms=set(chrom_to_sub))
    # gene ids must be globally unique (interact keys burdens by gene id; a
    # duplicate would silently merge SNPs from distinct loci / subgenomes)
    from collections import Counter
    dup = [gid for gid, n in Counter(g["gene_id"] for g in genes_all).items()
           if n > 1]
    if dup:
        raise SystemExit(
            f"ERR: GFF has {len(dup)} duplicate gene ids under --id-attr "
            f"{id_attr!r} (e.g. {dup[:5]}); gene ids must be unique")
    # group genes by subgenome via their chrom
    by_sub: dict[str, list[dict]] = defaultdict(list)
    for g in genes_all:
        by_sub[chrom_to_sub[g["chrom"]]].append(g)

    summary: dict = {"subgenomes": {}}
    for sub, prefix in bed_by_sub.items():
        bim_chrom, bim_pos = _read_bim(prefix)
        # per-chrom sorted gene intervals for this subgenome
        per_chrom: dict[str, list[dict]] = defaultdict(list)
        for g in by_sub.get(sub, []):
            per_chrom[g["chrom"]].append(g)
        for c in per_chrom:
            per_chrom[c].sort(key=lambda g: (g["start"], g["end"], g["gene_id"]))
        # assign each SNP (bim row order) to overlapping genes
        gene_snp: dict[str, list[int]] = defaultdict(list)
        n_assigned = 0
        # bucket SNP rows by chrom to scan once per chrom
        rows_by_chrom: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(bim_chrom):
            rows_by_chrom[c].append(i)
        for c, rows in rows_by_chrom.items():
            glist = per_chrom.get(c)
            if not glist:
                continue
            starts = [g["start"] - flank_bp for g in glist]
            ends = [g["end"] + flank_bp for g in glist]
            order_end = sorted(range(len(glist)), key=lambda k: ends[k])
            ends_sorted = [ends[k] for k in order_end]
            for i in rows:
                p = int(bim_pos[i])
                # candidate genes: start<=p and end>=p (flanked)
                hit = False
                lo = bisect_left(ends_sorted, p)   # genes with end>=p
                for k in order_end[lo:]:
                    if starts[k] <= p <= ends[k]:
                        gene_snp[glist[k]["gene_id"]].append(i)
                        hit = True
                if hit:
                    n_assigned += 1
        # order genes deterministically (chrom, start, end, id) and apply min_snp
        gmeta = {g["gene_id"]: g for g in by_sub.get(sub, [])}
        kept = [gid for gid in
                sorted(gene_snp, key=lambda gid: (gmeta[gid]["chrom"],
                                                  gmeta[gid]["start"],
                                                  gmeta[gid]["end"], gid))
                if len(gene_snp[gid]) >= min_snp]
        gene_ids = np.array(kept, dtype=object)
        snp_idx = np.empty(len(kept), dtype=object)
        for j, gid in enumerate(kept):
            snp_idx[j] = np.asarray(gene_snp[gid], dtype=np.int64)
        npz_path = out / f"snp_to_gene_{sub}.npz"
        np.savez(npz_path, gene_ids=gene_ids, snp_idx=snp_idx)
        # readable gene table
        rows_tsv = [{"gene_id": gid, "subgenome": sub,
                     "chrom": gmeta[gid]["chrom"], "start": gmeta[gid]["start"],
                     "end": gmeta[gid]["end"], "strand": gmeta[gid]["strand"],
                     "n_snp": int(len(gene_snp[gid]))} for gid in kept]
        genes_tsv = out / f"genes_{sub}.tsv"
        pd.DataFrame(rows_tsv, columns=["gene_id", "subgenome", "chrom",
                                        "start", "end", "strand",
                                        "n_snp"]).to_csv(
            genes_tsv, sep="\t", index=False)
        summary["subgenomes"][sub] = {
            "n_snp_bim": int(bim_chrom.size), "n_snp_in_genes": int(n_assigned),
            "n_genes_total": int(len(by_sub.get(sub, []))),
            "n_genes_with_snp": int(len(kept)),
            "npz": str(npz_path), "genes_tsv": str(genes_tsv)}
    return summary


# ----------------------------------------------------------------------
# prep-homoeologs
# ----------------------------------------------------------------------


def _load_gene_universe(genes_template: str, subs: list[str]) -> dict[str, dict]:
    """Read genes_<S>.tsv files → {subgenome: {gene_id: n_snp}} (authoritative)."""
    uni: dict[str, dict] = {}
    for s in subs:
        path = genes_template.replace("{S}", s)
        df = pd.read_csv(path, sep="\t", dtype={"gene_id": str})
        uni[s] = dict(zip(df["gene_id"], df.get("n_snp", pd.Series([1] * len(df))), strict=False))
    return uni


def _gene_to_sub(uni: dict[str, dict]) -> dict[str, str]:
    g2s: dict[str, str] = {}
    for s, genes in uni.items():
        for g in genes:
            g2s[g] = s
    return g2s


def homoeologs_from_table(table: str, table_format: str, subs: list[str],
                          uni: dict[str, dict], *, gene_col: str = "gene",
                          group_col: str = "group",
                          drop_missing: bool = False) -> pd.DataFrame:
    """Assemble gene_<S> rows from a user orthology table.

    ``long``: two columns (gene, group). ``wide``: a group column plus one
    column per subgenome holding comma-separated gene lists. Only genes present
    in the ``genes_<S>.tsv`` universe are emitted (so the output gene ids always
    match the NPZ); other table ids are ignored — normal, since the universe is
    only the SNP-carrying genes. The ignored count is reported; with
    ``drop_missing=False`` (default) it is printed as a heads-up.
    """
    g2s = _gene_to_sub(uni)
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    df = pd.read_csv(table, sep="\t", dtype=str)
    n_ignored = 0
    if table_format == "long":
        for col in (gene_col, group_col):
            if col not in df.columns:
                raise SystemExit(f"ERR: --from-table long needs columns "
                                 f"{gene_col!r},{group_col!r}; has {list(df.columns)}")
        for gene, grp in zip(df[gene_col], df[group_col], strict=False):
            gene = str(gene)
            if gene in g2s:                  # only genes in the SNP universe map
                groups[str(grp)][g2s[gene]].append(gene)
            else:
                n_ignored += 1
    elif table_format == "wide":
        for _, row in df.iterrows():
            grp = str(row[group_col]) if group_col in df.columns else str(_)
            for s in subs:
                if s in df.columns and pd.notna(row[s]):
                    for gene in str(row[s]).split(","):
                        gene = gene.strip()
                        if not gene:
                            continue
                        if gene in uni.get(s, {}):
                            groups[grp][s].append(gene)
                        else:
                            n_ignored += 1
    else:
        raise SystemExit(f"ERR: unknown --table-format {table_format!r} "
                         "(use long|wide)")
    if n_ignored and not drop_missing:
        print(f"  note: {n_ignored} table gene id(s) not in any genes_<S>.tsv "
              "(ignored; pass --drop-missing to silence)")
    return _assemble_rows(groups, subs, uni, drop_missing=drop_missing)


def _assemble_rows(groups: dict, subs: list[str], uni: dict[str, dict], *,
                   drop_missing: bool) -> pd.DataFrame:
    """One row per group with exactly one gene per subgenome (most SNPs wins)."""
    rows, ambiguous = [], []
    for grp, persub in groups.items():
        chosen: dict[str, str] = {}
        ok = True
        for s in subs:
            cands = [g for g in persub.get(s, []) if g in uni[s]]
            if not cands:
                ok = False
                break
            # pick the gene with the most callable SNPs (deterministic tiebreak)
            cands.sort(key=lambda g: (-int(uni[s].get(g, 0)), g))
            chosen[s] = cands[0]
            if len(cands) > 1:
                ambiguous.append(grp)
        if ok:
            rows.append({f"gene_{s}": chosen[s] for s in subs})
    out = pd.DataFrame(rows, columns=[f"gene_{s}" for s in subs])
    out.attrs["n_ambiguous_groups"] = len(set(ambiguous))
    return out


def _diamond_rbh(faa_a: str, faa_b: str, diamond: str, threads: int,
                 tmp: Path) -> dict[str, str]:
    """Reciprocal best hits A↔B by bitscore → {gene_a: gene_b} (mutual only)."""
    def _run(db_faa, q_faa, tag):
        db = tmp / f"{tag}.dmnd"
        subprocess.run([diamond, "makedb", "--in", db_faa, "-d", str(db),
                        "--quiet"], check=True)
        out = tmp / f"{tag}.m8"
        subprocess.run([diamond, "blastp", "-q", q_faa, "-d", str(db),
                        "-o", str(out), "-p", str(threads), "--quiet",
                        "-k", "1", "--outfmt", "6", "qseqid", "sseqid",
                        "bitscore"], check=True)
        best: dict[str, tuple[str, float]] = {}
        with open(out) as fh:
            for line in fh:
                q, s, bs = line.rstrip("\n").split("\t")[:3]
                bs = float(bs)
                if q not in best or bs > best[q][1]:
                    best[q] = (s, bs)
        return {q: v[0] for q, v in best.items()}
    a2b = _run(faa_b, faa_a, "b")   # query A vs db B
    b2a = _run(faa_a, faa_b, "a")   # query B vs db A
    return {a: b for a, b in a2b.items() if b2a.get(b) == a}


def homoeologs_diamond(proteins: dict[str, str], subs: list[str],
                       uni: dict[str, dict], *, diamond: str | None = None,
                       threads: int = 8, mode: str = "triad",
                       gene_base_group: dict[str, str] | None = None
                       ) -> pd.DataFrame:
    """Build triad/pair rows from DIAMOND reciprocal best hits across subgenomes.

    Pairwise = the RBH set of the two subgenomes. Triad = genes linked by RBH
    across all three subgenome pairs consistently (A↔B, B↔C, A↔C agree). When
    ``gene_base_group`` is given (gene_id -> base chromosome group), only pairs
    whose genes share a base group are kept — this stops an autopolyploid's
    paralogs on non-homologous chromosomes from being mistaken for homoeologs.
    """
    diamond = diamond or shutil.which("diamond")
    if not diamond or not shutil.which(diamond):
        raise SystemExit("ERR: diamond not found; install it or use --from-table")
    g2s = _gene_to_sub(uni)
    bg = gene_base_group or {}
    _NA = {None, "", "nan", "NaN", "NA", "na"}

    def _same_group(*genes) -> bool:
        if not bg:
            return True
        groups = {bg.get(g) for g in genes}
        # reject if any gene lacks a real base group, or the groups disagree
        return len(groups) == 1 and not (groups & _NA)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if mode == "pairwise":
            sx, sy = subs
            rbh = _diamond_rbh(proteins[sx], proteins[sy], diamond, threads, tmp)
            rows = [{f"gene_{sx}": a, f"gene_{sy}": b} for a, b in rbh.items()
                    if a in uni[sx] and b in uni[sy] and g2s.get(a) == sx
                    and g2s.get(b) == sy and _same_group(a, b)]
            return pd.DataFrame(rows, columns=[f"gene_{sx}", f"gene_{sy}"])
        a, b, c = subs
        ab = _diamond_rbh(proteins[a], proteins[b], diamond, threads, tmp)
        bc = _diamond_rbh(proteins[b], proteins[c], diamond, threads, tmp)
        ac = _diamond_rbh(proteins[a], proteins[c], diamond, threads, tmp)
        rows = []
        for ga, gb in ab.items():
            gc = bc.get(gb)
            if gc is not None and ac.get(ga) == gc:        # consistent triangle
                if (ga in uni[a] and gb in uni[b] and gc in uni[c]
                        and _same_group(ga, gb, gc)):
                    rows.append({f"gene_{a}": ga, f"gene_{b}": gb,
                                 f"gene_{c}": gc})
        return pd.DataFrame(rows, columns=[f"gene_{a}", f"gene_{b}",
                                           f"gene_{c}"])


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _parse_kv(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"ERR: expected SUB=VALUE, got {it!r}")
        k, _, v = it.partition("=")
        out[k] = v
    return out


def add_prep_subparsers(sub) -> None:
    ps = sub.add_parser("prep-snps", help="build snp_to_gene NPZ + genes TSV "
                                          "for `interact` from a GFF + BEDs")
    ps.add_argument("--gff", required=True)
    ps.add_argument("--subgenome-map", required=True,
                    help="TSV mapping chrom -> subgenome (+ optional base_group)")
    ps.add_argument("--bed", action="append", default=[], required=True,
                    metavar="SUB=PREFIX", help="per-subgenome PLINK prefix; repeat")
    ps.add_argument("--chrom-col", default="chrom")
    ps.add_argument("--subgenome-col", default="subgenome")
    ps.add_argument("--base-group-col", default=None)
    ps.add_argument("--feature", default="gene")
    ps.add_argument("--id-attr", default="ID")
    ps.add_argument("--flank-bp", type=int, default=0)
    ps.add_argument("--min-snp", type=int, default=1)
    ps.add_argument("--out-dir", required=True)

    ph = sub.add_parser("prep-homoeologs", help="assemble the gene_<S> triad/"
                                                "pair TSV for `interact`")
    ph.add_argument("--mode", choices=["triad", "pairwise"], default="triad")
    ph.add_argument("--subgenomes", required=True, help="comma list, e.g. A,B,C")
    ph.add_argument("--genes", required=True,
                    help="genes_{S}.tsv template from prep-snps (use {S})")
    ph.add_argument("--out", required=True)
    ph.add_argument("--from-table", default=None,
                    help="user orthology table (preferred)")
    ph.add_argument("--table-format", choices=["long", "wide"], default="long")
    ph.add_argument("--gene-col", default="gene")
    ph.add_argument("--group-col", default="group")
    ph.add_argument("--method", choices=["diamond-rbh"], default=None,
                    help="compute homoeologs via DIAMOND RBH (needs --proteins)")
    ph.add_argument("--proteins", action="append", default=[],
                    metavar="SUB=FAA", help="per-subgenome protein FASTA; repeat")
    ph.add_argument("--diamond", default=None, help="path to diamond binary")
    ph.add_argument("--threads", type=int, default=8)
    ph.add_argument("--restrict-base-group", action="store_true",
                    help="only pair genes sharing a base chromosome group "
                         "(diamond-rbh; needs --subgenome-map with base_group)")
    ph.add_argument("--subgenome-map", default=None,
                    help="subgenome map TSV (for --restrict-base-group)")
    ph.add_argument("--base-group-col", default="base_group")
    ph.add_argument("--drop-missing", action="store_true",
                    help="silence the heads-up about table gene ids not present "
                         "in genes_<S>.tsv (they are ignored either way)")


def cmd_prep_snps(args) -> int:
    sgmap = load_subgenome_map(args.subgenome_map, chrom_col=args.chrom_col,
                               subgenome_col=args.subgenome_col,
                               base_group_col=args.base_group_col)
    bed_by_sub = _parse_kv(args.bed)
    print(f"=== homoeogwas prep-snps — subgenomes={list(bed_by_sub)} ===",
          flush=True)
    summ = build_snp_to_gene(args.gff, bed_by_sub, sgmap, feature=args.feature,
                             id_attr=args.id_attr, flank_bp=args.flank_bp,
                             min_snp=args.min_snp, out_dir=args.out_dir)
    for s, d in summ["subgenomes"].items():
        print(f"  [{s}] {d['n_genes_with_snp']}/{d['n_genes_total']} genes with "
              f"SNPs; {d['n_snp_in_genes']}/{d['n_snp_bim']} SNPs in genes")
        print(f"       wrote {d['npz']}")
        print(f"       wrote {d['genes_tsv']}")
    return 0


def cmd_prep_homoeologs(args) -> int:
    subs = [s.strip() for s in args.subgenomes.split(",") if s.strip()]
    if args.mode == "triad" and len(subs) != 3:
        raise SystemExit("ERR: --mode triad needs 3 subgenomes")
    if args.mode == "pairwise" and len(subs) != 2:
        raise SystemExit("ERR: --mode pairwise needs 2 subgenomes")
    uni = _load_gene_universe(args.genes, subs)
    print(f"=== homoeogwas prep-homoeologs (mode={args.mode}) "
          f"subgenomes={subs} ===", flush=True)
    if args.from_table:
        df = homoeologs_from_table(args.from_table, args.table_format, subs, uni,
                                   gene_col=args.gene_col,
                                   group_col=args.group_col,
                                   drop_missing=args.drop_missing)
        src = f"table:{args.table_format}"
    elif args.method == "diamond-rbh":
        proteins = _parse_kv(args.proteins)
        missing = [s for s in subs if s not in proteins]
        if missing:
            raise SystemExit(f"ERR: --proteins missing for {missing}")
        gene_bg = None
        if args.restrict_base_group:
            if not args.subgenome_map:
                raise SystemExit("ERR: --restrict-base-group needs --subgenome-map")
            sg = load_subgenome_map(args.subgenome_map,
                                    base_group_col=args.base_group_col)
            chrom_bg = dict(zip(sg["chrom"], sg["base_group"], strict=False))
            gene_bg = {}
            for s in subs:
                gt = pd.read_csv(args.genes.replace("{S}", s), sep="\t",
                                 dtype={"gene_id": str})
                for gid, c in zip(gt["gene_id"], gt["chrom"], strict=False):
                    gene_bg[gid] = chrom_bg.get(str(c))
        df = homoeologs_diamond(proteins, subs, uni, diamond=args.diamond,
                                threads=args.threads, mode=args.mode,
                                gene_base_group=gene_bg)
        src = "diamond-rbh" + ("+base_group" if gene_bg else "")
    else:
        raise SystemExit("ERR: provide --from-table or --method diamond-rbh")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    n_amb = df.attrs.get("n_ambiguous_groups", 0)
    print(f"  [{src}] wrote {len(df)} {args.mode} groups -> {args.out}"
          + (f"  ({n_amb} groups had >1 candidate, kept most-SNP gene)"
             if n_amb else ""))
    return 0
