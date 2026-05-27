"""Generalized per-subgenome split for any allopolyploid species (Phase 5a Step 2).

Drives ``bcftools`` (chrom subset + concat) and ``plink2`` (BED/pgen
import + QC) from a ``configs/species/<species>.yaml``. The intent is
that adding a new species (e.g. strawberry 8n, peanut 4n, Jerusalem
artichoke 6n) requires writing a YAML — never editing this module.

Pipeline per subgenome (j ∈ {A, B, D, …}):

    layout=single:
        bcftools view -r <chr_list_j> --threads N -Oz src.vcf.gz \\
            > <out>/<j>/all.subset.vcf.gz
    layout=per_chrom:
        bcftools concat --threads N -Oz <chr_files_j> \\
            > <out>/<j>/all.subset.vcf.gz
    (common tail)
        bcftools index -t <out>/<j>/all.subset.vcf.gz
        plink2 --vcf <subset> --maf <qc.maf_min> --geno <qc.missingness_max> \\
               [--hwe <qc.hwe_min_p>] --max-alleles 2 --snps-only \\
               --set-all-var-ids @:# --new-id-max-allele-len 200 missing \\
               --make-pgen --make-bed --threads N --out <out>/<j>/all
        [optional] plink2 --pfile … --export vcf-4.2 bgz id-paste=iid

Output per subgenome:
    <out>/<j>/all.pgen,pvar,psam  (always)
    <out>/<j>/all.bed,bim,fam     (always; GRM/LOCO pipelines consume BED)
    <out>/<j>/all.vcf.gz[.tbi]    (only when --keep-vcf or not --no-vcf)
    <out>/<j>/split_log.json      (provenance for reproducibility)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from .species_config import SpeciesConfig, load_species_config

# ---------------------------------------------------------------------------
# Plan structures
# ---------------------------------------------------------------------------

@dataclass
class SubgenomeWork:
    """One subgenome's split commands + expected outputs.

    plink2 v2 disallows ``--make-pgen --make-bed`` in a single invocation,
    so when both formats are requested the splitter emits two commands:
    ``plink2_cmd`` does the VCF → pgen import with QC filters, and the
    optional ``plink2_bed_cmd`` then converts pgen → bed with no extra QC."""

    sg_id: str
    chroms: list[str]
    subset_vcf: Path
    pgen_prefix: Path
    bed_prefix: Path
    out_vcf: Path | None
    bcftools_cmd: list[str]            # bcftools view/concat → subset_vcf
    bcftools_index_cmd: list[str]      # bcftools index -t subset_vcf
    plink2_cmd: list[str]              # plink2 import → pgen (or bed if --skip-pgen)
    plink2_bed_cmd: list[str] | None = None        # plink2 pgen → bed
    plink2_export_vcf_cmd: list[str] | None = None  # only when keep_vcf


@dataclass
class SplitPlan:
    """Full per-species plan."""

    species_id: str
    out_dir: Path
    subgenomes: list[SubgenomeWork] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def _chrom_file_map(vcf_dir: Path) -> dict[str, Path]:
    """Map chrom_name → vcf file, for per_chrom layout.

    Convention: file name contains ``<chrom>`` somewhere; we glob *.vcf.gz
    and match by chrom-as-substring (handles ``chr1A.vcf.gz`` /
    ``chr_chr1A.vcf.gz`` / ``A01.norm.vcf.gz`` etc.).
    """
    out: dict[str, Path] = {}
    if not vcf_dir.exists():
        return out
    for f in sorted(vcf_dir.glob("*.vcf.gz")):
        # heuristic: strip suffixes, hope chrom is in basename
        out[f.stem.replace(".vcf", "")] = f
    return out


def _route_per_chrom(vcf_dir: Path, chroms: list[str]) -> list[Path]:
    """For per_chrom layout: return ordered list of VCF files matching the
    subgenome's chrom list. Raises if a chrom has no matching file."""
    available = _chrom_file_map(vcf_dir)
    matched: list[Path] = []
    for c in chroms:
        # Try exact stem match first
        hit = available.get(c) or available.get(f"chr{c}") or available.get(c.replace("chr", ""))
        if hit is None:
            # fall back: substring search across all files
            hits = [p for stem, p in available.items() if c in stem]
            if not hits:
                raise FileNotFoundError(
                    f"per_chrom layout: no VCF found for chrom {c!r} in {vcf_dir} "
                    f"(available stems: {sorted(available)[:5]}…)")
            hit = hits[0]
        matched.append(hit)
    return matched


def plan_split(
    cfg: SpeciesConfig,
    out_dir: Path,
    *,
    threads: int = 8,
    keep_vcf: bool = False,
    skip_pgen: bool = False,
    project_root: Path | None = None,
) -> SplitPlan:
    """Build the SplitPlan for a species (one SubgenomeWork per subgenome).

    No I/O is performed; pure planning. Call ``execute_plan`` to run.
    """
    if project_root is None:
        project_root = Path.cwd()

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else (project_root / p)

    if cfg.geno.vcf is None:
        raise ValueError("species_split: geno.bed_root mode does not need splitting "
                         "(panel is already pre-split). Set geno.vcf for new panels.")

    vcf_src = _resolve(cfg.geno.vcf)
    plan = SplitPlan(species_id=cfg.id, out_dir=out_dir)

    for sg in cfg.subgenomes:
        sg_dir = out_dir / sg.id
        subset_vcf = sg_dir / "all.subset.vcf.gz"
        pgen_prefix = sg_dir / "all"
        bed_prefix = sg_dir / "all"
        out_vcf = (sg_dir / "all.vcf.gz") if keep_vcf else None

        # bcftools chrom selection
        if cfg.geno.layout == "single":
            bcftools_cmd = [
                "bcftools", "view",
                "-r", ",".join(sg.chroms),
                "--threads", str(threads),
                "-Oz", "-o", str(subset_vcf),
                str(vcf_src),
            ]
        elif cfg.geno.layout == "per_chrom":
            chrom_files = _route_per_chrom(vcf_src, sg.chroms)
            bcftools_cmd = [
                "bcftools", "concat",
                "--threads", str(threads),
                "-Oz", "-o", str(subset_vcf),
                *[str(p) for p in chrom_files],
            ]
        else:  # pragma: no cover - schema enforces
            raise ValueError(f"unknown layout: {cfg.geno.layout}")

        bcftools_index_cmd = ["bcftools", "index", "-t", "-f", str(subset_vcf)]

        # plink2 import → pgen + bed in one pass
        plink2_cmd = ["plink2", "--vcf", str(subset_vcf)]
        if cfg.qc.maf_min and cfg.qc.maf_min > 0:
            plink2_cmd += ["--maf", str(cfg.qc.maf_min)]
        if cfg.qc.missingness_max and cfg.qc.missingness_max < 1.0:
            plink2_cmd += ["--geno", str(cfg.qc.missingness_max)]
        if cfg.qc.hwe_min_p is not None:
            plink2_cmd += ["--hwe", str(cfg.qc.hwe_min_p)]
        plink2_cmd += [
            "--max-alleles", "2", "--snps-only",
            # Auto-fill SNP IDs (cotton "." pitfall from M3.4 v2 docs/06)
            "--set-all-var-ids", "@:#",
            "--new-id-max-allele-len", "200", "missing",
            "--threads", str(threads),
            "--out", str(pgen_prefix),
        ]
        # First pass writes pgen (the QC pass — only one pass should run QC
        # filters so the subsequent bed conversion stays bit-identical).
        # If --skip-pgen the user only wants bed; do everything in one pass.
        plink2_bed_cmd: list[str] | None = None
        if not skip_pgen:
            plink2_cmd += ["--make-pgen"]
            plink2_bed_cmd = [
                "plink2", "--pfile", str(pgen_prefix),
                "--make-bed",
                "--threads", str(threads),
                "--out", str(bed_prefix),
            ]
        else:
            plink2_cmd += ["--make-bed"]

        plink2_export_vcf_cmd = None
        if keep_vcf:
            plink2_export_vcf_cmd = [
                "plink2", "--pfile" if not skip_pgen else "--bfile", str(pgen_prefix),
                "--export", "vcf-4.2", "bgz", "id-paste=iid",
                "--threads", str(threads),
                "--out", str(pgen_prefix),
            ]

        plan.subgenomes.append(SubgenomeWork(
            sg_id=sg.id, chroms=list(sg.chroms),
            subset_vcf=subset_vcf,
            pgen_prefix=pgen_prefix,
            bed_prefix=bed_prefix,
            out_vcf=out_vcf,
            bcftools_cmd=bcftools_cmd,
            bcftools_index_cmd=bcftools_index_cmd,
            plink2_cmd=plink2_cmd,
            plink2_bed_cmd=plink2_bed_cmd,
            plink2_export_vcf_cmd=plink2_export_vcf_cmd,
        ))
    return plan


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run_cmd(cmd: list[str], log_file: Path | None = None) -> subprocess.CompletedProcess:
    """Run a single command and (optionally) append stdout/stderr to log."""
    t0 = time.time()
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if log_file is not None:
        with log_file.open("a") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            fh.write(res.stdout)
            if res.stderr:
                fh.write("\n--- stderr ---\n")
                fh.write(res.stderr)
            fh.write(f"\n[elapsed {time.time() - t0:.1f}s]\n")
    return res


def execute_plan(
    plan: SplitPlan,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """Run a SplitPlan. Returns a summary dict (for json log)."""
    summary = {
        "species_id": plan.species_id,
        "out_dir": str(plan.out_dir),
        "dry_run": dry_run,
        "subgenomes": [],
    }
    for sg in plan.subgenomes:
        sg_dir = plan.out_dir / sg.sg_id
        sg_summary = {
            "sg_id": sg.sg_id,
            "n_chroms": len(sg.chroms),
            "subset_vcf": str(sg.subset_vcf),
            "pgen_prefix": str(sg.pgen_prefix),
            "bed_prefix": str(sg.bed_prefix),
            "bcftools_cmd": " ".join(sg.bcftools_cmd),
            "plink2_cmd": " ".join(sg.plink2_cmd),
            "status": "PLANNED",
        }
        if dry_run:
            summary["subgenomes"].append(sg_summary)
            continue

        sg_dir.mkdir(parents=True, exist_ok=True)
        log_file = sg_dir / "split_log.txt"
        # Refuse to overwrite an existing run unless --force
        if (sg_dir / "all.bed").exists() and not force:
            sg_summary["status"] = "SKIPPED_EXISTING"
            summary["subgenomes"].append(sg_summary)
            continue

        try:
            _run_cmd(sg.bcftools_cmd, log_file=log_file)
            _run_cmd(sg.bcftools_index_cmd, log_file=log_file)
            _run_cmd(sg.plink2_cmd, log_file=log_file)
            if sg.plink2_bed_cmd is not None:
                _run_cmd(sg.plink2_bed_cmd, log_file=log_file)
            if sg.plink2_export_vcf_cmd is not None:
                _run_cmd(sg.plink2_export_vcf_cmd, log_file=log_file)
                # plink2's export-bgz writes to <prefix>.vcf.gz already; ensure index
                _run_cmd(["bcftools", "index", "-t", "-f", str(sg.out_vcf)], log_file=log_file)
            sg_summary["status"] = "OK"
            # Tidy: remove subset.vcf once pgen exists (it's an intermediate)
            if (sg.subset_vcf).exists():
                sg.subset_vcf.unlink()
                (sg_dir / "all.subset.vcf.gz.tbi").unlink(missing_ok=True)
        except subprocess.CalledProcessError as exc:
            sg_summary["status"] = "ERROR"
            sg_summary["error"] = exc.stderr[-500:] if exc.stderr else str(exc)

        summary["subgenomes"].append(sg_summary)

    # Persist run summary
    if not dry_run:
        plan.out_dir.mkdir(parents=True, exist_ok=True)
        with (plan.out_dir / "split_summary.json").open("w") as fh:
            json.dump(summary, fh, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI subcommand handler
# ---------------------------------------------------------------------------


def cmd_split(args: argparse.Namespace) -> int:
    """Handler for ``homoeogwas split --species-yaml <yaml>``."""
    yaml_path = Path(args.species_yaml).resolve()
    cfg = load_species_config(yaml_path, validate=not args.skip_validate,
                              project_root=Path(args.project_root) if args.project_root else None)
    # default out_dir is data/processed/<species_id>
    project_root = Path(args.project_root) if args.project_root else yaml_path.parents[2]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = project_root / "data" / "processed" / cfg.id

    plan = plan_split(
        cfg, out_dir,
        threads=args.threads,
        keep_vcf=args.keep_vcf,
        skip_pgen=args.skip_pgen,
        project_root=project_root,
    )

    if args.dry_run:
        print(json.dumps({
            "species_id": cfg.id,
            "out_dir": str(out_dir),
            "subgenomes": [
                {
                    "sg_id": sg.sg_id,
                    "n_chroms": len(sg.chroms),
                    "bcftools": " ".join(sg.bcftools_cmd),
                    "plink2": " ".join(sg.plink2_cmd),
                }
                for sg in plan.subgenomes
            ],
        }, indent=2))
        return 0

    # Verify external tools are on PATH before running
    for tool in ("bcftools", "plink2"):
        if shutil.which(tool) is None:
            raise SystemExit(
                f"ERR: required tool {tool!r} not found on PATH.\n"
                f"     install:  micromamba install -n polygwas-cpu -c bioconda {tool}")

    summary = execute_plan(plan, dry_run=False, force=args.force)
    n_ok = sum(1 for sg in summary["subgenomes"] if sg["status"] == "OK")
    n_err = sum(1 for sg in summary["subgenomes"] if sg["status"] == "ERROR")
    n_skip = sum(1 for sg in summary["subgenomes"] if sg["status"] == "SKIPPED_EXISTING")
    print(f"split summary: OK={n_ok}  SKIPPED={n_skip}  ERROR={n_err}  → {out_dir}")
    return 1 if n_err else 0


def add_split_subparser(sub):
    sp = sub.add_parser(
        "split",
        help="split a panel VCF into per-subgenome BED/pgen using a species YAML",
    )
    sp.add_argument("--species-yaml", required=True,
                    help="path to configs/species/<species>.yaml")
    sp.add_argument("-o", "--out-dir", default=None,
                    help="override default data/processed/<species_id>/")
    sp.add_argument("--threads", type=int, default=8)
    sp.add_argument("--keep-vcf", action="store_true",
                    help="emit per-sub all.vcf.gz (default: pgen+bed only)")
    sp.add_argument("--skip-pgen", action="store_true",
                    help="emit BED-only (skip pgen — useful when downstream is BED-only)")
    sp.add_argument("--skip-validate", action="store_true",
                    help="skip species YAML cross-file validation (CI tests)")
    sp.add_argument("--project-root", default=None,
                    help="override project root for resolving relative YAML paths")
    sp.add_argument("--dry-run", action="store_true",
                    help="print the planned commands as JSON; do not execute")
    sp.add_argument("--force", action="store_true",
                    help="overwrite existing per-subgenome outputs")
    return sp


__all__ = [
    "SubgenomeWork",
    "SplitPlan",
    "plan_split",
    "execute_plan",
    "cmd_split",
    "add_split_subparser",
]
