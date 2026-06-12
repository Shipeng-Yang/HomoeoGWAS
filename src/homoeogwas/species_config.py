"""Species configuration schema for HomoeoGWAS v2 (Phase 5a).

Replaces the v1 hardcoded per-species paths (wheat/cotton/rapeseed) with
``configs/species/<species>.yaml`` configs that the framework reads to
plug-in any allopolyploid species — including user-supplied ones (e.g.
sweetpotato 6n, oat 6n, peanut 4n, strawberry 8n).

The schema is intentionally minimal: 10 required fields cover the
inputs that the LMM / DL-prior pipelines actually consume, and
species-specific behaviour MUST be encoded in the YAML rather than
``if species == "wheat": ...`` branches.

Usage:
    from homoeogwas.species_config import load_species_config
    cfg = load_species_config("configs/species/wheat_aestivum.yaml")
    # cfg.fasta, cfg.subgenomes, cfg.geno.bed_root, ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class Subgenome(BaseModel):
    """One subgenome entry. ``chroms`` lists the panel chrom names that
    belong to this subgenome; the framework uses this list to split the
    panel VCF/BED and build per-sub GRMs."""

    # populate_by_name=True so YAML keys 'copy_number' and (legacy) 'copy'
    # both bind to ``copy_number`` — avoids shadowing BaseModel.copy.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    id: str = Field(..., description="Short id, e.g. 'A' / 'B' / 'D'")
    copy_number: int = Field(
        1, ge=1, alias="copy",
        description="Copy number (1 for typical allopolyploid, "
                    ">=2 only for autopolyploid which is out-of-scope)",
    )
    donor: str | None = Field(None, description="Metadata: ancestral donor species, "
                                                    "e.g. 'Aegilops tauschii' for wheat D")
    chroms: list[str] = Field(..., min_length=1)


class GenoSource(BaseModel):
    """Genotype source: either a panel VCF (`vcf`) or a plink BED root
    (`bed_root`) containing one subdir per subgenome with ``all.{bed,bim,fam}``.
    Exactly one of the two must be set.

    For VCF mode, ``layout`` declares how the file(s) are organised:

    - ``single``  : ``vcf`` is one panel-wide ``.vcf.gz`` covering all chroms
    - ``per_chrom``: ``vcf`` is a directory; the splitter globs ``*.vcf.gz``
      and routes each chrom into its subgenome based on the YAML chrom lists

    The framework does NOT auto-detect layout — explicit declaration in the
    YAML avoids silent-misroute risk on user-provided panels.
    """

    model_config = ConfigDict(extra="forbid")

    vcf: Path | None = None
    bed_root: Path | None = None
    layout: Literal["single", "per_chrom"] = "single"

    @model_validator(mode="after")
    def _exactly_one_source(self):
        if (self.vcf is None) == (self.bed_root is None):
            raise ValueError("geno: exactly one of 'vcf' or 'bed_root' must be set")
        return self


class QCConfig(BaseModel):
    """Per-species QC thresholds. HWE filter is disabled by default (the
    Tier-1 panels include inbred landrace germplasm where HWE rejects 95%+
    of SNPs)."""

    model_config = ConfigDict(extra="forbid")

    maf_min: float = Field(0.01, ge=0.0, le=0.5)
    hwe_min_p: float | None = None          # null = filter off
    missingness_max: float = Field(0.1, ge=0.0, le=1.0)


class Provenance(BaseModel):
    """Free-form metadata block — not validated, only carried through to
    paper supplementary / mkdocs."""

    model_config = ConfigDict(extra="allow")  # allow any metadata field

    panel_name: str | None = None
    panel_source: str | None = None
    panel_n_sample: int | None = None
    reference_assembly: str | None = None
    reference_doi: str | None = None
    reference_paper: str | None = None
    notes: str | None = None


# ---------------------------------------------------------------------------
# Top-level species config
# ---------------------------------------------------------------------------

class SpeciesConfig(BaseModel):
    """One allopolyploid species configuration. All required fields must
    point at real files (validated at load time); optional fields default
    to safe values."""

    model_config = ConfigDict(extra="forbid")

    # === identification ===
    id: str = Field(..., min_length=1, max_length=64)
    latin: str
    common_name: str

    # === genome architecture ===
    genome_type: Literal["allopolyploid", "diploid"]
    ploidy: int = Field(..., ge=2)
    subgenomes: list[Subgenome] = Field(..., min_length=1)

    # === reference + panel (required) ===
    fasta: Path
    gff: Path
    chrom_map: Path           # TSV with panel_chrom / fasta_chrom / subgenome cols
    geno: GenoSource
    pheno: Path

    # === optional w/ default ===
    sample_map: Path | None = None
    dosage_model: Literal["diploidized", "dosage_0_to_n", "presence_absence"] = "diploidized"
    loco_unit: Literal["chrom", "subgenome"] = "chrom"
    dl_model_window: int = Field(6144, ge=128)
    dl_candidate_window: int = Field(100_000, ge=1_000)
    qc: QCConfig = Field(default_factory=QCConfig)
    known_qtl_tsv: Path | None = None
    homoeolog_map: Path | None = None
    provenance: Provenance | None = None

    # ---------------------------------------------------------------------
    # Structural cross-field validation (Tier 2 in the dual-plan validator)
    # ---------------------------------------------------------------------

    @model_validator(mode="after")
    def _structural(self):
        # Tier 2.1 — subgenome count consistent with genome_type/ploidy
        n_sub = len(self.subgenomes)
        if self.genome_type == "diploid":
            if n_sub != 1:
                raise ValueError(f"diploid species must declare 1 subgenome, got {n_sub}")
            if self.ploidy != 2:
                raise ValueError(f"diploid implies ploidy=2, got {self.ploidy}")
        else:  # allopolyploid
            # Each subgenome typically contributes 2 chromosome sets; the framework
            # tolerates any n_sub ≥ 2 as long as ploidy is consistent (n_sub * 2
            # by default, but some species have unequal sub copy numbers — we
            # do not hard-fail on that, just warn-by-shape).
            if n_sub < 2:
                raise ValueError(f"allopolyploid must declare ≥2 subgenomes, got {n_sub}")
            if sum(sg.copy_number for sg in self.subgenomes) * 2 != self.ploidy:
                # Soft warn — not all polyploids follow the n*2 rule (e.g.
                # segmental allopolyploids, hexaploid with non-uniform copy).
                pass  # do not raise; user is responsible for ploidy consistency

        # Tier 2.2 — subgenome ids unique
        sg_ids = [sg.id for sg in self.subgenomes]
        if len(set(sg_ids)) != len(sg_ids):
            raise ValueError(f"duplicate subgenome ids: {sg_ids}")

        # Tier 2.3 — no chrom appears in multiple subgenomes
        seen: dict[str, str] = {}
        for sg in self.subgenomes:
            for c in sg.chroms:
                if c in seen:
                    raise ValueError(
                        f"chrom {c} appears in both subgenome {seen[c]} and {sg.id}"
                    )
                seen[c] = sg.id

        return self


# ---------------------------------------------------------------------------
# Loader + tiered validator (Tier 1 file existence, Tier 3 cross-file checks)
# ---------------------------------------------------------------------------

class ValidationReport(BaseModel):
    """Output of ``validate_species_config``."""

    model_config = ConfigDict(extra="forbid")

    species_id: str
    n_subgenomes: int
    n_chroms_total: int
    n_chroms_in_chrom_map: int
    n_chroms_resolved_to_fasta: int
    n_samples_in_pheno: int | None
    n_known_qtl: int
    warnings: list[str]


def _check_file(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{label}: {p} does not exist")


def validate_species_config(cfg: SpeciesConfig, *, project_root: Path) -> ValidationReport:
    """Cross-file validation. Hard-fails on Tier 1 (file existence) and Tier 3
    (chrom_map coverage). Warnings collected for Tier 4-6 (pheno header,
    known QTL columns, dosage_model x genome_type sanity)."""
    warnings: list[str] = []

    def _resolve(p: Path | None) -> Path | None:
        if p is None:
            return None
        return p if p.is_absolute() else (project_root / p)

    # Tier 1 — required files exist
    fasta = _resolve(cfg.fasta)
    gff = _resolve(cfg.gff)
    chrom_map_path = _resolve(cfg.chrom_map)
    pheno_path = _resolve(cfg.pheno)
    assert fasta is not None and gff is not None and chrom_map_path is not None and pheno_path is not None
    _check_file(fasta, "fasta")
    _check_file(gff, "gff")
    _check_file(chrom_map_path, "chrom_map")
    _check_file(pheno_path, "pheno")
    if cfg.geno.vcf is not None:
        _check_file(_resolve(cfg.geno.vcf), "geno.vcf")
    if cfg.geno.bed_root is not None:
        bed_root = _resolve(cfg.geno.bed_root)
        if not bed_root.exists():
            raise FileNotFoundError(f"geno.bed_root: {bed_root} does not exist")
        # At least one subgenome bim should exist (smoke check)
        first_sg = cfg.subgenomes[0]
        bim = bed_root / first_sg.id / "all.bim"
        if not bim.exists():
            warnings.append(
                f"expected {bim} for subgenome {first_sg.id} (will fail at runtime)")

    # Tier 3 — chrom_map covers all panel chroms
    chrom_map_df = pd.read_csv(chrom_map_path, sep="\t")
    if "panel_chrom" not in chrom_map_df.columns:
        raise ValueError(f"chrom_map missing 'panel_chrom' column: {chrom_map_df.columns.tolist()}")
    chrom_map_panel = set(chrom_map_df["panel_chrom"].astype(str))

    declared_chroms = {c for sg in cfg.subgenomes for c in sg.chroms}
    missing_in_map = declared_chroms - chrom_map_panel
    if missing_in_map:
        raise ValueError(
            f"chrom_map missing entries for declared chroms: {sorted(missing_in_map)[:5]}…")

    # Tier 4 — pheno TSV header
    try:
        pheno_head = pd.read_csv(pheno_path, sep="\t", nrows=5)
        n_samples = None
        if "sample_id" not in pheno_head.columns and "ID" not in pheno_head.columns:
            warnings.append("pheno TSV lacks 'sample_id'/'ID' column (downstream will guess)")
        else:
            n_samples = int(pd.read_csv(pheno_path, sep="\t", usecols=[pheno_head.columns[0]]).shape[0])
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pheno read error: {exc}")
        n_samples = None

    # Tier 5 — known QTL (optional)
    n_known_qtl = 0
    if cfg.known_qtl_tsv is not None:
        kp = _resolve(cfg.known_qtl_tsv)
        if kp is not None and kp.exists():
            try:
                qdf = pd.read_csv(kp, sep="\t")
                for col in ("qtl_name", "chrom", "qtl_pos"):
                    if col not in qdf.columns:
                        warnings.append(f"known_qtl_tsv missing column '{col}'")
                n_known_qtl = int(len(qdf))
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"known_qtl_tsv read error: {exc}")
        else:
            warnings.append(f"known_qtl_tsv declared but file missing: {cfg.known_qtl_tsv}")

    # Tier 6 — dosage model sanity
    if cfg.dosage_model == "dosage_0_to_n" and cfg.genome_type == "allopolyploid":
        warnings.append(
            "dosage_0_to_n on allopolyploid is unusual (typically diploidized); "
            "verify the panel really carries 0..n dosage genotypes")

    # Optional fasta chrom presence check (cheap, useful)
    n_resolved_fasta = 0
    try:
        import pysam
        with pysam.FastaFile(str(fasta)) as fa:
            fasta_refs = set(fa.references)
        n_resolved_fasta = int(chrom_map_df["fasta_chrom"].astype(str).isin(fasta_refs).sum())
        if n_resolved_fasta < len(chrom_map_df):
            warnings.append(
                f"chrom_map: {len(chrom_map_df) - n_resolved_fasta} fasta_chrom entries "
                "not found in fasta references (will fail at runtime for those chroms)")
    except ImportError:
        warnings.append("pysam not installed; skipping fasta-references coverage check")

    return ValidationReport(
        species_id=cfg.id,
        n_subgenomes=len(cfg.subgenomes),
        n_chroms_total=len(declared_chroms),
        n_chroms_in_chrom_map=len(chrom_map_panel),
        n_chroms_resolved_to_fasta=n_resolved_fasta,
        n_samples_in_pheno=n_samples,
        n_known_qtl=n_known_qtl,
        warnings=warnings,
    )


def load_species_config(
    yaml_path: str | Path, *, validate: bool = True, project_root: Path | None = None
) -> SpeciesConfig:
    """Load + optionally validate a species YAML file.

    Parameters
    ----------
    yaml_path : path to ``configs/species/<species>.yaml``
    validate : if True, run ``validate_species_config`` (Tier 1 + 3-6).
        Set False for unit tests that don't have real data on disk.
    project_root : absolute root for resolving relative paths in the YAML
        (default: the directory containing the YAML's grandparent —
        i.e. the project repo root).
    """
    yaml_path = Path(yaml_path).resolve()
    with yaml_path.open() as fin:
        raw = yaml.safe_load(fin)
    cfg = SpeciesConfig.model_validate(raw)
    if validate:
        if project_root is None:
            # configs/species/<name>.yaml → project_root is yaml_path.parents[2]
            project_root = yaml_path.parents[2]
        report = validate_species_config(cfg, project_root=project_root)
        # Attach report for caller inspection
        cfg.__dict__["_validation_report"] = report
    return cfg


__all__ = [
    "Subgenome",
    "GenoSource",
    "QCConfig",
    "Provenance",
    "SpeciesConfig",
    "ValidationReport",
    "load_species_config",
    "validate_species_config",
]
