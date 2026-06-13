"""HomoeoGWAS command-line interface (``homoeogwas fit``).

A YAML-config-driven wrapper around the core engine. The whole
subgenome-stratified LMM GWAS runs from one config plus a few flags:

    homoeogwas fit --config run.yaml [--out-dir DIR] [--backend cpu|gpu|auto]
                   [--dry-run] [--force]

Pipeline (generalised to J subgenomes, e.g. A/C, A/B/D):

    per-subgenome VanRaden GRM (cached .npz or computed from BED)
      -> normalize_kernel  [+ optional homoeolog Hadamard kernel]
      -> fit_multi_reml    (multi-kernel REML variance components)
      -> build_scan_context (fixed-V projection P)
      -> scan_snps (in-memory) | scan_bed_stream (streaming)
      -> per-SNP sumstats + QQ + Manhattan + lambda_GC + summary JSON
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .grm import GRMPart, compute_grm, compute_loco_grm_parts, loco_grm_from_parts
from .io import load_bed_hardcall
from .kernel import hadamard_kernel, normalize_kernel
from .lmm import fit_multi_reml
from .scan import (
    LOCOContext,
    build_loco_scan_contexts,
    build_scan_context,
    lambda_gc,
    scan_bed_stream,
    scan_bed_stream_loco,
    scan_snps,
    scan_snps_loco,
)

# config loading


def _require_yaml():
    """Import PyYAML with an actionable message if it is missing."""
    try:
        import yaml  # noqa: F401
        return yaml
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise SystemExit(
            "ERR: PyYAML is required to read HomoeoGWAS run configs.\n"
            "     install:  micromamba install -n polygwas-cpu -c conda-forge pyyaml"
        ) from e


def _get(d: dict, path: str, default=None):
    """Nested dict get by dotted path; returns default if any hop is missing."""
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur or cur[key] is None:
            return default
        cur = cur[key]
    return cur


def load_config(config_path: str | Path) -> dict:
    """Load a fit run-config (YAML), merging defaults from an optional
    ``panel_manifest`` reference, and return the resolved config dict."""
    yaml = _require_yaml()
    config_path = Path(config_path)
    if not config_path.exists():
        raise SystemExit(f"ERR: config not found: {config_path}")
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise SystemExit(f"ERR: config {config_path} is not a YAML mapping")

    # optional panel-manifest reference: fills panel.subgenomes when absent
    manifest_rel = cfg.get("panel_manifest")
    if manifest_rel:
        man_path = (config_path.parent / manifest_rel
                    if not Path(manifest_rel).is_absolute() else Path(manifest_rel))
        if not man_path.exists():
            # also try project-root-relative
            man_path = Path(manifest_rel)
        if man_path.exists():
            with open(man_path) as fh:
                manifest = yaml.safe_load(fh) or {}
            cfg.setdefault("panel", {})
            if not cfg["panel"].get("subgenomes") and manifest.get("subgenomes"):
                cfg["panel"]["subgenomes"] = manifest["subgenomes"]
            if not cfg["panel"].get("name") and manifest.get("panel"):
                cfg["panel"]["name"] = manifest["panel"]
            cfg["_panel_manifest_resolved"] = str(man_path)
        else:
            cfg["_panel_manifest_warning"] = f"panel_manifest not found: {manifest_rel}"
    return cfg


def validate_config(cfg: dict) -> None:
    """Raise SystemExit with an actionable message for an invalid config."""
    subg = _get(cfg, "panel.subgenomes")
    if not subg or not isinstance(subg, list):
        raise SystemExit("ERR: config panel.subgenomes must be a non-empty list "
                          "(or provide a panel_manifest that defines it)")
    if len(set(subg)) != len(subg):
        raise SystemExit(f"ERR: panel.subgenomes has duplicates: {subg}")
    if not _get(cfg, "phenotype.path"):
        raise SystemExit("ERR: config phenotype.path is required")
    if not _get(cfg, "phenotype.trait"):
        raise SystemExit("ERR: config phenotype.trait is required")
    if not _get(cfg, "genotype.scan_bed_prefix_template"):
        raise SystemExit("ERR: config genotype.scan_bed_prefix_template is required")
    mode = _get(cfg, "scan.mode", "auto")
    if mode not in ("auto", "memory", "stream"):
        raise SystemExit(f"ERR: scan.mode must be auto|memory|stream, got {mode!r}")
    grm_source = _get(cfg, "genotype.grm.source", "bed")
    if grm_source not in ("bed", "npz"):
        raise SystemExit(f"ERR: genotype.grm.source must be bed|npz, got {grm_source!r}")
    if grm_source == "npz" and not _get(cfg, "genotype.grm.npz_path"):
        raise SystemExit("ERR: genotype.grm.source=npz requires genotype.grm.npz_path")
    had_name = _get(cfg, "kernels.hadamard_name", "hom")
    if had_name in subg:
        raise SystemExit(f"ERR: kernels.hadamard_name {had_name!r} collides with "
                          f"a subgenome name")
    norm = _get(cfg, "kernels.normalize", "trace")
    if norm not in ("trace", "frobenius"):
        raise SystemExit(f"ERR: kernels.normalize must be trace|frobenius, "
                         f"got {norm!r}")
    backend = _get(cfg, "scan.backend", "auto")
    if backend not in ("auto", "cpu", "gpu"):
        raise SystemExit(f"ERR: scan.backend must be auto|cpu|gpu, got {backend!r}")
    for key, lo in (("scan.batch_size", 1), ("scan.chunk_size", 1)):
        v = _get(cfg, key)
        if v is not None and (not isinstance(v, int) or v < lo):
            raise SystemExit(f"ERR: config {key} must be an integer >= {lo}, "
                             f"got {v!r}")
    for key in ("scan.maf_min", "scan.call_rate_min", "genotype.grm.maf_min"):
        v = _get(cfg, key)
        if v is not None and not (isinstance(v, (int, float))
                                  and 0.0 <= float(v) <= 1.0):
            raise SystemExit(f"ERR: config {key} must be a number in [0,1], "
                             f"got {v!r}")
    nstarts = _get(cfg, "reml.n_starts")
    if nstarts is not None and (not isinstance(nstarts, int) or nstarts < 1):
        raise SystemExit(f"ERR: config reml.n_starts must be an integer >= 1, "
                         f"got {nstarts!r}")
    # LOCO config
    loco_cfg = _get(cfg, "scan.loco")
    if loco_cfg is not None:
        if not isinstance(loco_cfg, dict):
            raise SystemExit(
                f"ERR: scan.loco must be a mapping, got "
                f"{type(loco_cfg).__name__}")
        enabled = _get(cfg, "scan.loco.enabled", False)
        if not isinstance(enabled, bool):
            raise SystemExit(
                f"ERR: scan.loco.enabled must be true|false, got {enabled!r}")
        fallback = _get(cfg, "scan.loco.fallback", "error")
        if fallback not in ("error", "emmax"):
            raise SystemExit(
                f"ERR: scan.loco.fallback must be error|emmax, got {fallback!r}")
        if fallback == "emmax":
            raise SystemExit(
                "ERR: scan.loco.fallback=emmax is reserved but not implemented in "
                "Phase 3 M3.1; Phase 3 acceptance requires fallback=error so an "
                "unknown chrom or degenerate K_(-c) fails the run loudly")
        mdf = _get(cfg, "scan.loco.min_denominator_fraction", 1.0e-6)
        if not (isinstance(mdf, (int, float))
                and 0.0 < float(mdf) <= 1.0):
            raise SystemExit(
                f"ERR: scan.loco.min_denominator_fraction must be in (0, 1], "
                f"got {mdf!r}")
        if enabled and grm_source == "npz":
            raise SystemExit(
                "ERR: scan.loco.enabled=true requires "
                "genotype.grm.source=bed; an npz GRM cache stores only the "
                "fully-normalised global GRM, not the per-chrom partials LOCO "
                "needs. Set genotype.grm.source=bed (and point "
                "bed_prefix_template at the pruned per-subgenome BED).")


# preflight


def _scan_bed_prefix(cfg: dict, sg: str) -> Path:
    return Path(_get(cfg, "genotype.scan_bed_prefix_template").format(subgenome=sg))


def _grm_bed_prefix(cfg: dict, sg: str) -> Path:
    tmpl = _get(cfg, "genotype.grm.bed_prefix_template")
    if not tmpl:
        tmpl = _get(cfg, "genotype.scan_bed_prefix_template")
    return Path(tmpl.format(subgenome=sg))


def preflight(cfg: dict) -> list[str]:
    """Check every input path exists; return a list of human-readable problems."""
    problems: list[str] = []
    subg = _get(cfg, "panel.subgenomes")
    pheno = Path(_get(cfg, "phenotype.path", ""))
    if not pheno.exists():
        problems.append(f"phenotype.path missing: {pheno}")
    for sg in subg:
        bed = _scan_bed_prefix(cfg, sg).with_suffix(".bed")
        if not bed.exists():
            problems.append(f"scan BED missing for subgenome {sg}: {bed}")
    if _get(cfg, "genotype.grm.source", "bed") == "npz":
        npz = Path(_get(cfg, "genotype.grm.npz_path", ""))
        if not npz.exists():
            problems.append(f"genotype.grm.npz_path missing: {npz}")
    else:
        for sg in subg:
            bed = _grm_bed_prefix(cfg, sg).with_suffix(".bed")
            if not bed.exists():
                problems.append(f"GRM BED missing for subgenome {sg}: {bed}")
    return problems


# phenotype + sample join


def _check_unique(samples: np.ndarray, label: str) -> np.ndarray:
    """Reject duplicate sample IDs — they would silently misalign rows."""
    ids = [str(s) for s in samples]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"ERR: GRM source {label} has duplicate sample IDs — "
                         "cannot align genotypes unambiguously")
    return np.asarray(ids, dtype=object)


def _grm_source_samples(cfg: dict, subg: list[str]) -> dict[str, np.ndarray]:
    """Per-subgenome sample IDs of the GRM source (npz or BED)."""
    out: dict[str, np.ndarray] = {}
    if _get(cfg, "genotype.grm.source", "bed") == "npz":
        npz = np.load(_get(cfg, "genotype.grm.npz_path"), allow_pickle=True)
        skey = _get(cfg, "genotype.grm.samples_key", "samples")
        if skey not in npz.files:
            raise SystemExit(f"ERR: GRM npz missing samples key {skey!r} "
                             f"(have {list(npz.files)})")
        samp = _check_unique(np.asarray(npz[skey], dtype=object),
                             f"npz[{skey}]")
        for sg in subg:
            out[sg] = samp
    else:
        from bed_reader import open_bed
        for sg in subg:
            with open_bed(str(_grm_bed_prefix(cfg, sg).with_suffix(".bed"))) as bed:
                out[sg] = _check_unique(np.asarray(bed.iid, dtype=object),
                                        f"BED {sg}")
    return out


def join_samples(cfg: dict):
    """Inner-join GRM samples (all subgenomes) with non-missing phenotype.

    Returns (analysis_samples, y, X). Canonical order = first subgenome's
    GRM-source sample order.
    """
    subg = _get(cfg, "panel.subgenomes")
    sample_col = _get(cfg, "phenotype.sample_col", "sample")
    trait = _get(cfg, "phenotype.trait")
    pheno = pd.read_csv(_get(cfg, "phenotype.path"), sep="\t")
    if sample_col not in pheno.columns:
        raise SystemExit(f"ERR: phenotype.sample_col {sample_col!r} not in "
                         f"{_get(cfg, 'phenotype.path')}")
    if trait not in pheno.columns:
        raise SystemExit(f"ERR: trait {trait!r} not in phenotype; have "
                         f"{[c for c in pheno.columns if c != sample_col]}")
    # average duplicate-sample rows (Horvath has repeated-site entries)
    ph = pheno.groupby(sample_col)[trait].mean()
    ph = ph[ph.notna()]

    grm_samp = _grm_source_samples(cfg, subg)
    canonical = [str(s) for s in grm_samp[subg[0]]]
    in_all_grm = set(canonical)
    for sg in subg[1:]:
        in_all_grm &= {str(s) for s in grm_samp[sg]}
    analysis = [s for s in canonical if s in in_all_grm and s in ph.index]
    if len(analysis) < 20:
        raise SystemExit(
            f"ERR: only {len(analysis)} samples overlap GRM ∩ phenotype "
            f"(trait {trait!r}); need >= 20")
    y = ph.loc[analysis].to_numpy(dtype=np.float64)
    X = np.ones((len(analysis), 1), dtype=np.float64)
    return np.array(analysis, dtype=object), y, X


# GRM + kernels


def build_kernels(cfg: dict, analysis_samples: np.ndarray):
    """Build the trace-/frobenius-normalized kernel dict for the analysis set.

    Returns (kernels, grm_info). kernels keys = subgenomes [+ hadamard name].
    """
    subg = _get(cfg, "panel.subgenomes")
    norm = _get(cfg, "kernels.normalize", "trace")
    maf_min = float(_get(cfg, "genotype.grm.maf_min", 0.01))
    source = _get(cfg, "genotype.grm.source", "bed")
    grm_info: dict = {"source": source, "normalize": norm, "raw": {}}

    raw: dict[str, np.ndarray] = {}
    if source == "npz":
        npz = np.load(_get(cfg, "genotype.grm.npz_path"), allow_pickle=True)
        skey = _get(cfg, "genotype.grm.samples_key", "samples")
        ktmpl = _get(cfg, "genotype.grm.npz_key_template", "G_{subgenome}")
        if skey not in npz.files:
            raise SystemExit(f"ERR: GRM npz missing samples key {skey!r} "
                             f"(have {list(npz.files)})")
        full_samp = [str(s) for s in np.asarray(npz[skey], dtype=object)]
        if len(set(full_samp)) != len(full_samp):
            raise SystemExit(f"ERR: GRM npz[{skey}] has duplicate sample IDs")
        pos = {s: i for i, s in enumerate(full_samp)}
        idx = np.array([pos[str(s)] for s in analysis_samples], dtype=np.int64)
        ns = len(full_samp)
        for sg in subg:
            key = ktmpl.format(subgenome=sg)
            if key not in npz.files:
                raise SystemExit(f"ERR: GRM npz missing key {key!r} "
                                 f"(have {list(npz.files)})")
            G_full = np.asarray(npz[key], dtype=np.float64)
            if G_full.shape != (ns, ns):
                raise SystemExit(f"ERR: GRM npz[{key}] shape {G_full.shape} "
                                 f"!= ({ns},{ns}) (n samples)")
            if not np.all(np.isfinite(G_full)):
                raise SystemExit(f"ERR: GRM npz[{key}] contains non-finite values")
            raw[sg] = G_full[np.ix_(idx, idx)]
            grm_info["raw"][sg] = {"n_markers": None}
    else:
        for sg in subg:
            geno = load_bed_hardcall(_grm_bed_prefix(cfg, sg))
            bed_samp = [str(s) for s in geno.samples]
            pos = {s: i for i, s in enumerate(bed_samp)}
            missing = [s for s in analysis_samples if str(s) not in pos]
            if missing:
                raise SystemExit(
                    f"ERR: GRM BED for subgenome {sg} missing "
                    f"{len(missing)} analysis samples (e.g. {missing[:3]})")
            idx = np.array([pos[str(s)] for s in analysis_samples], dtype=np.int64)
            from .io import GenoChunk
            # cast to float64: a float32 VanRaden GRM carries ~1e-7 eigenvalue
            # noise that trips the REML PSD check (eig_tol 1e-8).
            sub = GenoChunk(samples=np.asarray(analysis_samples, dtype=object),
                            variant_ids=geno.variant_ids, chrom=geno.chrom,
                            pos=geno.pos,
                            dosage=geno.dosage[idx, :].astype(np.float64))
            G, info = compute_grm(sub, maf_min=maf_min)
            raw[sg] = G
            grm_info["raw"][sg] = {"n_markers": int(info.get("m_used", -1))}

    kernels = {sg: normalize_kernel(raw[sg], mode=norm) for sg in subg}
    if _get(cfg, "kernels.include_hadamard", False):
        had_name = _get(cfg, "kernels.hadamard_name", "hom")
        K_hom = hadamard_kernel({sg: raw[sg] for sg in subg})
        kernels[had_name] = normalize_kernel(K_hom, mode=norm)
        grm_info["hadamard"] = had_name
    grm_info["kernel_names"] = list(kernels.keys())
    return kernels, grm_info


def build_loco_kernels(
    cfg: dict, analysis_samples: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray],
           dict[str, str], dict]:
    """Build per-chrom LOCO kernels and the global kernels.

    LOCO needs both ``global_kernels`` (``{name -> K (n,n)}``, used to fit the
    variance components, which are global and not refit per chrom) and
    ``kernels_by_chrom`` (``{chrom -> {name -> K_(-c) (n,n)}}``, consumed by
    :func:`scan.build_loco_scan_contexts`).

    Reads each subgenome's GRM BED once, runs ``compute_loco_grm_parts`` for the
    global part plus per-chrom partials, then builds K_j(-c) by subtraction. A
    subgenome not affected at chrom c (``sub_j != sub_c``) reuses the global raw
    GRM, so only the Hadamard kernel is recomputed, not the full GRM.

    Returns ``(kernels_by_chrom, global_kernels, chrom_to_subgenome, grm_info)``.
    """
    from .io import GenoChunk

    subg = _get(cfg, "panel.subgenomes")
    source = _get(cfg, "genotype.grm.source", "bed")
    norm = _get(cfg, "kernels.normalize", "trace")
    maf_min = float(_get(cfg, "genotype.grm.maf_min", 0.01))
    include_hadamard = bool(_get(cfg, "kernels.include_hadamard", False))
    had_name = _get(cfg, "kernels.hadamard_name", "hom")
    min_denom_frac = float(_get(cfg, "scan.loco.min_denominator_fraction", 1.0e-6))

    if source != "bed":
        # validate_config rejects this earlier; keep the guard for direct callers
        raise SystemExit(
            f"ERR: build_loco_kernels requires genotype.grm.source=bed, "
            f"got {source!r}")

    grm_info: dict = {"source": source, "normalize": norm,
                       "loco_enabled": True, "raw": {}, "loco": {}}

    # read each subgenome's BED, align samples, accumulate parts
    global_parts: dict[str, GRMPart] = {}
    parts_per_sub: dict[str, dict[str, GRMPart]] = {}
    chrom_to_sub: dict[str, str] = {}
    for sg in subg:
        geno = load_bed_hardcall(_grm_bed_prefix(cfg, sg))
        bed_samp = [str(s) for s in geno.samples]
        pos_map = {s: i for i, s in enumerate(bed_samp)}
        missing = [s for s in analysis_samples if str(s) not in pos_map]
        if missing:
            raise SystemExit(
                f"ERR: GRM BED for subgenome {sg} missing "
                f"{len(missing)} analysis samples (e.g. {missing[:3]})")
        idx = np.array(
            [pos_map[str(s)] for s in analysis_samples], dtype=np.int64)
        sub_chunk = GenoChunk(
            samples=np.asarray(analysis_samples, dtype=object),
            variant_ids=geno.variant_ids,
            chrom=geno.chrom,
            pos=geno.pos,
            dosage=geno.dosage[idx, :].astype(np.float64),
        )
        global_part, parts = compute_loco_grm_parts(sub_chunk, maf_min=maf_min)
        global_parts[sg] = global_part
        parts_per_sub[sg] = parts
        for c in parts:
            if c in chrom_to_sub and chrom_to_sub[c] != sg:
                raise SystemExit(
                    f"ERR: chrom {c!r} appears in multiple subgenome BEDs "
                    f"({chrom_to_sub[c]!r} and {sg!r}); each chrom must "
                    "belong to exactly one subgenome — check the GRM "
                    "bed_prefix_template input.")
            chrom_to_sub[c] = sg
        grm_info["raw"][sg] = {
            "n_markers": int(global_part.n_variants_used),
            "n_chrom": len(parts),
            "denominator_total": float(global_part.denominator),
        }

    # global normalised kernels (for the REML variance-component fit)
    raw_global = {sg: global_parts[sg].grm for sg in subg}
    global_kernels = {
        sg: normalize_kernel(raw_global[sg], mode=norm) for sg in subg
    }
    if include_hadamard:
        K_hom_global = hadamard_kernel({sg: raw_global[sg] for sg in subg})
        global_kernels[had_name] = normalize_kernel(K_hom_global, mode=norm)

    # per-chrom LOCO kernels
    kernels_by_chrom: dict[str, dict[str, np.ndarray]] = {}
    for c, sub_c in chrom_to_sub.items():
        # subgenome sub_c is LOCO-subtracted; the others reuse the global GRM
        raw_loco: dict[str, np.ndarray] = {}
        loco_info_c: dict = {}
        for sg in subg:
            if sg == sub_c:
                K_raw, info_loco = loco_grm_from_parts(
                    global_parts[sg], parts_per_sub[sg], c,
                    min_denominator_fraction=min_denom_frac)
                raw_loco[sg] = K_raw
                loco_info_c = info_loco
            else:
                raw_loco[sg] = raw_global[sg]
        kernels_c = {sg: normalize_kernel(raw_loco[sg], mode=norm)
                     for sg in subg}
        if include_hadamard:
            K_hom_c = hadamard_kernel({sg: raw_loco[sg] for sg in subg})
            kernels_c[had_name] = normalize_kernel(K_hom_c, mode=norm)
        kernels_by_chrom[c] = kernels_c
        grm_info["loco"][c] = {
            "subgenome": sub_c,
            "denom_retained": float(loco_info_c.get("denom_retained", 0.0)),
            "denom_retained_fraction": float(
                loco_info_c.get("denom_retained_fraction", 0.0)),
            "n_variants_retained": int(
                loco_info_c.get("n_variants_retained", 0)),
            "denom_removed": float(loco_info_c.get("denom_removed", 0.0)),
            "denom_removed_fraction": float(
                loco_info_c.get("denom_removed_fraction", 0.0)),
            "n_variants_removed": int(
                loco_info_c.get("n_variants_removed", 0)),
            "retained_fraction_low_risk": bool(
                loco_info_c.get("retained_fraction_low_risk", False)),
        }
    grm_info["chrom_to_subgenome"] = chrom_to_sub
    grm_info["kernel_names"] = list(global_kernels.keys())
    grm_info["n_chrom"] = len(kernels_by_chrom)
    return kernels_by_chrom, global_kernels, chrom_to_sub, grm_info


# scan


def _count_markers(cfg: dict, subg: list[str]) -> int:
    from bed_reader import open_bed
    total = 0
    for sg in subg:
        with open_bed(str(_scan_bed_prefix(cfg, sg).with_suffix(".bed"))) as bed:
            total += int(bed.sid_count)
    return total


def resolve_scan_mode(cfg: dict, subg: list[str], n: int) -> tuple[str, dict]:
    """Resolve scan.mode (auto -> memory|stream by a size heuristic)."""
    mode = _get(cfg, "scan.mode", "auto")
    info = {"requested": mode}
    if mode != "auto":
        info["resolved"] = mode
        return mode, info
    total_markers = _count_markers(cfg, subg)
    est_gb = n * total_markers * 4 / 1024 ** 3
    limit_m = int(_get(cfg, "scan.memory_marker_limit", 2_000_000))
    limit_gb = float(_get(cfg, "scan.memory_dosage_gb_limit", 4.0))
    resolved = ("memory" if total_markers <= limit_m and est_gb <= limit_gb
                else "stream")
    info.update({"resolved": resolved, "total_markers": total_markers,
                 "estimated_dosage_gb": round(est_gb, 3)})
    return resolved, info


def run_scan(cfg: dict, ctx, subg: list[str], backend: str, out_dir: Path,
             prefix: str, scan_mode: str) -> dict:
    """Run the per-SNP scan; returns a dict with sumstats path(s) + counts.

    Dispatches between the standard EMMAX scan (``ScanContext``) and the
    leave-one-chromosome-out scan (``LOCOContext``) by ``isinstance``.
    """
    maf_min = float(_get(cfg, "scan.maf_min", 0.01))
    cr_min = float(_get(cfg, "scan.call_rate_min", 0.9))
    is_loco = isinstance(ctx, LOCOContext)
    suffix = "_loco" if is_loco else ""
    if scan_mode == "memory":
        batch = int(_get(cfg, "scan.batch_size", 20000))
        frames, n_in, n_kept, backend_used = [], 0, 0, None
        filt: dict = {}
        for sg in subg:
            geno = load_bed_hardcall(_scan_bed_prefix(cfg, sg))
            if is_loco:
                res = scan_snps_loco(ctx, geno, backend=backend,
                                     batch_size=batch, maf_min=maf_min,
                                     call_rate_min=cr_min)
            else:
                res = scan_snps(ctx, geno, backend=backend, batch_size=batch,
                                maf_min=maf_min, call_rate_min=cr_min)
            backend_used = res.backend_used
            n_in += res.n_input
            n_kept += res.n_kept
            for k, v in res.filter_counts.items():
                filt[k] = filt.get(k, 0) + v
            frames.append(pd.DataFrame({
                "snp_id": res.snp_id, "subgenome": sg, "chrom": res.chrom,
                "pos": res.pos, "beta": res.beta, "se": res.se,
                "chi2": res.chi2, "p": res.p, "n_obs": res.n_obs,
                "maf": res.maf, "call_rate": res.call_rate}))
        df = pd.concat(frames, ignore_index=True).sort_values(
            ["subgenome", "chrom", "pos"]).reset_index(drop=True)
        ss_path = out_dir / f"sumstats_{prefix}{suffix}.tsv"
        df.to_csv(ss_path, sep="\t", index=False)
        return {"mode": "memory", "backend_used": backend_used,
                "sumstats": [str(ss_path)], "n_markers_input": n_in,
                "n_markers_kept": n_kept, "filter_counts": filt,
                "loco": is_loco}
    # streaming
    chunk = int(_get(cfg, "scan.chunk_size", 200_000))
    gzip_out = bool(_get(cfg, "scan.gzip", True))
    paths, n_in, n_kept, backend_used = [], 0, 0, None
    filt = {}
    for sg in subg:
        ext = ".tsv.gz" if gzip_out else ".tsv"
        sg_path = out_dir / f"sumstats_{prefix}{suffix}_{sg}{ext}"
        if is_loco:
            summ = scan_bed_stream_loco(
                ctx, _scan_bed_prefix(cfg, sg), sg_path,
                backend=backend, chunk_size=chunk,
                maf_min=maf_min, call_rate_min=cr_min,
                subgenome=sg, gzip_out=gzip_out)
        else:
            summ = scan_bed_stream(
                ctx, _scan_bed_prefix(cfg, sg), sg_path,
                backend=backend, chunk_size=chunk,
                maf_min=maf_min, call_rate_min=cr_min,
                subgenome=sg, gzip_out=gzip_out)
        backend_used = summ.backend_used
        n_in += summ.n_input
        n_kept += summ.n_kept
        for k, v in summ.filter_counts.items():
            filt[k] = filt.get(k, 0) + v
        paths.append(str(sg_path))
    return {"mode": "stream", "backend_used": backend_used, "sumstats": paths,
            "n_markers_input": n_in, "n_markers_kept": n_kept,
            "filter_counts": filt, "loco": is_loco}


# scan summary (bounded memory) + plots


def scan_summary(scan_out: dict, subg: list[str], *,
                 plot_target: int = 200_000, plot_keep_p: float = 1e-3):
    """One bounded-memory pass over the scan output.

    Returns ``(lambda_df, plot_df, qc)``. chi2 is streamed into a single
    preallocated float64 array (sized from ``n_markers_kept``) plus an int16
    subgenome-code array, so memory stays ~0.85 GB even at the 83M-marker wheat
    scale. The plot frame is thinned to roughly ``plot_target`` markers plus
    every marker with p <= ``plot_keep_p``; the full sumstats DataFrame is never
    materialised.
    """
    n_total = max(int(scan_out.get("n_markers_kept", 0)), 1)
    thin = max(1, n_total // max(plot_target, 1))
    sg_index = {sg: i for i, sg in enumerate(subg)}
    chi2_buf = np.empty(n_total, dtype=np.float64)
    sgcode_buf = np.empty(n_total, dtype=np.int16)
    plot_parts: list[pd.DataFrame] = []
    p_lo, p_hi, chi2_lo = np.inf, -np.inf, np.inf
    n_bad_p = n_bad_chi2 = 0
    cursor = 0

    def _ingest(ch: pd.DataFrame):
        nonlocal cursor, p_lo, p_hi, chi2_lo, n_bad_p, n_bad_chi2
        nonlocal chi2_buf, sgcode_buf
        k = len(ch)
        if k == 0:
            return
        p = ch["p"].to_numpy(dtype=np.float64)
        c = ch["chi2"].to_numpy(dtype=np.float64)
        fin_p, fin_c = np.isfinite(p), np.isfinite(c)
        n_bad_p += int((~fin_p).sum())
        n_bad_chi2 += int((~fin_c).sum())
        if fin_p.any():
            p_lo = min(p_lo, float(p[fin_p].min()))
            p_hi = max(p_hi, float(p[fin_p].max()))
        if fin_c.any():
            chi2_lo = min(chi2_lo, float(c[fin_c].min()))
        # grow the buffers only if the files hold more rows than advertised
        if cursor + k > chi2_buf.size:
            extra = cursor + k - chi2_buf.size
            chi2_buf = np.concatenate([chi2_buf, np.empty(extra, np.float64)])
            sgcode_buf = np.concatenate([sgcode_buf, np.empty(extra, np.int16)])
        chi2_buf[cursor:cursor + k] = c
        sg_arr = ch["subgenome"].to_numpy().astype(str)
        codes = np.full(k, -1, dtype=np.int16)
        for sg, i in sg_index.items():
            codes[sg_arr == sg] = i
        sgcode_buf[cursor:cursor + k] = codes
        idx = np.arange(cursor, cursor + k)
        keep = (idx % thin == 0) | (fin_p & (p <= plot_keep_p))
        if keep.any():
            plot_parts.append(
                ch.loc[keep, ["subgenome", "chrom", "pos", "p", "chi2"]].copy())
        cursor += k

    if scan_out["mode"] == "memory":
        _ingest(pd.read_csv(scan_out["sumstats"][0], sep="\t"))
    else:
        for path in scan_out["sumstats"]:
            for ch in pd.read_csv(path, sep="\t", chunksize=500_000):
                _ingest(ch)

    chi2 = chi2_buf[:cursor]
    sgc = sgcode_buf[:cursor]
    rows = [{"scope": "all", "level": "all", "n_markers": int(cursor),
             "lambda_gc": lambda_gc(chi2)}]
    for sg, i in sg_index.items():
        cc = chi2[sgc == i]
        rows.append({"scope": "subgenome", "level": sg,
                     "n_markers": int(cc.size), "lambda_gc": lambda_gc(cc)})
    lam_df = pd.DataFrame(rows)
    plot_df = (pd.concat(plot_parts, ignore_index=True) if plot_parts
               else pd.DataFrame(columns=["subgenome", "chrom", "pos",
                                          "p", "chi2"]))
    qc = {"p_ok": bool(cursor == 0 or (n_bad_p == 0
                                       and p_lo >= 0.0 and p_hi <= 1.0)),
          "chi2_ok": bool(cursor == 0 or (n_bad_chi2 == 0 and chi2_lo >= 0.0)),
          "n_markers": int(cursor), "plot_thin": int(thin),
          "n_nonfinite_p": int(n_bad_p), "n_nonfinite_chi2": int(n_bad_chi2)}
    return lam_df, plot_df, qc


# fit command


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def cmd_fit(args) -> int:
    t0 = time.time()
    cfg = load_config(args.config)
    validate_config(cfg)

    # flag overrides
    if args.out_dir:
        cfg.setdefault("outputs", {})["out_dir"] = args.out_dir
    if args.backend:
        cfg.setdefault("scan", {})["backend"] = args.backend
    out_dir = Path(_get(cfg, "outputs.out_dir", "results/homoeogwas_fit"))
    prefix = _get(cfg, "outputs.prefix") or _get(cfg, "phenotype.trait")
    subg = _get(cfg, "panel.subgenomes")
    trait = _get(cfg, "phenotype.trait")
    backend = _get(cfg, "scan.backend", "auto")

    problems = preflight(cfg)
    print(f"=== homoeogwas fit — trait={trait}  subgenomes={subg} ===", flush=True)
    print(f"  config: {args.config}")
    if cfg.get("_panel_manifest_resolved"):
        print(f"  panel_manifest: {cfg['_panel_manifest_resolved']}")
    if cfg.get("_panel_manifest_warning"):
        print(f"  WARN: {cfg['_panel_manifest_warning']}")

    if args.dry_run:
        print("  [dry-run] preflight:",
              "OK" if not problems else f"{len(problems)} problem(s)")
        for p in problems:
            print(f"    - {p}")
        return 0 if not problems else 1
    if problems:
        for p in problems:
            print(f"  ERR preflight: {p}")
        raise SystemExit("ERR: preflight failed — fix the paths above")

    # refuse a non-empty output dir without --force, so a crashed partial run
    # is not silently overwritten.
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise SystemExit(
            f"ERR: output dir {out_dir} is not empty; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    acceptance: list[dict] = []

    def check(name, ok, msg=""):
        acceptance.append({"check": name, "passed": bool(ok), "message": msg})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {msg}" if msg else ""))

    # samples + phenotype
    analysis, y, X = join_samples(cfg)
    n = len(analysis)
    print(f"\n[1] analysis set: n={n}  var(y)={np.var(y, ddof=1):.4g}")
    pd.DataFrame({"sample": analysis}).to_csv(
        out_dir / "analysis_samples.tsv", sep="\t", index=False)

    # GRM + kernels
    loco_enabled = bool(_get(cfg, "scan.loco.enabled", False))
    kernels_by_chrom: dict[str, dict[str, np.ndarray]] | None = None
    chrom_to_sub: dict[str, str] | None = None
    if loco_enabled:
        print("[2] per-subgenome GRM + per-chrom LOCO kernels")
        kernels_by_chrom, kernels, chrom_to_sub, grm_info = \
            build_loco_kernels(cfg, analysis)
        print(f"  global kernels: {list(kernels.keys())}")
        print(f"  LOCO chrom count: {len(kernels_by_chrom)}")
        check("loco_chrom_to_subgenome_unique",
              all(c in chrom_to_sub for c in kernels_by_chrom),
              f"{len(chrom_to_sub)} chrom routed to subgenomes")
    else:
        print("[2] per-subgenome GRM + kernels")
        kernels, grm_info = build_kernels(cfg, analysis)
        print(f"  kernels: {list(kernels.keys())}")

    # multi-kernel REML (global variance components; LOCO does not refit per chrom)
    print("[3] multi-kernel REML")
    n_starts = int(_get(cfg, "reml.n_starts", 10))
    seed = int(_get(cfg, "reml.seed", 2026))
    reml = fit_multi_reml(y, X, kernels, n_starts=n_starts, random_state=seed)
    print(f"  σ²: { {k: round(v, 4) for k, v in reml.sigma2.items()} }")
    print(f"  PVE: { {k: round(v, 4) for k, v in reml.pve.items()} }")
    check("reml_converged", bool(reml.optimizer_status))
    check("reml_pve_sums_to_1", abs(sum(reml.pve.values()) - 1.0) < 1e-6)

    # scan context
    if loco_enabled:
        assert kernels_by_chrom is not None
        ctx = build_loco_scan_contexts(
            y, X, kernels_by_chrom, reml.sigma2, sample_ids=analysis)
        jitter_vals = list(ctx.jitter_by_chrom.values())
        max_jitter = float(max(jitter_vals)) if jitter_vals else 0.0
        # checks each per-chrom V_(-c) Cholesky succeeds without jitter; K_(-c)
        # is PSD by construction, so a stricter min-eigenvalue gate is redundant.
        check("loco_all_chrom_V_pd",
              max_jitter < 1e-6,
              f"max per-chrom V Cholesky jitter {max_jitter:.3g}")
        # warn on chroms with thin retained denominator (statistical, not
        # numerical -- degeneracy would already have raised in loco_grm_from_parts)
        low_risk = [c for c, info in grm_info["loco"].items()
                    if info.get("retained_fraction_low_risk", False)]
        check("loco_all_chrom_retained_above_warn",
              len(low_risk) == 0,
              f"{len(low_risk)} chrom with denom_retained_fraction < 1%"
              + (f" (e.g. {low_risk[:3]})" if low_risk else ""))
    else:
        ctx = build_scan_context(y, X, kernels, reml.sigma2,
                                 sample_ids=analysis)

    # scan
    scan_mode, mode_info = resolve_scan_mode(cfg, subg, n)
    print(f"[4] per-SNP scan  mode={scan_mode}  backend={backend}")
    scan_out = run_scan(cfg, ctx, subg, backend, out_dir, prefix, scan_mode)
    print(f"  {scan_out['n_markers_input']} markers -> "
          f"{scan_out['n_markers_kept']} kept  backend={scan_out['backend_used']}")
    check("scan_has_markers", scan_out["n_markers_kept"] > 0)

    # lambda + plots (bounded-memory pass over scan output)
    lam_df, plot_df, qc = scan_summary(scan_out, subg)
    check("p_in_unit_interval", qc["p_ok"])
    check("chi2_nonneg", qc["chi2_ok"])
    lam_df.to_csv(out_dir / f"lambda_gc_{prefix}.tsv", sep="\t", index=False)
    lam_all = float(lam_df.loc[lam_df["scope"] == "all", "lambda_gc"].iloc[0])
    print(f"[5] λ_GC overall = {lam_all:.4f}"
          + (f"  (plot thinned 1/{qc['plot_thin']})" if qc["plot_thin"] > 1
             else ""))
    check("lambda_gc_finite", np.isfinite(lam_all), f"λ_GC={lam_all:.4f}")

    plot_paths = []
    if _get(cfg, "plots.enabled", True):
        from .plots import DEFAULT_FORMATS, make_all_plots
        fmts = _get(cfg, "plots.formats", list(DEFAULT_FORMATS))
        plot_paths = make_all_plots(
            plot_df, subg,
            sigma2=reml.sigma2, pve=reml.pve,
            boundary_components=reml.boundary_components,
            lambda_df=lam_df, trait=trait, out_dir=out_dir, prefix=prefix,
            formats=fmts, dpi=int(_get(cfg, "plots.dpi", 300)),
            top_n=int(_get(cfg, "plots.top_n", 5)))
        for pth in plot_paths:
            print(f"  wrote {pth}")

    # resolved config + summary
    yaml = _require_yaml()
    clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(out_dir / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(clean_cfg, fh, sort_keys=False, allow_unicode=True)

    all_passed = all(c["passed"] for c in acceptance)
    loco_block: dict | None = None
    if loco_enabled:
        loco_block = {
            "enabled": True,
            "fallback": _get(cfg, "scan.loco.fallback", "error"),
            "min_denominator_fraction": float(
                _get(cfg, "scan.loco.min_denominator_fraction", 1.0e-6)),
            "chrom_count": len(ctx.contexts),
            "chrom_to_subgenome": chrom_to_sub,
            "jitter_by_chrom": dict(ctx.jitter_by_chrom),
            "max_jitter": float(max(ctx.jitter_by_chrom.values()) or 0.0)
            if ctx.jitter_by_chrom else 0.0,
            "loco_chrom_provenance": grm_info.get("loco", {}),
        }
    summary = {
        "tool": "homoeogwas", "command": "fit", "version": __version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_sec": round(time.time() - t0, 1),
        "config": str(args.config),
        "panel": _get(cfg, "panel.name"), "trait": trait,
        "subgenomes": subg, "n_analysis": n,
        "grm_info": grm_info,
        "kernel_names": list(kernels.keys()),
        "include_hadamard": bool(_get(cfg, "kernels.include_hadamard", False)),
        "reml": {"sigma2": reml.sigma2, "pve": reml.pve,
                 "log_lik": reml.log_lik,
                 "optimizer_status": bool(reml.optimizer_status),
                 "boundary_components": reml.boundary_components,
                 "n_starts": n_starts},
        "scan": {**mode_info, "backend_requested": backend,
                 "backend_used": scan_out["backend_used"],
                 "n_markers_input": scan_out["n_markers_input"],
                 "n_markers_kept": scan_out["n_markers_kept"],
                 "filter_counts": scan_out["filter_counts"],
                 "loco_enabled": loco_enabled},
        "loco": loco_block,
        "lambda_gc": {r["level"]: r["lambda_gc"]
                      for _, r in lam_df.iterrows()},
        "outputs": {"out_dir": str(out_dir), "sumstats": scan_out["sumstats"],
                    "lambda_gc_tsv": str(out_dir / f"lambda_gc_{prefix}.tsv"),
                    "plots": plot_paths,
                    "analysis_samples": str(out_dir / "analysis_samples.tsv"),
                    "resolved_config": str(out_dir / "resolved_config.yaml")},
        "acceptance": acceptance, "acceptance_all_passed": all_passed,
    }
    summary_path = out_dir / f"summary_{prefix}.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    print(f"\nwrote {summary_path}")
    n_pass = sum(c["passed"] for c in acceptance)
    print(f"acceptance: {n_pass}/{len(acceptance)} checks passed  "
          f"(runtime {summary['runtime_sec']}s)")
    if all_passed:
        print("homoeogwas fit completed")
        return 0
    failed = [c["check"] for c in acceptance if not c["passed"]]
    print(f"ERROR: homoeogwas fit — failed checks: {failed}")
    return 1


# parser / entry point


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="homoeogwas",
        description="HomoeoGWAS — allopolyploid subgenome-stratified LMM GWAS")
    ap.add_argument("--version", action="version",
                    version=f"homoeogwas {__version__}")
    sub = ap.add_subparsers(dest="subcommand", required=True)

    fit = sub.add_parser("fit", help="run the end-to-end subgenome-stratified "
                                     "LMM GWAS from a YAML config")
    fit.add_argument("-c", "--config", required=True,
                     help="YAML run-config path")
    fit.add_argument("-o", "--out-dir", default=None,
                     help="override outputs.out_dir")
    fit.add_argument("--backend", default=None, choices=["auto", "cpu", "gpu"],
                     help="override scan.backend")
    fit.add_argument("--dry-run", action="store_true",
                     help="validate config + input paths, then exit")
    fit.add_argument("--force", action="store_true",
                     help="overwrite a non-empty output directory")

    # generalized subgenome split driven by a species YAML
    from .species_split import add_split_subparser
    add_split_subparser(sub)

    # homoeolog-pair burden-product interaction scan
    from .interact import add_interact_subparser
    add_interact_subparser(sub)

    # build the inputs that `interact` consumes (snp_to_gene NPZ + triad TSV)
    from .prep import add_prep_subparsers
    add_prep_subparsers(sub)

    # agent-facing MCP server (optional 'mcp' dependency)
    sub.add_parser("mcp", help="run the MCP server so any agent/LLM client can "
                               "drive HomoeoGWAS (needs: pip install homoeogwas[mcp])")

    val = sub.add_parser("validate", help="load + validate a run config and "
                                          "check input paths, without running")
    val.add_argument("-c", "--config", required=True, help="YAML run-config path")

    dem = sub.add_parser("demo", help="generate a tiny synthetic dataset and run "
                                      "an end-to-end fit (install self-test)")
    dem.add_argument("-o", "--out", default="demo_run",
                     help="output directory for the demo dataset + fit (default: demo_run)")
    dem.add_argument("--keep", action="store_true",
                     help="keep the generated demo dataset after the fit")

    plt = sub.add_parser("plot", help="regenerate publication figures from a "
                                      "finished run directory (no model refit)")
    plt.add_argument("results_dir", help="a finished homoeogwas fit output dir")
    plt.add_argument("--prefix", default=None,
                     help="select summary_<prefix>.json (needed if several)")
    plt.add_argument("--formats", default="png,pdf,svg",
                     help="comma list of output formats (default png,pdf,svg)")
    plt.add_argument("--dpi", type=int, default=300, help="PNG raster dpi")
    plt.add_argument("--top-n", type=int, default=5,
                     help="Manhattan hits to label per subgenome (0 disables)")
    plt.add_argument("--max-points", type=int, default=300_000,
                     help="thin background markers to ~this many for plotting")
    plt.add_argument("--figures", default="all",
                     help="comma subset of pve,manhattan,qq,lambda (or all)")
    plt.add_argument("--no-update-summary", action="store_true",
                     help="do not rewrite summary.json outputs.plots")

    loc = sub.add_parser("locus", help="draw LocusZoom-style regional plots "
                                       "(per-hit zoom) from a finished run")
    loc.add_argument("results_dir", help="a finished homoeogwas fit output dir")
    loc.add_argument("--prefix", default=None,
                     help="select summary_<prefix>.json (needed if several)")
    loc.add_argument("--lead-snp", default=None,
                     help="centre on this SNP id (else --chrom/--pos or top hit)")
    loc.add_argument("--chrom", default=None, help="centre chromosome")
    loc.add_argument("--pos", type=int, default=None, help="centre bp position")
    loc.add_argument("--subgenome", default=None,
                     help="restrict to this subgenome")
    loc.add_argument("--window-kb", type=int, default=500,
                     help="half-window in kb around the centre (default 500)")
    loc.add_argument("--top-n", type=int, default=3,
                     help="number of top independent hits to plot (default 3)")
    loc.add_argument("--genotype", default=None,
                     help="PLINK bed prefix for r^2-to-lead colouring "
                          "(else auto-resolved from the run config; else "
                          "flat colour)")
    loc.add_argument("--genes", default=None,
                     help="gene table (.tsv/.csv) or .gff/.gff3/.gtf for a "
                          "gene-model track")
    loc.add_argument("--formats", default="png,pdf,svg",
                     help="comma list of output formats (default png,pdf,svg)")
    loc.add_argument("--dpi", type=int, default=300, help="PNG raster dpi")
    loc.add_argument("--out-dir", default=None,
                     help="output dir (default: the results dir)")
    return ap


def cmd_plot(args) -> int:
    """Regenerate figures from a finished run directory without refitting."""
    from .plots import ALL_FIGURES, plot_from_results
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    figures = (ALL_FIGURES if args.figures.strip() == "all"
               else tuple(f.strip() for f in args.figures.split(",")
                          if f.strip()))
    paths = plot_from_results(
        Path(args.results_dir), prefix=args.prefix, formats=formats,
        dpi=args.dpi, top_n=args.top_n, max_points=args.max_points,
        figures=figures, update_summary=not args.no_update_summary)
    if not paths:
        print(f"[plot] no figures written from {args.results_dir}")
        return 1
    for p in paths:
        print(f"  wrote {p}")
    print(f"[plot] {len(paths)} files written")
    return 0


def _resolve_genotype_template(results_dir: Path, prefix: str | None) -> str | None:
    """Read a finished run's config to get its scan_bed_prefix_template.

    Returns the ``{subgenome}`` template (e.g. ``data/processed/cotton/{subgenome}/all``)
    so locus plots can find the per-subgenome bed for r^2 colouring, or ``None``
    if the config or key is unavailable (locus plots then fall back to flat).
    """
    import json
    results_dir = Path(results_dir)
    try:
        from .plots import _discover_summary
        sp = _discover_summary(results_dir, prefix)
        summary = json.loads(sp.read_text())
        cfg_path = summary.get("config")
        if not cfg_path:
            return None
        name = Path(cfg_path).name
        cands = [Path(cfg_path), results_dir / cfg_path, results_dir / name,
                 results_dir / "configs" / name]
        # best-effort: the config tree may have moved since the run
        for root in (Path("configs"), Path("reproducibility")):
            if root.is_dir():
                cands.extend(sorted(root.rglob(name))[:1])
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            return None
        cfg = load_config(p)
        return cfg.get("genotype", {}).get("scan_bed_prefix_template")
    except Exception:
        return None


def cmd_locus(args) -> int:
    """Draw LocusZoom-style regional plots from a finished run directory."""
    from .plots import plot_loci_from_results
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    gtmpl = (None if args.genotype
             else _resolve_genotype_template(Path(args.results_dir), args.prefix))
    paths = plot_loci_from_results(
        Path(args.results_dir), prefix=args.prefix, top_n=args.top_n,
        window_kb=args.window_kb, lead_snp=args.lead_snp, chrom=args.chrom,
        pos=args.pos, subgenome=args.subgenome, genotype_template=gtmpl,
        genotype=args.genotype, genes=args.genes, formats=formats,
        dpi=args.dpi, out_dir=args.out_dir)
    if not paths:
        print(f"[locus] no locus figures written from {args.results_dir}")
        return 1
    for p in paths:
        print(f"  wrote {p}")
    print(f"[locus] {len(paths)} files written")
    return 0


def cmd_validate(args) -> int:
    """Load + validate a config and run path preflight; report, don't compute."""
    cfg = load_config(args.config)
    validate_config(cfg)
    print(f"[validate] schema OK: {args.config}")
    if cfg.get("_panel_manifest_resolved"):
        print(f"  panel_manifest: {cfg['_panel_manifest_resolved']}")
    problems = preflight(cfg)
    if not problems:
        print("[validate] input preflight: OK")
        return 0
    print(f"[validate] input preflight: {len(problems)} problem(s):")
    for p in problems:
        print(f"  - {p}")
    return 1


def cmd_demo(args) -> int:
    """Self-test: synthesise a tiny dataset, run a full fit, list outputs."""
    import shutil

    from .demo_data import make_demo
    out = Path(args.out)
    print(f"[demo] generating synthetic allotetraploid dataset in {out} ...")
    cfg_path = make_demo(out)
    fit_args = argparse.Namespace(config=str(cfg_path), out_dir=str(out / "demo_out"),
                                  backend="cpu", dry_run=False, force=True)
    print("[demo] running homoeogwas fit ...")
    rc = cmd_fit(fit_args)
    if rc == 0:
        print(f"\n[demo] SUCCESS — outputs in {out / 'demo_out'}:")
        for f in sorted((out / "demo_out").glob("*")):
            print(f"    {f.name}")
        print("[demo] install verified end-to-end.")
    if not args.keep and rc == 0:
        shutil.rmtree(out, ignore_errors=True)
        print(f"[demo] cleaned up {out} (use --keep to retain).")
    return rc


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.subcommand == "fit":
        return cmd_fit(args)
    if args.subcommand == "split":
        from .species_split import cmd_split
        return cmd_split(args)
    if args.subcommand == "interact":
        from .interact import cmd_interact
        return cmd_interact(args)
    if args.subcommand == "validate":
        return cmd_validate(args)
    if args.subcommand == "demo":
        return cmd_demo(args)
    if args.subcommand == "plot":
        return cmd_plot(args)
    if args.subcommand == "locus":
        return cmd_locus(args)
    if args.subcommand == "prep-snps":
        from .prep import cmd_prep_snps
        return cmd_prep_snps(args)
    if args.subcommand == "prep-homoeologs":
        from .prep import cmd_prep_homoeologs
        return cmd_prep_homoeologs(args)
    if args.subcommand == "mcp":
        from .mcp_server import main as mcp_main
        return mcp_main()
    return 1


if __name__ == "__main__":
    sys.exit(main())
