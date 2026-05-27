"""Genomic Relationship Matrices (GRM) for HomoeoGWAS.

最小实现:VanRaden (2008) GRM,per-subgenome 计算.

VanRaden GRM:
    Z = X - 2p          (centered dosage, X ∈ {0,1,2}, p = counted allele freq)
    G = Z Z' / Σ 2p(1-p)

Properties:
- trace(G) ≈ n_samples (常数)
- diagonal = sample inbreeding * (relative to assumed HWE)
- works on any standard biallelic SNP matrix

LOCO support (Phase 3, M3.1):
    ``compute_grm_parts`` returns the unnormalised numerator (Z Z') and
    denominator (Σ 2pq) as a ``GRMPart``; ``compute_loco_grm_parts`` does the
    same per ``chrom`` of the input chunk in one pass; ``loco_grm_from_parts``
    constructs the leave-one-chromosome-out raw GRM by subtraction
    (algebraically identical to ``sum_other_chrom_parts`` in exact arithmetic,
    with a build-time checksum that catches any chrom-dependent QC drift).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .io import GenoChunk


def _impute_mean(dosage: np.ndarray) -> np.ndarray:
    """Impute NaN dosage with per-variant mean. Returns new array."""
    X = dosage.copy()
    # 用 nanmean (按列, axis=0) 算每个 variant 的 mean
    col_means = np.nanmean(X, axis=0)
    # broadcast 替换 NaN
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X


# ---------------------------------------------------------------------
# LOCO building blocks (Phase 3 — M3.1)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class GRMPart:
    """Unnormalised VanRaden GRM building-block.

    Holds ``numerator = Z Z'`` (centred, no division) and
    ``denominator = Σ 2p(1-p)`` separately so leave-one-out variants can be
    built by subtraction without ever materialising a normalised partial GRM.
    The canonical GRM is ``numerator / denominator`` (``.grm`` property).
    """
    numerator: np.ndarray          # (n,n) Z Z'
    denominator: float              # Σ 2p(1-p)
    n_variants_input: int
    n_variants_used: int
    chrom: str | None = None       # None for a whole-chunk/global part
    info: dict = field(default_factory=dict)

    @property
    def grm(self) -> np.ndarray:
        """The canonical normalised VanRaden GRM."""
        if self.denominator <= 0.0 or self.n_variants_used == 0:
            raise ValueError(
                f"GRMPart has no informative variants "
                f"(chrom={self.chrom!r}, denominator={self.denominator})")
        return self.numerator / self.denominator


def _compute_grm_part_from_dosage(
    X: np.ndarray, maf_min: float, *, chrom: str | None, allow_empty: bool,
) -> GRMPart:
    """Shared core: NaN-impute → low-MAF drop → Z Z' numerator + Σ 2pq denom.

    Returns a zero-filled ``GRMPart`` when ``allow_empty`` and the (sub-)chunk
    contributes nothing — needed so a chrom block with no surviving variants
    in some subgenome does not break the LOCO accumulator.
    """
    n, m = X.shape
    zero = GRMPart(
        numerator=np.zeros((n, n), dtype=np.float64),
        denominator=0.0,
        n_variants_input=int(m),
        n_variants_used=0,
        chrom=chrom,
    )
    if m == 0:
        if allow_empty:
            return zero
        raise ValueError(
            f"Cannot compute GRM with zero variants (chrom={chrom!r})")
    X = X.copy()
    col_means = np.nanmean(X, axis=0)
    nan_inds = np.where(np.isnan(X))
    X[nan_inds] = np.take(col_means, nan_inds[1])
    p = X.sum(axis=0) / (2.0 * n)
    maf = np.minimum(p, 1.0 - p)
    keep = maf >= maf_min
    X = X[:, keep]
    p = p[keep]
    m_used = X.shape[1]
    if m_used == 0 or not np.all(np.isfinite(p)):
        if allow_empty:
            return zero
        raise ValueError(
            f"Cannot compute GRM: 0 informative variants "
            f"(chrom={chrom!r}, m_input={m}, maf_min={maf_min})")
    Z = X - 2.0 * p[np.newaxis, :]
    denom = float(np.sum(2.0 * p * (1.0 - p)))
    if not np.isfinite(denom) or denom <= 0.0:
        if allow_empty:
            return zero
        raise ValueError(
            f"Cannot compute GRM with non-positive denominator "
            f"(chrom={chrom!r}, denom={denom})")
    S = Z @ Z.T
    S = 0.5 * (S + S.T)                                       # enforce symmetry
    return GRMPart(
        numerator=S.astype(np.float64),
        denominator=float(denom),
        n_variants_input=int(m),
        n_variants_used=int(m_used),
        chrom=chrom,
    )


def compute_grm_parts(
    chunk: GenoChunk, maf_min: float = 0.01, *, allow_empty: bool = False,
) -> GRMPart:
    """Compute the unnormalised GRM (numerator + denominator) for a chunk.

    Mathematically identical to ``compute_grm`` modulo the
    ``.numerator / .denominator`` division, but with a richer return type so
    LOCO callers can build leave-one-out variants without recomputing.
    """
    X = np.asarray(chunk.dosage, dtype=np.float64)
    return _compute_grm_part_from_dosage(
        X, maf_min, chrom=None, allow_empty=allow_empty)


def compute_loco_grm_parts(
    chunk: GenoChunk, maf_min: float = 0.01,
    *,
    checksum_atol: float = 1e-8,
) -> tuple[GRMPart, dict[str, GRMPart]]:
    """Compute global GRM part + per-chrom partials in one chunk pass.

    Args:
        chunk: GenoChunk; ``chunk.chrom`` (m,) names the chromosome of each
            variant (string-castable).
        maf_min: per-variant MAF filter, applied identically at the chrom-block
            level and at the global level. With a per-variant filter this is
            chrom-independent, so Σ_c m_used_c == m_used_global by
            construction; the checksum verifies that invariant.
        checksum_atol: max allowed relative error
            ``|Σ_c denom_c - denom_global| / denom_global``. PSD-sum subtraction
            in float64 is bounded by ~ε·m·var, so 1e-8 leaves >7 decades of
            headroom.

    Returns:
        (global_part, parts_by_chrom). ``parts_by_chrom`` is ordered by the
        first appearance of each chrom in ``chunk.chrom`` (preserved via
        ``dict.fromkeys``). Useful when ``chunk.chrom`` is already sorted —
        the typical plink2 layout.
    """
    if chunk.dosage.ndim != 2:
        raise ValueError(f"chunk.dosage must be 2D, got {chunk.dosage.shape}")
    n, m = chunk.dosage.shape
    if m == 0:
        raise ValueError("chunk has zero variants — nothing to compute")
    if chunk.chrom is None or len(chunk.chrom) != m:
        raise ValueError(
            f"chunk.chrom must align with {m} variants, "
            f"got len={0 if chunk.chrom is None else len(chunk.chrom)}")
    chroms = np.asarray(chunk.chrom).astype(str)
    unique_chroms = list(dict.fromkeys(chroms.tolist()))       # stable order

    parts: dict[str, GRMPart] = {}
    for c in unique_chroms:
        mask = (chroms == c)
        sub = np.asarray(chunk.dosage[:, mask], dtype=np.float64)
        parts[c] = _compute_grm_part_from_dosage(
            sub, maf_min, chrom=c, allow_empty=True)

    global_part = _compute_grm_part_from_dosage(
        np.asarray(chunk.dosage, dtype=np.float64),
        maf_min, chrom=None, allow_empty=False)

    # Consistency: per-variant MAF filter is chrom-blind, so Σ m_used_c must
    # match m_used_global exactly; denominators must agree within float-eps.
    sum_used = sum(p.n_variants_used for p in parts.values())
    if global_part.n_variants_used != sum_used:
        raise RuntimeError(
            f"LOCO checksum: global m_used={global_part.n_variants_used} != "
            f"Σ chrom m_used={sum_used} (per-variant MAF filter went out of sync)")
    sum_denom = sum(p.denominator for p in parts.values())
    if global_part.denominator > 0:
        rel = abs(sum_denom - global_part.denominator) / global_part.denominator
        if rel > checksum_atol:
            raise RuntimeError(
                f"LOCO checksum: Σ denom_c={sum_denom:.6g} vs "
                f"global={global_part.denominator:.6g} (rel err {rel:.3g}, "
                f"tol={checksum_atol:.3g})")
    return global_part, parts


_LOCO_RETAINED_FRACTION_WARN = 0.01     # warn if leave-one-out keeps < 1 %


def loco_grm_from_parts(
    global_part: GRMPart,
    parts_by_chrom: dict[str, GRMPart],
    target_chrom: str,
    *,
    min_denominator_fraction: float = 1e-6,
    warn_retained_fraction: float = _LOCO_RETAINED_FRACTION_WARN,
) -> tuple[np.ndarray, dict]:
    """Construct the leave-one-chromosome-out **raw** GRM via subtraction.

    .. math:: K_j(-c) = \\frac{Z_j Z_j' - Z_{j,c} Z_{j,c}'}
                              {\\mathrm{denom}_j - \\mathrm{denom}_{j,c}}

    Algebraically identical to ``sum_other_chrom_parts``. Float64 cancellation
    on PSD sums is bounded by ~ε·n·||S||, which for n≤10⁴ is ~1e-11 relative —
    far below the REML PSD jitter ladder. The build-time checksum in
    :func:`compute_loco_grm_parts` already verifies that the parts add up
    cleanly, so subtraction is safe and O(1) per chrom (~20× faster than
    summing all other chrom parts for a 21-chrom panel).

    Args:
        global_part: ``GRMPart`` for the full subgenome (all chrom).
        parts_by_chrom: ``{chrom -> GRMPart}`` from
            :func:`compute_loco_grm_parts`.
        target_chrom: chrom to leave out.
        min_denominator_fraction: hard floor on the **retained** denominator
            fraction (denom_retained / denom_global). Below this the LOCO
            GRM is dominated by the missing chrom; raise rather than build a
            degenerate kernel. ``1e-6`` is a numerical-stability floor; the
            statistical-quality threshold is the looser
            ``warn_retained_fraction``.
        warn_retained_fraction: emit ``info["retained_fraction_low_risk"]
            = True`` if the leave-one-out denominator keeps less than this
            fraction of the global one. Default ``0.01`` (1 %). Use to
            surface chroms whose LOCO kernel is statistically thin but not
            mathematically degenerate (callers can downgrade them at
            acceptance time).

    Returns:
        ``(K_raw, info)`` — ``K_raw`` is the *raw* (un-renormalised) VanRaden
        GRM for the leave-one-out marker set; ``info`` records both the
        **retained** quantities (used to build K) and the **removed**
        quantities (the chrom-c contribution), plus a low-retained flag.
    """
    if target_chrom not in parts_by_chrom:
        keys = list(parts_by_chrom.keys())
        preview = keys[:5] + (["..."] if len(keys) > 5 else [])
        raise KeyError(
            f"LOCO: target chrom {target_chrom!r} not in parts {preview}")
    part_c = parts_by_chrom[target_chrom]
    if global_part.denominator <= 0:
        raise ValueError("LOCO: global denominator is zero (no informative variants)")
    denom_retained = global_part.denominator - part_c.denominator
    retained_frac = denom_retained / global_part.denominator
    if retained_frac < min_denominator_fraction:
        raise ValueError(
            f"LOCO chrom {target_chrom!r}: leave-one-out denominator "
            f"{denom_retained:.4g} retains only {retained_frac:.3g} of global "
            f"{global_part.denominator:.4g} (< min_denominator_fraction="
            f"{min_denominator_fraction:g}); chrom dominates the GRM — "
            "insufficient diversity outside it")
    num_retained = global_part.numerator - part_c.numerator
    num_retained = 0.5 * (num_retained + num_retained.T)        # re-symmetrise
    K = num_retained / denom_retained
    info = {
        "target_chrom": target_chrom,
        "denom_retained": float(denom_retained),
        "denom_retained_fraction": float(retained_frac),
        "n_variants_retained": int(global_part.n_variants_used
                                    - part_c.n_variants_used),
        "denom_removed": float(part_c.denominator),
        "denom_removed_fraction": float(
            part_c.denominator / global_part.denominator),
        "n_variants_removed": int(part_c.n_variants_used),
        "n_variants_global": int(global_part.n_variants_used),
        "retained_fraction_low_risk": bool(retained_frac < warn_retained_fraction),
    }
    return K, info


def compute_grm(chunk: GenoChunk, maf_min: float = 0.01) -> tuple[np.ndarray, dict]:
    """Compute VanRaden GRM from a GenoChunk.

    Args:
        chunk: GenoChunk with dosage shape (n, m), values in {0,1,2}∪{NaN}.
        maf_min: drop variants with MAF < maf_min (already QC'd upstream,
            this is defensive).

    Returns:
        (G, info):
        - G: shape (n, n), float64, trace(G) ≈ n
        - info: dict with n_samples, n_variants_used, mean_diag, trace

    Notes:
        - NaN imputed by per-variant mean (standard practice when missingness is low).
        - Σ 2p(1-p) 用 imputed-recomputed allele freq.
        - Internally delegates to :func:`compute_grm_parts` for a single,
          float64-throughout code path shared with the LOCO building blocks.
    """
    n, m = chunk.dosage.shape
    part = compute_grm_parts(chunk, maf_min=maf_min)
    if part.n_variants_used == 0 or part.denominator == 0.0:
        raise ValueError(
            "Cannot compute GRM with no informative variants "
            f"(input_shape={chunk.dosage.shape}, m_used={part.n_variants_used}, "
            f"denominator_sum_2pq={part.denominator})"
        )
    G = part.grm

    info = {
        "n_samples": int(n),
        "n_variants_input": int(m),
        "n_variants_used": int(part.n_variants_used),
        "n_variants_dropped_maf": int(m - part.n_variants_used),
        "denominator_sum_2pq": float(part.denominator),
        "trace": float(np.trace(G)),
        "mean_diag": float(np.mean(np.diag(G))),
        "min_offdiag": float(np.min(G - np.diag(np.diag(G)))),
        "max_offdiag": float(np.max(G - np.diag(np.diag(G)))),
    }
    return G, info


def _assert_unique_sample_iids(samples: np.ndarray, subgenome: str) -> None:
    """Require unique IID keys before sample-to-index mapping."""
    seen: set[object] = set()
    duplicates: list[object] = []
    duplicate_seen: set[object] = set()
    for sample in samples:
        if sample in seen:
            if sample not in duplicate_seen:
                duplicates.append(sample)
                duplicate_seen.add(sample)
        else:
            seen.add(sample)
    if duplicates:
        preview = ", ".join(str(s) for s in duplicates[:5])
        extra = "" if len(duplicates) <= 5 else f", ... ({len(duplicates)} total)"
        raise ValueError(
            f"chunk.samples must be unique for subgenome {subgenome}; "
            f"duplicate IID(s): {preview}{extra}"
        )


def compute_grm_panel(
    panel_dir: str | Path,
    subgenomes: list[str],
    maf_min: float = 0.01,
    intersect_samples: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, dict], np.ndarray]:
    """Compute one GRM per subgenome from a panel's processed dir.

    Expects layout:
        <panel_dir>/<SUB>/all.{bed,bim,fam}
    或 if .bed missing, vcf_to_bed will convert from all.vcf.gz.

    Args:
        panel_dir: e.g. data/processed/rapeseed/horvath
        subgenomes: e.g. ["A", "C"] or ["A", "B", "D"]
        maf_min: per-subgenome MAF filter for GRM construction
        intersect_samples: if True, restrict to samples present in ALL subgenomes

    Returns:
        (grms, infos, common_samples):
        - grms: {sub -> G (n, n)}
        - infos: {sub -> info dict}
        - common_samples: np.ndarray of sample IIDs (length n)
    """
    from .io import load_bed_hardcall, vcf_to_bed

    panel_dir = Path(panel_dir)
    chunks: dict[str, GenoChunk] = {}
    for sub in subgenomes:
        sub_dir = panel_dir / sub
        bed = sub_dir / "all.bed"
        if not bed.exists():
            vcf_gz = sub_dir / "all.vcf.gz"
            assert vcf_gz.exists(), f"missing both .bed and .vcf.gz in {sub_dir}"
            vcf_to_bed(vcf_gz, sub_dir / "all")
        chunk = load_bed_hardcall(sub_dir / "all")
        _assert_unique_sample_iids(chunk.samples, sub)
        chunks[sub] = chunk

    # Find common sample set (in order of first subgenome)
    samples_first = chunks[subgenomes[0]].samples
    if intersect_samples:
        common_set = set(samples_first)
        for sub in subgenomes[1:]:
            common_set &= set(chunks[sub].samples)
        # Preserve order from first subgenome
        common_samples = np.asarray(
            [s for s in samples_first if s in common_set], dtype=object
        )
    else:
        common_samples = samples_first

    # Compute GRM per subgenome on common samples
    grms = {}
    infos = {}
    for sub in subgenomes:
        chunk = chunks[sub]
        # Map chunk.samples -> indices matching common_samples
        sample_to_idx = {s: i for i, s in enumerate(chunk.samples)}
        idx = np.asarray([sample_to_idx[s] for s in common_samples], dtype=np.int64)
        sub_chunk = GenoChunk(
            samples=common_samples,
            variant_ids=chunk.variant_ids,
            chrom=chunk.chrom,
            pos=chunk.pos,
            dosage=chunk.dosage[idx, :],
        )
        G, info = compute_grm(sub_chunk, maf_min=maf_min)
        grms[sub] = G
        infos[sub] = info

    return grms, infos, common_samples
