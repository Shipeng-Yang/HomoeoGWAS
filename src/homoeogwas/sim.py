"""Simulation benchmark utilities for HomoeoGWAS (Phase 2 M2.6).

This module holds the *pure* (no external binaries, no torch) building blocks
for the per-subgenome causal-SNP simulation benchmark:

  * phenotype simulation under a subgenome-stratified random-effect model,
  * causal-SNP truth windows (physical + LD-tagging),
  * power / FDR evaluation with discovery-locus collapsing,
  * paired bootstrap for "HomoeoGWAS beats baseline" significance.

Generative model (replicate r, arm a)
-------------------------------------
    y = Xβ + η_A + η_C + u_bg + ε

    η_s   = Σ_{j∈causal_s} z_sj b_sj         per-subgenome oligogenic QTL signal
    u_bg  = polygenic background, one of:
              stratified : u_A ~ N(0,σ²_A K_A) + u_C ~ N(0,σ²_C K_C)
              pooled     : u   ~ N(0,σ²  K_sum)
              hadamard   : stratified + u_hom ~ N(0,σ²_hom K_hom)
              null       : same background as stratified, but η_A=η_C=0
    ε     ~ N(0,σ²_e I)

Each component is drawn then *rescaled* so its realized sample variance equals
the target heritability fraction — so realized PVE is exact per replicate. The
benchmark's scientific claim: when the true background has unequal subgenome
variance (σ²_A ≠ σ²_C), a subgenome-stratified multi-kernel LMM has higher
power at matched empirical FDR than single pooled-GRM methods. The pooled /
balanced / null arms are honesty controls where the stratified model must NOT
inflate FDR or lose power.

Everything here is unit-testable with toy arrays — no Horvath data, no GEMMA.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np

# =====================================================================
# Arm / causal-set / phenotype dataclasses
# =====================================================================


@dataclass(frozen=True)
class SimArm:
    """One simulation arm = a fixed variance-component budget.

    Heritability fractions are on the unit-variance phenotype scale and must
    sum to ≤ 1; the remainder is the residual variance ``h_e``.
    """
    name: str
    kind: str                 # "stratified" | "pooled" | "hadamard" | "null"
    h_qtl_A: float
    h_qtl_C: float
    h_poly_A: float = 0.0
    h_poly_C: float = 0.0
    h_poly_pooled: float = 0.0
    h_poly_hom: float = 0.0
    description: str = ""

    _KINDS = ("stratified", "pooled", "hadamard", "null")

    def __post_init__(self):
        if self.kind not in self._KINDS:
            raise ValueError(f"arm {self.name!r}: kind must be one of {self._KINDS}")
        fracs = (self.h_qtl_A, self.h_qtl_C, self.h_poly_A, self.h_poly_C,
                 self.h_poly_pooled, self.h_poly_hom)
        for f in fracs:
            if not np.isfinite(f) or f < 0:
                raise ValueError(f"arm {self.name!r}: heritability fractions must "
                                 f"be finite and ≥ 0; got {fracs}")
        if sum(fracs) > 1.0 + 1e-9:
            raise ValueError(f"arm {self.name!r}: heritability fractions sum to "
                             f"{sum(fracs):.4f} > 1")
        if self.kind == "pooled" and (self.h_poly_A > 0 or self.h_poly_C > 0):
            raise ValueError(f"arm {self.name!r}: pooled arm must not set "
                             "h_poly_A / h_poly_C (use h_poly_pooled)")
        if self.kind != "pooled" and self.h_poly_pooled > 0:
            raise ValueError(f"arm {self.name!r}: only the pooled arm may set "
                             "h_poly_pooled")
        if self.kind != "hadamard" and self.h_poly_hom > 0:
            raise ValueError(f"arm {self.name!r}: only the hadamard arm may set "
                             "h_poly_hom")
        if self.kind == "null" and (self.h_qtl_A > 0 or self.h_qtl_C > 0):
            raise ValueError(f"arm {self.name!r}: null arm must have h_qtl == 0")

    @property
    def h_e(self) -> float:
        """Residual variance fraction."""
        return 1.0 - (self.h_qtl_A + self.h_qtl_C + self.h_poly_A
                      + self.h_poly_C + self.h_poly_pooled + self.h_poly_hom)


def default_arms() -> dict[str, SimArm]:
    """The canonical M2.6 benchmark arms (see docs/09_phase2_progress.md).

    Pre-registered parameters (fixed before the production run by a pilot
    power calculation, not tuned to method-vs-method results): per-subgenome
    QTL heritability h_qtl = 0.18 split equally over q = 4 causal SNPs gives
    per-SNP h² ≈ 0.045. The Horvath pilot measured per-SNP χ² ≈ 400·h²_snp at
    n = 429, so this lands the median causal SNP at χ² ≈ 16 — the informative
    mid-power regime where roughly half the causal loci clear genome-wide
    significance and method curves are not at floor or ceiling.
    """
    return {
        "S1_C_dominant": SimArm(
            "S1_C_dominant", "stratified", h_qtl_A=0.18, h_qtl_C=0.18,
            h_poly_A=0.09, h_poly_C=0.33,
            description="stratified, C-dominant polygenic background"),
        "S2_A_dominant": SimArm(
            "S2_A_dominant", "stratified", h_qtl_A=0.18, h_qtl_C=0.18,
            h_poly_A=0.33, h_poly_C=0.09,
            description="stratified mirror, A-dominant polygenic background"),
        "S3_pooled": SimArm(
            "S3_pooled", "pooled", h_qtl_A=0.18, h_qtl_C=0.18,
            h_poly_pooled=0.42,
            description="pooled-truth honesty arm (single-GRM world)"),
        "S4_balanced": SimArm(
            "S4_balanced", "stratified", h_qtl_A=0.18, h_qtl_C=0.18,
            h_poly_A=0.21, h_poly_C=0.21,
            description="balanced stratified honesty arm (σ²_A = σ²_C)"),
        "S5_null": SimArm(
            "S5_null", "null", h_qtl_A=0.0, h_qtl_C=0.0,
            h_poly_A=0.09, h_poly_C=0.33,
            description="null arm — no causal SNPs, calibration check"),
        "S6_hadamard": SimArm(
            "S6_hadamard", "hadamard", h_qtl_A=0.18, h_qtl_C=0.18,
            h_poly_A=0.09, h_poly_C=0.23, h_poly_hom=0.10,
            description="homoeolog-interaction stress arm (secondary, not "
                        "an exit-gate claim)"),
    }


@dataclass(frozen=True)
class CausalSet:
    """Causal SNPs of one replicate (both subgenomes concatenated)."""
    causal_id: np.ndarray     # (q,) int, 0..q-1
    subgenome: np.ndarray     # (q,) "A"/"C"
    snp_id: np.ndarray        # (q,) original per-subgenome variant id
    chrom: np.ndarray         # (q,) original chromosome label
    pos: np.ndarray           # (q,) int64 bp
    marker_index: np.ndarray  # (q,) column index into that subgenome's matrix
    effect: np.ndarray        # (q,) signed standardized QTL effect (filled in
                              #      by simulate_phenotype; zeros until then)

    @property
    def n_causal(self) -> int:
        return int(self.causal_id.size)


@dataclass(frozen=True)
class SimPhenotype:
    """Result of one phenotype simulation."""
    y: np.ndarray                     # (n,) centered phenotype
    causal: CausalSet                 # with .effect filled in
    realized_var: dict[str, float]    # realized sample variance per component
    arm_name: str


# =====================================================================
# Genotype standardization & QC
# =====================================================================


def standardize_dosage(
    dosage_raw: np.ndarray,
    *,
    maf_min: float = 0.05,
    call_rate_min: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean-impute, center and unit-scale a {0,1,2,NaN} dosage matrix.

    Args:
        dosage_raw: (n, m) hard-call dosage with NaN for missing.
        maf_min, call_rate_min: per-SNP QC thresholds for the ``keep`` mask.

    Returns:
        (Z, keep, maf, call_rate):
          Z          (n, m) float64, column-standardized to unit variance
                     (ddof=0); monomorphic columns are all-zero.
          keep       (m,) bool — finite, call-rate, MAF and non-monomorphic.
          maf        (m,) float64 minor-allele frequency from imputed dosage.
          call_rate  (m,) float64.
    """
    G = np.asarray(dosage_raw, dtype=np.float64)
    if G.ndim != 2:
        raise ValueError(f"dosage_raw must be 2D, got {G.shape}")
    n, m = G.shape
    obs = np.isfinite(G)
    n_obs = obs.sum(axis=0)
    call_rate = n_obs / n
    with np.errstate(invalid="ignore", divide="ignore"):
        col_sum = np.where(obs, G, 0.0).sum(axis=0)
        col_mean = np.where(n_obs > 0, col_sum / np.maximum(n_obs, 1), np.nan)
    Gi = np.where(obs, G, col_mean[None, :])           # mean-imputed
    with np.errstate(invalid="ignore"):
        af = np.nanmean(Gi, axis=0) / 2.0
        sd = np.nanstd(Gi, axis=0)                      # ddof=0
    maf = np.minimum(af, 1.0 - af)
    centered = Gi - col_mean[None, :]
    Z = np.zeros((n, m), dtype=np.float64)
    nz = np.isfinite(sd) & (sd > 0)
    Z[:, nz] = centered[:, nz] / sd[None, nz]
    keep = (
        (n_obs > 0)
        & (call_rate >= call_rate_min)
        & np.isfinite(maf) & (maf >= maf_min)
        & nz
    )
    return Z, keep, maf, call_rate


# =====================================================================
# Causal-SNP selection
# =====================================================================


def select_causal_snps(
    rng: np.random.Generator,
    eligible: np.ndarray,
    chrom: np.ndarray,
    pos: np.ndarray,
    q: int,
    *,
    min_sep_bp: int = 1_000_000,
    max_tries: int = 10_000,
) -> np.ndarray:
    """Pick ``q`` causal SNP indices, spaced ≥ min_sep_bp apart on a chromosome.

    The spacing keeps each causal SNP in its own discovery neighbourhood so the
    power / FDR truth windows do not overlap ambiguously.

    Args:
        rng: numpy Generator (reproducible).
        eligible: (k,) int indices of QC-passing candidate markers.
        chrom: (m,) chromosome label per marker (full marker axis).
        pos: (m,) bp position per marker (full marker axis).
        q: number of causal SNPs to pick.
        min_sep_bp: minimum within-chromosome physical separation.
        max_tries: cap on rejection-sampling iterations.

    Returns:
        (q,) int array of chosen marker indices (sorted).
    """
    eligible = np.asarray(eligible, dtype=np.int64)
    if q < 0:
        raise ValueError(f"q must be ≥ 0, got {q}")
    if q == 0:
        return np.empty(0, dtype=np.int64)
    if eligible.size < q:
        raise ValueError(f"only {eligible.size} eligible markers for q={q}")
    chrom = np.asarray(chrom, dtype=object)
    pos = np.asarray(pos, dtype=np.int64)

    chosen: list[int] = []
    pool = eligible.copy()
    rng.shuffle(pool)
    cursor = 0
    tries = 0
    while len(chosen) < q and tries < max_tries:
        tries += 1
        if cursor >= pool.size:                       # exhausted: reshuffle
            rng.shuffle(pool)
            cursor = 0
        cand = int(pool[cursor])
        cursor += 1
        ok = True
        for c in chosen:
            if chrom[c] == chrom[cand] and abs(int(pos[cand]) - int(pos[c])) < min_sep_bp:
                ok = False
                break
        if ok:
            chosen.append(cand)
    if len(chosen) < q:
        raise ValueError(
            f"could not place {q} causal SNPs with min_sep_bp={min_sep_bp} "
            f"after {max_tries} tries (got {len(chosen)}); relax spacing")
    return np.sort(np.array(chosen, dtype=np.int64))


# =====================================================================
# Component draws & phenotype simulation
# =====================================================================


def kernel_factor(K: np.ndarray, *, eig_tol: float = 1e-8) -> np.ndarray:
    """Return L such that L @ L.T ≈ K (eigen square-root, PSD-projected).

    Used to draw u ~ N(0, K) as u = L @ z, z ~ N(0, I). Negative eigenvalues
    in [-eig_tol, 0) are clipped to 0; anything more negative raises.
    """
    K = np.asarray(K, dtype=np.float64)
    Ks = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(Ks)
    if (w < -eig_tol).any():
        raise ValueError(f"kernel not PSD: min eigval {w.min():.3g} < -{eig_tol}")
    w = np.clip(w, 0.0, None)
    return V * np.sqrt(w)[None, :]


def _draw_component(rng: np.random.Generator, factor: np.ndarray,
                    h2_target: float) -> np.ndarray:
    """Draw u = factor @ z, then rescale to realized sample variance h2_target."""
    n = factor.shape[0]
    if h2_target <= 0:
        return np.zeros(n, dtype=np.float64)
    u = factor @ rng.standard_normal(factor.shape[1])
    u = u - u.mean()
    v = float(np.var(u, ddof=1))
    if v <= 0:
        raise ValueError("degenerate component draw (zero variance)")
    return u * np.sqrt(h2_target / v)


def _draw_qtl(rng: np.random.Generator, Z_causal: np.ndarray,
              h2_target: float) -> tuple[np.ndarray, np.ndarray]:
    """Draw oligogenic QTL signal η = Z_causal @ b, rescaled to var = h2_target.

    Effects are *equal-magnitude with random sign* — the standard design for a
    power benchmark: every injected causal SNP is then an equally hard test
    (per-SNP h² ≈ h2_target / q), instead of the N(0,1) effect-size lottery
    where most causal SNPs are undetectably small. The shared magnitude is set
    by rescaling η to the exact target variance.

    Returns (eta, effects) where effects is the post-rescale signed b vector.
    """
    n, q = Z_causal.shape
    if q == 0 or h2_target <= 0:
        return np.zeros(n, dtype=np.float64), np.zeros(q, dtype=np.float64)
    b = rng.choice(np.array([-1.0, 1.0]), size=q)        # equal magnitude
    eta = Z_causal @ b
    eta = eta - eta.mean()
    v = float(np.var(eta, ddof=1))
    if v <= 0:
        raise ValueError("degenerate QTL draw (zero variance) — causal SNPs "
                          "may be collinear or monomorphic")
    scale = np.sqrt(h2_target / v)
    return eta * scale, b * scale


def simulate_phenotype(
    rng: np.random.Generator,
    arm: SimArm,
    causal: CausalSet,
    Z_causal_A: np.ndarray,
    Z_causal_C: np.ndarray,
    factors: dict[str, np.ndarray],
) -> SimPhenotype:
    """Simulate one replicate phenotype under ``arm``.

    Args:
        rng: numpy Generator.
        arm: the SimArm variance budget.
        causal: CausalSet for this replicate (effect field is overwritten).
        Z_causal_A: (n, q_A) standardized dosage of the A causal SNPs.
        Z_causal_C: (n, q_C) standardized dosage of the C causal SNPs.
        factors: kernel square-root factors, keys among
            {"A","C","pooled","hom"} as required by the arm kind.

    Each random component is drawn then rescaled to its *exact* target sample
    variance; the phenotype is the sum, then standardized to unit variance.
    Because the components are independent draws their finite-sample
    cross-covariances are small but nonzero, so the *realized* PVE of each
    component (``pve_*`` in ``realized_var``) deviates slightly from its target
    fraction — these realized values, not the nominal targets, are what the
    benchmark reports.

    Returns:
        SimPhenotype with the unit-variance phenotype, the causal set with
        realized effects, and ``realized_var`` carrying both per-component
        sample variances (``var_*``) and realized PVE fractions (``pve_*``).
    """
    n = Z_causal_A.shape[0]
    eta_A, eff_A = _draw_qtl(rng, Z_causal_A, arm.h_qtl_A)
    eta_C, eff_C = _draw_qtl(rng, Z_causal_C, arm.h_qtl_C)

    u_A = u_C = u_pool = u_hom = np.zeros(n, dtype=np.float64)
    if arm.kind in ("stratified", "hadamard", "null"):
        if arm.h_poly_A > 0:
            u_A = _draw_component(rng, factors["A"], arm.h_poly_A)
        if arm.h_poly_C > 0:
            u_C = _draw_component(rng, factors["C"], arm.h_poly_C)
    if arm.kind == "pooled" and arm.h_poly_pooled > 0:
        u_pool = _draw_component(rng, factors["pooled"], arm.h_poly_pooled)
    if arm.kind == "hadamard" and arm.h_poly_hom > 0:
        u_hom = _draw_component(rng, factors["hom"], arm.h_poly_hom)

    eps = _draw_component(rng, np.eye(n), arm.h_e) if arm.h_e > 0 \
        else np.zeros(n, dtype=np.float64)

    y = eta_A + eta_C + u_A + u_C + u_pool + u_hom + eps
    y = y - y.mean()
    total_var = float(np.var(y, ddof=1))
    if total_var <= 0:
        raise ValueError("degenerate phenotype draw (zero total variance)")
    y = y / np.sqrt(total_var)                          # unit-variance scale

    comp_var = {
        "qtl_A": float(np.var(eta_A, ddof=1)),
        "qtl_C": float(np.var(eta_C, ddof=1)),
        "poly_A": float(np.var(u_A, ddof=1)),
        "poly_C": float(np.var(u_C, ddof=1)),
        "poly_pooled": float(np.var(u_pool, ddof=1)),
        "poly_hom": float(np.var(u_hom, ddof=1)),
        "residual": float(np.var(eps, ddof=1)),
    }
    realized = {f"var_{k}": v for k, v in comp_var.items()}
    # realized PVE = component variance / total phenotype variance (pre-scale);
    # Σ pve ≈ 1, deviating by the finite-sample cross-covariance term.
    realized.update({f"pve_{k}": v / total_var for k, v in comp_var.items()})
    realized["total_var_prescale"] = total_var
    # effects divided by the same sqrt(total_var) as y, so the stored signed
    # effects are on the returned unit-variance phenotype scale.
    effect = (np.concatenate([eff_A, eff_C]) / np.sqrt(total_var)
              if causal.n_causal else np.zeros(0, dtype=np.float64))
    causal_out = CausalSet(
        causal_id=causal.causal_id, subgenome=causal.subgenome,
        snp_id=causal.snp_id, chrom=causal.chrom, pos=causal.pos,
        marker_index=causal.marker_index, effect=effect)
    return SimPhenotype(y=y, causal=causal_out, realized_var=realized,
                        arm_name=arm.name)


# =====================================================================
# LD & causal truth windows
# =====================================================================


def ld_r2(z_target: np.ndarray, Z_block: np.ndarray) -> np.ndarray:
    """r² between one standardized SNP and a block of standardized SNPs.

    Both inputs must be unit-variance standardized over the SAME n samples
    (as produced by standardize_dosage). r² = corr².
    """
    n = z_target.shape[0]
    corr = (Z_block.T @ z_target) / n
    return np.clip(corr * corr, 0.0, 1.0)


def build_truth_windows(
    causal: CausalSet,
    geno_meta: dict[str, dict],
    Z_by_sub: dict[str, np.ndarray],
    *,
    window_bp: int = 500_000,
    r2_min: float = 0.20,
    r2_search_bp: int = 2_000_000,
) -> dict[int, tuple[str, frozenset]]:
    """Build, per causal SNP, the set of marker snp_ids that count as a hit.

    A marker tags causal ``c`` iff it is on the same subgenome+chromosome and
    either within ``window_bp`` physically or in LD r² ≥ ``r2_min`` (LD only
    evaluated within ±``r2_search_bp`` for speed). The causal marker itself
    always tags itself.

    Args:
        causal: the replicate CausalSet.
        geno_meta: {subgenome: {"snp_id","chrom","pos"}} arrays aligned to the
            columns of Z_by_sub[subgenome].
        Z_by_sub: {subgenome: standardized dosage (n, m_sub)}.
        window_bp, r2_min, r2_search_bp: truth-window parameters.

    Returns:
        {causal_id: (subgenome, frozenset_of_tagged_snp_ids)}.
    """
    out: dict[int, tuple[str, frozenset]] = {}
    for k in range(causal.n_causal):
        cid = int(causal.causal_id[k])
        sub = str(causal.subgenome[k])
        c_chrom = causal.chrom[k]
        c_pos = int(causal.pos[k])
        c_mi = int(causal.marker_index[k])
        meta = geno_meta[sub]
        m_chrom = np.asarray(meta["chrom"], dtype=object)
        m_pos = np.asarray(meta["pos"], dtype=np.int64)
        m_sid = np.asarray(meta["snp_id"], dtype=object)
        Z = Z_by_sub[sub]
        same_chr = m_chrom == c_chrom
        dpos = np.abs(m_pos - c_pos)
        phys = same_chr & (dpos <= window_bp)
        tagged = phys.copy()
        ld_cand = np.where(same_chr & (dpos <= r2_search_bp) & ~phys)[0]
        if ld_cand.size:
            r2 = ld_r2(Z[:, c_mi], Z[:, ld_cand])
            tagged[ld_cand[r2 >= r2_min]] = True
        tagged[c_mi] = True
        out[cid] = (sub, frozenset(m_sid[tagged].tolist()))
    return out


# =====================================================================
# Discovery-locus collapsing & power / FDR evaluation
# =====================================================================


@dataclass(frozen=True)
class ThresholdEval:
    """Power / FDR at one p-value threshold for one (replicate, method)."""
    threshold: float
    n_sig: int
    n_loci: int
    n_tp_loci: int
    n_fp_loci: int            # includes duplicate loci (counted as false)
    n_duplicate_loci: int     # loci tagging only already-claimed causals
    n_detected_causal: int
    n_causal: int
    power: float              # NaN if n_causal == 0
    fdp: float
    loci_extra: dict = field(default_factory=dict)


def _collapse_loci(sub: np.ndarray, chrom: np.ndarray, pos: np.ndarray,
                   p: np.ndarray, collapse_window_bp: int) -> list[np.ndarray]:
    """Greedily collapse significant markers into discovery loci.

    Markers are taken most-significant first; a marker joins an existing locus
    when it is on the same subgenome+chromosome and within collapse_window_bp
    of any marker already in that locus. Returns a list of index arrays (into
    the passed significant-marker arrays).
    """
    order = np.argsort(p, kind="stable")
    loci: list[list[int]] = []
    locus_meta: list[list[tuple]] = []   # per locus: list of (sub,chrom,pos)
    for i in order:
        si, ci, pi = sub[i], chrom[i], int(pos[i])
        placed = False
        for li, meta in enumerate(locus_meta):
            for (ms, mc, mp) in meta:
                if ms == si and mc == ci and abs(mp - pi) <= collapse_window_bp:
                    loci[li].append(i)
                    locus_meta[li].append((si, ci, pi))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            loci.append([i])
            locus_meta.append([(si, ci, pi)])
    return [np.array(x, dtype=np.int64) for x in loci]


def evaluate_threshold(
    sumstats: dict[str, np.ndarray],
    truth: dict[int, tuple[str, frozenset]],
    threshold: float,
    *,
    collapse_window_bp: int = 500_000,
) -> ThresholdEval:
    """Power / FDR for one method at one p-value threshold.

    Args:
        sumstats: normalized association results, dict of equal-length arrays
            with keys "subgenome","snp_id","chrom","pos","p".
        truth: build_truth_windows output for this replicate.
        threshold: p-value cutoff.
        collapse_window_bp: physical window for collapsing significant markers.

    Returns:
        ThresholdEval. ``power`` is NaN when there are no causal SNPs (null
        arm); ``fdp`` is then the fraction of discovery loci (all false).
        A discovery locus counts as a true positive only if it is the *first*
        locus to claim a given causal SNP — a second locus tagging an
        already-claimed causal is a duplicate and counts toward the FP total.
    """
    sub = np.asarray(sumstats["subgenome"], dtype=object)
    snp = np.asarray(sumstats["snp_id"], dtype=object)
    chrom = np.asarray(sumstats["chrom"], dtype=object)
    pos = np.asarray(sumstats["pos"], dtype=np.int64)
    p = np.asarray(sumstats["p"], dtype=np.float64)

    n_causal = len(truth)
    sig = np.isfinite(p) & (p <= threshold)
    if not sig.any():
        return ThresholdEval(threshold, 0, 0, 0, 0, 0, 0, n_causal,
                             float("nan") if n_causal == 0 else 0.0, 0.0)

    s_sub, s_snp = sub[sig], snp[sig]
    s_chr, s_pos, s_p = chrom[sig], pos[sig], p[sig]
    loci = _collapse_loci(s_sub, s_chr, s_pos, s_p, collapse_window_bp)

    # reverse map: snp_id -> set of causal ids it tags (per subgenome)
    tag_index: dict[tuple[str, object], set[int]] = {}
    for cid, (csub, sids) in truth.items():
        for sid in sids:
            tag_index.setdefault((csub, sid), set()).add(cid)

    # claim causals strongest-locus-first so the lead hit owns the causal
    loci_sorted = sorted(loci, key=lambda idx: float(s_p[idx].min()))
    detected: set[int] = set()
    n_tp = n_fp = n_dup = 0
    for idx in loci_sorted:
        locus_causals: set[int] = set()
        for j in idx:
            hit = tag_index.get((s_sub[j], s_snp[j]))
            if hit:
                locus_causals |= hit
        new = locus_causals - detected
        if new:                              # claims >=1 fresh causal -> TP
            n_tp += 1
            detected |= new
        elif locus_causals:                  # only already-claimed -> duplicate
            n_dup += 1
            n_fp += 1
        else:                                # tags nothing -> false positive
            n_fp += 1

    n_loci = len(loci)
    power = (len(detected) / n_causal) if n_causal > 0 else float("nan")
    fdp = n_fp / max(n_tp + n_fp, 1)
    return ThresholdEval(
        threshold=float(threshold), n_sig=int(sig.sum()), n_loci=n_loci,
        n_tp_loci=n_tp, n_fp_loci=n_fp, n_duplicate_loci=n_dup,
        n_detected_causal=len(detected),
        n_causal=n_causal, power=power, fdp=fdp)


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR q-values (monotone-enforced)."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    if n == 0:
        return p.copy()
    finite = np.isfinite(p)
    q = np.full(n, np.nan)
    pf = p[finite]
    k = pf.size
    if k == 0:
        return q
    order = np.argsort(pf)
    ranks = np.arange(1, k + 1)
    qs = pf[order] * k / ranks
    qs = np.minimum.accumulate(qs[::-1])[::-1]
    qs = np.clip(qs, 0.0, 1.0)
    qf = np.empty(k)
    qf[order] = qs
    q[finite] = qf
    return q


def bh_threshold(p: np.ndarray, q_level: float) -> float:
    """Largest p-value whose BH q-value ≤ q_level (0.0 if none significant)."""
    p = np.asarray(p, dtype=np.float64)
    q = bh_adjust(p)
    ok = np.isfinite(q) & (q <= q_level)
    if not ok.any():
        return 0.0
    return float(np.max(p[ok]))


def power_fdr_curve(
    sumstats: dict[str, np.ndarray],
    truth: dict[int, tuple[str, frozenset]],
    thresholds: np.ndarray,
    *,
    collapse_window_bp: int = 500_000,
) -> list[ThresholdEval]:
    """Evaluate one method over a grid of p-value thresholds."""
    return [evaluate_threshold(sumstats, truth, float(t),
                               collapse_window_bp=collapse_window_bp)
            for t in thresholds]


def power_at_fdr(mean_power: np.ndarray, mean_fdr: np.ndarray,
                 target_fdr: float) -> float:
    """Max mean power over thresholds whose mean empirical FDR ≤ target_fdr."""
    mean_power = np.asarray(mean_power, dtype=np.float64)
    mean_fdr = np.asarray(mean_fdr, dtype=np.float64)
    ok = np.isfinite(mean_fdr) & (mean_fdr <= target_fdr) & np.isfinite(mean_power)
    if not ok.any():
        return 0.0
    return float(np.max(mean_power[ok]))


def partial_auc(mean_power: np.ndarray, mean_fdr: np.ndarray,
                fdr_cap: float = 0.10) -> float:
    """Partial area under the power-vs-FDR curve for FDR ∈ [0, fdr_cap].

    Normalized by fdr_cap so the value is a mean power, in [0,1].
    """
    mean_power = np.asarray(mean_power, dtype=np.float64)
    mean_fdr = np.asarray(mean_fdr, dtype=np.float64)
    ok = np.isfinite(mean_power) & np.isfinite(mean_fdr)
    fp, pw = mean_fdr[ok], mean_power[ok]
    if fp.size == 0:
        return 0.0
    order = np.argsort(fp)
    fp, pw = fp[order], pw[order]
    # monotone envelope: best power achievable at FDR ≤ x
    pw_env = np.maximum.accumulate(pw)
    # anchor at (0,0): power is 0 below the smallest attainable FDR, so the
    # pAUC never credits power at an FDR no operating point can reach.
    fp = np.concatenate([[0.0], fp])
    pw_env = np.concatenate([[0.0], pw_env])
    grid = np.linspace(0.0, fdr_cap, 101)
    vals = np.interp(grid, fp, pw_env, left=0.0, right=pw_env[-1])
    # np.trapz was renamed np.trapezoid in NumPy 2.0
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(vals, grid) / fdr_cap)


# =====================================================================
# Paired significance tests
# =====================================================================


def paired_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    rng: np.random.Generator,
    n_boot: int = 2000,
    alternative: str = "greater",
) -> dict:
    """Paired bootstrap of mean(a − b) over replicates.

    Args:
        values_a, values_b: per-replicate paired scalars (same length).
        rng: numpy Generator.
        n_boot: bootstrap resamples.
        alternative: "greater" tests H1: mean(a−b) > 0.

    Returns:
        dict(delta, ci_lo, ci_hi, p_value, n).
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired arrays differ in shape {a.shape} vs {b.shape}")
    d = a - b
    n = d.size
    delta = float(np.mean(d)) if n else float("nan")
    if n == 0:
        return {"delta": delta, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_value": float("nan"), "n": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = d[idx].mean(axis=1)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    if alternative == "greater":
        p = float((np.sum(boot <= 0.0) + 1) / (n_boot + 1))
    elif alternative == "less":
        p = float((np.sum(boot >= 0.0) + 1) / (n_boot + 1))
    else:
        p = float((np.sum(np.abs(boot) >= abs(delta)) + 1) / (n_boot + 1))
    return {"delta": delta, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),
            "p_value": p, "n": int(n)}


def holm_correct(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment of a small set of p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adj[k] = running
    return adj
