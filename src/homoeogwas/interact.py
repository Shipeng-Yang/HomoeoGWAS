"""homoeogwas ``interact`` — gene-resolution homoeolog-pair burden-product interaction scan.

This promotes the Phase-7 research per-pair interaction method into a first-class CLI
module. For each homoeolog pair (X in subgenome S_x, Y in S_y) it fits the whitened GLS

    y_w ~ 1 + b_X + b_Y + b_X * b_Y

where b_* are gene burdens (capped block-mean of standardized dosages) and the whitener
comes from a subgenome-stratified null LMM (multi-kernel REML over {K_sub}); it then tests
the interaction coefficient and aggregates per-pair p-values with ACAT. INT is the primary
transform; ``raw`` is a sensitivity transform. Empirical calibration uses y-shuffle
permutation (lambda_perm + empirical ACAT). Extension hooks (frozen, y-independent):
``--weights`` (DL/HEB pair priors -> weighted hypothesis testing) and ``--multi-trait``
(ACAT across a predeclared trait set) are layered on top without touching this core.

Config (YAML)::

    interact:
      subgenomes: [A, D]               # 2 (disomic) or 3 (hexaploid)
      genotype: {A: <plink_prefix>, D: <plink_prefix>}
      snp_to_gene: {A: <npz>, D: <npz>}    # gene_ids + snp_idx into that subgenome's X
      pairs: <tsv>                          # columns gene_<S> per subgenome (e.g. gene_A,gene_D)
      phenotype: <tsv>
      sample_col: sample
      trait: <name>                         # single-trait; OR predeclare a frozen set below
      # multi_trait: [t1, t2, t3]           # pairwise-only pleiotropy: ACAT across a FROZEN trait set
      burden: {cap: 150, min_snp: 3}
      transform: INT                        # INT primary; also runs raw sensitivity
      calibration: {perm_B: 2000}
    outputs: {out_dir: <dir>}
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------- stats helpers


def _acat_tan_terms(p: np.ndarray) -> np.ndarray:
    """Cauchy terms tan((0.5-p)*pi) with precision-safe branches at the extremes (Liu & Xie 2020).

    For p < 1e-15, ``(0.5 - p)`` rounds to exactly 0.5 in float64 so ``tan`` saturates near pi/2 and
    loses all information; use the cotangent identity tan((0.5-p)*pi) = cot(p*pi) ~ 1/(p*pi). Symmetric
    branch for p near 1. In the normal range the plain ``tan`` term is returned, so ACAT is bit-exact
    for non-extreme p (the only values that change are raw-scale outlier p far below 1e-15)."""
    small = p < 1e-15
    big = p > 1.0 - 1e-15
    t = np.tan((0.5 - p) * np.pi)
    t = np.where(small, 1.0 / (p * np.pi), t)
    t = np.where(big, -1.0 / ((1.0 - p) * np.pi), t)
    return t


def _acat_p_from_t(tbar: float) -> float:
    """Invert the mean Cauchy statistic to a p-value; the |t|-large branches p ~ 1/(t*pi) (t>0) and
    p ~ 1 - 1/(|t|*pi) (t<0) avoid arctan saturation when one extreme p dominates the combination."""
    if tbar > 1e15:
        return float(1.0 / (tbar * np.pi))
    if tbar < -1e15:
        return float(1.0 - 1.0 / (abs(tbar) * np.pi))
    return float(0.5 - np.arctan(tbar) / np.pi)


def acat(pvals: np.ndarray) -> float:
    """Aggregated Cauchy association test combination of p-values (equal weights)."""
    p = np.asarray(pvals, float)
    p = np.clip(p[np.isfinite(p)], 1e-300, 1.0 - 1e-16)
    if p.size == 0:
        return float("nan")
    return _acat_p_from_t(float(np.mean(_acat_tan_terms(p))))


def acat_weighted(pvals: np.ndarray, weights: np.ndarray) -> float:
    """Weighted ACAT combination. Weights must be y-INDEPENDENT (e.g. zero-shot DL variant
    priors or homoeolog expression bias, frozen before the association test) so the
    combination stays calibrated while up-weighting prior-favoured pairs."""
    p = np.asarray(pvals, float)
    w = np.asarray(weights, float)
    m = np.isfinite(p) & np.isfinite(w) & (w > 0)
    p = np.clip(p[m], 1e-300, 1.0 - 1e-16)
    w = w[m]
    if p.size == 0:
        return float("nan")
    w = w / w.sum()
    return _acat_p_from_t(float(np.sum(w * _acat_tan_terms(p))))


def lambda_gc(pvals: np.ndarray) -> float:
    """Genomic-control lambda from p-values via the median chi-square(1) statistic."""
    p = np.asarray(pvals, float)
    p = np.clip(p[np.isfinite(p)], 1e-300, 1.0)
    if p.size == 0:
        return float("nan")
    chi2 = stats.norm.isf(p / 2.0) ** 2
    return float(np.median(chi2) / 0.4549364)


def rank_int(y: np.ndarray) -> np.ndarray:
    """Rank-based inverse-normal transform (Blom)."""
    y = np.asarray(y, float)
    n = y.size
    r = stats.rankdata(y, method="average")
    return stats.norm.ppf((r - 0.5) / n)


def block_burden_capped(X: np.ndarray, snp_idx, cap: int, rng) -> np.ndarray:
    """Mean of column-standardized dosages over a gene's SNPs; subsample to ``cap`` if larger."""
    idx = np.asarray(snp_idx, int)
    if cap and idx.size > cap:
        idx = np.sort(rng.choice(idx, size=cap, replace=False))
    M = X[:, idx].astype(float)
    mu = np.nanmean(M, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Mi = np.where(np.isnan(M), mu, M)
    sd = Mi.std(0, ddof=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (Mi - mu) / sd_safe
    Z[:, sd <= 1e-12] = 0.0
    return Z.mean(1)


def scols_safe(M: np.ndarray) -> np.ndarray:
    mu = M.mean(0)
    sd = M.std(0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (M - mu) / sd


def grm_from_X(X: np.ndarray) -> np.ndarray:
    """Additive GRM from a dosage matrix (mean-impute, standardize, XX'/m), trace-normed + PSD-clipped."""
    mu = np.nanmean(X, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    Xi = np.where(np.isnan(X), mu, X)
    sd = Xi.std(0, ddof=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = (Xi - mu) / sd
    K = Z @ Z.T / Z.shape[1]
    K = K / (np.trace(K) / K.shape[0])
    w, Q = np.linalg.eigh(0.5 * (K + K.T))
    w = np.clip(w, 1e-6, None)
    K = (Q * w) @ Q.T
    return K / (np.trace(K) / K.shape[0])


def whiten_multi(kernels: dict[str, np.ndarray], y: np.ndarray, X: np.ndarray = None, seed: int = 42):
    """Subgenome-stratified whitener from a multi-kernel null LMM (REML variance components).

    ``X`` is the fixed-effect mean model for the null LMM. Default ``None`` => intercept-only
    ``ones((n,1))`` (reproduces the legacy covariate-free behaviour bit-exactly). Passing a
    covariate block ``[1, PCs, covariates]`` makes the variance components estimated *conditional*
    on those fixed effects, so population/environment structure is not absorbed into sigma^2_sub
    (the statistically complete covariate-adjusted whitener)."""
    from .lmm import fit_multi_reml

    n = next(iter(kernels.values())).shape[0]
    if X is None:
        X = np.ones((n, 1))
    res = fit_multi_reml(y, X, kernels, n_starts=3, random_state=int(seed) % 100000)
    cv = res.component_var
    V = max(cv.get("e", 1e-6), 1e-6) * np.eye(n)
    for name, K in kernels.items():
        V = V + cv.get(name, 0.0) * K
    w, Q = np.linalg.eigh(0.5 * (V + V.T))
    w = np.clip(w, 1e-10, None)
    return (Q * (1.0 / np.sqrt(w))) @ Q.T, cv


def pairwise_pvals(Wh: np.ndarray, y: np.ndarray, BX: np.ndarray, BY: np.ndarray,
                   C: np.ndarray = None) -> np.ndarray:
    """Per-pair interaction p (whitened GLS t-test on the b_X*b_Y coefficient).

    ``C`` is the fixed-effect covariate block (n, p_c) that ALREADY INCLUDES the intercept
    column (e.g. ``[1, PC1..PCk, env...]``). Default ``None`` => intercept-only, which runs the
    exact legacy design ``[1, b_X, b_Y, b_X*b_Y]`` (bit-exact reproduction of covariate-free runs).
    When ``C`` is given, the design is ``[C, b_X, b_Y, b_X*b_Y]`` (covariates as fixed effects,
    so the interaction is tested conditional on them) and the residual df is ``n - (p_c + 3)``."""
    n, G = BX.shape
    yw = Wh @ y
    BXw = Wh @ BX
    BYw = Wh @ BY
    INTw = Wh @ (BX * BY)
    if C is None:
        Cw = (Wh @ np.ones(n)).reshape(-1, 1)
    else:
        Cw = Wh @ np.asarray(C, float).reshape(n, -1)
    p_c = Cw.shape[1]
    j_int = p_c + 2                          # index of the interaction coef in [C, bX, bY, INT]
    p_full = p_c + 3                         # full column count when the design is full rank
    pv = np.empty(G)
    for g in range(G):
        Xw = np.column_stack([Cw, BXw[:, g], BYw[:, g], INTw[:, g]])
        beta, _res, rank, _sv = np.linalg.lstsq(Xw, yw, rcond=None)
        df = n - rank                        # rank-aware df (== n-(p_c+3) when full rank)
        if rank < p_full or df < 1:          # rank-deficient design => interaction not estimable
            pv[g] = 1.0                      # conservative: cannot test a non-identified coefficient
            continue
        resid = yw - Xw @ beta
        s2 = float(resid @ resid) / df
        se = np.sqrt(max(s2 * np.linalg.pinv(Xw.T @ Xw)[j_int, j_int], 1e-30))
        pv[g] = 2.0 * stats.t.sf(abs(beta[j_int] / se), df)
    return pv


def _neglog10(p: np.ndarray) -> np.ndarray:
    """-log10(p) with p floored at 1e-300 (consistent finite cap; never +inf)."""
    return -np.log10(np.clip(np.asarray(p, float), 1e-300, 1.0))


def _decile_bin(x: np.ndarray) -> np.ndarray:
    """Decile bin (0..9) of a 1-D vector by rank (descriptive convenience column only — the
    authoritative matched-null binning is re-derived downstream from the pre-registered design).
    Ties share a bin; an all-constant vector maps to bin 0."""
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return np.zeros(0, int)
    r = stats.rankdata(x, method="average")          # 1..n, ties averaged
    return np.clip(((r - 0.5) / n * 10).astype(int), 0, 9)


def _write_ranking_tsv(path, header: list, rows: list) -> None:
    """Persist the FULL genome-wide ranking (every callable unit), deterministically ordered.
    This is the only side effect of the dump flag; legacy JSON output is untouched (bit-exact)."""
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as fh:
        wtr = csv.writer(fh, delimiter="\t")
        wtr.writerow(header)
        wtr.writerows(rows)


def _rank_with_ties(pv: np.ndarray, keys: list) -> tuple:
    """Deterministic ascending-p order (p, then original index via stable sort) and a tie-group id
    per row (rows sharing an identical p get the same tie_group). Returns (order, rank_of_row,
    tie_group_of_row) where rank/tie are indexed by ORIGINAL row position."""
    pv = np.asarray(pv, float)
    G = pv.size
    order = np.argsort(pv, kind="stable")            # stable => ties keep original (gene) order
    rank_of = np.empty(G, int)
    tie_of = np.empty(G, int)
    rank_of[order] = np.arange(G)
    tg = 0
    prev = None
    for j, i in enumerate(order):
        if prev is None or pv[i] != prev:
            tg = j                                   # tie group = rank of the first member
            prev = pv[i]
        tie_of[i] = tg
    return order, rank_of, tie_of


# ----------------------------------------------------------------------------- data structures


@dataclass
class SubgenomeData:
    X: np.ndarray                       # (n_samples, n_snp) for this subgenome
    gene_snp: dict                      # gene_id -> snp_idx (into X)
    samples: list
    chunk: object = None                # GenoChunk (for grm.compute_grm)


@dataclass(frozen=True)
class FrozenTraitSet:
    """Immutable, ordered, predeclared trait set for multi-trait (pleiotropy) scans.

    Freezing the trait list in a hashable object (insertion order preserved, content digest
    recorded) makes the "multiplicity = G not G x T" claim auditable: the set must be declared
    before the association test and cannot be silently reordered, deduplicated, or expanded.
    This is the code-level firewall for the leakage risk flagged for the multi-trait extension."""

    traits: tuple
    digest: str

    @classmethod
    def from_list(cls, traits) -> FrozenTraitSet:
        ts = list(traits)
        if not ts:
            raise ValueError("multi_trait must be a non-empty ordered list of trait names")
        if any(not isinstance(t, str) for t in ts):
            raise ValueError(f"multi_trait must be trait-name strings; got {ts}")
        dup = sorted({t for t in ts if ts.count(t) > 1})
        if dup:
            raise ValueError(f"multi_trait has duplicate traits {dup}; the trait set must be unique")
        import hashlib

        digest = hashlib.sha256("\x00".join(ts).encode()).hexdigest()[:16]
        return cls(traits=tuple(ts), digest=digest)


@dataclass
class InteractResult:
    trait: str
    transform: str
    n: int
    G: int
    pair_acat: float
    pair_acat_emp: float
    min_p: float
    lambda_gc_obs: float
    lambda_gc_perm_median: float
    bonferroni_alpha: float
    n_sig: int
    sig: list = field(default_factory=list)
    top: list = field(default_factory=list)
    sigma_hat: dict = field(default_factory=dict)
    weighted: dict = None
    covariates: dict = None


# ----------------------------------------------------------------------------- core engine


def _load_subgenome(plink_prefix: str, npz_path: str) -> SubgenomeData:
    from .io import load_bed_hardcall

    bed = load_bed_hardcall(plink_prefix)
    X = np.asarray(bed.dosage, dtype=np.float64)
    samples = [str(s) for s in np.asarray(bed.samples)]
    z = np.load(npz_path, allow_pickle=True)
    gene_ids = z["gene_ids"].tolist()
    snp_idx = z["snp_idx"]
    gene_snp = {g: np.asarray(snp_idx[i], int) for i, g in enumerate(gene_ids)}
    return SubgenomeData(X=X, gene_snp=gene_snp, samples=samples, chunk=bed)


def _build_grm(sd: SubgenomeData, sample_idx: np.ndarray, method: str, maf_min: float) -> np.ndarray:
    """Subgenome GRM restricted to valid samples, trace-normed. ``method`` is explicit
    (Codex: GRM choice must be a documented config knob, not an implicit implementation
    detail) — ``compute_grm_maf`` reuses the package GRM (reproduces the research scripts);
    ``grm_from_X`` is the all-SNP PSD-clipped variant (sensitivity)."""
    n_t = sample_idx.size
    if method == "compute_grm_maf":
        from .grm import compute_grm
        K, _ = compute_grm(sd.chunk, maf_min=maf_min)
        K = np.asarray(K)[np.ix_(sample_idx, sample_idx)]
        return K / (np.trace(K) / n_t)
    if method == "grm_from_X":
        K = grm_from_X(sd.X)[np.ix_(sample_idx, sample_idx)]
        return K / (np.trace(K) / n_t)
    raise ValueError(f"unknown grm.method '{method}' (use compute_grm_maf | grm_from_X)")


def genotype_pcs(kernels: dict[str, np.ndarray], n_pcs: int) -> np.ndarray:
    """Top-``n_pcs`` genotype principal components from the combined (mean) subgenome GRM.

    Reuses the already-built per-subgenome GRMs (no genotype re-read): the additive relatedness
    matrix K_comb = mean_s K_s shares its leading eigenvectors with standard genotype PCA. PCs are
    a y-INDEPENDENT covariate (functions of genotype only) and are standardized to unit scale so
    the fixed-effect block is well conditioned."""
    n = next(iter(kernels.values())).shape[0]
    if n_pcs <= 0:
        return np.empty((n, 0))
    K = np.mean([np.asarray(K_, float) for K_ in kernels.values()], axis=0)
    w, Q = np.linalg.eigh(0.5 * (K + K.T))
    order = np.argsort(w)[::-1]
    n_pos = int(np.sum(w > 1e-8 * max(abs(w).max(), 1.0)))   # positive-eigenvalue capacity
    k = min(n_pcs, n_pos, max(n - 4, 0))                     # cap by rank AND leave design headroom
    idx = order[:k]
    P = Q[:, idx]
    sd = P.std(0, ddof=0)
    return P / np.where(sd > 1e-12, sd, 1.0)


def build_covariate_block(kernels: dict[str, np.ndarray], n_t: int, *, n_pcs: int = 0,
                          extra: np.ndarray = None) -> tuple:
    """Assemble the fixed-effect covariate design ``C = [1, PC1..PCk, extra...]`` (intercept first).

    Returns ``(C, meta)``. With ``n_pcs==0`` and ``extra is None`` returns ``(None, ...)`` so the
    engine takes the exact legacy intercept-only path (bit-exact). ``extra`` is an optional already
    sample-aligned, y-independent covariate matrix (n_t, q) (e.g. environment/batch), standardized
    here. The intercept is always column 0; the interaction df becomes ``n - (C.shape[1] + 3)``."""
    blocks = [np.ones((n_t, 1))]
    cols = ["intercept"]
    actual_pcs = 0
    if n_pcs and n_pcs > 0:
        P = genotype_pcs(kernels, n_pcs)            # may be clamped to positive-eigenvalue capacity
        actual_pcs = int(P.shape[1])
        if actual_pcs:
            blocks.append(P)
            cols += [f"PC{i+1}" for i in range(actual_pcs)]
    n_extra = 0
    if extra is not None:
        E = np.asarray(extra, float).reshape(n_t, -1)
        sd = E.std(0, ddof=0)
        keep = sd > 1e-12                            # drop zero-variance (constant) covariates
        E = E[:, keep]
        if E.shape[1]:
            E = (E - E.mean(0)) / E.std(0, ddof=0)
            blocks.append(E)
            n_extra = int(E.shape[1])
            cols += [f"cov{i+1}" for i in range(n_extra)]
    if len(blocks) == 1:                             # nothing usable => legacy intercept-only
        return None, dict(policy="none", n_pcs=0, n_extra=0, columns=cols, rank=1)
    C = np.column_stack(blocks)
    rank = int(np.linalg.matrix_rank(C))
    if rank < C.shape[1]:                            # rank-deficient covariate block is a config error
        raise ValueError(f"covariate block is rank-deficient (rank {rank} < {C.shape[1]} cols "
                         f"{cols[1:]}); covariates must be linearly independent (a covariate may "
                         "duplicate a genotype PC or another covariate)")
    meta = dict(policy=f"pcs={actual_pcs}+extra={n_extra}", n_pcs=actual_pcs, n_extra=n_extra,
                n_pcs_requested=int(n_pcs), columns=cols, n_cols=int(C.shape[1]), rank=rank)
    return C, meta


def run_pair_scan(
    subdata: dict[str, SubgenomeData],
    pairs: list[tuple],                 # list of dict {sub: gene_id} OR (sx, gx, sy, gy)
    y_raw: np.ndarray,
    sample_idx: np.ndarray,             # rows (into subdata samples) for valid phenotype
    *,
    cap: int = 150,
    transform: str = "INT",
    perm_B: int = 2000,
    n_jobs: int = 8,
    seed: int = 7,
    pair_subs: tuple = None,            # (sx, sy) for 2-col pair tuples
    grm_method: str = "compute_grm_maf",
    maf_min: float = 0.01,
    pair_weights: dict = None,          # {(gx,gy): w}  y-INDEPENDENT prior (DL/HEB), frozen
    covariates: dict = None,            # {n_pcs:int, extra:(n_t,q) array} fixed effects; None=legacy
    full_dump_path: str = None,         # if set, write FULL per-pair ranking TSV (else top-N only)
) -> InteractResult:
    """Whitened per-pair burden-product interaction scan with ACAT + permutation calibration.
    Optional ``pair_weights`` (y-independent DL/HEB priors) enable weighted-hypothesis testing:
    weighted Bonferroni p_i < alpha*w_i/G and weighted ACAT, reported ALONGSIDE the unweighted
    results (never replacing them). FWER stays controlled because weights do not depend on y.
    Optional ``covariates`` (y-independent genotype PCs and/or extra fixed effects) enters BOTH the
    null-LMM mean model (whitener) AND the per-pair GLS design; ``None`` => exact legacy intercept-
    only path. Permutation switches to Freedman-Lane (residualize on C, permute residuals) so the
    covariate-phenotype structure is preserved under the null while the genotype link is broken;
    with C=intercept this reduces exactly to the legacy y-shuffle."""
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    subs = list(subdata.keys())
    n_t = sample_idx.size
    # subgenome GRMs restricted to valid samples (GRM method is an explicit config knob)
    kernels = {s: _build_grm(subdata[s], sample_idx, grm_method, maf_min) for s in subs}
    # fixed-effect covariate block C = [1, PCs, extra] (None = legacy intercept-only, bit-exact)
    cov_meta = dict(policy="none")
    C = None
    if covariates:
        C, cov_meta = build_covariate_block(kernels, n_t, n_pcs=int(covariates.get("n_pcs", 0)),
                                            extra=covariates.get("extra"))

    # build paired burden matrices BX (subgenome sx) / BY (subgenome sy)
    sx, sy = pair_subs
    bx_cols, by_cols, kept_pairs = [], [], []
    nsnp_x, nsnp_y = [], []                          # callable SNP count per gene (= len snp_idx)
    for p in pairs:
        gx, gy = p
        if gx in subdata[sx].gene_snp and gy in subdata[sy].gene_snp:
            bx_cols.append(block_burden_capped(subdata[sx].X, subdata[sx].gene_snp[gx], cap, rng)[sample_idx])
            by_cols.append(block_burden_capped(subdata[sy].X, subdata[sy].gene_snp[gy], cap, rng)[sample_idx])
            kept_pairs.append((gx, gy))
            nsnp_x.append(int(np.asarray(subdata[sx].gene_snp[gx]).size))
            nsnp_y.append(int(np.asarray(subdata[sy].gene_snp[gy]).size))
    BX = scols_safe(np.column_stack(bx_cols))
    BY = scols_safe(np.column_stack(by_cols))
    G = len(kept_pairs)

    # y-independent prior weights aligned to kept pairs, normalized to mean 1 (missing -> 1)
    w = None
    if pair_weights:
        w = np.array([float(pair_weights.get(kp, 1.0)) for kp in kept_pairs])
        w = np.where((w > 0) & np.isfinite(w), w, 1.0)
        w = w * (G / w.sum())

    y = rank_int(y_raw) if transform == "INT" else y_raw.astype(float)
    Wh, cv = whiten_multi(kernels, y, X=C, seed=42)
    pv = pairwise_pvals(Wh, y, BX, BY, C=C)
    pv = np.where(np.isfinite(pv), pv, 1.0)
    p_acat_obs = acat(pv)
    minp_obs = float(pv.min())
    lam_obs = lambda_gc(pv)
    bonf = 0.05 / G
    order = np.argsort(pv)
    sig = [dict(pair=kept_pairs[int(i)], p=float(pv[i])) for i in order if pv[i] < bonf]
    top = [dict(pair=kept_pairs[int(i)], p=float(pv[i])) for i in order[:5]]

    if full_dump_path:
        # FULL genome-wide ranking (descriptive; inference is the pre-registered enrichment, NOT
        # per-row Bonferroni). gene_len is NOT emitted here (the engine has no coordinates) -> NA;
        # it is joined downstream from the annotation (no fabrication).
        nx, ny = np.asarray(nsnp_x), np.asarray(nsnp_y)
        _, rank_of, tie_of = _rank_with_ties(pv, kept_pairs)
        nl = _neglog10(pv)
        cbin = _decile_bin(nx + ny)
        rows = [[int(rank_of[i]), kept_pairs[i][0], kept_pairs[i][1], sx, sy,
                 repr(float(pv[i])), repr(float(nl[i])), int(nx[i]), int(ny[i]), int(nx[i] + ny[i]),
                 int(cbin[i]), int(tie_of[i]), int(pv[i] >= 1.0),
                 int(pv[i] < bonf), "NA", "NA", "NA"]
                for i in np.argsort(pv, kind="stable")]
        _write_ranking_tsv(full_dump_path,
                           ["rank", f"gene_{sx}", f"gene_{sy}", "sub_x", "sub_y", "p_interaction",
                            "neglog10p", f"n_snp_{sx}", f"n_snp_{sy}", "n_snp_pair",
                            "callable_snp_decile", "tie_group", "p_is_one", "bonferroni_sig",
                            f"gene_len_{sx}", f"gene_len_{sy}", "gene_len_pair_sum"], rows)

    weighted = None
    if w is not None:
        # weighted Bonferroni: pair i significant if p_i < alpha * w_i / G
        wsig = [dict(pair=kept_pairs[int(i)], p=float(pv[i]), weight=float(w[i]),
                     p_weighted=float(pv[i] / w[i]))
                for i in order if pv[i] < bonf * w[i]]
        n_cov = int(sum(kp in pair_weights for kp in kept_pairs))
        weighted = dict(
            acat_weighted=float(acat_weighted(pv, w)), bonferroni_n_sig=len(wsig), sig=wsig,
            audit=dict(n_pairs=int(G), n_covered=n_cov, n_missing_default1=int(G - n_cov),
                       weight_min=float(w.min()), weight_mean=float(w.mean()),
                       weight_max=float(w.max())),
            note=("weighted Bonferroni controls pair-level FWER; weighted ACAT is a y-independent "
                  "prior-weighted omnibus. Valid ONLY if weights were frozen before this scan and "
                  "are y-independent (DL zero-shot / HEB). Missing pairs default to weight 1; after "
                  "normalization (sum w = G) up-weighting prioritized pairs reallocates alpha from "
                  "the rest (their effective weight < 1)."))

    # Freedman-Lane permutation calibration: y* = C@beta_hat + (y - C@beta_hat)[perm].
    # With C=None (intercept only) beta_hat=mean(y) so y* == y[perm] exactly (legacy y-shuffle);
    # with covariates it preserves the covariate mean structure while breaking the genotype link.
    if C is None:
        fl_fit, fl_resid = None, None
    else:
        b_fl, *_ = np.linalg.lstsq(C, y, rcond=None)
        fl_fit = C @ b_fl
        fl_resid = y - fl_fit

    def _perm(seed_i):
        r = np.random.default_rng(seed_i)
        perm = r.permutation(n_t)
        ys = y[perm] if C is None else (fl_fit + fl_resid[perm])
        try:
            Whp, _ = whiten_multi(kernels, ys, X=C, seed=seed_i % 100000)
            pp = pairwise_pvals(Whp, ys, BX, BY, C=C)
            return acat(pp), lambda_gc(pp)
        except Exception:  # noqa: BLE001
            return None

    p_acat_emp = float("nan")
    lam_perm = float("nan")
    if perm_B and perm_B > 0:
        res = Parallel(n_jobs=n_jobs)(delayed(_perm)(900000 + i) for i in range(perm_B))
        res = [r for r in res if r is not None]
        if res:
            ap = np.array([r[0] for r in res])
            lam = np.array([r[1] for r in res])
            p_acat_emp = float((1 + int((ap <= p_acat_obs).sum())) / (len(ap) + 1))
            lam_perm = float(np.median(lam))

    return InteractResult(
        trait="", transform=transform, n=int(n_t), G=int(G),
        pair_acat=float(p_acat_obs), pair_acat_emp=p_acat_emp, min_p=minp_obs,
        lambda_gc_obs=float(lam_obs), lambda_gc_perm_median=lam_perm,
        bonferroni_alpha=float(bonf), n_sig=len(sig), sig=sig, top=top,
        sigma_hat={s: float(cv.get(s, 0.0)) for s in subs} | {"e": float(cv.get("e", 0.0))},
        weighted=weighted, covariates=cov_meta)


def run_triad_scan(
    subdata: dict[str, SubgenomeData],
    triads: list[tuple],                # list of (gene_s0, gene_s1, gene_s2) aligned to subs order
    y_raw: np.ndarray,
    sample_idx: np.ndarray,
    *,
    cap: int = 150,
    transform: str = "INT",
    perm_B: int = 2000,
    n_jobs: int = 8,
    seed: int = 7,
    grm_method: str = "compute_grm_maf",
    maf_min: float = 0.01,
    triad_weights: dict = None,         # {(g0,g1,g2): w}  y-INDEPENDENT prior (HEB/DL), frozen
    covariates: dict = None,            # {n_pcs:int, extra:(n_t,q)} fixed effects; None=legacy
    full_dump_path: str = None,         # if set, write FULL per-triad ranking TSV (else top-N only)
) -> dict:
    """Hexaploid triad burden-product scan: full {K_A,K_B,K_D} whitening, the 3 within-triad
    pairwise interactions (s0-s1, s0-s2, s1-s2), and a triad-level ACAT (= hexaploid omnibus).
    Requires all three homoeologs of a triad retained (dense panels, e.g. wheat WGS).
    ``covariates`` (genotype PCs / extra fixed effects) enters the whitener mean model AND every
    pairwise GLS; ``None`` = exact legacy intercept-only path. Permutation is Freedman-Lane."""
    from joblib import Parallel, delayed

    rng = np.random.default_rng(seed)
    subs = list(subdata.keys())            # 3 subgenomes in config order
    s0, s1, s2 = subs
    n_t = sample_idx.size
    kernels = {s: _build_grm(subdata[s], sample_idx, grm_method, maf_min) for s in subs}
    cov_meta = dict(policy="none")
    C = None
    if covariates:
        C, cov_meta = build_covariate_block(kernels, n_t, n_pcs=int(covariates.get("n_pcs", 0)),
                                            extra=covariates.get("extra"))

    # keep triads with all three homoeologs retained; build aligned burden matrices
    cols = {s: [] for s in subs}
    nsnp = {s: [] for s in subs}                     # callable SNP count per gene, per subgenome
    kept = []
    for g0, g1, g2 in triads:
        gmap = {s0: g0, s1: g1, s2: g2}
        if all(gmap[s] in subdata[s].gene_snp for s in subs):
            for s in subs:
                cols[s].append(block_burden_capped(subdata[s].X, subdata[s].gene_snp[gmap[s]],
                                                    cap, rng)[sample_idx])
                nsnp[s].append(int(np.asarray(subdata[s].gene_snp[gmap[s]]).size))
            kept.append((g0, g1, g2))
    G = len(kept)
    if G < 1:
        return dict(G=0, note="no triads with all three homoeologs retained")
    Bd = {s: scols_safe(np.column_stack(cols[s])) for s in subs}

    # y-independent per-triad prior weights (HEB/DL), normalized to mean 1 (missing -> 1)
    w = None
    if triad_weights:
        w = np.array([float(triad_weights.get(t, 1.0)) for t in kept])
        w = np.where((w > 0) & np.isfinite(w), w, 1.0)
        w = w * (G / w.sum())

    y = rank_int(y_raw) if transform == "INT" else y_raw.astype(float)
    Wh, cv = whiten_multi(kernels, y, X=C, seed=42)

    pairwise_defs = [(s0, s1), (s0, s2), (s1, s2)]
    pw_p = {}
    for sx, sy in pairwise_defs:
        pw_p[f"{sx}{sy}"] = pairwise_pvals(Wh, y, Bd[sx], Bd[sy], C=C)
    # triad-level ACAT over the 3 pairwise p per triad
    triad_acat = np.array([acat([pw_p[f"{sx}{sy}"][i] for sx, sy in pairwise_defs])
                           for i in range(G)])
    triad_omnibus = acat(triad_acat)

    if full_dump_path:
        # FULL genome-wide per-triad ranking, ordered by ascending triad-ACAT (descriptive; the
        # inferential test is the pre-registered enrichment, NOT per-triad Bonferroni). gene_len is
        # NOT emitted (engine has no coordinates) -> NA, joined downstream from the GFF.
        tags = [f"{sx}{sy}" for sx, sy in pairwise_defs]
        pmat = np.column_stack([pw_p[t] for t in tags])      # (G, 3) per-pairwise p
        pmat = np.where(np.isfinite(pmat), pmat, 1.0)         # Fix4: sanitize before ranking
        # local finite copy of the per-triad ACAT for ranking (persisted stats untouched). rank is a
        # 0-based ordinal after a STABLE ascending sort; tie_group = the first ordinal of an exact-p
        # tie (exact float equality, meaningful only for genuine ties e.g. p==1.0), NOT a shared rank.
        acat_rank = np.where(np.isfinite(triad_acat), triad_acat, 1.0)
        min_pair = pmat.min(1)
        min_tag = np.array(tags)[pmat.argmin(1)]
        bonf_ac = 0.05 / G
        _, rank_of, tie_of = _rank_with_ties(acat_rank, kept)
        nl = _neglog10(acat_rank)
        ns = {s: np.asarray(nsnp[s]) for s in subs}
        ns_sum = sum(ns[s] for s in subs)
        cbin = _decile_bin(ns_sum)
        rows = [[int(rank_of[i]), kept[i][0], kept[i][1], kept[i][2],
                 repr(float(pmat[i, 0])), repr(float(pmat[i, 1])), repr(float(pmat[i, 2])),
                 repr(float(acat_rank[i])), repr(float(nl[i])), repr(float(min_pair[i])),
                 str(min_tag[i]), int(ns[s0][i]), int(ns[s1][i]), int(ns[s2][i]), int(ns_sum[i]),
                 int(cbin[i]), int(tie_of[i]), int(acat_rank[i] < bonf_ac), "NA", "NA", "NA"]
                for i in np.argsort(acat_rank, kind="stable")]
        _write_ranking_tsv(full_dump_path,
                           ["rank", f"gene_{s0}", f"gene_{s1}", f"gene_{s2}",
                            f"p_{tags[0]}", f"p_{tags[1]}", f"p_{tags[2]}", "p_acat", "neglog10p_acat",
                            "min_pair_p", "min_pair_tag", f"n_snp_{s0}", f"n_snp_{s1}", f"n_snp_{s2}",
                            "n_snp_triad", "callable_snp_decile", "tie_group", "bonferroni_sig_acat",
                            f"gene_len_{s0}", f"gene_len_{s1}", f"gene_len_{s2}"], rows)

    pw_res = {}
    for sx, sy in pairwise_defs:
        tag = f"{sx}{sy}"
        pv = pw_p[tag]
        bonf = 0.05 / G
        order = np.argsort(pv)
        pw_res[tag] = dict(
            G=int(G), acat=float(acat(pv)), min_p=float(pv.min()), lambda_gc_obs=float(lambda_gc(pv)),
            bonferroni_alpha=float(bonf), n_sig=int((pv < bonf).sum()),
            sig=[dict(triad=kept[int(i)], p=float(pv[i])) for i in order if pv[i] < bonf],
            top=[dict(triad=kept[int(i)], p=float(pv[i])) for i in order[:5]])
        if w is not None:
            wsig = [dict(triad=kept[int(i)], p=float(pv[i]), weight=float(w[i]))
                    for i in order if pv[i] < bonf * w[i]]
            pw_res[tag]["weighted"] = dict(acat_weighted=float(acat_weighted(pv, w)),
                                           bonferroni_n_sig=len(wsig), sig=wsig)

    # Freedman-Lane permutation (reduces to legacy y-shuffle when C is intercept-only)
    if C is None:
        fl_fit, fl_resid = None, None
    else:
        b_fl, *_ = np.linalg.lstsq(C, y, rcond=None)
        fl_fit = C @ b_fl
        fl_resid = y - fl_fit

    def _perm(seed_i):
        r = np.random.default_rng(seed_i)
        perm = r.permutation(n_t)
        ys = y[perm] if C is None else (fl_fit + fl_resid[perm])
        try:
            Whp, _ = whiten_multi(kernels, ys, X=C, seed=seed_i % 100000)
            pp = {f"{sx}{sy}": pairwise_pvals(Whp, ys, Bd[sx], Bd[sy], C=C) for sx, sy in pairwise_defs}
            tacat = np.array([acat([pp[f"{sx}{sy}"][i] for sx, sy in pairwise_defs]) for i in range(G)])
            return acat(tacat), {f"{sx}{sy}": lambda_gc(pp[f"{sx}{sy}"]) for sx, sy in pairwise_defs}
        except Exception:  # noqa: BLE001
            return None

    triad_emp = float("nan")
    lam_perm = {}
    if perm_B and perm_B > 0:
        res = [r for r in Parallel(n_jobs=n_jobs)(delayed(_perm)(900000 + i) for i in range(perm_B))
               if r is not None]
        if res:
            om = np.array([r[0] for r in res])
            triad_emp = float((1 + int((om <= triad_omnibus).sum())) / (len(om) + 1))
            lam_perm = {tag: float(np.median([r[1][tag] for r in res]))
                        for tag in (f"{sx}{sy}" for sx, sy in pairwise_defs)}

    out = dict(transform=transform, n=int(n_t), G=int(G),
               triad_acat_omnibus=float(triad_omnibus), triad_acat_omnibus_emp=triad_emp,
               pairwise=pw_res, lambda_gc_perm_median=lam_perm, covariates=cov_meta,
               sigma_hat={s: float(cv.get(s, 0.0)) for s in subs} | {"e": float(cv.get("e", 0.0))})
    if w is not None:
        n_cov = int(sum(t in triad_weights for t in kept))
        out["weighted"] = dict(
            triad_acat_omnibus_weighted=float(acat_weighted(triad_acat, w)),
            audit=dict(n_triads=int(G), n_covered=n_cov, n_missing_default1=int(G - n_cov),
                       weight_min=float(w.min()), weight_mean=float(w.mean()), weight_max=float(w.max())),
            note=("per-triad y-independent prior (HEB/DL) frozen pre-association; weighted "
                  "Bonferroni controls per-pairwise FWER, weighted ACAT is the prior-weighted omnibus."))
    return out


def _ols_rss(Xd: np.ndarray, yv: np.ndarray):
    """OLS fit; return (beta, residual-sum-of-squares)."""
    beta, *_ = np.linalg.lstsq(Xd, yv, rcond=None)
    resid = yv - Xd @ beta
    return beta, float(resid @ resid)


def pair_conditional_diagnostics(Wh: np.ndarray, y: np.ndarray, bx: np.ndarray, by: np.ndarray,
                                 C: np.ndarray = None) -> dict:
    """Conditional-on-marginals sanity panel for ONE homoeolog pair in the *whitened* GLS
    ``y_w ~ C + bX + bY + bX*bY`` (the exact design the scan tests). ``bx``/``by`` are the
    column-standardized gene burdens; ``C`` is the fixed-effect covariate block INCLUDING the
    intercept (default ``None`` => intercept-only ``[1]``, bit-exact legacy). Returns:
      - the interaction Wald p/t/beta (must reproduce the deployed hit p);
      - projection-based collinearity in whitened space: R2_int|marg = 1 - SSE(cI~C0+cX+cY)/
        SSE(cI~C0) and VIF_int = 1/(1-R2) (C0 = Wh@C is the whitened covariate block, not a
        constant — use partial/projection R2, not TSS-around-mean);
      - a nested-model F panel: (a) interaction | marginals (= headline, F==t^2 check),
        (b) JOINT main effects null[C0] vs [C0,cX,cY] (single-gene visibility), (c) total pair model;
      - single-gene **burden-GLS marginal** p (each burden alone under the same covariance + C):
        the 'pair-only / single-gene-invisible' evidence.
    All quantities use the identical whitener/burdens/covariates as the production scan, so the
    panel stays a faithful persisted diagnostic under covariate (PC) adjustment too."""
    n = int(y.shape[0])
    C0 = (Wh @ np.ones(n)).reshape(-1, 1) if C is None else Wh @ np.asarray(C, float).reshape(n, -1)
    p_c = C0.shape[1]                                   # covariate-block width (1 = intercept only)
    cX = Wh @ bx
    cY = Wh @ by
    cI = Wh @ (bx * by)
    yw = Wh @ y

    Xfull = np.column_stack([C0, cX, cY, cI])           # interaction at index p_c+2
    Xmain = np.column_stack([C0, cX, cY])
    Xnull = C0
    j_int = p_c + 2
    bfull, rss_full = _ols_rss(Xfull, yw)
    bmain, rss_main = _ols_rss(Xmain, yw)
    _, rss_null = _ols_rss(Xnull, yw)

    # rank-aware df (== n-(p_c+3) when the design is full rank; guards collinear covariates)
    rank_full = int(np.linalg.matrix_rank(Xfull))
    rank_main = int(np.linalg.matrix_rank(Xmain))
    estimable = bool(rank_full == p_c + 3)
    df = n - rank_full
    s2 = rss_full / df
    se_I = np.sqrt(max(s2 * np.linalg.pinv(Xfull.T @ Xfull)[j_int, j_int], 1e-30))
    t_I = float(bfull[j_int] / se_I)
    p_int = float(2.0 * stats.t.sf(abs(t_I), df))

    # projection-based partial R2 / VIF of the interaction column vs the covariate+marginal columns
    _, sse0 = _ols_rss(Xnull, cI)                       # cI ~ C0
    _, sse1 = _ols_rss(Xmain, cI)                       # cI ~ C0 + cX + cY
    r2_im = float(1.0 - sse1 / sse0) if sse0 > 0 else float("nan")
    vif = float(1.0 / (1.0 - r2_im)) if np.isfinite(r2_im) and r2_im < 1 else float("inf")

    def _F(rss_r, rss_f, df_num, df_den):
        F = ((rss_r - rss_f) / df_num) / (rss_f / df_den)
        return float(F), float(stats.f.sf(F, df_num, df_den))

    df_main = n - rank_main
    F_int, pF_int = _F(rss_main, rss_full, 1, df)        # interaction | marginals (== Wald)
    F_main, pF_main = _F(rss_null, rss_main, 2, df_main)  # joint main effects (single-gene visibility)
    F_pair, pF_pair = _F(rss_null, rss_full, 3, df)      # total pair model

    # single-gene burden-GLS marginal p (each burden alone, same whitener + covariates)
    def _marg_p(c):
        Xm = np.column_stack([C0, c])
        bm, rss_m = _ols_rss(Xm, yw)
        dfm = n - (p_c + 1)
        se = np.sqrt(max((rss_m / dfm) * np.linalg.pinv(Xm.T @ Xm)[p_c, p_c], 1e-30))
        return float(2.0 * stats.t.sf(abs(bm[p_c] / se), dfm))

    # joint main-effect coefficient p (each in presence of the other + covariates)
    se_main = np.sqrt(np.maximum((rss_main / df_main) * np.diag(np.linalg.pinv(Xmain.T @ Xmain)), 1e-30))
    p_mainX = float(2.0 * stats.t.sf(abs(bmain[p_c] / se_main[p_c]), df_main))
    p_mainY = float(2.0 * stats.t.sf(abs(bmain[p_c + 1] / se_main[p_c + 1]), df_main))

    # Frisch-Waugh-Lovell confirmation: the interaction t computed on the part of cI ORTHOGONAL to
    # the marginals (+covariates) equals the full-model interaction t.
    b_proj, _ = _ols_rss(Xmain, cI)
    cI_resid = cI - Xmain @ b_proj
    Xres = np.column_stack([C0, cX, cY, cI_resid])
    bres, rss_res = _ols_rss(Xres, yw)
    se_res = np.sqrt(max((rss_res / df) * np.linalg.pinv(Xres.T @ Xres)[j_int, j_int], 1e-30))
    t_resid = float(bres[j_int] / se_res)

    return dict(
        n=n, n_covariates=p_c, design_rank=rank_full, interaction_estimable=estimable,
        interaction_p_wald=p_int, interaction_t=t_I,
        interaction_beta=float(bfull[j_int]), vif_int=vif, r2_int_given_marginals=r2_im,
        residualized_interaction=dict(t=t_resid, absdiff_vs_full_t=float(abs(t_resid - t_I))),
        corr_int_bX=float(np.corrcoef(cI, cX)[0, 1]), corr_int_bY=float(np.corrcoef(cI, cY)[0, 1]),
        nested_interaction_given_marginals=dict(
            F=F_int, p=pF_int, df=[1, df],
            check_F_eq_t2=dict(t2=float(t_I ** 2), absdiff=float(abs(F_int - t_I ** 2)),
                               p_wald=p_int, p_F_minus_p_wald=float(abs(pF_int - p_int)))),
        nested_joint_main_effects=dict(F=F_main, p=pF_main, df=[2, df_main]),
        nested_total_pair_model=dict(F=F_pair, p=pF_pair, df=[3, df]),
        single_gene_marginal=dict(p_X_alone=_marg_p(cX), p_Y_alone=_marg_p(cY),
                                  p_X_joint=p_mainX, p_Y_joint=p_mainY))


def run_multitrait_pair_scan(
    subdata: dict[str, SubgenomeData],
    pairs: list[tuple],
    y_by_trait: dict[str, np.ndarray],  # trait -> y_raw on complete-case samples (order = trait_set)
    sample_idx: np.ndarray,
    *,
    trait_set: FrozenTraitSet,          # frozen, ordered; keys of y_by_trait MUST match
    cap: int = 150,
    transform: str = "INT",
    perm_B: int = 2000,
    n_jobs: int = 8,
    seed: int = 7,
    pair_subs: tuple = None,
    grm_method: str = "compute_grm_maf",
    maf_min: float = 0.01,
) -> dict:
    """Multi-trait (pleiotropy) pairwise scan: ACAT-across-traits.

    For each homoeolog pair, the per-trait whitened interaction p-values over a PREDECLARED,
    frozen trait set are combined with ACAT into a single per-pair pleiotropy p. Multiplicity
    is over G pairs only (each pair yields exactly one combined p), NOT G x T. ACAT is
    dependence-robust so the combination stays calibrated under cross-trait correlation; it is
    not the optimal combiner for dense weak same-direction effects, so the discover-more gain is
    conditional (relative to G x T single-trait Bonferroni when signal is shared across traits).

    Calibration uses a SHARED permutation: one row permutation per replicate applied to every
    trait's phenotype vector, preserving the empirical cross-trait correlation while breaking the
    genotype-phenotype link. The whitener depends on y (REML), so each trait is re-whitened in
    each permutation (exact procedural null). ``audit_min_trait_*`` fields are reported for
    transparency and are NOT inferential (no significance is called on them)."""
    from joblib import Parallel, delayed

    if pair_subs is None:
        raise ValueError("pair_subs=(sx, sy) is required for the pairwise multi-trait scan")
    rng = np.random.default_rng(seed)
    subs = list(subdata.keys())
    sx, sy = pair_subs
    n_t = sample_idx.size
    traits = list(trait_set.traits)
    if list(y_by_trait.keys()) != traits:
        raise ValueError("y_by_trait key order must match the frozen trait set "
                         f"{traits}; got {list(y_by_trait.keys())}")
    for t in traits:
        yt = np.asarray(y_by_trait[t], float)
        if yt.shape[0] != n_t:
            raise ValueError(f"trait '{t}' has {yt.shape[0]} values; expected {n_t} "
                             "(complete-case sample count)")
        if not np.all(np.isfinite(yt)):
            raise ValueError(f"trait '{t}' contains non-finite values; pass complete-case "
                             "(NA-filtered across the whole trait set) phenotypes")

    # genotype-only objects built ONCE (trait-independent)
    kernels = {s: _build_grm(subdata[s], sample_idx, grm_method, maf_min) for s in subs}
    bx_cols, by_cols, kept_pairs = [], [], []
    for gx, gy in pairs:
        if gx in subdata[sx].gene_snp and gy in subdata[sy].gene_snp:
            bx_cols.append(block_burden_capped(subdata[sx].X, subdata[sx].gene_snp[gx], cap, rng)[sample_idx])
            by_cols.append(block_burden_capped(subdata[sy].X, subdata[sy].gene_snp[gy], cap, rng)[sample_idx])
            kept_pairs.append((gx, gy))
    G = len(kept_pairs)
    if G < 1:
        raise ValueError("no homoeolog pairs retained (none present in both subgenomes); "
                         "check pairs table vs snp_to_gene gene IDs")
    BX = scols_safe(np.column_stack(bx_cols))
    BY = scols_safe(np.column_stack(by_cols))

    # per-trait observed transform + whitened per-pair interaction p (whitener depends on y).
    # Observed whitening uses a deterministic per-trait seed (perms use the rep seed); REML is
    # 3-start (picks best logL) so the optimum is stable regardless of seed.
    y_t = {t: (rank_int(np.asarray(y_by_trait[t], float)) if transform == "INT"
               else np.asarray(y_by_trait[t], float)) for t in traits}
    P = np.empty((G, len(traits)))
    cv_by_trait = {}
    for j, t in enumerate(traits):
        Wh, cv = whiten_multi(kernels, y_t[t], seed=42 + j)
        pv = pairwise_pvals(Wh, y_t[t], BX, BY)
        P[:, j] = np.where(np.isfinite(pv), pv, 1.0)
        cv_by_trait[t] = {s: float(cv.get(s, 0.0)) for s in subs} | {"e": float(cv.get("e", 0.0))}

    pleio_p = np.array([acat(P[i]) for i in range(G)])
    audit_min_p = P.min(1)
    audit_min_trait = [traits[int(j)] for j in P.argmin(1)]

    pleio_omnibus = acat(pleio_p)
    minp_obs = float(pleio_p.min())
    lam_obs = lambda_gc(pleio_p)
    bonf = 0.05 / G                                     # multiplicity over G pairs, not G x T
    order = np.argsort(pleio_p)
    sig = [dict(pair=kept_pairs[int(i)], pleio_p=float(pleio_p[i]),
                per_trait_p={t: float(P[i, j]) for j, t in enumerate(traits)})
           for i in order if pleio_p[i] < bonf]
    top = [dict(pair=kept_pairs[int(i)], pleio_p=float(pleio_p[i]),
                audit_min_trait_p=float(audit_min_p[i]), audit_min_trait_name=audit_min_trait[int(i)],
                per_trait_p={t: float(P[i, j]) for j, t in enumerate(traits)})
           for i in order[:5]]

    # shared-permutation calibration (same row permutation across all traits)
    def _perm(seed_i):
        r = np.random.default_rng(seed_i)
        perm = r.permutation(n_t)
        cols = []
        for t in traits:
            ys = y_t[t][perm]
            try:
                Whp, _ = whiten_multi(kernels, ys, seed=seed_i % 100000)
                pp = pairwise_pvals(Whp, ys, BX, BY)
            except Exception:  # noqa: BLE001
                return None
            cols.append(np.where(np.isfinite(pp), pp, 1.0))
        Pp = np.column_stack(cols)
        pleio_perm = np.array([acat(Pp[i]) for i in range(G)])
        return acat(pleio_perm), float(pleio_perm.min())

    pleio_emp = float("nan")
    minp_emp = float("nan")
    if perm_B and perm_B > 0:
        res = [r for r in Parallel(n_jobs=n_jobs)(delayed(_perm)(900000 + i) for i in range(perm_B))
               if r is not None]
        if res:
            om = np.array([r[0] for r in res])
            mp = np.array([r[1] for r in res])
            pleio_emp = float((1 + int((om <= pleio_omnibus).sum())) / (len(om) + 1))
            minp_emp = float((1 + int((mp <= minp_obs).sum())) / (len(mp) + 1))

    return dict(
        transform=transform, n=int(n_t), G=int(G), traits=list(traits),
        trait_set_digest=trait_set.digest, single_trait=bool(len(traits) == 1),
        pleio_acat_omnibus=float(pleio_omnibus), pleio_acat_omnibus_emp=pleio_emp,
        min_p=minp_obs, min_p_emp=minp_emp, lambda_gc_obs=float(lam_obs),
        bonferroni_alpha=float(bonf), n_sig=len(sig), sig=sig, top=top,
        sigma_hat_by_trait=cv_by_trait,
        note=("DEGENERATE single-trait set: pleiotropy ACAT reduces to the single-trait scan. "
              if len(traits) == 1 else "")
        + ("multiplicity is over G pairs (one ACAT-combined pleiotropy p per pair) for ONE "
              "frozen multi-trait family; audit_min_trait_* are reported for transparency and are "
              "NOT inferential. ACAT is dependence-robust (valid under cross-trait correlation) but "
              "is not the optimal combiner for dense weak same-direction effects, so the "
              "discover-more claim is conditional on signal being shared across the frozen traits, "
              "relative to G x T single-trait Bonferroni."))


# ----------------------------------------------------------------------------- CLI


def _load_pairs(path: str, subs: list[str]):
    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    # accept gene_<S> columns; for 2 subgenomes use (gene_<s0>, gene_<s1>)
    cols = [f"gene_{s}" for s in subs]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"pairs table missing columns {missing}; has {list(df.columns)}")
    return [tuple(r) for r in df[cols].itertuples(index=False, name=None)]


def _load_pair_weights(path: str, subs: list[str]) -> dict:
    """Load y-independent pair priors -> {(gene_s0, gene_s1): weight}. Columns:
    gene_<S> per subgenome + 'weight'. Weights must be computed WITHOUT the phenotype
    (DL zero-shot scores, homoeolog expression bias), frozen before this scan."""
    import pandas as pd

    df = pd.read_csv(path, sep="\t")
    cols = [f"gene_{s}" for s in subs]
    if "weight" not in df.columns or any(c not in df.columns for c in cols):
        raise ValueError(f"weights table needs columns {cols + ['weight']}; has {list(df.columns)}")
    return {tuple(r[:-1]): float(r[-1])
            for r in df[cols + ["weight"]].itertuples(index=False, name=None)}


def _build_cov_arg(cov_cfg, valid: list):
    """Parse the ``interact.covariates`` config into the ``covariates`` dict {n_pcs, extra} that the
    scan functions accept. Returns ``(None, "none")`` when absent (exact legacy reproduction).

    Config::

        covariates:
          n_pcs: 10                 # genotype PCs from the combined subgenome GRM (y-independent)
          file: covars.tsv          # optional y-independent covariate table (sample_col + columns)
          sample_col: sample
          columns: [env, batch]     # optional subset; default = all non-sample columns

    The covariate file MUST be y-independent and frozen before association (same firewall as
    weights). Extra covariates are aligned to the complete-case ``valid`` sample order."""
    if not cov_cfg:
        return None, "none"
    n_pcs = int(cov_cfg.get("n_pcs", 0))
    extra = None
    label = f"n_pcs={n_pcs}"
    if cov_cfg.get("file"):
        import pandas as pd
        cf = pd.read_csv(cov_cfg["file"], sep="\t").set_index(cov_cfg.get("sample_col", "sample"))
        cols = cov_cfg.get("columns") or [c for c in cf.columns]
        missing_s = [s for s in valid if s not in cf.index]
        if missing_s:
            raise ValueError(f"covariate file missing {len(missing_s)} samples (e.g. {missing_s[:3]})")
        extra = cf.loc[valid, cols].to_numpy(dtype=float)
        if not np.all(np.isfinite(extra)):
            raise ValueError("covariate file contains non-finite values for the complete-case samples")
        label += f"+file[{','.join(map(str, cols))}]"
    if n_pcs <= 0 and extra is None:
        return None, "none"
    return dict(n_pcs=n_pcs, extra=extra), label


def _run_multitrait(args, ic, subs, out_dir, subdata, samples, ph, t0) -> int:
    """Multi-trait (pleiotropy) CLI path: complete-case across a frozen trait set, ACAT-across-traits.

    The trait set is frozen (FrozenTraitSet) before any association; missing traits raise rather
    than silently drop (a leakage guard), so multiplicity is honestly over G pairs."""
    import hashlib

    import pandas as pd

    trait_set = FrozenTraitSet.from_list(ic["multi_trait"])
    tlist = list(trait_set.traits)
    missing_tr = [t for t in tlist if t not in ph.columns]
    if missing_tr:
        print(f"❌ interact multi_trait: traits not in phenotype {missing_tr} "
              f"(available {list(ph.columns)[:20]}...)")
        return 1

    # complete-case: rows non-NA across ALL frozen traits (one shared sample set)
    valid = [s for s in samples if s in ph.index and bool(ph.loc[s, tlist].notna().all())]
    if len(valid) < 10:
        print(f"❌ interact multi_trait: only {len(valid)} complete-case samples across "
              f"{len(tlist)} traits (need >=10)")
        return 1
    sample_idx = np.array([samples.index(s) for s in valid])
    y_by_trait = {t: np.array([float(ph.loc[s, t]) for s in valid]) for t in tlist}
    per_trait_present = {t: int(sum((s in ph.index) and pd.notna(ph.loc[s, t]) for s in samples))
                         for t in tlist}

    pairs = _load_pairs(ic["pairs"], subs)
    burden = ic.get("burden", {})
    cap = int(burden.get("cap", 150))
    perm_B = int(ic.get("calibration", {}).get("perm_B", 2000))
    grm_cfg = ic.get("grm", {})
    grm_method = grm_cfg.get("method", "compute_grm_maf")
    maf_min = float(grm_cfg.get("maf_min", 0.01))
    n_jobs = int(args.n_jobs)

    print(f"  multi_trait n_complete_case={len(valid)} traits={len(tlist)} digest={trait_set.digest} "
          f"pairs(raw)={len(pairs)} ({time.time()-t0:.1f}s)", flush=True)
    results = {}
    for transform in ("INT", "raw"):
        r = run_multitrait_pair_scan(subdata, pairs, y_by_trait, sample_idx, trait_set=trait_set,
                                     cap=cap, transform=transform,
                                     perm_B=(perm_B if transform == "INT" else 0), n_jobs=n_jobs,
                                     pair_subs=(subs[0], subs[1]), grm_method=grm_method,
                                     maf_min=maf_min)
        results[transform] = r
        print(f"  [{transform}] G={r['G']} pleio_ACAT_omnibus={r['pleio_acat_omnibus']:.3g} "
              f"emp={r['pleio_acat_omnibus_emp']} minP={r['min_p']:.3g} λ_obs={r['lambda_gc_obs']:.3f} "
              f"nsig(α={r['bonferroni_alpha']:.1e})={r['n_sig']}", flush=True)
        for h in r["sig"]:
            print(f"      HIT {h['pair']} pleio_p={h['pleio_p']:.3g}")

    from . import __version__
    cc_hash = hashlib.sha256("\x00".join(valid).encode()).hexdigest()[:16]
    pairs_path = ic["pairs"]
    pairs_sha = (hashlib.sha256(Path(pairs_path).read_bytes()).hexdigest()[:16]
                 if Path(pairs_path).exists() else None)
    provenance = dict(
        version=__version__, mode="pairwise", multi_trait=True, transform="INT(primary)+raw(sens)",
        grm_method=grm_method, maf_min=maf_min, burden_cap=cap, perm_B=perm_B,
        covariate_policy="none: subgenome-stratified GRMs only (no PCs/covariates)",
        trait_set=tlist, trait_set_digest=trait_set.digest, n_traits=len(tlist),
        n_complete_case=len(valid), complete_case_sha256=cc_hash, complete_case_sample_order=valid,
        per_trait_present=per_trait_present, n_pairs_raw=len(pairs),
        pairs_source=pairs_path, pairs_sha256=pairs_sha,
        psd_floor=1e-6 if grm_method == "grm_from_X" else None, config_path=str(args.config),
        firewall=("trait set is predeclared and FROZEN before association (digest recorded); it must "
                  "NOT be chosen by per-trait GWAS significance, and covariate/PC count must be fixed "
                  "a priori by variance explained. Multiplicity is over G pairs only because each pair "
                  "yields exactly one ACAT-combined p across the frozen trait family. No trait "
                  "selection occurs inside the scan; missing traits raise rather than silently drop."))
    payload = dict(tool="homoeogwas", command="interact", mode="pairwise", multi_trait=True,
                   subgenomes=subs, traits=tlist, provenance=provenance, results=results)
    fp = out_dir / f"interact_multitrait_{trait_set.digest}.json"
    fp.write_text(json.dumps(payload, indent=2, default=float))
    print(f"✅ homoeogwas interact (multi-trait) -> {fp} ({time.time()-t0:.1f}s)")
    return 0


def cmd_interact(args) -> int:
    import pandas as pd
    import yaml

    t0 = time.time()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    ic = cfg["interact"]
    subs = list(ic["subgenomes"])
    mode = ic.get("mode", "pairwise")
    if mode == "pairwise" and len(subs) != 2:
        print(f"❌ interact mode=pairwise needs exactly 2 subgenomes; got {subs}. "
              f"For hexaploids use mode=triad with 3 subgenomes.")
        return 1
    if mode == "triad" and len(subs) != 3:
        print(f"❌ interact mode=triad needs exactly 3 subgenomes; got {subs}.")
        return 1
    out_dir = Path(args.out_dir or cfg.get("outputs", {}).get("out_dir", "results/interact"))
    out_dir.mkdir(parents=True, exist_ok=True)

    trait_label = ic.get("trait") if ic.get("multi_trait") is None else f"multi:{ic['multi_trait']}"
    print(f"=== homoeogwas interact (mode={mode}) — subgenomes={subs} trait={trait_label} ===",
          flush=True)
    subdata = {s: _load_subgenome(ic["genotype"][s], ic["snp_to_gene"][s]) for s in subs}
    samples = subdata[subs[0]].samples
    for s in subs[1:]:
        if subdata[s].samples != samples:
            print(f"❌ interact: sample order mismatch between {subs[0]} and {s}")
            return 1

    ph = pd.read_csv(ic["phenotype"], sep="\t").set_index(ic.get("sample_col", "sample"))

    if ic.get("multi_trait") is not None:
        if mode != "pairwise":
            print(f"❌ interact: multi_trait is pairwise-only; mode={mode} not supported (future).")
            return 1
        return _run_multitrait(args, ic, subs, out_dir, subdata, samples, ph, t0)

    trait = ic["trait"]
    valid = [s for s in samples if s in ph.index and pd.notna(ph.loc[s, trait])]
    sample_idx = np.array([samples.index(s) for s in valid])
    y_raw = np.array([float(ph.loc[s, trait]) for s in valid])

    burden = ic.get("burden", {})
    cap = int(burden.get("cap", 150))
    perm_B = int(ic.get("calibration", {}).get("perm_B", 2000))
    grm_cfg = ic.get("grm", {})
    grm_method = grm_cfg.get("method", "compute_grm_maf")   # default reproduces research scripts
    maf_min = float(grm_cfg.get("maf_min", 0.01))
    n_jobs = int(args.n_jobs)

    # optional fixed-effect covariates (genotype PCs and/or a y-independent covariate file).
    # Absent => covariate_policy "none" (exact legacy reproduction). PCs are opt-in.
    cov_arg, cov_label = _build_cov_arg(ic.get("covariates"), valid)
    if cov_arg:
        print(f"  covariates: {cov_label}", flush=True)

    # opt-in FULL genome-wide ranking dump (outputs.full_ranking: true). Off => legacy bit-exact.
    dump_on = bool(cfg.get("outputs", {}).get("full_ranking", False))

    def _dump_path(transform):
        return str(out_dir / f"interact_{trait}_ranking_{mode}_{transform}.tsv") if dump_on else None

    if mode == "triad":
        triads = _load_pairs(ic["triads"], subs)            # gene_<S> columns -> (g0,g1,g2)
        triad_weights = _load_pair_weights(ic.get("weights"), subs) if ic.get("weights") else None
        print(f"  n={len(valid)} triads(raw)={len(triads)}"
              + (f" weights={len(triad_weights)}" if triad_weights else "")
              + f" ({time.time()-t0:.1f}s)", flush=True)
        results = {}
        for transform in ("INT", "raw"):
            r = run_triad_scan(subdata, triads, y_raw, sample_idx, cap=cap, transform=transform,
                               perm_B=(perm_B if transform == "INT" else 0), n_jobs=n_jobs,
                               grm_method=grm_method, maf_min=maf_min, triad_weights=triad_weights,
                               covariates=cov_arg, full_dump_path=_dump_path(transform))
            results[transform] = r
            print(f"  [{transform}] G={r.get('G')} triad_ACAT_omnibus={r.get('triad_acat_omnibus'):.3g} "
                  f"emp={r.get('triad_acat_omnibus_emp')}"
                  + (f" | weighted_omnibus={r['weighted']['triad_acat_omnibus_weighted']:.3g}"
                     if r.get("weighted") else ""), flush=True)
            for tag, pw in r.get("pairwise", {}).items():
                wstr = (f" | wACAT={pw['weighted']['acat_weighted']:.3g} "
                        f"wnsig={pw['weighted']['bonferroni_n_sig']}" if pw.get("weighted") else "")
                print(f"    {tag}: ACAT={pw['acat']:.3g} minP={pw['min_p']:.3g} "
                      f"λ_obs={pw['lambda_gc_obs']:.3f} nsig(α={pw['bonferroni_alpha']:.1e})={pw['n_sig']}{wstr}")
                for h in pw["sig"]:
                    print(f"        HIT {h['triad']} p={h['p']:.3g}")
        n_units = len(triads)
    else:
        pairs = _load_pairs(ic["pairs"], subs)
        pair_weights = _load_pair_weights(ic.get("weights"), subs) if ic.get("weights") else None
        print(f"  n={len(valid)} pairs(raw)={len(pairs)}"
              + (f" weights={len(pair_weights)}" if pair_weights else "")
              + f" ({time.time()-t0:.1f}s)", flush=True)
        results = {}
        for transform in ("INT", "raw"):
            r = run_pair_scan(subdata, pairs, y_raw, sample_idx, cap=cap, transform=transform,
                              perm_B=(perm_B if transform == "INT" else 0), n_jobs=n_jobs,
                              pair_subs=(subs[0], subs[1]), grm_method=grm_method, maf_min=maf_min,
                              pair_weights=pair_weights, covariates=cov_arg,
                              full_dump_path=_dump_path(transform))
            r.trait = trait
            results[transform] = r.__dict__
            print(f"  [{transform}] G={r.G} pair_ACAT={r.pair_acat:.3g} emp={r.pair_acat_emp} "
                  f"minP={r.min_p:.3g} λ_obs={r.lambda_gc_obs:.3f} λ_perm={r.lambda_gc_perm_median} "
                  f"nsig(α={r.bonferroni_alpha:.1e})={r.n_sig}", flush=True)
            for h in r.sig:
                print(f"      HIT {h['pair']} p={h['p']:.3g}")
            if r.weighted:
                print(f"      [weighted] ACAT={r.weighted['acat_weighted']:.3g} "
                      f"nsig={r.weighted['bonferroni_n_sig']}", flush=True)
                for h in r.weighted["sig"]:
                    print(f"        WHIT {h['pair']} p={h['p']:.3g} w={h['weight']:.2f}")
        n_units = len(pairs)

    from . import __version__
    weights_path = ic.get("weights")
    weights_sha = None
    if weights_path and Path(weights_path).exists():
        import hashlib
        weights_sha = hashlib.sha256(Path(weights_path).read_bytes()).hexdigest()[:16]
    provenance = dict(version=__version__, mode=mode, grm_method=grm_method, maf_min=maf_min,
                      burden_cap=cap, perm_B=perm_B, n_samples=len(valid), n_units_raw=n_units,
                      psd_floor=1e-6 if grm_method == "grm_from_X" else None,
                      covariate_policy=(cov_label if cov_arg else
                                        "none: subgenome-stratified GRMs only (no PCs/covariates)"),
                      covariates_detail=results["INT"].get("covariates"),
                      covariate_permutation=("freedman-lane (residualize on C, permute residuals; "
                                             "reduces to y-shuffle when intercept-only)"
                                             if cov_arg else "y-shuffle"),
                      covariate_firewall=("covariates (genotype PCs / environment) must be "
                                          "y-independent and the PC count fixed a priori; PCs enter "
                                          "both the null-LMM mean model and the per-pair GLS design"),
                      weights_source=weights_path, weights_sha256=weights_sha,
                      weights_firewall="weights must be y-independent (DL/HEB) and frozen "
                                       "pre-association; not enforced by the tool",
                      full_ranking=dump_on,
                      full_ranking_note=("full per-unit ranking TSV is descriptive (every callable "
                                         "unit); inference is the pre-registered enrichment, not "
                                         "per-row Bonferroni; gene_len emitted as NA (engine has no "
                                         "coordinates) and joined downstream from annotation"),
                      config_path=str(args.config))
    payload = dict(tool="homoeogwas", command="interact", mode=mode, subgenomes=subs, trait=trait,
                   provenance=provenance, results=results)
    fp = out_dir / f"interact_{trait}.json"
    fp.write_text(json.dumps(payload, indent=2, default=float))
    print(f"✅ homoeogwas interact -> {fp} ({time.time()-t0:.1f}s)")
    return 0


def add_interact_subparser(sub) -> None:
    ap = sub.add_parser("interact", help="gene-resolution homoeolog-pair burden-product "
                                         "interaction scan from a YAML config")
    ap.add_argument("-c", "--config", required=True, help="YAML run-config path")
    ap.add_argument("-o", "--out-dir", default=None, help="override outputs.out_dir")
    ap.add_argument("--n-jobs", type=int, default=8, help="parallel workers for permutation")
