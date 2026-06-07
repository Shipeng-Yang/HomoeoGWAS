"""Per-SNP association scan under the subgenome-stratified LMM.

EMMAX / P3D style: variance components are estimated once genome-wide
(multi-kernel REML), then each marker is tested with the covariance V held
fixed. The per-SNP test reduces to a fixed projection:

    V   = σ²_e I + Σ_j σ²_j K_j                    (absolute variance comps)
    P   = V⁻¹ − V⁻¹X (X'V⁻¹X)⁻¹ X'V⁻¹             (n×n, built once)
    U_g = g'P y,   I_g = g'P g
    γ̂   = U_g / I_g,   SE = √(1/I_g),   χ² = U_g²/I_g  ~  χ²₁

Batched over markers this is one GEMM ``P @ G``.

The scanned marker also sits in the GRM (proximal contamination); the plain
EMMAX/P3D scan does not correct for it. Use the LOCO variants for that.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_CHI2_1_MEDIAN = 0.4549364231195724   # median of the χ²₁ distribution


@dataclass(frozen=True)
class ScanContext:
    """Fixed-V projection for a per-SNP scan (built once, reused per marker)."""
    P: np.ndarray                       # (n,n) projection V⁻¹ − V⁻¹X(X'V⁻¹X)⁻¹X'V⁻¹
    Py: np.ndarray                      # (n,) P @ y
    n: int
    p: int
    sigma2: dict[str, float]            # absolute variance components (incl. "e")
    kernel_names: list[str]
    sample_ids: np.ndarray | None       # (n,) IID order P/Py were built in
    jitter_used: float
    dtype: str = "float64"


def _cholesky_jitter(V: np.ndarray, mean_diag: float,
                     ladder=(0.0, 1e-10, 1e-8, 1e-6)) -> tuple[np.ndarray, float]:
    """Cholesky factor with a progressive jitter ladder; returns (c, jitter)."""
    from scipy.linalg import cho_factor
    n = V.shape[0]
    last: Exception | None = None
    for j in ladder:
        try:
            Vj = V + j * mean_diag * np.eye(n) if j > 0 else V
            c = cho_factor(Vj, lower=True, check_finite=False)
            return c, j
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"Cholesky of V failed even at jitter {ladder[-1]}: {last}")


def build_scan_context(
    y: np.ndarray,
    X: np.ndarray,
    kernels: dict[str, np.ndarray],
    sigma2: dict[str, float],
    *,
    sample_ids: np.ndarray | None = None,
    dtype: str = "float64",
    check_tol: float = 1e-6,
) -> ScanContext:
    """Build the fixed-V projection P from genome-wide REML variance components.

    Args:
        y: phenotype (n,).
        X: fixed-effect design (n,p); must include an intercept column.
        kernels: {name: K (n,n)} GRM-like kernels of the null model.
        sigma2: absolute variance components from fit_multi_reml, keys =
            kernel names + "e". V = σ²_e I + Σ σ²_j K_j (phenotype scale).
        sample_ids: optional (n,) IID order, stored so scan_snps can align
            genotypes to it.
        check_tol: tolerance for the P symmetry / P@X≈0 checks.

    Returns:
        ScanContext.
    """
    y = np.ascontiguousarray(y, dtype=np.float64).ravel()
    X = np.ascontiguousarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError(f"y length {y.shape[0]} != X rows {n}")
    if p >= n:
        raise ValueError(f"design rank issue: p={p} >= n={n}")
    if not kernels:
        raise ValueError("kernels dict is empty")
    kernel_names = list(kernels.keys())
    for k in (*kernel_names, "e"):
        if k not in sigma2:
            raise ValueError(f"sigma2 missing component {k!r}")
        if not np.isfinite(sigma2[k]) or sigma2[k] < 0:
            raise ValueError(f"sigma2[{k!r}]={sigma2[k]} must be finite and >= 0")

    # V = σ²_e I + Σ σ²_j K_j   (symmetrized)
    V = float(sigma2["e"]) * np.eye(n, dtype=np.float64)
    for name in kernel_names:
        K = np.asarray(kernels[name], dtype=np.float64)
        if K.shape != (n, n):
            raise ValueError(f"kernels[{name!r}] must be ({n},{n}), got {K.shape}")
        V = V + float(sigma2[name]) * K
    V = 0.5 * (V + V.T)

    mean_diag = float(np.trace(V) / n)
    c = _cholesky_jitter(V, mean_diag)
    cfac, jitter = c
    from scipy.linalg import cho_solve

    Vinv = cho_solve(cfac, np.eye(n), check_finite=False)
    Vinv = 0.5 * (Vinv + Vinv.T)
    VinvX = cho_solve(cfac, X, check_finite=False)          # (n,p)
    XtVinvX = X.T @ VinvX                                   # (p,p)
    # P = V⁻¹ − VinvX (X'V⁻¹X)⁻¹ VinvX'
    mid = np.linalg.solve(XtVinvX, VinvX.T)                 # (p,n)
    P = Vinv - VinvX @ mid
    P = 0.5 * (P + P.T)

    # P must be symmetric and annihilate the fixed effects (P @ X ≈ 0)
    sym_err = float(np.max(np.abs(P - P.T)))
    px_err = float(np.max(np.abs(P @ X)))
    scale = max(float(np.max(np.abs(P))), 1.0)
    if sym_err > check_tol * scale:
        raise RuntimeError(f"P not symmetric: max|P-P'|={sym_err:.3g}")
    if px_err > check_tol * scale:
        raise RuntimeError(f"P @ X not ~0: max|PX|={px_err:.3g}")

    Py = P @ y

    sids = None if sample_ids is None else np.asarray(sample_ids, dtype=object)
    if sids is not None:
        if sids.shape[0] != n:
            raise ValueError(f"sample_ids length {sids.shape[0]} != n {n}")
        if len(set(sids.tolist())) != sids.shape[0]:
            raise ValueError("sample_ids must be unique — duplicates would "
                             "misalign genotype rows to P / Py")

    return ScanContext(
        P=np.ascontiguousarray(P), Py=np.ascontiguousarray(Py),
        n=n, p=p,
        sigma2={k: float(v) for k, v in sigma2.items()},
        kernel_names=kernel_names, sample_ids=sids,
        jitter_used=float(jitter), dtype=dtype,
    )


@dataclass(frozen=True)
class ScanResult:
    """Per-SNP association statistics (kept markers only)."""
    snp_id: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    chi2: np.ndarray
    p: np.ndarray
    n_obs: np.ndarray
    maf: np.ndarray
    call_rate: np.ndarray
    backend_used: str
    n_input: int
    n_kept: int
    filter_counts: dict[str, int] = field(default_factory=dict)

    @property
    def lambda_gc(self) -> float:
        return lambda_gc(self.chi2)


def lambda_gc(chi2: np.ndarray) -> float:
    """Genomic inflation factor λ_GC = median(χ²) / median(χ²₁)."""
    chi2 = np.asarray(chi2, dtype=np.float64)
    chi2 = chi2[np.isfinite(chi2)]
    if chi2.size == 0:
        return float("nan")
    return float(np.median(chi2) / _CHI2_1_MEDIAN)


def _scan_batch_cpu(P: np.ndarray, Py: np.ndarray, G: np.ndarray):
    """U = G'Py, info = diag(G'PG) for a SNP batch G (n,B). numpy float64."""
    PG = P @ G
    U = G.T @ Py
    info = np.einsum("ij,ij->j", G, PG, optimize=True)
    return U, info


def _scan_batch_gpu(P_t, Py_t, G: np.ndarray, device):
    """GPU counterpart of _scan_batch_cpu (torch)."""
    import torch
    G_t = torch.as_tensor(np.ascontiguousarray(G), dtype=P_t.dtype, device=device)
    PG = P_t @ G_t
    U = G_t.t() @ Py_t
    info = (G_t * PG).sum(dim=0)
    return U.detach().cpu().numpy(), info.detach().cpu().numpy()


def _resolve_backend(backend: str) -> str:
    """Resolve 'auto'/'cpu'/'gpu' to a concrete backend."""
    if backend == "cpu":
        return "cpu"
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except ImportError:
        has_gpu = False
    if backend == "gpu":
        if not has_gpu:
            raise RuntimeError("backend='gpu' but torch/CUDA not available")
        return "gpu"
    if backend == "auto":
        return "gpu" if has_gpu else "cpu"
    raise ValueError(f"unknown backend={backend!r}")


def _impute_filter_batch(G_raw: np.ndarray, maf_min: float, call_rate_min: float):
    """Mean-impute missing, compute QC, return (G, keep, cr, maf, n_obs, var).

    G_raw: (n, B) with NaN missing; per SNP, NaN is filled with the
    non-missing mean. keep is False on all-missing / call_rate / MAF /
    zero-variance failure.
    """
    n = G_raw.shape[0]
    obs = np.isfinite(G_raw)
    n_obs = obs.sum(axis=0).astype(np.int64)              # (B,)
    call_rate = n_obs / n
    with np.errstate(invalid="ignore", divide="ignore"):
        col_sum = np.where(obs, G_raw, 0.0).sum(axis=0)
        col_mean = np.where(n_obs > 0, col_sum / np.maximum(n_obs, 1), np.nan)
    G = np.where(obs, G_raw, col_mean[None, :])           # mean-imputed
    # allele frequency from imputed dosage in {0,1,2}; all-missing -> nan
    with np.errstate(invalid="ignore"):
        af = G.mean(axis=0) / 2.0
        var = G.var(axis=0)
    maf = np.minimum(af, 1.0 - af)
    keep = (
        (n_obs > 0)
        & (call_rate >= call_rate_min)
        & np.isfinite(maf) & (maf >= maf_min)
        & np.isfinite(var) & (var > 0.0)
    )
    return G, keep, call_rate, maf, n_obs, var


def scan_snps(
    context: ScanContext,
    geno,
    *,
    backend: str = "auto",
    batch_size: int = 20000,
    maf_min: float = 0.01,
    call_rate_min: float = 0.9,
) -> ScanResult:
    """Run the per-SNP association scan for an in-memory genotype block.

    Args:
        context: ScanContext from build_scan_context.
        geno: an io.GenoChunk with dosage (n,m) in {0,1,2,NaN} plus
            samples / variant_ids / chrom / pos. Rows are reordered to match
            context.sample_ids when both that and geno.samples are present.
        backend: "cpu" / "gpu" / "auto".
        batch_size: SNPs per GEMM batch.
        maf_min, call_rate_min: per-SNP QC thresholds; failing SNPs are dropped.

    Returns:
        ScanResult over the kept markers.
    """
    from scipy import stats

    dosage = np.asarray(geno.dosage, dtype=np.float64)
    if dosage.ndim != 2:
        raise ValueError(f"geno.dosage must be 2D, got {dosage.shape}")

    # align genotype rows to context.sample_ids
    if context.sample_ids is not None and getattr(geno, "samples", None) is not None:
        geno_ids = np.asarray(geno.samples, dtype=object)
        if len(set(geno_ids.tolist())) != geno_ids.shape[0]:
            raise ValueError("genotype block has duplicate sample IDs; "
                             "cannot align unambiguously to context.sample_ids")
        pos = {s: i for i, s in enumerate(geno_ids)}
        missing = [s for s in context.sample_ids if s not in pos]
        if missing:
            raise ValueError(
                f"{len(missing)} context samples absent from genotype block "
                f"(e.g. {missing[:3]})"
            )
        order = np.array([pos[s] for s in context.sample_ids], dtype=np.int64)
        dosage = dosage[order, :]
    if dosage.shape[0] != context.n:
        raise ValueError(
            f"genotype rows {dosage.shape[0]} != context.n {context.n}; "
            "pass aligned genotypes or set context.sample_ids + geno.samples"
        )

    m = dosage.shape[1]
    snp_id = np.asarray(getattr(geno, "variant_ids", np.arange(m)), dtype=object)
    chrom = np.asarray(getattr(geno, "chrom", np.full(m, "NA")), dtype=object)
    pos_arr = np.asarray(getattr(geno, "pos", np.zeros(m, dtype=np.int64)))
    for nm, arr in (("variant_ids", snp_id), ("chrom", chrom), ("pos", pos_arr)):
        if arr.shape[0] != m:
            raise ValueError(f"geno.{nm} length {arr.shape[0]} != n_markers {m}")

    backend_used = _resolve_backend(backend)
    P, Py = context.P, context.Py
    P_t = Py_t = device = None
    if backend_used == "gpu":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False    # exact small p-values
        device = torch.device("cuda:0")
        P_t = torch.as_tensor(P, dtype=torch.float64, device=device)
        Py_t = torch.as_tensor(Py, dtype=torch.float64, device=device)

    cols = {k: [] for k in ("snp_id", "chrom", "pos", "beta", "se",
                            "chi2", "p", "n_obs", "maf", "call_rate")}
    filt = {"call_rate": 0, "maf": 0, "zero_var_or_allmiss": 0, "bad_info": 0}

    for start in range(0, m, batch_size):
        end = min(start + batch_size, m)
        G_raw = dosage[:, start:end]
        G, keep, cr, maf, n_obs, var = _impute_filter_batch(G_raw, maf_min, call_rate_min)
        # a SNP can fail several tests; attribute the drop to the first
        for j in np.where(~keep)[0]:
            if n_obs[j] == 0 or not np.isfinite(var[j]) or var[j] <= 0.0:
                filt["zero_var_or_allmiss"] += 1
            elif cr[j] < call_rate_min:
                filt["call_rate"] += 1
            else:
                filt["maf"] += 1
        if not keep.any():
            continue
        Gk = np.ascontiguousarray(G[:, keep])
        if backend_used == "gpu":
            U, info = _scan_batch_gpu(P_t, Py_t, Gk, device)
        else:
            U, info = _scan_batch_cpu(P, Py, Gk)
        good = np.isfinite(info) & (info > 0) & np.isfinite(U)
        filt["bad_info"] += int((~good).sum())
        if not good.any():
            continue
        U, info = U[good], info[good]
        beta = U / info
        se = np.sqrt(1.0 / info)
        chi2 = U * U / info
        pval = stats.chi2.sf(chi2, df=1)

        idx = np.arange(start, end)[keep][good]
        cols["snp_id"].append(snp_id[idx])
        cols["chrom"].append(chrom[idx])
        cols["pos"].append(pos_arr[idx])
        cols["beta"].append(beta)
        cols["se"].append(se)
        cols["chi2"].append(chi2)
        cols["p"].append(pval)
        cols["n_obs"].append(n_obs[keep][good])
        cols["maf"].append(maf[keep][good])
        cols["call_rate"].append(cr[keep][good])

    def _cat(key, dt):
        return (np.concatenate(cols[key]).astype(dt) if cols[key]
                else np.array([], dtype=dt))

    return ScanResult(
        snp_id=_cat("snp_id", object),
        chrom=_cat("chrom", object),
        pos=_cat("pos", np.int64),
        beta=_cat("beta", np.float64),
        se=_cat("se", np.float64),
        chi2=_cat("chi2", np.float64),
        p=_cat("p", np.float64),
        n_obs=_cat("n_obs", np.int64),
        maf=_cat("maf", np.float64),
        call_rate=_cat("call_rate", np.float64),
        backend_used=backend_used,
        n_input=m,
        n_kept=int(_cat("p", np.float64).size),
        filter_counts=filt,
    )


@dataclass(frozen=True)
class StreamScanSummary:
    """Lightweight summary of a streaming BED scan (no per-SNP arrays kept)."""
    out_path: str
    backend_used: str
    subgenome: str | None
    n_input: int
    n_kept: int
    n_chunks: int
    filter_counts: dict[str, int]
    runtime_sec: float
    chunk_size: int


def scan_bed_stream(
    context: ScanContext,
    bed_prefix,
    out_path,
    *,
    backend: str = "auto",
    chunk_size: int = 200_000,
    maf_min: float = 0.01,
    call_rate_min: float = 0.9,
    subgenome: str | None = None,
    gzip_out: bool = True,
    progress_every: int = 25,
) -> StreamScanSummary:
    """Stream a per-SNP scan over a BED too large to hold in memory.

    Reads the BED in ``chunk_size``-variant chunks (io.iter_bed_chunks), runs
    the batched fixed-V score test per chunk, and appends results to a TSV
    (gzip by default). Returns a StreamScanSummary rather than a per-SNP
    ScanResult, which would not fit in memory at tens of millions of markers.

    Args:
        context: ScanContext from build_scan_context (carries P, Py, sample_ids).
        bed_prefix: PLINK1 BED prefix to scan.
        out_path: output TSV(.gz) path, written incrementally.
        backend: "cpu" / "gpu" / "auto".
        chunk_size: variants per IO+compute chunk.
        maf_min, call_rate_min: per-SNP QC thresholds.
        subgenome: tag written into the ``subgenome`` column.
        gzip_out: gzip the output stream.

    Returns:
        StreamScanSummary.
    """
    import gzip
    import time as _time

    import pandas as pd
    from scipy import stats

    if context.sample_ids is None:
        raise ValueError("context.sample_ids is required for streaming scan "
                         "(build_scan_context with sample_ids=...)")
    from .io import iter_bed_chunks

    t0 = _time.time()
    backend_used = _resolve_backend(backend)
    P, Py = context.P, context.Py
    P_t = Py_t = device = None
    if backend_used == "gpu":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")
        P_t = torch.as_tensor(P, dtype=torch.float64, device=device)
        Py_t = torch.as_tensor(Py, dtype=torch.float64, device=device)

    ctx_ids = list(context.sample_ids)
    cols = ["snp_id", "subgenome", "chrom", "pos", "variant_index",
            "beta", "se", "chi2", "p", "n_obs", "maf", "call_rate"]
    opener = gzip.open if gzip_out else open
    n_input = n_kept = n_chunks = 0
    filt = {"call_rate": 0, "maf": 0, "zero_var_or_allmiss": 0, "bad_info": 0}
    order = None

    with opener(out_path, "wt") as fh:
        fh.write("\t".join(cols) + "\n")
        for start, chunk in iter_bed_chunks(bed_prefix, chunk_size=chunk_size):
            n_chunks += 1
            if order is None:                       # align genotype rows once
                geno_ids = np.asarray(chunk.samples, dtype=object)
                if len(set(geno_ids.tolist())) != geno_ids.shape[0]:
                    raise ValueError("BED has duplicate sample IDs")
                gpos = {s: i for i, s in enumerate(geno_ids)}
                missing = [s for s in ctx_ids if s not in gpos]
                if missing:
                    raise ValueError(
                        f"{len(missing)} context samples absent from BED "
                        f"(e.g. {missing[:3]})")
                order = np.array([gpos[s] for s in ctx_ids], dtype=np.int64)

            dosage = np.asarray(chunk.dosage, dtype=np.float64)[order, :]
            m = dosage.shape[1]
            n_input += m
            G, keep, cr, maf, n_obs, var = _impute_filter_batch(
                dosage, maf_min, call_rate_min)
            for j in np.where(~keep)[0]:
                if n_obs[j] == 0 or not np.isfinite(var[j]) or var[j] <= 0.0:
                    filt["zero_var_or_allmiss"] += 1
                elif cr[j] < call_rate_min:
                    filt["call_rate"] += 1
                else:
                    filt["maf"] += 1
            if not keep.any():
                continue
            Gk = np.ascontiguousarray(G[:, keep])
            if backend_used == "gpu":
                U, info = _scan_batch_gpu(P_t, Py_t, Gk, device)
            else:
                U, info = _scan_batch_cpu(P, Py, Gk)
            good = np.isfinite(info) & (info > 0) & np.isfinite(U)
            filt["bad_info"] += int((~good).sum())
            if not good.any():
                continue
            U, info = U[good], info[good]
            chi2 = U * U / info
            idx_local = np.arange(start, start + m)[keep][good]
            df = pd.DataFrame({
                "snp_id": np.asarray(chunk.variant_ids, dtype=object)[keep][good],
                "subgenome": subgenome if subgenome is not None else "",
                "chrom": np.asarray(chunk.chrom, dtype=object)[keep][good],
                "pos": np.asarray(chunk.pos, dtype=np.int64)[keep][good],
                "variant_index": idx_local,
                "beta": U / info,
                "se": np.sqrt(1.0 / info),
                "chi2": chi2,
                "p": stats.chi2.sf(chi2, df=1),
                "n_obs": n_obs[keep][good],
                "maf": maf[keep][good],
                "call_rate": cr[keep][good],
            })
            df.to_csv(fh, sep="\t", header=False, index=False)
            n_kept += len(df)
            if progress_every and n_chunks % progress_every == 0:
                print(f"  [scan_bed_stream {subgenome or ''}] chunk {n_chunks}, "
                      f"{n_input} scanned / {n_kept} kept")

    return StreamScanSummary(
        out_path=str(out_path), backend_used=backend_used, subgenome=subgenome,
        n_input=n_input, n_kept=n_kept, n_chunks=n_chunks,
        filter_counts=filt, runtime_sec=round(_time.time() - t0, 2),
        chunk_size=chunk_size,
    )


def write_scan_tsv(result: ScanResult, path) -> None:
    """Write a ScanResult to a TSV (snp_id, chrom, pos, beta, se, chi2, p, ...)."""
    import pandas as pd
    df = pd.DataFrame({
        "snp_id": result.snp_id,
        "chrom": result.chrom,
        "pos": result.pos,
        "beta": result.beta,
        "se": result.se,
        "chi2": result.chi2,
        "p": result.p,
        "n_obs": result.n_obs,
        "maf": result.maf,
        "call_rate": result.call_rate,
    })
    df.to_csv(path, sep="\t", index=False)


# Leave-one-chromosome-out (LOCO) scan.
#
# The plain fixed-V scan uses a single genome-wide GRM, so the test SNP sits
# inside the polygenic kernel and its cis-effect is partially absorbed into u,
# costing power. LOCO rebuilds K_j(-c) per chrom c by excluding chrom-c markers
# from the GRM and forms a fresh P_(-c). Variance components σ² are not re-fit
# (EMMAX/FastLMM/BOLT-LOCO convention): they drift <1% across leave-one-chrom
# subsets, and refitting would couple null-model changes into the contrast.


@dataclass(frozen=True)
class LOCOContext:
    """Per-chromosome ``ScanContext`` bundle for a LOCO scan.

    ``contexts[c]`` is the fixed projection for "test SNPs on chrom c with
    chrom-c markers excluded from the GRM". Shared scalars (n, p, sigma2,
    kernel_names, sample_ids) come from the first context but are validated
    against all per-chrom contexts at build time.
    """
    contexts: dict[str, ScanContext]    # chrom -> P_(-c) ScanContext
    chroms: list[str]                   # ordered chrom list
    n: int
    p: int
    sigma2: dict[str, float]            # global REML σ²  (reused across chroms)
    kernel_names: list[str]
    sample_ids: np.ndarray | None
    jitter_by_chrom: dict[str, float]
    dtype: str = "float64"


def build_loco_scan_contexts(
    y: np.ndarray,
    X: np.ndarray,
    kernels_by_chrom: dict[str, dict[str, np.ndarray]],
    sigma2: dict[str, float],
    *,
    sample_ids: np.ndarray | None = None,
    dtype: str = "float64",
    check_tol: float = 1e-6,
) -> LOCOContext:
    """Build one ``ScanContext`` per chrom; wrap them in a ``LOCOContext``.

    Args:
        y: phenotype (n,).
        X: fixed-effect design (n, p) including intercept.
        kernels_by_chrom: ``{chrom -> {kernel_name -> K_(-c) (n,n)}}`` — the
            leave-one-chromosome-out kernels. ``kernel_names`` must be the
            same set in every chrom dict.
        sigma2: GLOBAL REML σ² (reused; not refit per chrom).
        sample_ids, dtype, check_tol: forwarded to ``build_scan_context``.

    Returns:
        LOCOContext.
    """
    if not isinstance(kernels_by_chrom, dict) or not kernels_by_chrom:
        raise ValueError("kernels_by_chrom is empty")
    chroms = [str(c) for c in kernels_by_chrom.keys()]
    if len(set(chroms)) != len(chroms):
        raise ValueError(f"kernels_by_chrom has duplicate chrom keys: {chroms}")
    base_kernel_names: list[str] | None = None
    contexts: dict[str, ScanContext] = {}
    jitter_by_chrom: dict[str, float] = {}
    for c in chroms:
        kernels_c = kernels_by_chrom[c]
        if not isinstance(kernels_c, dict) or not kernels_c:
            raise ValueError(f"kernels_by_chrom[{c!r}] is empty")
        names_c = list(kernels_c.keys())
        if base_kernel_names is None:
            base_kernel_names = names_c
        elif names_c != base_kernel_names:
            raise ValueError(
                f"kernel_names mismatch at chrom {c!r}: "
                f"{names_c} vs {base_kernel_names}")
        ctx_c = build_scan_context(
            y, X, kernels_c, sigma2,
            sample_ids=sample_ids, dtype=dtype, check_tol=check_tol)
        contexts[c] = ctx_c
        jitter_by_chrom[c] = ctx_c.jitter_used

    n = contexts[chroms[0]].n
    p = contexts[chroms[0]].p
    for c in chroms[1:]:
        ctx_c = contexts[c]
        if ctx_c.n != n or ctx_c.p != p:
            raise ValueError(
                f"context shape mismatch at chrom {c!r}: "
                f"({ctx_c.n},{ctx_c.p}) vs ({n},{p})")
    sids = (np.asarray(sample_ids, dtype=object)
            if sample_ids is not None else None)
    return LOCOContext(
        contexts=contexts,
        chroms=chroms,
        n=n, p=p,
        sigma2={k: float(v) for k, v in sigma2.items()},
        kernel_names=list(base_kernel_names or []),
        sample_ids=sids,
        jitter_by_chrom=jitter_by_chrom,
        dtype=dtype,
    )


def _chrom_runs(chrom_arr: np.ndarray) -> list[tuple[str, int, int]]:
    """Run-length split: ``[(chrom, start_incl, end_excl), ...]``.

    A 'run' is a maximal contiguous block of identical chrom values. When the
    input is already sorted by chrom (standard plink2 layout) this yields one
    block per chrom; otherwise each contiguous block is its own run and the
    same chrom name may appear in multiple runs.
    """
    chrom_arr = np.asarray(chrom_arr).astype(str)
    m = chrom_arr.shape[0]
    if m == 0:
        return []
    runs: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, m):
        if chrom_arr[i] != chrom_arr[i - 1]:
            runs.append((str(chrom_arr[start]), start, i))
            start = i
    runs.append((str(chrom_arr[start]), start, m))
    return runs


def _align_geno_rows(geno, ctx_sample_ids, ctx_n: int) -> np.ndarray:
    """Reorder genotype rows to match context.sample_ids; return dosage view."""
    dosage = np.asarray(geno.dosage, dtype=np.float64)
    if dosage.ndim != 2:
        raise ValueError(f"geno.dosage must be 2D, got {dosage.shape}")
    if ctx_sample_ids is not None and getattr(geno, "samples", None) is not None:
        geno_ids = np.asarray(geno.samples, dtype=object)
        if len(set(geno_ids.tolist())) != geno_ids.shape[0]:
            raise ValueError("genotype block has duplicate sample IDs; "
                             "cannot align unambiguously to context.sample_ids")
        pos = {s: i for i, s in enumerate(geno_ids)}
        missing = [s for s in ctx_sample_ids if s not in pos]
        if missing:
            raise ValueError(
                f"{len(missing)} context samples absent from genotype block "
                f"(e.g. {missing[:3]})")
        order = np.array([pos[s] for s in ctx_sample_ids], dtype=np.int64)
        dosage = dosage[order, :]
    if dosage.shape[0] != ctx_n:
        raise ValueError(
            f"genotype rows {dosage.shape[0]} != context.n {ctx_n}; "
            "pass aligned genotypes or set context.sample_ids + geno.samples")
    return dosage


def scan_snps_loco(
    loco_ctx: LOCOContext,
    geno,
    *,
    backend: str = "auto",
    batch_size: int = 20000,
    maf_min: float = 0.01,
    call_rate_min: float = 0.9,
) -> ScanResult:
    """In-memory LOCO per-SNP scan.

    Each variant is tested against the ``P_(-c)`` of its chrom. SNPs are
    grouped into chrom runs (see :func:`_chrom_runs`) and each run is
    batch-scanned with the corresponding ``loco_ctx.contexts[chrom]``. GPU
    tensors for ``P`` / ``Py`` are uploaded once per chrom run.

    Args mirror :func:`scan_snps`; LOCO routing is automatic.

    Raises:
        KeyError: if a variant's chrom is not in ``loco_ctx.contexts``; LOCO
            has no defined behaviour there, so fail fast rather than fall back
            to a global P.
    """
    from scipy import stats

    dosage = _align_geno_rows(geno, loco_ctx.sample_ids, loco_ctx.n)
    m = dosage.shape[1]
    snp_id = np.asarray(getattr(geno, "variant_ids", np.arange(m)), dtype=object)
    chrom_arr = np.asarray(getattr(geno, "chrom", np.full(m, "NA"))).astype(str)
    pos_arr = np.asarray(getattr(geno, "pos", np.zeros(m, dtype=np.int64)))
    for nm, arr in (("variant_ids", snp_id), ("chrom", chrom_arr), ("pos", pos_arr)):
        if arr.shape[0] != m:
            raise ValueError(f"geno.{nm} length {arr.shape[0]} != n_markers {m}")

    runs = _chrom_runs(chrom_arr)
    unknown = sorted({c for c, _, _ in runs if c not in loco_ctx.contexts})
    if unknown:
        raise KeyError(
            f"LOCO scan: {len(unknown)} chrom(s) in genotype not in "
            f"LOCOContext (have {len(loco_ctx.contexts)} chrom; "
            f"missing e.g. {unknown[:3]})")

    backend_used = _resolve_backend(backend)
    P_t = Py_t = device = None
    if backend_used == "gpu":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")

    cols = {k: [] for k in ("snp_id", "chrom", "pos", "beta", "se",
                            "chi2", "p", "n_obs", "maf", "call_rate")}
    filt = {"call_rate": 0, "maf": 0, "zero_var_or_allmiss": 0, "bad_info": 0}

    for c_name, r_start, r_end in runs:
        ctx_c = loco_ctx.contexts[c_name]
        if backend_used == "gpu":
            import torch
            P_t = torch.as_tensor(ctx_c.P, dtype=torch.float64, device=device)
            Py_t = torch.as_tensor(ctx_c.Py, dtype=torch.float64, device=device)
        for b_start in range(r_start, r_end, batch_size):
            b_end = min(b_start + batch_size, r_end)
            G_raw = dosage[:, b_start:b_end]
            G, keep, cr, maf, n_obs, var = _impute_filter_batch(
                G_raw, maf_min, call_rate_min)
            for j in np.where(~keep)[0]:
                if n_obs[j] == 0 or not np.isfinite(var[j]) or var[j] <= 0.0:
                    filt["zero_var_or_allmiss"] += 1
                elif cr[j] < call_rate_min:
                    filt["call_rate"] += 1
                else:
                    filt["maf"] += 1
            if not keep.any():
                continue
            Gk = np.ascontiguousarray(G[:, keep])
            if backend_used == "gpu":
                U, info = _scan_batch_gpu(P_t, Py_t, Gk, device)
            else:
                U, info = _scan_batch_cpu(ctx_c.P, ctx_c.Py, Gk)
            good = np.isfinite(info) & (info > 0) & np.isfinite(U)
            filt["bad_info"] += int((~good).sum())
            if not good.any():
                continue
            U, info = U[good], info[good]
            chi2 = U * U / info
            idx = np.arange(b_start, b_end)[keep][good]
            cols["snp_id"].append(snp_id[idx])
            cols["chrom"].append(chrom_arr[idx])
            cols["pos"].append(pos_arr[idx])
            cols["beta"].append(U / info)
            cols["se"].append(np.sqrt(1.0 / info))
            cols["chi2"].append(chi2)
            cols["p"].append(stats.chi2.sf(chi2, df=1))
            cols["n_obs"].append(n_obs[keep][good])
            cols["maf"].append(maf[keep][good])
            cols["call_rate"].append(cr[keep][good])

    def _cat(key, dt):
        return (np.concatenate(cols[key]).astype(dt) if cols[key]
                else np.array([], dtype=dt))

    return ScanResult(
        snp_id=_cat("snp_id", object),
        chrom=_cat("chrom", object),
        pos=_cat("pos", np.int64),
        beta=_cat("beta", np.float64),
        se=_cat("se", np.float64),
        chi2=_cat("chi2", np.float64),
        p=_cat("p", np.float64),
        n_obs=_cat("n_obs", np.int64),
        maf=_cat("maf", np.float64),
        call_rate=_cat("call_rate", np.float64),
        backend_used=backend_used,
        n_input=m,
        n_kept=int(_cat("p", np.float64).size),
        filter_counts=filt,
    )


def scan_bed_stream_loco(
    loco_ctx: LOCOContext,
    bed_prefix,
    out_path,
    *,
    backend: str = "auto",
    chunk_size: int = 200_000,
    maf_min: float = 0.01,
    call_rate_min: float = 0.9,
    subgenome: str | None = None,
    gzip_out: bool = True,
    progress_every: int = 25,
) -> StreamScanSummary:
    """Streaming LOCO scan over a BED too large to hold in memory.

    Each ``chunk_size``-variant chunk from :func:`io.iter_bed_chunks` is split
    by chrom run; each run uses the corresponding ``loco_ctx.contexts[chrom]``.
    GPU ``P_t`` / ``Py_t`` are re-uploaded only when chrom changes
    (≤ ~21 uploads per wheat subgenome). Output TSV(.gz) is appended
    incrementally.
    """
    import gzip
    import time as _time

    import pandas as pd
    from scipy import stats

    if loco_ctx.sample_ids is None:
        raise ValueError(
            "loco_ctx.sample_ids is required for streaming LOCO scan "
            "(build_loco_scan_contexts with sample_ids=...)")
    from .io import iter_bed_chunks

    t0 = _time.time()
    backend_used = _resolve_backend(backend)
    ctx_ids = list(loco_ctx.sample_ids)
    cols = ["snp_id", "subgenome", "chrom", "pos", "variant_index",
            "beta", "se", "chi2", "p", "n_obs", "maf", "call_rate"]
    opener = gzip.open if gzip_out else open
    n_input = n_kept = n_chunks = 0
    filt = {"call_rate": 0, "maf": 0, "zero_var_or_allmiss": 0, "bad_info": 0}
    order = None

    cached_chrom: str | None = None
    P_t = Py_t = device = None
    if backend_used == "gpu":
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device("cuda:0")

    with opener(out_path, "wt") as fh:
        fh.write("\t".join(cols) + "\n")
        for chunk_start, chunk in iter_bed_chunks(bed_prefix, chunk_size=chunk_size):
            n_chunks += 1
            if order is None:
                geno_ids = np.asarray(chunk.samples, dtype=object)
                if len(set(geno_ids.tolist())) != geno_ids.shape[0]:
                    raise ValueError("BED has duplicate sample IDs")
                gpos = {s: i for i, s in enumerate(geno_ids)}
                missing = [s for s in ctx_ids if s not in gpos]
                if missing:
                    raise ValueError(
                        f"{len(missing)} context samples absent from BED "
                        f"(e.g. {missing[:3]})")
                order = np.array([gpos[s] for s in ctx_ids], dtype=np.int64)

            dosage = np.asarray(chunk.dosage, dtype=np.float64)[order, :]
            chrom_arr = np.asarray(chunk.chrom).astype(str)
            m = dosage.shape[1]
            n_input += m
            runs = _chrom_runs(chrom_arr)
            unknown = sorted({c for c, _, _ in runs if c not in loco_ctx.contexts})
            if unknown:
                raise KeyError(
                    f"LOCO stream scan: chunk {n_chunks} has unknown chrom(s) "
                    f"{unknown[:3]}; LOCOContext has {len(loco_ctx.contexts)} chrom")

            for c_name, r_start, r_end in runs:
                ctx_c = loco_ctx.contexts[c_name]
                if c_name != cached_chrom:
                    if backend_used == "gpu":
                        import torch
                        P_t = torch.as_tensor(ctx_c.P, dtype=torch.float64, device=device)
                        Py_t = torch.as_tensor(ctx_c.Py, dtype=torch.float64, device=device)
                    cached_chrom = c_name
                # sub-batch within a run to keep GPU GEMM tiles predictable
                run_batch = min(chunk_size, r_end - r_start)
                if run_batch <= 0:
                    continue
                for b_start in range(r_start, r_end, run_batch):
                    b_end = min(b_start + run_batch, r_end)
                    G_raw = dosage[:, b_start:b_end]
                    G, keep, cr, maf, n_obs, var = _impute_filter_batch(
                        G_raw, maf_min, call_rate_min)
                    for j in np.where(~keep)[0]:
                        if n_obs[j] == 0 or not np.isfinite(var[j]) or var[j] <= 0.0:
                            filt["zero_var_or_allmiss"] += 1
                        elif cr[j] < call_rate_min:
                            filt["call_rate"] += 1
                        else:
                            filt["maf"] += 1
                    if not keep.any():
                        continue
                    Gk = np.ascontiguousarray(G[:, keep])
                    if backend_used == "gpu":
                        U, info = _scan_batch_gpu(P_t, Py_t, Gk, device)
                    else:
                        U, info = _scan_batch_cpu(ctx_c.P, ctx_c.Py, Gk)
                    good = np.isfinite(info) & (info > 0) & np.isfinite(U)
                    filt["bad_info"] += int((~good).sum())
                    if not good.any():
                        continue
                    U, info = U[good], info[good]
                    chi2 = U * U / info
                    idx_local = np.arange(
                        chunk_start + b_start, chunk_start + b_end)[keep][good]
                    df = pd.DataFrame({
                        "snp_id": np.asarray(chunk.variant_ids[b_start:b_end],
                                             dtype=object)[keep][good],
                        "subgenome": subgenome if subgenome is not None else "",
                        "chrom": np.asarray(chunk.chrom[b_start:b_end],
                                            dtype=object)[keep][good],
                        "pos": np.asarray(chunk.pos[b_start:b_end],
                                          dtype=np.int64)[keep][good],
                        "variant_index": idx_local,
                        "beta": U / info,
                        "se": np.sqrt(1.0 / info),
                        "chi2": chi2,
                        "p": stats.chi2.sf(chi2, df=1),
                        "n_obs": n_obs[keep][good],
                        "maf": maf[keep][good],
                        "call_rate": cr[keep][good],
                    })
                    df.to_csv(fh, sep="\t", header=False, index=False)
                    n_kept += len(df)
            if progress_every and n_chunks % progress_every == 0:
                print(f"  [scan_bed_stream_loco {subgenome or ''}] chunk "
                      f"{n_chunks}, {n_input} scanned / {n_kept} kept")

    return StreamScanSummary(
        out_path=str(out_path), backend_used=backend_used, subgenome=subgenome,
        n_input=n_input, n_kept=n_kept, n_chunks=n_chunks,
        filter_counts=filt, runtime_sec=round(_time.time() - t0, 2),
        chunk_size=chunk_size,
    )
