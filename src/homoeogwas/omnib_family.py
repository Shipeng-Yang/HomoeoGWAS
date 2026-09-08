"""Shared edge/group omniB score matrices for canonical homoeolog families."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .group_family import ExpandedEdgeFamily, MasterGroupFamily, expand_pair_edges
from .parallel import run_fork_blocks

OMNIB_COMPONENT_NAMES = ("minor_burden", "pc1", "kernel_hadamard")
INDEXED_SCORE_MICROBLOCK = 25
PREPARED_SCORE_ALGORITHM = "homoeogwas-omnib-prepared-response-v2"
FEATURE_SEED_SCHEME = "homoeogwas-feature-v1"
COMPONENT_RANK_ATOL = 1.0e-10
_OMNIB_WORKER_STATE: dict | None = None


def _resolve_feature_seed(feature_seed, bootstrap_seed: int) -> tuple[int, str]:
    policy = "explicit"
    if feature_seed is None:
        feature_seed = bootstrap_seed
        policy = "legacy_seed_fallback"
    if isinstance(feature_seed, bool) or not isinstance(
        feature_seed, (int, np.integer)
    ) or int(feature_seed) < 0:
        raise ValueError("feature_seed must be a non-negative integer")
    return int(feature_seed), policy


def _gene_feature_seed(
    feature_seed: int,
    subgenome: str,
    gene_id: str,
    feature_type: str,
) -> int:
    """Derive a stable per-gene, per-feature uint64 RNG seed."""
    if feature_type not in {"minor_burden", "gene_pc"}:
        raise ValueError("unknown feature_type")
    payload = "\0".join((
        FEATURE_SEED_SCHEME,
        str(int(feature_seed)),
        str(subgenome),
        str(gene_id),
        feature_type,
    )).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _select_capped_indices(
    size: int,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    local = np.arange(size, dtype=int)
    if cap and local.size > cap:
        local = np.sort(rng.choice(local, size=cap, replace=False))
    return local


def _little_endian_float64_identity(values: np.ndarray) -> dict:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return {
        "shape": list(array.shape),
        "dtype": "<f8",
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _build_keyed_gene_feature(
    interaction_module,
    Xg: np.ndarray,
    retained_indices: np.ndarray,
    *,
    feature_seed: int,
    subgenome: str,
    gene_id: str,
    cap: int,
    n_pc: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], dict]:
    burden_seed = _gene_feature_seed(
        feature_seed, subgenome, gene_id, "minor_burden")
    pc_seed = _gene_feature_seed(feature_seed, subgenome, gene_id, "gene_pc")
    burden_rng = np.random.default_rng(burden_seed)
    pc_rng = np.random.default_rng(pc_seed)
    burden_local = _select_capped_indices(
        retained_indices.size, cap, burden_rng)
    pc_local = _select_capped_indices(
        retained_indices.size, cap, pc_rng)
    burden = interaction_module.block_burden_capped(
        Xg, burden_local, 0, burden_rng, minor=True,
    ).reshape(-1, 1)
    pcs = interaction_module.gene_pc_scores(
        Xg, pc_local, 0, pc_rng, n_pc,
    )
    feature = (burden, pcs[:, :1], pcs)
    identity = {
        "subgenome": str(subgenome),
        "gene_id": str(gene_id),
        "root_feature_seed": int(feature_seed),
        "child_seeds": {
            "minor_burden": int(burden_seed),
            "gene_pc": int(pc_seed),
        },
        "cap": int(cap),
        "n_pc": int(n_pc),
        "retained_global_variant_indices": [
            int(value) for value in retained_indices
        ],
        "selected_global_variant_indices": {
            "minor_burden": [
                int(value) for value in retained_indices[burden_local]
            ],
            "gene_pc": [int(value) for value in retained_indices[pc_local]],
        },
        "minor_burden": _little_endian_float64_identity(burden),
        "gene_pc": _little_endian_float64_identity(pcs),
    }
    return feature, identity


def _feature_cache_sha256(feature_identity: dict) -> str:
    records = [
        feature_identity[key]
        for key in sorted(
            feature_identity,
            key=lambda value: (str(value[0]), str(value[1])),
        )
    ]
    payload = json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalize_retained_variant_masks(
    subdata: dict,
    retained_variant_masks,
) -> dict[str, np.ndarray] | None:
    if retained_variant_masks is None:
        return None
    if not isinstance(retained_variant_masks, Mapping):
        raise ValueError("retained_variant_masks must be a subgenome mapping")
    expected_subgenomes = set(subdata)
    supplied_subgenomes = set(retained_variant_masks)
    missing = sorted(expected_subgenomes - supplied_subgenomes)
    if missing:
        raise ValueError(
            "retained_variant_masks missing subgenomes: " + ", ".join(missing))
    extra = sorted(supplied_subgenomes - expected_subgenomes)
    if extra:
        raise ValueError(
            "retained_variant_masks has unknown subgenomes: " + ", ".join(extra))

    normalized = {}
    for sub in subdata:
        entry = retained_variant_masks[sub]
        expected_hash = None
        if isinstance(entry, Mapping):
            if "mask" not in entry or "sha256" not in entry:
                raise ValueError(
                    f"retained variant mask record for {sub} requires mask and sha256")
            mask = np.asarray(entry["mask"])
            expected_hash = entry["sha256"]
        else:
            mask = np.asarray(entry)
        if mask.dtype != np.bool_:
            raise ValueError(
                f"retained variant mask for {sub} must have boolean dtype")
        n_variants = int(np.asarray(subdata[sub].X).shape[1])
        if mask.ndim != 1 or mask.size != n_variants:
            raise ValueError(
                f"retained variant mask length for {sub} must equal {n_variants}")
        mask = np.ascontiguousarray(mask, dtype=bool)
        digest = hashlib.sha256(mask.astype(np.uint8).tobytes()).hexdigest()
        if expected_hash is not None and expected_hash != digest:
            raise ValueError(f"retained variant mask hash mismatch for {sub}")
        normalized[sub] = mask
    return normalized


def _normalized_svd_basis(
    matrix: np.ndarray,
    *,
    atol: float = COMPONENT_RANK_ATOL,
) -> np.ndarray:
    matrix = np.asarray(matrix, float)
    if matrix.ndim != 2:
        raise ValueError("rank matrix must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("rank matrix contains non-finite values")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("rank atol must be finite and non-negative")
    if matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0), float)
    norms = np.linalg.norm(matrix, axis=0)
    normalized = np.zeros_like(matrix, dtype=float)
    nonzero = norms > 0.0
    normalized[:, nonzero] = matrix[:, nonzero] / norms[nonzero]
    U, singular_values, _ = np.linalg.svd(normalized, full_matrices=False)
    return U[:, singular_values > atol]


def _normalized_svd_rank(
    matrix: np.ndarray,
    *,
    atol: float = COMPONENT_RANK_ATOL,
) -> int:
    return int(_normalized_svd_basis(matrix, atol=atol).shape[1])


def _component_design_signature(
    gsx,
    gsy,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranks = np.empty(len(OMNIB_COMPONENT_NAMES), dtype=int)
    numerator_df = np.empty(len(OMNIB_COMPONENT_NAMES), dtype=int)
    denominator_df = np.empty(len(OMNIB_COMPONENT_NAMES), dtype=int)
    n = int(np.asarray(C).shape[0])
    for component, (ax, ay) in enumerate(zip(gsx, gsy, strict=True)):
        reduced = np.column_stack((C, ax, ay))
        cross = (ax[:, :, None] * ay[:, None, :]).reshape(n, -1)
        rank_reduced = _normalized_svd_rank(reduced)
        rank_full = _normalized_svd_rank(np.column_stack((reduced, cross)))
        ranks[component] = rank_reduced
        numerator_df[component] = rank_full - rank_reduced
        denominator_df[component] = n - rank_full
    return ranks, numerator_df, denominator_df


def _fixed_component_mask_sha256(scores) -> str:
    payload = {
        "algorithm": "normalized-column-svd-absolute-v1",
        "atol": COMPONENT_RANK_ATOL,
        "component_names": list(OMNIB_COMPONENT_NAMES),
        "rank_reduced": np.asarray(scores.component_rank_reduced, int).tolist(),
        "numerator_df": np.asarray(scores.component_dfn, int).tolist(),
        "denominator_df": np.asarray(scores.component_dfd, int).tolist(),
        "estimable": np.asarray(scores.component_estimable, bool).tolist(),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _set_omnib_worker_state(state: dict) -> None:
    global _OMNIB_WORKER_STATE
    _OMNIB_WORKER_STATE = state


def _clear_omnib_worker_state() -> None:
    global _OMNIB_WORKER_STATE
    _OMNIB_WORKER_STATE = None


def _require_omnib_worker_state(mode: str) -> dict:
    state = _OMNIB_WORKER_STATE
    if state is None or state.get("mode") != mode:
        raise RuntimeError(f"omniB {mode} worker state is not installed")
    return state


def _score_subset_block(bounds):
    from . import interact as I

    state = _require_omnib_worker_state("subset")
    start, stop = bounds
    selected = state["valid_indices"][start:stop]
    values = np.full(selected.size, np.nan)
    component_values = np.full(
        (selected.size, len(OMNIB_COMPONENT_NAMES)), np.nan)
    for local, edge_index in enumerate(selected):
        edge = state["expanded"].edges[int(edge_index)]
        try:
            left_full = state["feature_cache"][(edge.sub_x, edge.gene_x)]
            right_full = state["feature_cache"][(edge.sub_y, edge.gene_y)]
        except KeyError as exc:
            raise RuntimeError(
                f"frozen omniB feature missing for {edge.edge_id}") from exc
        left = tuple(np.asarray(value)[state["keep"]] for value in left_full)
        right = tuple(np.asarray(value)[state["keep"]] for value in right_full)
        result = omnib_components_over_Y(
            state["W"], state["Yw"], state["Cw"], left, right)[:, 0]
        component_values[local] = result
        values[local] = I.acat(result)
    return selected, values, component_values


def _score_prepared_block(bounds):
    from . import interact as I

    state = _require_omnib_worker_state("prepared")
    lo, hi = bounds
    edge_indices = state["valid_indices"][lo:hi]
    response_count = state["response_count"]
    values = np.full((edge_indices.size, response_count), np.nan)
    component_values = np.full(
        (edge_indices.size, len(OMNIB_COMPONENT_NAMES), response_count),
        np.nan,
    )
    for local, edge_index in enumerate(edge_indices):
        prepared = state["projection_cache"][int(edge_index)]
        for start, stop, response_block in state["whitened_blocks"]:
            components = _prepared_components_over_Y(
                response_block, prepared)[:, :stop - start]
            component_values[local, :, start:stop] = components
            values[local, start:stop] = np.asarray([
                I.acat(components[:, column])
                for column in range(stop - start)
            ], float)
    return edge_indices, values, component_values


@dataclass
class OmniBFamilyScores:
    """Observed/bootstrap scores retaining the complete predeclared family."""

    edge_p: np.ndarray
    group_p: np.ndarray
    edge_components_obs: np.ndarray
    edge_estimable: np.ndarray
    group_estimable: np.ndarray
    W: np.ndarray
    y: np.ndarray
    covariance_components: dict[str, float]
    group_partial: np.ndarray = field(default_factory=lambda: np.empty(0, bool))
    gated_snp: dict = field(default_factory=dict, repr=False)
    feature_cache: dict = field(default_factory=dict, repr=False)
    feature_identity: dict = field(default_factory=dict)
    feature_cache_sha256: str = ""
    feature_seed_provenance: dict = field(default_factory=dict)
    component_rank_reduced: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), int))
    component_dfn: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), int))
    component_dfd: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), int))
    component_estimable: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), bool))
    fixed_mask_sha256: str = ""
    null_fit_identity: dict = field(default_factory=dict, repr=False)
    null_fit_sha256: str = ""
    prepared_design_identity: dict = field(default_factory=dict, repr=False)
    prepared_design_sha256: str = ""
    covariate_block: np.ndarray | None = field(default=None, repr=False)
    covariate_metadata: dict = field(default_factory=dict)
    null_covariance: np.ndarray | None = field(default=None, repr=False)
    null_beta: np.ndarray | None = field(default=None, repr=False)
    null_design: np.ndarray | None = field(default=None, repr=False)
    null_kernels: dict = field(default_factory=dict, repr=False)
    projection_cache: dict = field(default_factory=dict, repr=False)
    edge_membership: np.ndarray = field(
        default_factory=lambda: np.empty(0, bool), repr=False)
    grm_provenance: dict = field(default_factory=dict)
    parallel_execution: dict = field(default_factory=dict)
    response_diagnostics: OmniBResponseDiagnostics | None = None


@dataclass(frozen=True)
class OmniBResponseDiagnostics:
    """Response-level failures over the fixed edge-component family."""

    failed_response_mask: np.ndarray
    failed_response_indices: tuple[int, ...]
    failed_response_indices_by_component: Mapping[str, tuple[int, ...]]
    nonfinite_component_counts: tuple[int, ...]
    attempted: int
    successful: int
    retried: int
    terminal_failures: int

    def __post_init__(self) -> None:
        mask = np.array(self.failed_response_mask, dtype=bool, copy=True)
        if mask.ndim != 1 or mask.size != self.attempted:
            raise ValueError("failed_response_mask must align to attempted responses")
        mask.setflags(write=False)
        object.__setattr__(self, "failed_response_mask", mask)

    def as_dict(self) -> dict:
        return {
            "failed_response_indices": list(self.failed_response_indices),
            "failed_response_indices_by_component": {
                key: list(value)
                for key, value in self.failed_response_indices_by_component.items()
            },
            "nonfinite_component_counts": list(self.nonfinite_component_counts),
            "attempted": self.attempted,
            "successful": self.successful,
            "retried": self.retried,
            "terminal_failures": self.terminal_failures,
        }


@dataclass(frozen=True)
class OmniBSubsetScores:
    """Candidate-sensitivity scores from one frozen prepared omniB family."""

    edge_p: np.ndarray
    group_p: np.ndarray
    edge_components: np.ndarray
    covariance_components: dict[str, float]


def _response_failure_diagnostics(
    scores: OmniBFamilyScores,
    edge_components: np.ndarray,
) -> OmniBResponseDiagnostics:
    values = np.asarray(edge_components, float)
    expected = (
        scores.component_estimable.shape[0],
        scores.component_estimable.shape[1],
    )
    if values.ndim != 3 or values.shape[:2] != expected:
        raise ValueError("edge component responses do not match the fixed family")
    fixed = np.asarray(scores.component_estimable, bool)[:, :, None]
    failed_cells = fixed & ~np.isfinite(values)
    failed_mask = failed_cells.any(axis=(0, 1))
    by_component = {
        name: tuple(
            int(index)
            for index in np.flatnonzero(failed_cells[:, component, :].any(axis=0))
        )
        for component, name in enumerate(OMNIB_COMPONENT_NAMES)
    }
    attempted = int(values.shape[2])
    terminal = int(failed_mask.sum())
    return OmniBResponseDiagnostics(
        failed_response_mask=failed_mask,
        failed_response_indices=tuple(
            int(index) for index in np.flatnonzero(failed_mask)
        ),
        failed_response_indices_by_component=by_component,
        nonfinite_component_counts=tuple(
            int(value) for value in failed_cells.sum(axis=(0, 1))
        ),
        attempted=attempted,
        successful=attempted - terminal,
        retried=0,
        terminal_failures=terminal,
    )


def omnib_components_over_Y(Wh, Yw, Cw, gsx, gsy):
    """Return minor-burden, PC1 and kernel-Hadamard p-values by response."""
    # Delayed import avoids a module cycle while keeping the established
    # nested-F implementation as the single numerical primitive.
    from . import interact as I

    bx, p1x, PX = gsx
    by, p1y, PY = gsy
    components = []
    for ax, ay in ((bx, by), (p1x, p1y), (PX, PY)):
        reduced = np.column_stack([Cw, Wh @ ax, Wh @ ay])
        cross = (ax[:, :, None] * ay[:, None, :]).reshape(ax.shape[0], -1)
        components.append(I._batch_nested_f(Yw, reduced, Wh @ cross))
    return np.vstack(components)


def _null_bootstrap_responses(
    V: np.ndarray,
    beta: np.ndarray,
    C: np.ndarray,
    B: int,
    seed: int,
) -> np.ndarray:
    """Draw all null responses from the already-fitted covariance."""
    values, vectors = np.linalg.eigh(0.5 * (V + V.T))
    root = (vectors * np.sqrt(np.clip(values, 1e-12, None))) @ vectors.T
    rng = np.random.default_rng(seed)
    # Preserve the frozen legacy stream exactly: the historical implementation
    # drew one length-n vector per replicate in a Python loop.  NumPy fills a
    # (B, n) array in that same row-major order; drawing (n, B) would assign a
    # different random stream to every replicate despite using the same seed.
    return C @ beta[:, None] + root @ rng.standard_normal((B, V.shape[0])).T


def score_omnib_family(
    subdata: dict,
    family: MasterGroupFamily,
    y_raw: np.ndarray,
    sample_idx: np.ndarray,
    *,
    cap: int = 150,
    n_pc: int = 3,
    transform: str = "INT",
    bootstrap_B: int = 2000,
    bootstrap_seed: int = 2026,
    feature_seed: int | None = None,
    n_jobs: int = 8,
    grm_method: str = "compute_grm_maf",
    maf_min: float = 0.01,
    burden_maf: float = 0.01,
    min_snp: int = 3,
    covariates: dict = None,
    retained_variant_masks: Mapping[str, np.ndarray] | None = None,
) -> tuple[OmniBFamilyScores, ExpandedEdgeFamily]:
    """Prepare once, then score observed and optional bootstrap responses."""
    if isinstance(bootstrap_B, bool) or int(bootstrap_B) != bootstrap_B:
        raise ValueError("bootstrap_B must be an integer")
    bootstrap_B = int(bootstrap_B)
    if bootstrap_B < 0:
        raise ValueError("bootstrap_B must be >= 0")
    resolved_feature_seed, feature_seed_policy = _resolve_feature_seed(
        feature_seed, bootstrap_seed)
    if isinstance(n_jobs, bool) or int(n_jobs) != n_jobs or int(n_jobs) < 1:
        raise ValueError("n_jobs must be an integer >= 1")
    n_jobs = int(n_jobs)
    scores, expanded = prepare_omnib_design(
        subdata,
        family,
        y_raw,
        sample_idx,
        cap=cap,
        n_pc=n_pc,
        transform=transform,
        feature_seed=resolved_feature_seed,
        retained_variant_masks=retained_variant_masks,
        grm_method=grm_method,
        maf_min=maf_min,
        burden_maf=burden_maf,
        min_snp=min_snp,
        covariates=covariates,
    )
    scores.feature_seed_provenance["policy"] = feature_seed_policy
    response_count = bootstrap_B + 1
    responses = np.empty((scores.y.size, response_count), float)
    responses[:, 0] = scores.y
    if bootstrap_B:
        responses[:, 1:] = _null_bootstrap_responses(
            scores.null_covariance,
            scores.null_beta,
            scores.null_design,
            bootstrap_B,
            bootstrap_seed,
        )
    edge_p, group_p, edge_components, diagnostics = score_omnib_responses(
        scores,
        family,
        expanded,
        responses,
        n_jobs=n_jobs,
        return_diagnostics=True,
    )
    scores.response_diagnostics = diagnostics
    failed = scores.edge_estimable & ~np.isfinite(edge_p[:, 0])
    if failed.any():
        failed_ids = [expanded.edges[i].edge_id for i in np.flatnonzero(failed)[:5]]
        raise RuntimeError(
            "design-valid edge produced a post-whitening non-finite observed "
            f"omniB score: {', '.join(failed_ids)}")
    failed_groups = scores.group_estimable & ~np.isfinite(group_p[:, 0])
    if failed_groups.any():
        failed_ids = [
            family.group_ids[i] for i in np.flatnonzero(failed_groups)[:5]
        ]
        raise RuntimeError(
            "design-valid group produced a post-whitening non-finite observed "
            f"omniB score: {', '.join(failed_ids)}")
    scores.edge_p = edge_p
    scores.group_p = group_p
    scores.edge_components_obs = edge_components[:, :, 0]
    return scores, expanded


def score_omnib_subset(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
    keep,
    y_raw,
    *,
    n_jobs: int = 8,
) -> OmniBSubsetScores:
    """Refit a deleted-sample null while preserving the formal genotype features.

    This is an internal-sensitivity primitive. It emits raw candidate-family
    scores only and deliberately has no bootstrap or rejection interface.
    """
    from . import interact as I

    keep = np.asarray(keep, int)
    if keep.ndim != 1 or keep.size < 10:
        raise ValueError("omniB deletion sensitivity requires at least 10 samples")
    if not np.array_equal(keep, np.unique(keep)):
        raise ValueError("keep indices must be unique sorted integers")
    n_full = int(scores.W.shape[0])
    if keep[0] < 0 or keep[-1] >= n_full:
        raise ValueError("keep indices are outside the prepared analysis cohort")
    y_raw = np.asarray(y_raw, float)
    if y_raw.ndim != 1 or y_raw.size != keep.size:
        raise ValueError("y_raw must be one-dimensional and aligned to keep")
    if not np.all(np.isfinite(y_raw)):
        raise ValueError("y_raw contains non-finite values")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) \
            or int(n_jobs) < 1:
        raise ValueError("n_jobs must be an integer >= 1")
    n_jobs = int(n_jobs)
    if scores.covariate_block is not None:
        raise ValueError(
            "deletion sensitivity with fixed covariates is not yet supported")
    if not scores.null_kernels:
        raise ValueError("prepared omniB context does not retain its null kernels")

    all_rows = np.array_equal(keep, np.arange(n_full))
    kernels = {}
    for sub, full in scores.null_kernels.items():
        value = full if all_rows else np.asarray(full)[np.ix_(keep, keep)]
        if not all_rows:
            scale = float(np.trace(value) / keep.size)
            if not np.isfinite(scale) or scale <= 1e-12:
                raise ValueError(
                    f"subgenome {sub}: deleted-sample kernel has invalid trace")
            value = value / scale
        kernels[sub] = value

    y = I.rank_int(y_raw)
    W, _V, _beta, covariance = I.null_lmm_fit(kernels, y, seed=42)
    Cw = (W @ np.ones(keep.size)).reshape(-1, 1)
    Yw = W @ y.reshape(-1, 1)
    membership = (
        scores.edge_membership
        if scores.edge_membership.size == len(expanded.edges)
        else scores.edge_estimable
    )
    valid_indices = np.flatnonzero(membership)
    edge_p = np.full(len(expanded.edges), np.nan)
    components = np.full(
        (len(expanded.edges), len(OMNIB_COMPONENT_NAMES)), np.nan)

    step = max(1, valid_indices.size // (n_jobs * 8))
    blocks = [
        (start, min(start + step, valid_indices.size))
        for start in range(0, valid_indices.size, step)
    ]
    worker_state = {
        "mode": "subset",
        "valid_indices": valid_indices,
        "expanded": expanded,
        "feature_cache": scores.feature_cache,
        "keep": keep,
        "W": W,
        "Yw": Yw,
        "Cw": Cw,
    }
    results, execution = run_fork_blocks(
        blocks, _score_subset_block, n_jobs=n_jobs,
        state_setter=lambda: _set_omnib_worker_state(worker_state),
        state_clearer=_clear_omnib_worker_state,
    )
    scores.parallel_execution = execution.as_dict()
    for selected, values, component_values in results:
        edge_p[selected] = values
        components[selected] = component_values

    group_p = np.full(len(family.group_ids), np.nan)
    for group_index, edge_indices in enumerate(expanded.group_edge_indices):
        selected = np.asarray(edge_indices, int)
        group_p[group_index] = (
            edge_p[selected[0]] if selected.size == 1
            else I.acat(edge_p[selected])
        )
    return OmniBSubsetScores(
        edge_p=edge_p,
        group_p=group_p,
        edge_components=components,
        covariance_components={
            str(name): float(value) for name, value in covariance.items()
        },
    )


def prepare_omnib_design(
    subdata: dict,
    family: MasterGroupFamily,
    y_raw: np.ndarray,
    sample_idx: np.ndarray,
    *,
    cap: int,
    n_pc: int,
    transform: str,
    feature_seed: int,
    grm_method: str,
    maf_min: float,
    burden_maf: float,
    min_snp: int,
    covariates: dict | None = None,
    retained_variant_masks: Mapping[str, np.ndarray] | None = None,
) -> tuple[OmniBFamilyScores, ExpandedEdgeFamily]:
    """Prepare one immutable raw-family/null/feature omniB design."""
    from . import interact as I

    sample_idx = np.asarray(sample_idx, int)
    y_raw = np.asarray(y_raw, float)
    if y_raw.ndim != 1 or y_raw.size != sample_idx.size:
        raise ValueError("y_raw must be one-dimensional and aligned to sample_idx")
    if not np.all(np.isfinite(y_raw)):
        raise ValueError("phenotype contains non-finite values")
    if feature_seed is None:
        raise ValueError("prepare_omnib_design requires an explicit feature_seed")
    feature_seed, feature_seed_policy = _resolve_feature_seed(feature_seed, 0)
    missing = [sub for sub in family.subgenomes if sub not in subdata]
    if missing:
        raise ValueError(
            "master family references missing subgenomes: " + ", ".join(missing))
    retained_variant_masks = _normalize_retained_variant_masks(
        subdata, retained_variant_masks)

    expanded = expand_pair_edges(family)
    if not expanded.edges:
        raise ValueError("master homoeolog family contains no pair edges")

    n = sample_idx.size
    kernels = {}
    grm_provenance = {}
    for sub in subdata:
        kernel, provenance = I._build_grm(
            subdata[sub], sample_idx, grm_method, maf_min,
            retained_variant_mask=(
                None
                if retained_variant_masks is None
                else retained_variant_masks[sub]
            ),
            return_provenance=True)
        kernels[sub] = kernel
        grm_provenance[sub] = {
            key: value for key, value in provenance.items()
            if key != "retained_variant_mask"
        }
    C = None
    covariate_metadata = {"policy": "none"}
    if covariates:
        C, covariate_metadata = I.build_covariate_block(
            kernels, n, n_pcs=int(covariates.get("n_pcs", 0)),
            extra=covariates.get("extra"))
    C_design = (
        np.ones((n, 1))
        if C is None else np.asarray(C, float).reshape(n, -1)
    )
    y = I.rank_int(y_raw) if transform == "INT" else y_raw.astype(float)

    gated: dict[tuple[str, str], np.ndarray] = {}
    features: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    feature_identity: dict[tuple[str, str], dict] = {}

    def gated_snps(sub: str, gene: str) -> np.ndarray | None:
        key = (sub, gene)
        if key in gated:
            return gated[key]
        if gene not in subdata[sub].gene_snp:
            return None
        indices = np.asarray(subdata[sub].gene_snp[gene], int)
        if retained_variant_masks is not None:
            indices = indices[retained_variant_masks[sub][indices]]
        if indices.size == 0:
            gated[key] = indices
            return gated[key]
        means = np.nanmean(
            subdata[sub].X[np.ix_(sample_idx, indices)], axis=0) / 2.0
        gated[key] = indices[np.minimum(means, 1.0 - means) >= burden_maf]
        return gated[key]

    def feature(sub: str, gene: str, indices: np.ndarray):
        key = (sub, gene)
        if key not in features:
            Xg = subdata[sub].X[np.ix_(sample_idx, indices)]
            features[key], feature_identity[key] = _build_keyed_gene_feature(
                I,
                Xg,
                indices,
                feature_seed=feature_seed,
                subgenome=sub,
                gene_id=gene,
                cap=cap,
                n_pc=n_pc,
            )
        return features[key]

    component_rank_reduced = np.full(
        (len(expanded.edges), len(OMNIB_COMPONENT_NAMES)), -1, int)
    component_dfn = np.full_like(component_rank_reduced, -1)
    component_dfd = np.full_like(component_rank_reduced, -1)
    component_estimable = np.zeros_like(component_rank_reduced, dtype=bool)
    for edge_index, edge in enumerate(expanded.edges):
        ix = gated_snps(edge.sub_x, edge.gene_x)
        iy = gated_snps(edge.sub_y, edge.gene_y)
        if ix is None or iy is None or ix.size < min_snp or iy.size < min_snp:
            continue
        fx = feature(edge.sub_x, edge.gene_x, ix)
        fy = feature(edge.sub_y, edge.gene_y, iy)
        ranks, dfn, dfd = _component_design_signature(fx, fy, C_design)
        component_rank_reduced[edge_index] = ranks
        component_dfn[edge_index] = dfn
        component_dfd[edge_index] = dfd
        component_estimable[edge_index] = (dfn >= 1) & (dfd >= 1)

    edge_membership = component_estimable.any(axis=1)
    group_estimable = np.asarray([
        bool(edge_membership[np.asarray(indices, int)].any())
        for indices in expanded.group_edge_indices
    ])
    W, V, beta, covariance_components = I.null_lmm_fit(
        kernels, y, C, seed=42)

    scores = OmniBFamilyScores(
        edge_p=np.full((len(expanded.edges), 0), np.nan),
        group_p=np.full((len(family.group_ids), 0), np.nan),
        edge_components_obs=np.full(
            (len(expanded.edges), len(OMNIB_COMPONENT_NAMES)), np.nan),
        edge_estimable=edge_membership.copy(),
        group_estimable=group_estimable,
        W=W,
        y=y,
        covariance_components={
            str(name): float(value)
            for name, value in covariance_components.items()
        },
        group_partial=np.zeros(len(family.group_ids), bool),
        gated_snp=gated,
        feature_cache=features,
        feature_identity=feature_identity,
        feature_cache_sha256=_feature_cache_sha256(feature_identity),
        feature_seed_provenance={
            "scheme": FEATURE_SEED_SCHEME,
            "root_seed": feature_seed,
            "policy": feature_seed_policy,
        },
        component_rank_reduced=component_rank_reduced,
        component_dfn=component_dfn,
        component_dfd=component_dfd,
        component_estimable=component_estimable,
        covariate_block=C,
        covariate_metadata=covariate_metadata,
        null_covariance=V,
        null_beta=np.asarray(beta, float),
        null_design=C_design,
        null_kernels=kernels,
        edge_membership=edge_membership,
        grm_provenance=grm_provenance,
    )
    scores.fixed_mask_sha256 = _fixed_component_mask_sha256(scores)
    scores.null_fit_identity = {
        "W": _array_identity(W),
        "covariance": _array_identity(V),
        "beta": _array_identity(np.asarray(beta, float)),
        "design": _array_identity(C_design),
        "kernels": {
            sub: _array_identity(kernels[sub]) for sub in sorted(kernels)
        },
        "components": {
            str(key): float(value)
            for key, value in sorted(covariance_components.items())
        },
    }
    scores.null_fit_sha256 = _text_identity(scores.null_fit_identity)
    scores.prepared_design_identity = {
        "schema": "homoeogwas-omnib-prepared-design-v1",
        "family": _family_provenance(family, expanded),
        "sample_index": _array_identity(sample_idx),
        "phenotype_analyzed": _array_identity(y),
        "feature_cache_sha256": scores.feature_cache_sha256,
        "fixed_mask_sha256": scores.fixed_mask_sha256,
        "null_fit_sha256": scores.null_fit_sha256,
        "retained_variant_mask_sha256": {
            sub: grm_provenance[sub]["retained_variant_mask_sha256"]
            for sub in sorted(grm_provenance)
        },
        "transform": str(transform),
        "cap": int(cap),
        "n_pc": int(n_pc),
        "maf_min": float(maf_min),
        "burden_maf": float(burden_maf),
        "min_snp": int(min_snp),
    }
    scores.prepared_design_sha256 = _text_identity(
        scores.prepared_design_identity)
    _prepare_projection_cache(scores, expanded)
    _update_group_partial(scores, expanded)
    return scores, expanded


def _prepare_checkpoint_omnib(
    subdata: dict,
    family: MasterGroupFamily,
    y_raw: np.ndarray,
    sample_idx: np.ndarray,
    *,
    cap: int,
    n_pc: int,
    transform: str,
    bootstrap_seed: int,
    n_jobs: int,
    grm_method: str,
    maf_min: float,
    burden_maf: float,
    min_snp: int,
    covariates: dict | None,
    feature_seed: int | None = None,
    retained_variant_masks: Mapping[str, np.ndarray] | None = None,
) -> tuple[OmniBFamilyScores, ExpandedEdgeFamily]:
    """Compatibility wrapper around :func:`prepare_omnib_design`."""
    if isinstance(n_jobs, bool) or int(n_jobs) != n_jobs or int(n_jobs) < 1:
        raise ValueError("n_jobs must be an integer >= 1")
    resolved_seed, policy = _resolve_feature_seed(feature_seed, bootstrap_seed)
    scores, expanded = prepare_omnib_design(
        subdata,
        family,
        y_raw,
        sample_idx,
        cap=cap,
        n_pc=n_pc,
        transform=transform,
        feature_seed=resolved_seed,
        retained_variant_masks=retained_variant_masks,
        grm_method=grm_method,
        maf_min=maf_min,
        burden_maf=burden_maf,
        min_snp=min_snp,
        covariates=covariates,
    )
    scores.feature_seed_provenance["policy"] = policy
    return scores, expanded


def _prepare_projection_cache(
    scores: OmniBFamilyScores,
    expanded: ExpandedEdgeFamily,
) -> np.ndarray:
    """Prepare projections and assert agreement with the immutable raw mask."""
    Cw = scores.W @ scores.null_design
    if scores.component_estimable.shape != (
        len(expanded.edges), len(OMNIB_COMPONENT_NAMES)
    ):
        raise RuntimeError("prepared omniB context lacks a fixed component mask")
    for edge_index in np.flatnonzero(scores.edge_estimable):
        edge = expanded.edges[int(edge_index)]
        cache_key = int(edge_index)
        if cache_key not in scores.projection_cache:
            scores.projection_cache[cache_key] = _prepare_omnib_nested_designs(
                scores.W,
                Cw,
                scores.feature_cache[(edge.sub_x, edge.gene_x)],
                scores.feature_cache[(edge.sub_y, edge.gene_y)],
            )
        for component, prepared in enumerate(scores.projection_cache[cache_key]):
            _Qr, _Qa, rank_reduced, dfn, dfd = prepared
            observed = (
                int(rank_reduced),
                int(dfn),
                int(dfd),
                bool(dfn >= 1 and dfd >= 1),
            )
            expected = (
                int(scores.component_rank_reduced[edge_index, component]),
                int(scores.component_dfn[edge_index, component]),
                int(scores.component_dfd[edge_index, component]),
                bool(scores.component_estimable[edge_index, component]),
            )
            if observed != expected:
                raise RuntimeError(
                    "raw/whitened component rank mismatch for "
                    f"{edge.edge_id}/{OMNIB_COMPONENT_NAMES[component]}: "
                    f"expected={expected}, observed={observed}")
    return scores.edge_estimable.copy()


def _update_group_partial(
    scores: OmniBFamilyScores,
    expanded: ExpandedEdgeFamily,
) -> None:
    partial = np.zeros(len(expanded.group_edge_indices), bool)
    for group_index, edge_indices in enumerate(expanded.group_edge_indices):
        selected = np.asarray(edge_indices, int)
        valid_count = int(scores.edge_estimable[selected].sum())
        partial[group_index] = 0 < valid_count < selected.size
    scores.group_partial = partial


def _score_prepared_responses(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
    responses: np.ndarray,
    *,
    n_jobs: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score response columns through one frozen prepared omniB algorithm."""
    from . import interact as I

    responses = np.asarray(responses, float)
    if responses.ndim == 1:
        responses = responses.reshape(-1, 1)
    if responses.ndim != 2 or responses.shape[0] != scores.W.shape[0]:
        raise ValueError(
            "prepared omniB responses must be sample-by-response and aligned "
            "to the frozen null fit")
    response_count = int(responses.shape[1])
    if response_count == 0:
        return (
            np.empty((len(expanded.edges), 0), float),
            np.empty((len(family.group_ids), 0), float),
            np.empty((len(expanded.edges), len(OMNIB_COMPONENT_NAMES), 0), float),
        )
    # Whiten each response independently, then batch only column-independent
    # contractions below. General matrix multiplication may select a reduction
    # strategy from the block shape and thereby change the last bits.
    whitened = np.column_stack([
        scores.W @ responses[:, column] for column in range(response_count)])
    whitened_blocks = []
    for start in range(0, response_count, INDEXED_SCORE_MICROBLOCK):
        stop = min(start + INDEXED_SCORE_MICROBLOCK, response_count)
        width = stop - start
        padded = np.zeros(
            (whitened.shape[0], INDEXED_SCORE_MICROBLOCK), dtype=whitened.dtype)
        padded[:, :width] = whitened[:, start:stop]
        whitened_blocks.append((start, stop, padded))
    edge_p = np.full((len(expanded.edges), response_count), np.nan)
    edge_components = np.full(
        (len(expanded.edges), len(OMNIB_COMPONENT_NAMES), response_count),
        np.nan,
    )
    _prepare_projection_cache(scores, expanded)
    _update_group_partial(scores, expanded)
    valid_indices = np.flatnonzero(scores.edge_estimable)

    step = max(1, valid_indices.size // (int(n_jobs) * 8))
    blocks = [
        (lo, min(lo + step, valid_indices.size))
        for lo in range(0, valid_indices.size, step)
    ]
    worker_state = {
        "mode": "prepared",
        "valid_indices": valid_indices,
        "response_count": response_count,
        "projection_cache": scores.projection_cache,
        "whitened_blocks": whitened_blocks,
    }
    results, execution = run_fork_blocks(
        blocks, _score_prepared_block, n_jobs=int(n_jobs),
        state_setter=lambda: _set_omnib_worker_state(worker_state),
        state_clearer=_clear_omnib_worker_state,
    )
    scores.parallel_execution = execution.as_dict()
    for edge_indices, values, component_values in results:
        edge_p[edge_indices] = values
        edge_components[edge_indices] = component_values

    group_p = np.full((len(family.group_ids), response_count), np.nan)
    for group_index, edge_indices in enumerate(expanded.group_edge_indices):
        selected = np.asarray(edge_indices, int)
        if selected.size == 1:
            group_p[group_index] = edge_p[selected[0]]
        else:
            for column in range(response_count):
                group_p[group_index, column] = I.acat(
                    edge_p[selected, column])
    return edge_p, group_p, edge_components


def score_omnib_responses(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
    responses: np.ndarray,
    *,
    n_jobs: int = 8,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[
    np.ndarray, np.ndarray, np.ndarray, OmniBResponseDiagnostics
]:
    """Score an explicit response bank through one frozen production design.

    This returns raw edge/group/component p-value matrices. It does not create
    a discovery family; callers must calibrate a predeclared matrix separately.
    """
    edge_p, group_p, components = _score_prepared_responses(
        scores, family, expanded, responses, n_jobs=n_jobs)
    diagnostics = _response_failure_diagnostics(scores, components)
    scores.response_diagnostics = diagnostics
    if return_diagnostics:
        return edge_p, group_p, components, diagnostics
    return edge_p, group_p, components


def score_omnib_observed(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
    *,
    n_jobs: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score and retain the observed response through the prepared API."""
    edge_p, group_p, components, diagnostics = score_omnib_responses(
        scores,
        family,
        expanded,
        scores.y.reshape(-1, 1),
        n_jobs=n_jobs,
        return_diagnostics=True,
    )
    if diagnostics.failed_response_mask[0]:
        raise RuntimeError("native observed omniB response failed")
    scores.edge_p = edge_p
    scores.group_p = group_p
    scores.edge_components_obs = components[:, :, 0]
    return edge_p, group_p, components


def score_omnib_null_indices(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
    indices,
    *,
    base_seed: int,
    n_jobs: int = 8,
    return_components: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score only the requested indexed null responses using prepared features."""
    from . import interact as I

    requested = tuple(int(index) for index in indices)
    if not requested:
        empty = (
            np.empty((len(expanded.edges), 0), float),
            np.empty((len(family.group_ids), 0), float),
        )
        if return_components:
            return empty + (np.empty(
                (len(expanded.edges), len(OMNIB_COMPONENT_NAMES), 0), float),)
        return empty
    if (
        scores.null_covariance is None
        or scores.null_beta is None
        or scores.null_design is None
        or not scores.null_kernels
    ):
        raise RuntimeError("omniB score context lacks its frozen null fit")
    null_fit = (
        scores.W,
        scores.null_covariance,
        scores.null_beta,
        scores.covariance_components,
    )
    response_list, _, _ = I.null_replicates_by_index(
        scores.null_kernels, scores.y, scores.null_design,
        indices=requested, base_seed=base_seed, null_fit=null_fit)
    responses = np.column_stack(response_list)
    edge_p, group_p, components, _diagnostics = score_omnib_responses(
        scores,
        family,
        expanded,
        responses,
        n_jobs=n_jobs,
        return_diagnostics=True,
    )
    if return_components:
        return edge_p, group_p, components
    return edge_p, group_p


def _prepare_omnib_nested_designs(Wh, Cw, gsx, gsy):
    """Cache response-independent nested-model bases for one omniB edge."""
    prepared = []
    for ax, ay in zip(gsx, gsy, strict=True):
        reduced = np.column_stack([Cw, Wh @ ax, Wh @ ay])
        Qr = _normalized_svd_basis(reduced)
        cross = (ax[:, :, None] * ay[:, None, :]).reshape(ax.shape[0], -1)
        added = Wh @ cross
        added_residual = added - Qr @ (Qr.T @ added) if Qr.size else added
        Qa = _normalized_svd_basis(added_residual)
        prepared.append((
            Qr,
            Qa,
            int(Qr.shape[1]),
            int(Qa.shape[1]),
            int(reduced.shape[0] - Qr.shape[1] - Qa.shape[1]),
        ))
    return tuple(prepared)


def _prepared_components_over_Y(Yw, prepared):
    """Score a response block with fixed, column-independent reduction order."""
    from scipy import stats

    Yw = np.asarray(Yw, float)
    output = np.full((len(prepared), Yw.shape[1]), np.nan)
    for component, (Qr, Qa, _rank_reduced, dfn, dfd) in enumerate(prepared):
        if dfn < 1 or dfd < 1:
            continue
        if Qr.size:
            reduced_coef = np.einsum(
                "ij,ik->jk", Qr, Yw, optimize=False)
            y_residual = Yw - np.einsum(
                "ij,jk->ik", Qr, reduced_coef, optimize=False)
        else:
            y_residual = Yw.copy()
        added_coef = np.einsum(
            "ij,ik->jk", Qa, y_residual, optimize=False)
        added_ss = np.einsum(
            "ij,ij->j", added_coef, added_coef, optimize=False)
        full_residual = y_residual - np.einsum(
            "ij,jk->ik", Qa, added_coef, optimize=False)
        rss_full = np.einsum(
            "ij,ij->j", full_residual, full_residual, optimize=False)
        denominator = rss_full / dfd
        bad = denominator <= 1e-300
        f_stat = added_ss / dfn / np.where(bad, 1.0, denominator)
        p_value = stats.f.sf(np.maximum(f_stat, 0.0), dfn, dfd)
        output[component] = np.where(
            bad, np.where(added_ss > 1e-300, 0.0, np.nan), p_value)
    return output


def _interact_result(interact_module, **values):
    """Construct against both the frozen and extended compatibility schemas."""
    allowed = interact_module.InteractResult.__dataclass_fields__
    return interact_module.InteractResult(
        **{key: value for key, value in values.items() if key in allowed})


def _callable_group_rows(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    min_snp: int,
) -> np.ndarray:
    """Return master rows whose every declared copy passes the SNP gate."""
    callable_rows = []
    for group_index, genes in enumerate(family.genes):
        if all(
            (sub, gene) in scores.gated_snp
            and scores.gated_snp[(sub, gene)].size >= min_snp
            for sub, gene in zip(family.subgenomes, genes, strict=True)
        ):
            callable_rows.append(group_index)
    return np.asarray(callable_rows, int)


def _tsv_p(interact_module, value) -> str:
    """Serialize a probability without turning non-estimability into a number."""
    if value is None:
        return "NA"
    if hasattr(interact_module, "_tsv_p"):
        try:
            return interact_module._tsv_p(value)
        except (TypeError, ValueError):
            return "NA"
    try:
        return repr(float(value)) if np.isfinite(value) else "NA"
    except (TypeError, ValueError):
        return "NA"


def bootstrap_minp_calibration(
    p_obs: np.ndarray,
    p_null: np.ndarray,
    *,
    alpha: float = 0.05,
) -> dict:
    """Calibrate one fixed hypothesis matrix by plus-one single-step min-P."""
    p_obs = np.asarray(p_obs, float)
    p_null = np.asarray(p_null, float)
    if p_obs.ndim != 1:
        raise ValueError("p_obs must be one-dimensional")
    if p_null.ndim != 2 or p_null.shape[0] != p_obs.size:
        raise ValueError(
            "p_obs and p_null must contain the same fixed hypothesis family")
    if not np.isfinite(p_obs).all():
        raise ValueError("observed min-P family contains non-finite statistics")
    B = int(p_null.shape[1])
    if B < 1:
        raise ValueError("bootstrap min-P calibration requires at least one replicate")

    finite_null = np.isfinite(p_null)
    degenerate = ~finite_null.all(axis=0)
    safe = np.where(finite_null, p_null, np.inf)
    null_min = safe.min(axis=0)
    null_min[degenerate] = 0.0
    minp_obs = float(p_obs.min())
    empirical_p = float(
        (1 + int((null_min <= minp_obs).sum())) / (B + 1))
    k = int(np.floor(alpha * (B + 1)))
    threshold = float(np.sort(null_min)[k - 1]) if k >= 1 else None
    rejected_local = (
        np.flatnonzero(p_obs < threshold).astype(int).tolist()
        if threshold is not None else [])
    adjusted = (
        (1 + (null_min[None, :] <= p_obs[:, None]).sum(axis=1))
        / (B + 1)
    ).astype(float)
    rejected = bool(empirical_p <= alpha)
    if rejected != bool(rejected_local):
        raise RuntimeError(
            "bootstrap global decision and single-step rejection set disagree")
    if set(rejected_local) != set(np.flatnonzero(adjusted <= alpha).tolist()):
        raise RuntimeError(
            "bootstrap adjusted-p and threshold rejection sets disagree")
    return {
        "alpha": float(alpha),
        "method": "parametric_bootstrap_minp_plus_one",
        "B": B,
        "empirical_p": empirical_p,
        "threshold": threshold,
        "threshold_comparator": "strict_less_than",
        "rejected": rejected,
        "rejected_local": rejected_local,
        "adjusted_p_local": adjusted,
        "n_degenerate_replicates": int(degenerate.sum()),
        "degenerate_policy": "any_nonfinite_statistic_sets_null_min_to_zero",
    }


def _component_localization(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    component_p = {
        name: (float(values[index]) if np.isfinite(values[index]) else None)
        for index, name in enumerate(OMNIB_COMPONENT_NAMES)
    }
    finite = np.isfinite(values)
    smallest = int(np.nanargmin(values)) if finite.any() else None
    return {
        "component_p": component_p,
        "smallest_component": (
            OMNIB_COMPONENT_NAMES[smallest] if smallest is not None else None),
        "smallest_component_p": (
            float(values[smallest]) if smallest is not None else None),
    }


def _edge_identity_record(
    scores: OmniBFamilyScores,
    expanded: ExpandedEdgeFamily,
    index: int,
) -> dict:
    edge = expanded.edges[index]
    return {
        "hypothesis_id": f"edge:{edge.edge_id}",
        "hypothesis_unit": "edge",
        "edge_id": edge.edge_id,
        "group_ids": list(edge.source_group_ids),
        "direction": edge.direction,
        "sub_x": edge.sub_x,
        "sub_y": edge.sub_y,
        "gene_x": edge.gene_x,
        "gene_y": edge.gene_y,
    } | _component_localization(scores.edge_components_obs[index])


def _group_identity_record(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
    index: int,
) -> dict:
    edge_indices = np.asarray(expanded.group_edge_indices[index], int)
    edge_values = scores.edge_p[edge_indices, 0]
    finite_edges = np.isfinite(edge_values)
    driving_index = (
        int(edge_indices[int(np.nanargmin(edge_values))])
        if finite_edges.any() else None)
    driving = (
        _edge_identity_record(scores, expanded, driving_index)
        if driving_index is not None else {})
    return {
        "hypothesis_id": f"group:{family.group_ids[index]}",
        "hypothesis_unit": "group",
        "group_id": family.group_ids[index],
        "genes": {
            sub: gene
            for sub, gene in zip(
                family.subgenomes, family.genes[index], strict=True)
        },
        "ordered_genes": list(family.genes[index]),
        "driving_edge": driving.get("edge_id"),
        "driving_direction": driving.get("direction"),
        "driving_component": driving.get("smallest_component"),
        "driving_component_p": driving.get("smallest_component_p"),
        "edge_localization": [
            {
                "edge_id": expanded.edges[int(edge_index)].edge_id,
                "direction": expanded.edges[int(edge_index)].direction,
                "p_omnib": (
                    float(scores.edge_p[int(edge_index), 0])
                    if np.isfinite(scores.edge_p[int(edge_index), 0]) else None),
            }
            for edge_index in edge_indices
        ],
    }


def _select_primary_family(
    scores: OmniBFamilyScores,
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
    hypothesis_unit: str,
    family_scope: str,
) -> tuple[np.ndarray, list[dict], str, list[str]]:
    edge_records = [
        _edge_identity_record(scores, expanded, index)
        for index in range(len(expanded.edges))
    ]
    group_records = [
        _group_identity_record(scores, family, expanded, index)
        for index in range(len(family.group_ids))
    ]
    if family_scope == "joint":
        return (
            np.vstack([scores.edge_p, scores.group_p]),
            edge_records + group_records,
            "joint",
            ["edge", "group"],
        )
    if hypothesis_unit == "edge":
        return scores.edge_p, edge_records, "edge", ["edge"]
    return scores.group_p, group_records, "group", ["group"]


def _ranking_rows(
    interact_module,
    records: list[dict],
    order: list[int],
    family: MasterGroupFamily,
) -> tuple[list[str], list[list]]:
    header = [
        "rank", "hypothesis_id", "family_id", "hypothesis_unit",
        "group_id", "edge_id", "group_ids", "direction", "sub_x", "sub_y",
        *[f"gene_{sub}" for sub in family.subgenomes],
        "p_interaction", "p_adjusted_bootstrap_minp", "primary_sig",
        "p_unestimable", "driving_edge", "driving_component",
        "driving_component_p",
    ]
    rows = []
    for rank, index in enumerate(order):
        record = records[index]
        genes = record.get("genes", {})
        if record["hypothesis_unit"] == "edge":
            genes = {
                record["sub_x"]: record["gene_x"],
                record["sub_y"]: record["gene_y"],
            }
        group_ids = record.get("group_ids", [])
        rows.append([
            rank,
            record["hypothesis_id"],
            record["family_id"],
            record["hypothesis_unit"],
            record.get("group_id") or "|".join(group_ids) or "NA",
            record.get("edge_id") or "NA",
            "|".join(group_ids) or "NA",
            record.get("direction") or "NA",
            record.get("sub_x") or "NA",
            record.get("sub_y") or "NA",
            *[genes.get(sub, "NA") for sub in family.subgenomes],
            _tsv_p(interact_module, record["p_interaction"]),
            _tsv_p(
                interact_module, record["p_adjusted_bootstrap_minp"]),
            int(record["primary_sig"]),
            int(record["p_unestimable"]),
            record.get("driving_edge") or record.get("edge_id") or "NA",
            record.get("driving_component")
            or record.get("smallest_component") or "NA",
            _tsv_p(
                interact_module,
                record.get("driving_component_p")
                if record["hypothesis_unit"] == "group"
                else record.get("smallest_component_p")),
        ])
    return header, rows


def _family_provenance(
    family: MasterGroupFamily,
    expanded: ExpandedEdgeFamily,
) -> dict:
    """Return full ordered hashes for the biological and tested families."""
    group_records = [
        "\t".join((group_id, *genes))
        for group_id, genes in zip(family.group_ids, family.genes, strict=True)
    ]
    edge_records = [
        "\t".join((
            edge.edge_id, edge.sub_x, edge.sub_y, edge.gene_x, edge.gene_y,
            *edge.source_group_ids,
        ))
        for edge in expanded.edges
    ]
    return {
        "group_family_sha256": hashlib.sha256(
            "\x00".join(group_records).encode()).hexdigest(),
        "edge_family_sha256": hashlib.sha256(
            "\x00".join(edge_records).encode()).hexdigest(),
        "n_groups_raw": len(family.group_ids),
        "n_unique_edges": len(expanded.edges),
    }


def _array_identity(values: np.ndarray) -> dict:
    """Hash an array together with shape and dtype without phenotype semantics."""
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(b"\0")
    if array.flags.c_contiguous:
        digest.update(memoryview(array).cast("B"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": digest.hexdigest(),
    }


def _text_identity(values) -> str:
    body = json.dumps(
        values, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def _checkpoint_manifest(
    subdata,
    family,
    expanded,
    scores,
    y_raw,
    sample_idx,
    identities,
    *,
    hypothesis_unit,
    family_scope,
    bootstrap_B,
    bootstrap_seed,
    cap,
    n_pc,
    grm_method,
    maf_min,
    burden_maf,
    min_snp,
    covariates,
    alpha,
    inferential,
    manifest_context,
) -> dict:
    """Bind every inference-relevant canonical group input to one run ID."""
    from . import __version__
    from .resampling_checkpoint import CHECKPOINT_SCHEMA_VERSION

    subgenome_identity = {}
    for sub in family.subgenomes:
        data = subdata[sub]
        gene_map = [
            [str(gene), [int(index) for index in np.asarray(indices, int)]]
            for gene, indices in sorted(data.gene_snp.items())
        ]
        subgenome_identity[sub] = {
            "dosage": _array_identity(data.X),
            "samples_sha256": _text_identity([str(value) for value in data.samples]),
            "gene_snp_sha256": _text_identity(gene_map),
        }
    covariate_identity = {"configured": bool(covariates)}
    if covariates:
        covariate_identity["n_pcs"] = int(covariates.get("n_pcs", 0))
        extra = covariates.get("extra")
        covariate_identity["extra"] = (
            _array_identity(extra) if extra is not None else None)
    provenance = _family_provenance(family, expanded)
    return {
        "schema": "homoeogwas-omnib-bootstrap-manifest-v1",
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "score_algorithm": PREPARED_SCORE_ALGORITHM,
        "indexed_score_microblock": INDEXED_SCORE_MICROBLOCK,
        "implementation_version": __version__,
        "family": provenance | {
            "subgenomes": list(family.subgenomes),
            "group_ids": list(family.group_ids),
            "hypothesis_ids": [record["hypothesis_id"] for record in identities],
        },
        "hypothesis_unit": hypothesis_unit,
        "family_scope": family_scope,
        "subset_order": 2,
        "transform": "INT",
        "bootstrap": {"B": bootstrap_B, "seed": bootstrap_seed},
        "calibration": {
            "alpha": float(alpha),
            "inferential": bool(inferential),
            "method": "parametric_bootstrap_minp_plus_one",
        },
        "phenotype_raw": _array_identity(np.asarray(y_raw, float)),
        "phenotype_analyzed": _array_identity(scores.y),
        "sample_index": _array_identity(np.asarray(sample_idx, int)),
        "covariates": covariate_identity,
        "grm": {
            "method": grm_method,
            "maf_min": float(maf_min),
            "subgenomes": scores.grm_provenance,
        },
        "burden": {
            "cap": int(cap), "n_pc": int(n_pc), "maf_min": float(burden_maf),
            "min_snp": int(min_snp),
            "feature_seed": int(scores.feature_seed_provenance["root_seed"]),
            "feature_seed_policy": scores.feature_seed_provenance["policy"],
            "feature_seed_scheme": scores.feature_seed_provenance["scheme"],
            "feature_cache_sha256": scores.feature_cache_sha256,
        },
        "prepared_design": scores.prepared_design_identity | {
            "sha256": scores.prepared_design_sha256,
        },
        "subgenome_inputs": subgenome_identity,
        "context": manifest_context or {},
    }


def _select_primary_matrix(edge_p, group_p, hypothesis_unit, family_scope):
    if family_scope == "joint":
        return np.vstack([edge_p, group_p])
    return edge_p if hypothesis_unit == "edge" else group_p


def run_group_scan_omnib(
    subdata,
    family: MasterGroupFamily,
    y_raw,
    sample_idx,
    *,
    hypothesis_unit="group",
    family_scope="primary_only",
    cap=150,
    n_pc=3,
    transform="INT",
    bootstrap_B=2000,
    bootstrap_seed=2026,
    feature_seed=None,
    n_jobs=8,
    grm_method="grm_from_X",
    maf_min=0.01,
    burden_maf=0.01,
    min_snp=3,
    covariates=None,
    full_dump_path=None,
    alpha=0.05,
    inferential=True,
    checkpoint_dir=None,
    checkpoint_block_size=25,
    checkpoint_manifest_context=None,
):
    """Run an edge, group or jointly calibrated omniB family.

    ``inferential=False`` retains the bootstrap as a QA diagnostic but strips
    every field that could be interpreted as experiment-wide rejection
    authority.
    """
    from . import interact as I

    hypothesis_unit = str(hypothesis_unit).lower()
    family_scope = str(family_scope).lower()
    if hypothesis_unit not in {"edge", "group"}:
        raise ValueError("hypothesis_unit must be edge or group")
    if family_scope not in {"primary_only", "joint"}:
        raise ValueError("family_scope must be primary_only or joint")
    if not isinstance(inferential, bool):
        raise ValueError("inferential must be true or false")
    if str(transform).upper() != "INT":
        raise ValueError("formal omniB family calibration requires transform='INT'")
    if isinstance(bootstrap_B, bool) or int(bootstrap_B) != bootstrap_B:
        raise ValueError("formal calibration bootstrap_B must be an integer")
    bootstrap_B = int(bootstrap_B)
    if bootstrap_B < 1:
        raise ValueError("formal calibration requires at least one bootstrap replicate")
    _resolve_feature_seed(feature_seed, bootstrap_seed)
    checkpoint_metadata = None
    if checkpoint_dir is None:
        # Keep the historical, single-stream bootstrap byte-for-byte unchanged
        # unless indexed checkpointing is explicitly requested.
        scores, expanded = score_omnib_family(
            subdata, family, y_raw, sample_idx, cap=cap, n_pc=n_pc,
            transform="INT", bootstrap_B=bootstrap_B,
            bootstrap_seed=bootstrap_seed, feature_seed=feature_seed,
            n_jobs=n_jobs,
            grm_method=grm_method, maf_min=maf_min, burden_maf=burden_maf,
            min_snp=min_snp, covariates=covariates)
        primary_p, identities, family_id, calibrated_layers = (
            _select_primary_family(
                scores, family, expanded, hypothesis_unit, family_scope))
        if primary_p.ndim != 2 or primary_p.shape[1] != bootstrap_B + 1:
            raise RuntimeError(
                "omniB scorer returned an invalid observed/bootstrap matrix shape")
        observed = primary_p[:, 0]
        finite = np.isfinite(observed)
        if not finite.any():
            raise ValueError(
                "declared omniB primary family has no estimable hypotheses")
    else:
        from .resampling_checkpoint import (
            CHECKPOINT_SCHEMA_VERSION,
            CheckpointStore,
            canonical_manifest_id,
        )

        if (
            isinstance(checkpoint_block_size, bool)
            or not isinstance(checkpoint_block_size, (int, np.integer))
            or int(checkpoint_block_size) < 1
        ):
            raise ValueError("checkpoint_block_size must be an integer >= 1")
        checkpoint_block_size = int(checkpoint_block_size)
        scores, expanded = _prepare_checkpoint_omnib(
            subdata, family, y_raw, sample_idx, cap=cap, n_pc=n_pc,
            transform="INT",
            bootstrap_seed=bootstrap_seed, feature_seed=feature_seed,
            n_jobs=n_jobs,
            grm_method=grm_method, maf_min=maf_min, burden_maf=burden_maf,
            min_snp=min_snp, covariates=covariates)
        score_omnib_observed(scores, family, expanded, n_jobs=n_jobs)
        observed_matrix, identities, family_id, calibrated_layers = (
            _select_primary_family(
                scores, family, expanded, hypothesis_unit, family_scope))
        if observed_matrix.ndim != 2 or observed_matrix.shape[1] != 1:
            raise RuntimeError(
                "omniB scorer returned an invalid observed matrix shape")
        observed = observed_matrix[:, 0]
        finite = np.isfinite(observed)
        if not finite.any():
            raise ValueError(
                "declared omniB primary family has no estimable hypotheses")
        manifest = _checkpoint_manifest(
            subdata, family, expanded, scores, y_raw, sample_idx, identities,
            hypothesis_unit=hypothesis_unit,
            family_scope=family_scope,
            bootstrap_B=bootstrap_B,
            bootstrap_seed=bootstrap_seed,
            cap=cap,
            n_pc=n_pc,
            grm_method=grm_method,
            maf_min=maf_min,
            burden_maf=burden_maf,
            min_snp=min_snp,
            covariates=covariates,
            alpha=alpha,
            inferential=inferential,
            manifest_context=checkpoint_manifest_context,
        )
        manifest_id = canonical_manifest_id(manifest)
        store = CheckpointStore(
            checkpoint_dir, manifest_id, bootstrap_B,
            checkpoint_block_size, base_seed=bootstrap_seed)
        store.bind_manifest(manifest)
        hypothesis_ids = [record["hypothesis_id"] for record in identities]
        store.write_observed(observed, hypothesis_ids)
        for start, stop in store.missing_ranges():
            edge_null, group_null = score_omnib_null_indices(
                scores, family, expanded, range(start, stop),
                base_seed=bootstrap_seed, n_jobs=n_jobs)
            block = _select_primary_matrix(
                edge_null, group_null, hypothesis_unit, family_scope)
            store.write_block(start, stop, block[finite])
        primary_null = store.concatenate(require_complete=True)
        if primary_null.shape != (int(finite.sum()), bootstrap_B):
            raise RuntimeError(
                "checkpoint primary null-p matrix has an invalid shape")
        primary_p = np.full((observed.size, bootstrap_B + 1), np.nan)
        primary_p[:, 0] = observed
        primary_p[finite, 1:] = primary_null
        checkpoint_metadata = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "manifest_id": manifest_id,
            "block_size": checkpoint_block_size,
            "score_microblock_size": INDEXED_SCORE_MICROBLOCK,
            "score_algorithm": PREPARED_SCORE_ALGORITHM,
            "completed_ranges": [
                [int(start), int(stop)]
                for start, stop in store.completed_ranges()
            ],
            "primary_null_p_sha256": hashlib.sha256(
                np.ascontiguousarray(primary_null).tobytes(order="C")
            ).hexdigest(),
        }

    finite_indices = np.flatnonzero(finite)

    # This is deliberately the sole calibration call for primary_only and joint.
    calibration = bootstrap_minp_calibration(
        observed[finite], primary_p[finite, 1:], alpha=alpha)
    diagnostic_adjusted = np.full(observed.size, np.nan)
    diagnostic_adjusted[finite] = calibration["adjusted_p_local"]
    adjusted = (
        diagnostic_adjusted
        if inferential else np.full(observed.size, np.nan))
    rejected_indices = [
        int(finite_indices[int(local)])
        for local in calibration["rejected_local"]
    ]
    rejected_set = set(rejected_indices) if inferential else set()

    records = []
    for index, identity in enumerate(identities):
        records.append(identity | {
            "family_id": family_id,
            "p": float(observed[index]) if finite[index] else None,
            "p_interaction": (
                float(observed[index]) if finite[index] else None),
            "p_adjusted_bootstrap_minp": (
                float(adjusted[index]) if np.isfinite(adjusted[index]) else None),
            "primary_sig": index in rejected_set,
            "p_unestimable": not bool(finite[index]),
        })
    order = np.argsort(
        np.where(finite, observed, np.inf), kind="stable").astype(int).tolist()
    sig = (
        [records[index] for index in order if index in rejected_set]
        if inferential else None)
    top = [records[index] for index in order if finite[index]][:5]

    hypothesis_ids = [record["hypothesis_id"] for record in records]
    fwer = {
        key: value
        for key, value in calibration.items()
        if key not in {"rejected_local", "adjusted_p_local"}
    } | {
        "family_id": family_id,
        "family_scope": family_scope,
        "declared_hypothesis_unit": hypothesis_unit,
        "calibrated_layers": calibrated_layers,
        "formal_discovery_layer": inferential,
        "inferential": inferential,
        "n_hypotheses": int(observed.size),
        "n_calibrated": int(finite.sum()),
        "n_unestimable": int((~finite).sum()),
        "empirical_p": (
            float(calibration["empirical_p"]) if inferential else None),
        "threshold": calibration["threshold"] if inferential else None,
        "threshold_comparator": (
            calibration.get("threshold_comparator") if inferential else None),
        "hypothesis_ids": hypothesis_ids,
        "family_order_sha256": hashlib.sha256(
            "\x00".join(hypothesis_ids).encode()).hexdigest(),
        "observed_p": [
            float(value) if np.isfinite(value) else None for value in observed
        ],
        "adjusted_p": [
            float(value) if np.isfinite(value) else None for value in adjusted
        ],
        "rejected": bool(calibration["rejected"]) if inferential else None,
        "rejected_indices": (
            [int(index) for index in order if index in rejected_set]
            if inferential else None),
        "rejected_hypothesis_ids": (
            [hypothesis_ids[index] for index in order if index in rejected_set]
            if inferential else None),
        "n_rejected": len(sig) if inferential else None,
        "sig": sig,
        "note": (
            "This bootstrap min-P object is the sole calibrated discovery "
            "layer; edge and component decompositions are descriptive unless "
            "included in family_scope=joint."
            if inferential else
            "QA-only bootstrap diagnostics; this object has no formal "
            "discovery or rejection authority."),
    }
    if not inferential:
        fwer["qa_diagnostics"] = {
            "role": "noninferential_do_not_threshold",
            "empirical_p": float(calibration["empirical_p"]),
            "threshold": calibration["threshold"],
            "threshold_comparator": calibration.get("threshold_comparator"),
            "adjusted_p": [
                float(value) if np.isfinite(value) else None
                for value in diagnostic_adjusted
            ],
        }

    flags = omnib_fwer_consistency_flags({
        "n_sig": len(sig) if inferential else None,
        "sig": sig,
        "minp_boot_rejected": (
            bool(calibration["rejected"]) if inferential else None),
        "minp_boot_emp": (
            float(calibration["empirical_p"]) if inferential else None),
        "minp_boot_threshold": (
            calibration["threshold"] if inferential else None),
        "bootstrap_B": bootstrap_B,
        "model_diagnostics": {"bootstrap_fwer": fwer},
    })
    if flags:
        raise RuntimeError(
            "internal omniB FWER serialization inconsistency: " + ", ".join(flags))

    if full_dump_path:
        header, rows = _ranking_rows(I, records, order, family)
        I._write_ranking_tsv(full_dump_path, header, rows)

    G = int(observed.size)
    bonferroni_alpha = float(alpha / G)
    analytic_indices = [
        index for index in order
        if finite[index] and observed[index] < bonferroni_alpha
    ]
    analytic = [records[index] for index in analytic_indices]
    family_provenance = _family_provenance(family, expanded)
    if checkpoint_metadata is not None:
        family_provenance |= {
            "checkpoint_manifest_id": checkpoint_metadata["manifest_id"],
            "checkpoint_block_size": checkpoint_metadata["block_size"],
            "checkpoint_completed_ranges": checkpoint_metadata[
                "completed_ranges"],
            "primary_null_p_sha256": checkpoint_metadata[
                "primary_null_p_sha256"],
        }
    model_diagnostics = {
        "bootstrap_fwer": fwer,
        "family_provenance": family_provenance,
        "parallel_execution": dict(scores.parallel_execution),
        "grm_provenance": {
            "method": grm_method,
            "maf_min": float(maf_min),
            "subgenomes": scores.grm_provenance,
        },
        "feature_provenance": scores.feature_seed_provenance | {
            "feature_cache_sha256": scores.feature_cache_sha256,
        },
        "prepared_design": {
            "sha256": scores.prepared_design_sha256,
            "fixed_mask_sha256": scores.fixed_mask_sha256,
            "null_fit_sha256": scores.null_fit_sha256,
        },
    }
    if checkpoint_metadata is not None:
        model_diagnostics["resampling_checkpoint"] = checkpoint_metadata
    return _interact_result(
        I,
        trait="",
        transform="INT",
        n=int(np.asarray(sample_idx).size),
        G=G,
        pair_acat=float(I.acat(observed[finite])),
        pair_acat_emp=float("nan"),
        min_p=float(observed[finite].min()),
        lambda_gc_obs=float(I.lambda_gc(observed[finite])),
        lambda_gc_perm_median=float("nan"),
        bonferroni_alpha=bonferroni_alpha,
        n_sig=len(sig) if inferential else None,
        sig=sig,
        top=top,
        sigma_hat={
            sub: float(scores.covariance_components.get(sub, 0.0))
            for sub in family.subgenomes
        } | {"e": float(scores.covariance_components.get("e", 0.0))},
        weighted=None,
        covariates=scores.covariate_metadata,
        n_planned=G,
        n_valid=int(finite.sum()),
        n_unestimable=int((~finite).sum()),
        statistic="omniB",
        calibration_method="bootstrap",
        bootstrap_B=bootstrap_B,
        bootstrap_seed=int(bootstrap_seed),
        minp_boot_emp=(
            float(calibration["empirical_p"]) if inferential else None),
        minp_boot_threshold=(
            calibration["threshold"] if inferential else None),
        minp_boot_rejected=(
            bool(calibration["rejected"]) if inferential else None),
        component_diagnostics={
            "role": "descriptive_localization",
            "calibrated_layers": calibrated_layers,
        },
        model_diagnostics=model_diagnostics,
        analytic_screen_n=len(analytic),
        analytic_screen_sig=analytic,
    )


def omnib_fwer_consistency_flags(payload: dict) -> tuple[str, ...]:
    """Return stable audit codes for a serialized formal omniB result record."""
    flags: list[str] = []
    try:
        fwer = payload["model_diagnostics"]["bootstrap_fwer"]
    except (KeyError, TypeError):
        return ("OMNIB_FWER_OBJECT_MISSING",)
    if not isinstance(fwer, dict):
        return ("OMNIB_FWER_OBJECT_MISSING",)

    family_scope = fwer.get("family_scope")
    declared = fwer.get("declared_hypothesis_unit")
    expected_family = "joint" if family_scope == "joint" else declared
    if fwer.get("family_id") != expected_family:
        flags.append("OMNIB_FWER_FAMILY_ID_MISMATCH")

    expected_layers = (
        ["edge", "group"] if family_scope == "joint" else [declared])
    if fwer.get("calibrated_layers") != expected_layers:
        flags.append("OMNIB_FWER_UNCALIBRATED_SECOND_PRIMARY_LAYER")

    inferential = fwer.get("inferential")
    formal_discovery_layer = fwer.get("formal_discovery_layer")
    n_rejected = fwer.get("n_rejected")
    fwer_rejected = fwer.get("rejected")
    empirical_p = fwer.get("empirical_p")
    alpha = fwer.get("alpha")
    if inferential is False:
        qa_authority = (
            formal_discovery_layer is not False
            or payload.get("n_sig") is not None
            or payload.get("sig") is not None
            or payload.get("minp_boot_rejected") is not None
            or fwer_rejected is not None
            or n_rejected is not None
            or fwer.get("sig") is not None
            or fwer.get("rejected_indices") is not None
            or fwer.get("rejected_hypothesis_ids") is not None
        )
        if qa_authority:
            flags.append("OMNIB_FWER_QA_AUTHORITY_PRESENT")
    else:
        if formal_discovery_layer is not True or inferential is not True:
            flags.append("OMNIB_FWER_AUTHORITY_MODE_INVALID")
        if payload.get("n_sig") != n_rejected:
            flags.append("OMNIB_FWER_TOPLEVEL_COUNT_MISMATCH")
        if not (
            isinstance(fwer_rejected, bool)
            and payload.get("minp_boot_rejected") == fwer_rejected
            and isinstance(n_rejected, int)
            and fwer_rejected == (n_rejected > 0)
            and isinstance(empirical_p, (int, float))
            and isinstance(alpha, (int, float))
            and fwer_rejected == (float(empirical_p) <= float(alpha))
        ):
            flags.append("OMNIB_FWER_TOPLEVEL_DECISION_MISMATCH")
    if payload.get("minp_boot_emp") != empirical_p:
        flags.append("OMNIB_FWER_TOPLEVEL_EMPIRICAL_P_MISMATCH")
    if payload.get("minp_boot_threshold") != fwer.get("threshold"):
        flags.append("OMNIB_FWER_TOPLEVEL_THRESHOLD_MISMATCH")
    if payload.get("bootstrap_B") != fwer.get("B"):
        flags.append("OMNIB_FWER_TOPLEVEL_BOOTSTRAP_B_MISMATCH")

    ids = fwer.get("hypothesis_ids")
    observed = fwer.get("observed_p")
    adjusted = fwer.get("adjusted_p")
    if not (
        isinstance(ids, list)
        and isinstance(observed, list)
        and isinstance(adjusted, list)
        and all(isinstance(value, str) for value in ids)
        and len(ids) == len(observed) == len(adjusted)
        and fwer.get("n_hypotheses") == len(ids)
    ):
        flags.append("OMNIB_FWER_VECTOR_LENGTH_MISMATCH")
        return tuple(dict.fromkeys(flags))

    expected_hash = hashlib.sha256("\x00".join(ids).encode()).hexdigest()
    if fwer.get("family_order_sha256") != expected_hash:
        flags.append("OMNIB_FWER_FAMILY_HASH_MISMATCH")

    missing_adjusted = any(
        value is not None and adjusted[index] is None
        for index, value in enumerate(observed)
    )
    if inferential is not False and missing_adjusted:
        flags.append("OMNIB_FWER_ADJUSTED_P_MISSING")

    if inferential is False:
        def records_have_formal_adjustment(records):
            if records is None:
                return False
            if not isinstance(records, list):
                return True
            return any(
                not isinstance(record, dict)
                or record.get("p_adjusted_bootstrap_minp") is not None
                for record in records
            )

        qa_formal_adjustment = (
            payload.get("minp_boot_emp") is not None
            or payload.get("minp_boot_threshold") is not None
            or empirical_p is not None
            or fwer.get("threshold") is not None
            or fwer.get("threshold_comparator") is not None
            or any(value is not None for value in adjusted)
            or records_have_formal_adjustment(payload.get("top"))
            or records_have_formal_adjustment(
                payload.get("analytic_screen_sig"))
            or records_have_formal_adjustment(payload.get("sig"))
            or records_have_formal_adjustment(fwer.get("sig"))
        )
        if qa_formal_adjustment:
            flags.append("OMNIB_FWER_QA_FORMAL_ADJUSTMENT_PRESENT")

        qa = fwer.get("qa_diagnostics")
        qa_adjusted = qa.get("adjusted_p") if isinstance(qa, dict) else None
        qa_valid = (
            isinstance(qa, dict)
            and qa.get("role") == "noninferential_do_not_threshold"
            and isinstance(qa.get("empirical_p"), (int, float))
            and isinstance(qa_adjusted, list)
            and len(qa_adjusted) == len(ids)
            and all(
                value is None or isinstance(value, (int, float))
                for value in qa_adjusted)
        )
        if not qa_valid:
            flags.append("OMNIB_FWER_QA_DIAGNOSTICS_INVALID")
        return tuple(dict.fromkeys(flags))

    threshold = fwer.get("threshold")
    adjusted_ids = {
        ids[index]
        for index, value in enumerate(adjusted)
        if value is not None and alpha is not None and value <= alpha
    }
    threshold_ids = {
        ids[index]
        for index, value in enumerate(observed)
        if value is not None and threshold is not None and value < threshold
    }
    rejected_ids_raw = fwer.get("rejected_hypothesis_ids")
    serialized_ids = set(rejected_ids_raw or [])
    fwer_sig = fwer.get("sig")
    sig_ids = {
        record.get("hypothesis_id")
        for record in (fwer_sig or [])
        if isinstance(record, dict)
    }
    if not (
        adjusted_ids == threshold_ids == serialized_ids == sig_ids
        and n_rejected == len(serialized_ids)
    ):
        flags.append("OMNIB_FWER_THRESHOLD_HIT_MISMATCH")

    def ordered_hit_ids(value):
        if not isinstance(value, list):
            return None
        result = []
        for record in value:
            if not isinstance(record, dict) or not isinstance(
                record.get("hypothesis_id"), str
            ):
                return None
            result.append(record["hypothesis_id"])
        return result

    top_sig_ids = ordered_hit_ids(payload.get("sig"))
    fwer_sig_ids = ordered_hit_ids(fwer_sig)
    rejected_ids = (
        rejected_ids_raw
        if isinstance(rejected_ids_raw, list)
        and all(isinstance(value, str) for value in rejected_ids_raw)
        else None
    )
    if not (
        top_sig_ids is not None
        and fwer_sig_ids is not None
        and rejected_ids is not None
        and top_sig_ids == fwer_sig_ids == rejected_ids
    ):
        flags.append("OMNIB_FWER_TOPLEVEL_SIG_MISMATCH")

    rejected_indices = fwer.get("rejected_indices")
    valid_indices = (
        isinstance(rejected_indices, list)
        and all(
            isinstance(index, int)
            and not isinstance(index, bool)
            and 0 <= index < len(ids)
            for index in rejected_indices
        )
        and len(set(rejected_indices)) == len(rejected_indices)
    )
    mapped_ids = (
        [ids[index] for index in rejected_indices]
        if valid_indices else None
    )
    if mapped_ids is None or mapped_ids != rejected_ids:
        flags.append("OMNIB_FWER_REJECTED_INDEX_MISMATCH")

    top_sig = payload.get("sig")
    hit_records_ok = isinstance(top_sig, list) and top_sig == fwer_sig
    if (
        valid_indices
        and isinstance(fwer_sig, list)
        and len(fwer_sig) == len(rejected_indices)
    ):
        for full_index, hit in zip(rejected_indices, fwer_sig, strict=True):
            if not isinstance(hit, dict):
                hit_records_ok = False
                break
            try:
                hit_legacy_p = float(hit.get("p"))
                hit_p = float(hit.get("p_interaction"))
                hit_adjusted = float(hit.get("p_adjusted_bootstrap_minp"))
                expected_p = float(observed[full_index])
                expected_adjusted = float(adjusted[full_index])
            except (TypeError, ValueError):
                hit_records_ok = False
                break
            if not (
                hit.get("hypothesis_id") == ids[full_index]
                and np.isfinite(hit_legacy_p)
                and hit_legacy_p == expected_p
                and np.isfinite(hit_p)
                and np.isfinite(expected_p)
                and hit_p == expected_p
                and np.isfinite(hit_adjusted)
                and np.isfinite(expected_adjusted)
                and hit_adjusted == expected_adjusted
                and hit.get("primary_sig") is True
                and hit.get("p_unestimable") is False
            ):
                hit_records_ok = False
                break
    elif valid_indices:
        hit_records_ok = False
    if not hit_records_ok:
        flags.append("OMNIB_FWER_HIT_RECORD_MISMATCH")
    return tuple(dict.fromkeys(flags))


def run_pair_scan_omnib(
    subdata,
    pairs,
    y_raw,
    sample_idx,
    *,
    cap=150,
    n_pc=3,
    transform="INT",
    bootstrap_B=2000,
    bootstrap_seed=2026,
    n_jobs=8,
    pair_subs=None,
    grm_method="compute_grm_maf",
    maf_min=0.01,
    burden_maf=0.01,
    min_snp=3,
    covariates=None,
    tail_thresholds=(1e-2, 1e-3, 1e-4, 1e-5),
    inferential=True,
    full_dump_path=None,
    burden_dump_path=None,
    top_k_burden=5,
    primary_multiplicity="bonferroni",
):
    """Compatibility pair wrapper selecting the shared scorer's edge matrix."""
    from . import interact as I

    primary_multiplicity = str(primary_multiplicity).lower()
    if primary_multiplicity != "bonferroni":
        raise ValueError(
            "Task 3 pairwise omniB compatibility supports only "
            "primary_multiplicity='bonferroni'; bootstrap_minp becomes a "
            "formal discovery layer in Task 4")
    sx, sy = pair_subs
    family = MasterGroupFamily(
        (sx, sy), tuple(f"pair_{i}" for i in range(len(pairs))),
        tuple(tuple(pair) for pair in pairs))
    scores, expanded = score_omnib_family(
        subdata, family, y_raw, sample_idx, cap=cap, n_pc=n_pc,
        transform=transform, bootstrap_B=bootstrap_B,
        bootstrap_seed=bootstrap_seed, n_jobs=n_jobs,
        grm_method=grm_method, maf_min=maf_min, burden_maf=burden_maf,
        min_snp=min_snp, covariates=covariates)
    selected = _callable_group_rows(scores, family, min_snp)
    if not selected.size:
        raise ValueError(
            "no homoeolog pairs retained: each copy must be present with "
            f">= {min_snp} SNPs passing burden MAF >= {burden_maf}")
    # A pair master row has exactly one edge membership.  Index through the
    # memberships instead of selecting unique primitive edges so duplicate
    # planned rows remain distinct hypotheses in the legacy wrapper family.
    edge_indices = np.asarray([
        expanded.group_edge_indices[int(group_index)][0]
        for group_index in selected
    ], int)
    kept = [family.genes[int(group_index)] for group_index in selected]
    P = scores.edge_p[edge_indices]
    component_obs = scores.edge_components_obs[edge_indices]
    observed = P[:, 0]
    finite = np.isfinite(observed)
    if not finite.any():
        raise ValueError("no estimable pairs (every omniB edge was non-estimable)")
    order_all = np.argsort(
        np.where(finite, observed, np.inf), kind="stable").astype(int).tolist()
    order = [index for index in order_all if finite[index]]
    G = len(kept)
    bonferroni = 0.05 / G

    def hit(index):
        record = I._omnib_component_record(component_obs[index]) if hasattr(
            I, "_omnib_component_record") else {}
        return {
            "pair": kept[index], "p": float(observed[index]),
            "p_adjusted_bonferroni": float(min(observed[index] * G, 1.0)),
        } | record

    analytic_indices = [i for i in order if observed[i] < bonferroni]
    minp_emp = threshold = None
    tail_excess = None
    if bootstrap_B:
        null_family = P[finite, 1:]
        null_min = np.nanmin(null_family, axis=0)
        minp_emp = float(
            (1 + int((null_min <= np.nanmin(observed[finite])).sum()))
            / (bootstrap_B + 1))
        threshold = float(np.quantile(null_min, 0.05))
        tail_excess = {}
        for cutoff in tail_thresholds:
            counts = (null_family < cutoff).sum(axis=0)
            observed_count = int((observed[finite] < cutoff).sum())
            tail_excess[f"n_below_{cutoff:g}"] = {
                "observed": observed_count,
                "null_mean": float(counts.mean()),
                "null_q95": float(np.quantile(counts, 0.95)),
                "empirical_p": float(
                    (1 + int((counts >= observed_count).sum()))
                    / (bootstrap_B + 1)),
            }
    analytic = [hit(i) for i in analytic_indices]
    sig = analytic if inferential else None

    if full_dump_path:
        rows = []
        for rank, index in enumerate(order_all):
            record = (
                I._omnib_component_record(component_obs[index])
                if hasattr(I, "_omnib_component_record") else {})
            components = record.get("component_p", {})
            rows.append([
                rank, kept[index][0], kept[index][1], sx, sy,
                _tsv_p(I, observed[index]),
                _tsv_p(I, components.get("minor_burden", np.nan)),
                _tsv_p(I, components.get("pc1", np.nan)),
                _tsv_p(I, components.get("kernel_hadamard", np.nan)),
                record.get("smallest_component") or "NA",
                int(not finite[index]),
            ])
        I._write_ranking_tsv(
            full_dump_path,
            ["rank", f"gene_{sx}", f"gene_{sy}", "sub_x", "sub_y",
             "p_interaction", "p_minor_burden", "p_pc1",
             "p_kernel_hadamard", "smallest_component", "p_unestimable"], rows)

    if burden_dump_path:
        features = scores.feature_cache
        rows = []
        for pair_rank, index in enumerate(order[:max(int(top_k_burden), 1)]):
            gx, gy = kept[index]
            bx = features[(sx, gx)][0][:, 0]
            by = features[(sy, gy)][0][:, 0]
            if scores.covariate_block is None:
                residual = scores.y - float(np.mean(scores.y))
            else:
                coef, *_ = np.linalg.lstsq(
                    scores.covariate_block, scores.y, rcond=None)
                residual = scores.y - scores.covariate_block @ coef
            for sample_row in range(sample_idx.size):
                rows.append([
                    pair_rank, gx, gy, sx, sy, sample_row,
                    repr(float(bx[sample_row])), repr(float(by[sample_row])),
                    repr(float(scores.y[sample_row])),
                    repr(float(residual[sample_row])),
                ])
        I._write_ranking_tsv(
            burden_dump_path,
            ["pair_rank", "gene_x", "gene_y", "sub_x", "sub_y",
             "sample_row", "minor_burden_x", "minor_burden_y", "phenotype",
             "resid"],
            rows)

    diagnostics = (
        I._omnib_component_summary(component_obs)
        if hasattr(I, "_omnib_component_summary") else None)
    return _interact_result(
        I, trait="", transform=transform, n=int(sample_idx.size), G=G,
        pair_acat=float(I.acat(observed)), pair_acat_emp=float("nan"),
        min_p=float(np.nanmin(observed)),
        lambda_gc_obs=float(I.lambda_gc(observed[finite])),
        lambda_gc_perm_median=float("nan"), bonferroni_alpha=bonferroni,
        n_sig=(len(sig) if sig is not None else None), sig=sig,
        top=[hit(i) for i in order[:5]],
        sigma_hat={sub: float(scores.covariance_components.get(sub, 0.0))
                   for sub in subdata} | {
                       "e": float(scores.covariance_components.get("e", 0.0))},
        weighted=None, covariates=scores.covariate_metadata,
        n_planned=G, n_valid=int(finite.sum()), n_unestimable=int((~finite).sum()),
        statistic="omniB", calibration_method=("bootstrap" if bootstrap_B else "none"),
        bootstrap_B=int(bootstrap_B), bootstrap_seed=int(bootstrap_seed),
        minp_boot_emp=minp_emp, minp_boot_threshold=threshold,
        minp_boot_rejected=None,
        tail_excess=tail_excess, component_diagnostics=diagnostics,
        analytic_screen_n=len(analytic), analytic_screen_sig=analytic,
        model_diagnostics={
            "bootstrap_fwer": {
                "role": "diagnostic_only_task3_legacy_compatibility",
                "formal_discovery_layer": False,
            } if bootstrap_B else None})


def run_clique_scan_omnib(
    subdata,
    groups,
    y_raw,
    sample_idx,
    *,
    cap=150,
    n_pc=3,
    transform="INT",
    bootstrap_B=2000,
    bootstrap_seed=2026,
    n_jobs=8,
    grm_method="compute_grm_maf",
    maf_min=0.01,
    burden_maf=0.01,
    min_snp=3,
    covariates=None,
    tail_thresholds=(1e-2, 1e-3, 1e-4, 1e-5),
    inferential=True,
    full_dump_path=None,
):
    """Compatibility clique wrapper selecting the shared scorer's group matrix."""
    from . import interact as I

    subs = tuple(subdata)
    family = MasterGroupFamily(
        subs, tuple(f"group_{i}" for i in range(len(groups))),
        tuple(tuple(group) for group in groups))
    scores, expanded = score_omnib_family(
        subdata, family, y_raw, sample_idx, cap=cap, n_pc=n_pc,
        transform=transform, bootstrap_B=bootstrap_B,
        bootstrap_seed=bootstrap_seed, n_jobs=n_jobs,
        grm_method=grm_method, maf_min=maf_min, burden_maf=burden_maf,
        min_snp=min_snp, covariates=covariates)
    selected = _callable_group_rows(scores, family, min_snp)
    if not selected.size:
        raise ValueError(
            "no homoeolog groups retained (need all copies present with >= min_snp SNPs)")
    kept = [family.genes[i] for i in selected]
    P = scores.group_p[selected]
    observed = P[:, 0]
    finite = np.isfinite(observed)
    if not finite.any():
        raise ValueError("no estimable groups (every omniB group was non-estimable)")
    order_all = np.argsort(
        np.where(finite, observed, np.inf), kind="stable").astype(int).tolist()
    order = [index for index in order_all if finite[index]]
    pair_defs = list(__import__("itertools").combinations(subs, 2))
    pair_labels = [f"{left}{right}" for left, right in pair_defs]
    pair_obs = []
    components = []
    for family_index in selected:
        indices = np.asarray(expanded.group_edge_indices[family_index], int)
        pair_obs.append(scores.edge_p[indices, 0])
        components.append(scores.edge_components_obs[indices])
    pair_obs = np.asarray(pair_obs)
    components = np.asarray(components)
    G = len(kept)
    bonferroni = 0.05 / G

    def hit(index):
        pairwise = {}
        for pair_index, label in enumerate(pair_labels):
            record = (
                I._omnib_component_record(components[index, pair_index])
                if hasattr(I, "_omnib_component_record") else {})
            pairwise[label] = {"p_omnib": float(pair_obs[index, pair_index])} | record
        flat = components[index].reshape(-1)
        finite_flat = np.isfinite(flat)
        smallest_pair = smallest_component = None
        smallest_p = None
        if finite_flat.any():
            flat_index = int(np.nanargmin(np.where(finite_flat, flat, np.nan)))
            pair_index, component_index = np.unravel_index(
                flat_index, components[index].shape)
            smallest_pair = pair_labels[pair_index]
            smallest_component = OMNIB_COMPONENT_NAMES[component_index]
            smallest_p = float(flat[flat_index])
        return {
            "pair": kept[index], "p": float(observed[index]),
            "p_adjusted_bonferroni": float(min(observed[index] * G, 1.0)),
            "pairwise": pairwise,
            "smallest_pair": smallest_pair,
            "smallest_component": smallest_component,
            "smallest_component_p": smallest_p,
        }

    sig = [hit(i) for i in order if observed[i] < bonferroni] if inferential else None
    minp_emp = threshold = None
    tail_excess = None
    if bootstrap_B:
        null_family = P[finite, 1:]
        null_min = np.nanmin(null_family, axis=0)
        minp_emp = float(
            (1 + int((null_min <= np.nanmin(observed[finite])).sum()))
            / (bootstrap_B + 1))
        threshold = float(np.quantile(null_min, 0.05))
        tail_excess = {}
        for cutoff in tail_thresholds:
            counts = (null_family < cutoff).sum(axis=0)
            observed_count = int((observed[finite] < cutoff).sum())
            tail_excess[f"n_below_{cutoff:g}"] = {
                "observed": observed_count,
                "null_mean": float(counts.mean()),
                "null_q95": float(np.quantile(counts, 0.95)),
                "empirical_p": float(
                    (1 + int((counts >= observed_count).sum()))
                    / (bootstrap_B + 1)),
            }

    if full_dump_path:
        rows = []
        for rank, index in enumerate(order_all):
            record = hit(index)
            row = [
                rank, *kept[index], _tsv_p(I, observed[index]),
                record["smallest_pair"] or "NA",
                record["smallest_component"] or "NA",
                _tsv_p(I, record["smallest_component_p"]),
                int(not finite[index]),
            ]
            for label in pair_labels:
                pair_record = record["pairwise"][label]
                component_p = pair_record.get("component_p", {})
                row.extend([
                    _tsv_p(I, pair_record["p_omnib"]),
                    _tsv_p(I, component_p.get("minor_burden", np.nan)),
                    _tsv_p(I, component_p.get("pc1", np.nan)),
                    _tsv_p(I, component_p.get("kernel_hadamard", np.nan)),
                ])
            rows.append(row)
        I._write_ranking_tsv(
            full_dump_path,
            ["rank", *[f"gene_{sub}" for sub in subs], "p_interaction",
             "smallest_pair", "smallest_component", "smallest_component_p",
             "p_unestimable"]
            + [value for label in pair_labels for value in (
                f"p_omnib_{label}", f"p_minor_burden_{label}",
                f"p_pc1_{label}", f"p_kernel_hadamard_{label}")],
            rows)

    component_diagnostics = {
        "pair_labels": pair_labels,
        "per_pair": {
            label: (
                I._omnib_component_summary(components[:, pair_index, :])
                if hasattr(I, "_omnib_component_summary") else None)
            for pair_index, label in enumerate(pair_labels)
        },
        "interpretation": (
            "The group p-value is ACAT across pairwise omniB tests. Pair and "
            "component decomposition is descriptive localization, not an "
            "additional rejection family."),
    }

    return _interact_result(
        I, trait="", transform=transform, n=int(sample_idx.size), G=G,
        pair_acat=float(I.acat(observed)), pair_acat_emp=float("nan"),
        min_p=float(np.nanmin(observed)),
        lambda_gc_obs=float(I.lambda_gc(observed[finite])),
        lambda_gc_perm_median=float("nan"), bonferroni_alpha=bonferroni,
        n_sig=(len(sig) if sig is not None else None), sig=sig,
        top=[hit(i) for i in order[:5]],
        sigma_hat={sub: float(scores.covariance_components.get(sub, 0.0))
                   for sub in subdata} | {
                       "e": float(scores.covariance_components.get("e", 0.0))},
        weighted=None, covariates=scores.covariate_metadata,
        n_planned=G, n_valid=int(finite.sum()), n_unestimable=int((~finite).sum()),
        statistic="omniB", calibration_method=("bootstrap" if bootstrap_B else "none"),
        bootstrap_B=int(bootstrap_B), bootstrap_seed=int(bootstrap_seed),
        minp_boot_emp=minp_emp, minp_boot_threshold=threshold,
        tail_excess=tail_excess, component_diagnostics=component_diagnostics)
