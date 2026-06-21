#!/usr/bin/env python
"""Parametric sequencing-DEPTH design heuristic for HomoeoGWAS.

Allopolyploid genomes are huge, so blind whole-genome sequencing (WGS) is
expensive and a breeder reasonably asks, *before* paying for sequencing, roughly
what coverage depth (in x) is enough to have a chance of detecting signal. This
module answers that as a transparent pre-flight calculator.

Chain::

    depth (x)
      -> retention fraction f(depth)            # per-sample Poisson confident-call prob
      -> effective usable in-gene density        # lambda_eff = lambda_full * f(depth)
      -> predicted callable homoeolog-pair count # ride the validated H7 density curve
      -> discovery design-band                   # null / borderline / feasible / strong

Two modes are reported side by side: the MARGINAL subgenome-stratified scan, and
the stricter homoeolog-INTERACTION test. The interaction test must confidently
genotype *both* homoeolog copies *and* tell them apart, so it uses a higher
confident-call depth threshold and a lower effective mappability (homoeolog
cross-mapping corrupts exactly the copy-distinguishing sites) -> it needs more
depth. That marginal-vs-interaction gap is the allopolyploid-specific point.

PARAMETRIC PLANNING HEURISTIC ONLY -- read this. No raw sequencing reads were
used to fit any of this (the project's panels are all VCF/array-derived genotype
matrices). Coverage is modelled as Poisson; the confident-call depth threshold
and effective mappability are *explicit assumptions*; the full-depth endpoint is
*anchored* to each panel's observed usable density.
Treat the numbers as order-of-magnitude planning estimates with sensitivity
bands, NOT as an empirically validated, guaranteed depth optimiser. The single
biggest unknown is effective mappability/copy-assignability, which is assumed
here rather than learned from reads; the marginal-vs-interaction *relative*
ordering is far more robust than any absolute depth threshold.

The validated H7 density->callability curve (held-out median relative error
3.5%, four real-panel anchors) is baked in below as ``_LAMBDA_GRID`` /
``_PGENE_GRID`` so this module needs neither the raw wheat data nor a VCF at call
time -- exactly what a pre-sequencing calculator requires.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------- #
# Validated empirical per-gene callable-fraction curve (from h7_marker_density,
# min_snp=3 thinning of wheat chr1 WGS; held-out rel-error 3.5%). lambda_gene =
# usable, in-gene, copy-assigned SNPs per gene copy. A (0, 0) origin is prepended
# and the curve saturates at the full-WGS value for lambda beyond the grid.
# --------------------------------------------------------------------------- #
_LAMBDA_GRID = np.array([0.0, 0.494, 0.989, 2.472, 4.944, 9.889, 24.722, 49.445])
_PGENE_GRID = np.array([0.0, 0.0316, 0.1070, 0.3254, 0.5257, 0.6616, 0.7794, 0.8885])

# wheat Watkins WGS measured mean in-gene SNPs per gene copy at full depth.
WHEAT_LAMBDA_FULL = 49.44485384426431

# feasibility bands on callable homoeolog-pair count G (same empirical boundary
# as the H7 marker-density tool; heuristic categories, not formal thresholds).
BANDS = [(0, 70, "underpowered/null-likely"), (70, 300, "borderline/risky"),
         (300, 1000, "discovery-feasible"), (1000, math.inf, "strong-design")]
FEASIBLE_G = 300
STRONG_G = 1000


def classify_band(G: float) -> str:
    for lo, hi, lab in BANDS:
        if lo <= G < hi:
            return lab
    return "strong-design"


def pgene_at_lambda(lambda_gene: float) -> float:
    """Validated per-gene callable fraction at a given usable in-gene density.

    Linear interpolation on the baked curve; clamped to the full-WGS saturating
    value above the grid and to 0 below it.
    """
    lam = max(0.0, float(lambda_gene))
    return float(np.interp(lam, _LAMBDA_GRID, _PGENE_GRID,
                           left=0.0, right=float(_PGENE_GRID[-1])))


@dataclass(frozen=True)
class ModeParams:
    """Explicit, user-overridable assumptions for one test mode.

    min_dp        confident-call depth threshold per sample (a genotype counts
                  only if its read depth >= min_dp -- this already encodes
                  "callable *and* genotyped", so genotype error is not
                  double-counted in the recovery fraction).
    mappability   effective fraction of nominal depth that lands uniquely and
                  copy-assigned on a gene copy (homoeolog cross-mapping < 1).

    Note: sample count N does NOT enter the retention fraction -- callable-pair
    COUNT is a per-site marker-density quantity, independent of N (N enters
    statistical POWER, modelled separately). Folding a hard cross-sample
    missingness filter here would only sharpen the curve into a near step; we
    keep the soft per-genotype recovery as the planning quantity.
    """

    name: str
    min_dp: int
    mappability: float

    def __post_init__(self):
        if int(self.min_dp) < 1:
            raise ValueError(f"min_dp must be >= 1 (got {self.min_dp})")
        if not (0.0 < float(self.mappability) <= 1.0):
            raise ValueError(
                f"mappability must be in (0, 1] (got {self.mappability})")


# Defaults are PLANNING ASSUMPTIONS, deliberately conservative for interaction.
MARGINAL = ModeParams(name="marginal", min_dp=4, mappability=0.85)
INTERACTION = ModeParams(name="interaction", min_dp=8, mappability=0.55)
MODES = {"marginal": MARGINAL, "interaction": INTERACTION}


def site_callable_prob(depth: float, mappability: float, min_dp: int) -> float:
    """P(a true segregating site reaches >= min_dp reads) in one sample.

    Per-sample, per-site read count ~ Poisson(mean = depth * mappability).
    """
    mu = max(0.0, float(depth) * float(mappability))
    return float(stats.poisson.sf(int(min_dp) - 1, mu))


def retention_fraction(depth: float, mode: ModeParams) -> float:
    """Fraction of the saturating usable genotype calls retained at this depth.

    Soft per-genotype recovery = expected fraction of confidently-called
    genotypes at this depth (the per-site Poisson call probability). f -> 1 at
    high depth (so the full-depth endpoint reproduces the anchored panel
    density) and f -> 0 at depth 0; smooth and monotone in between.
    """
    return site_callable_prob(depth, mode.mappability, mode.min_dp)


def predict_G(depth: float, *, g_full: float, lambda_full: float,
              mode: ModeParams) -> dict:
    """Predicted callable homoeolog-pair count G at a given depth.

    Rides the validated density curve and rescales to the species' observed
    full-depth G: a pair is callable iff both gene copies are, so the relative
    erosion is (P_gene(lambda_eff) / P_gene(lambda_full))**2, capped at 1.
    """
    if float(depth) < 0:
        raise ValueError(f"depth must be >= 0 (got {depth})")
    if float(lambda_full) <= 0:
        raise ValueError(f"lambda_full must be > 0 (got {lambda_full})")
    if float(g_full) < 0:
        raise ValueError(f"g_full must be >= 0 (got {g_full})")
    f = retention_fraction(depth, mode)
    lam_eff = float(lambda_full) * f
    pg_full = pgene_at_lambda(lambda_full)
    pg_eff = pgene_at_lambda(lam_eff)
    rho = (pg_eff / pg_full) ** 2 if pg_full > 0 else 0.0
    rho = min(max(rho, 0.0), 1.0)
    G = float(g_full) * rho
    return dict(depth=float(depth), mode=mode.name, retention_f=float(f),
                lambda_eff=float(lam_eff), rel_to_full=float(rho),
                G_pred=float(G), band=classify_band(G))


def depth_for_band(target_G: float, *, g_full: float, lambda_full: float,
                   mode: ModeParams,
                   dmax: float = 60.0, dstep: float = 0.1) -> float | None:
    """Smallest depth whose predicted G reaches ``target_G``.

    Returns None if even saturating depth cannot reach it -- i.e. the binding
    constraint is marker density / capture design, not depth (the honest answer
    for sparse-GBS panels whose full-depth G is already below the band).
    """
    if float(g_full) < float(target_G):
        return None                                  # unreachable at any depth
    grid = np.arange(dstep, dmax + dstep, dstep)
    for d in grid:
        if predict_G(d, g_full=g_full, lambda_full=lambda_full,
                     mode=mode)["G_pred"] >= target_G:
            return float(round(d, 2))
    return None                                      # not reached within dmax


# --------------------------------------------------------------------------- #
# Species presets: the four anchored real panels (full-depth G observed). Use as
# `--like <name>` so a planner with no VCF yet still gets a grounded estimate.
# lambda_full defaults to the wheat-measured value (a stated assumption: we take
# saturating WGS to reach a comparable in-gene per-copy density; species with
# intrinsically lower polymorphism start lower and reach bands at higher depth --
# expose --lambda-full to test sensitivity).
# --------------------------------------------------------------------------- #
SPECIES_PRESETS = {
    "wheat": dict(species="wheat AABBDD", ploidy="6n", n_subgenomes=3,
                  genome_gb=16.0, g_full=1759, n_default=827, tech="WGS",
                  note="Watkins WGS; chr1 flank B-D scope; discovery (2 hits)"),
    "cotton": dict(species="cotton AADD", ploidy="4n", n_subgenomes=2,
                   genome_gb=2.3, g_full=372, n_default=419, tech="SNP-array",
                   note="hebau 498k array; discovery (2 hits)"),
    "oat": dict(species="oat AACCDD", ploidy="6n", n_subgenomes=3,
                genome_gb=11.0, g_full=69, n_default=737, tech="sparse GBS",
                note="OLD GBS; null (density-limited boundary)"),
    "rapeseed": dict(species="rapeseed AACC", ploidy="4n", n_subgenomes=2,
                     genome_gb=1.1, g_full=14, n_default=428, tech="GBS+imputation",
                     note="Horvath GBS; null (density-limited boundary)"),
}

CAVEATS = [
    "PARAMETRIC PLANNING HEURISTIC -- no raw reads were used to fit it; numbers "
    "are order-of-magnitude with sensitivity bands, not guaranteed thresholds.",
    "f(depth) is per-sample/per-genotype confident-call recovery; it does NOT "
    "model the cross-sample call-rate/missingness filter, allele DISCOVERY across "
    "the panel, residual genotype-likelihood uncertainty, or copy-misassignment "
    "error rates -- N is omitted from the COUNT only under this softened "
    "definition (N enters power separately).",
    "The non-WGS presets (cotton SNP-array, oat/rapeseed GBS) anchor each panel's "
    "OBSERVED full-depth callable-pair count; the WGS depth curve is then a "
    "RESCALED planning analogy ('what hypothetical WGS depth reaches this "
    "density'), not an empirical WGS depth calibration of those panels.",
    "lambda_full above the baked H7 grid is clamped: higher intrinsic "
    "polymorphism cannot raise P_gene beyond the full-WGS saturating value in "
    "this model.",
    "Coverage is modelled as Poisson (ignores library/GC/target overdispersion, "
    "which makes real low-depth recovery somewhat worse).",
    "Effective mappability and the confident-call depth threshold are ASSUMPTIONS "
    "(the dominant uncertainty); the marginal-vs-interaction ORDERING is far more "
    "robust than any absolute depth.",
    "The full-depth G is ANCHORED to each panel's observed callable-pair count; "
    "lambda_full defaults to the wheat-measured in-gene density (an assumption -- "
    "vary --lambda-full for species with lower intrinsic polymorphism).",
    "G predicts callable pairwise TEST opportunities (not independent evidence) "
    "and also sets the alpha/G multiple-testing burden.",
    "Homoeolog copy misassignment at low depth can create FALSE interaction "
    "signal, not only lose power -- a risk this yield model does not capture.",
    "If full-depth G is already below a band, NO depth fixes it: the limit is "
    "marker density / capture design, not coverage.",
]


def design_table(*, g_full: float, lambda_full: float,
                 depth_grid, modes=("marginal", "interaction")) -> list[dict]:
    """One row per (depth, mode)."""
    rows = []
    for d in depth_grid:
        for m in modes:
            rows.append(predict_G(d, g_full=g_full, lambda_full=lambda_full,
                                  mode=MODES[m]))
    return rows


def depth_band_range(target_G: float, *, g_full: float, lambda_full: float,
                     mode: ModeParams, map_delta: float = 0.15,
                     dmax: float = 60.0) -> list:
    """Depth-to-band under a +/-map_delta mappability sensitivity sweep.

    Returns [optimistic, point, pessimistic] depths: better mappability lowers
    the depth needed, worse mappability raises it. The dominant unknown is
    effective mappability, so a single threshold is never reported alone.
    """
    out = []
    for mp_val in (min(1.0, mode.mappability + map_delta), mode.mappability,
                   max(0.05, mode.mappability - map_delta)):
        out.append(depth_for_band(target_G, g_full=g_full, lambda_full=lambda_full,
                                  mode=replace(mode, mappability=mp_val), dmax=dmax))
    return out


def design_summary(*, g_full: float, lambda_full: float,
                   modes=("marginal", "interaction"), dmax: float = 60.0,
                   map_delta: float = 0.15) -> dict:
    """Per-mode 'depth to reach feasible/strong' inversion -- the headline.

    Each threshold also carries a mappability sensitivity range (the dominant
    unknown), not a single clean number.
    """
    out = {}
    for m in modes:
        mp = MODES[m]
        out[m] = dict(
            assumptions=asdict(mp), map_delta=map_delta,
            depth_for_feasible=depth_for_band(FEASIBLE_G, g_full=g_full,
                                              lambda_full=lambda_full,
                                              mode=mp, dmax=dmax),
            depth_for_feasible_range=depth_band_range(
                FEASIBLE_G, g_full=g_full, lambda_full=lambda_full, mode=mp,
                map_delta=map_delta, dmax=dmax),
            depth_for_strong=depth_for_band(STRONG_G, g_full=g_full,
                                            lambda_full=lambda_full,
                                            mode=mp, dmax=dmax),
            depth_for_strong_range=depth_band_range(
                STRONG_G, g_full=g_full, lambda_full=lambda_full, mode=mp,
                map_delta=map_delta, dmax=dmax),
            g_full=float(g_full))
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_depth_grid(s: str) -> list[float]:
    return [float(x) for x in str(s).split(",") if x.strip() != ""]


def cmd_design(args) -> int:
    # resolve species anchor
    if getattr(args, "like", None):
        if args.like not in SPECIES_PRESETS:
            print(f"unknown --like '{args.like}'; choose from "
                  f"{sorted(SPECIES_PRESETS)}", file=sys.stderr)
            return 2
        preset = dict(SPECIES_PRESETS[args.like])
    else:
        if args.g_full is None:
            print("provide --like <species> OR --g-full <observed full-depth "
                  "callable-pair count>", file=sys.stderr)
            return 2
        preset = dict(species=args.species or "custom", ploidy=args.ploidy or "?",
                      n_subgenomes=args.subgenomes, genome_gb=args.genome_gb,
                      g_full=args.g_full, n_default=None, tech="custom",
                      note="user-supplied anchor")

    g_full = float(args.g_full) if args.g_full is not None else float(preset["g_full"])
    n_samples = int(args.samples) if args.samples is not None else int(
        preset.get("n_default") or 400)
    lambda_full = float(args.lambda_full)
    depth_grid = _parse_depth_grid(args.depth_grid)

    # validate scalar anchors up front (loud, not silent nonsense)
    if g_full < 0:
        print(f"--g-full must be >= 0 (got {g_full})", file=sys.stderr)
        return 2
    if lambda_full <= 0:
        print(f"--lambda-full must be > 0 (got {lambda_full})", file=sys.stderr)
        return 2
    if any(d < 0 for d in depth_grid):
        print("--depth-grid values must be >= 0", file=sys.stderr)
        return 2

    # apply optional mode-assumption overrides (ModeParams validates ranges)
    marg, inter = MARGINAL, INTERACTION
    try:
        if args.marginal_min_dp is not None:
            marg = replace(marg, min_dp=args.marginal_min_dp)
        if args.marginal_mappability is not None:
            marg = replace(marg, mappability=args.marginal_mappability)
        if args.interaction_min_dp is not None:
            inter = replace(inter, min_dp=args.interaction_min_dp)
        if args.interaction_mappability is not None:
            inter = replace(inter, mappability=args.interaction_mappability)
    except ValueError as e:
        print(f"invalid mode override: {e}", file=sys.stderr)
        return 2
    modes_local = {"marginal": marg, "interaction": inter}

    lambda_clamped = lambda_full > float(_LAMBDA_GRID[-1])
    if lambda_clamped:
        print(f"# WARNING: lambda_full={lambda_full:g} exceeds the H7 grid max "
              f"({_LAMBDA_GRID[-1]:g}); P_gene is clamped at its full-WGS "
              f"saturating value -- higher polymorphism does not raise it here.",
              file=sys.stderr)

    rows = []
    for d in depth_grid:
        for mp in modes_local.values():
            rows.append(predict_G(d, g_full=g_full, lambda_full=lambda_full,
                                  mode=mp))
    summ = {m: dict(assumptions=asdict(mp), map_delta=0.15,
                    depth_for_feasible=depth_for_band(
                        FEASIBLE_G, g_full=g_full, lambda_full=lambda_full, mode=mp),
                    depth_for_feasible_range=depth_band_range(
                        FEASIBLE_G, g_full=g_full, lambda_full=lambda_full, mode=mp),
                    depth_for_strong=depth_for_band(
                        STRONG_G, g_full=g_full, lambda_full=lambda_full, mode=mp),
                    depth_for_strong_range=depth_band_range(
                        STRONG_G, g_full=g_full, lambda_full=lambda_full, mode=mp),
                    g_full=g_full)
            for m, mp in modes_local.items()}

    payload = dict(tool="homoeogwas", analysis="design_depth",
                   framing="parametric planning heuristic (no raw reads)",
                   species=preset["species"], ploidy=preset.get("ploidy"),
                   genome_gb=preset.get("genome_gb"), tech=preset.get("tech"),
                   n_samples=n_samples, n_samples_note="context for POWER only; "
                   "does not affect the callable-pair COUNT/band",
                   g_full=g_full, lambda_full=lambda_full,
                   lambda_full_clamped=lambda_clamped,
                   feasible_G=FEASIBLE_G, strong_G=STRONG_G,
                   depth_grid=depth_grid, table=rows, summary=summ,
                   caveats=CAVEATS, note=preset.get("note"))

    # human-readable report to stdout
    print("# HomoeoGWAS sequencing-depth design (PARAMETRIC PLANNING HEURISTIC)")
    print(f"# species={preset['species']}  genome~{preset.get('genome_gb')}Gb  "
          f"N={n_samples}  full-depth G={g_full:g}  lambda_full={lambda_full:g}")
    if preset.get("tech") not in (None, "WGS", "custom"):
        print(f"# NOTE: {preset['tech']} anchor -- 'depth' is the hypothetical WGS "
              f"depth to reach this panel's observed density (rescaled analogy).")
    print(f"{'depth':>6} {'mode':>12} {'ret_f':>7} {'lam_eff':>8} "
          f"{'G_pred':>8}  band")
    for r in rows:
        print(f"{r['depth']:>6g} {r['mode']:>12} {r['retention_f']:>7.3f} "
              f"{r['lambda_eff']:>8.2f} {r['G_pred']:>8.1f}  {r['band']}")
    print("\n# depth needed to reach a band [optimistic..pessimistic over "
          "mappability +/-0.15] (None = unreachable; density/capture-limited):")
    for m, s in summ.items():
        fr, sr = s["depth_for_feasible_range"], s["depth_for_strong_range"]
        print(f"  {m:>12}: feasible(G>={FEASIBLE_G}) -> {s['depth_for_feasible']}x "
              f"[{fr[0]}..{fr[2]}] ;  strong(G>={STRONG_G}) -> "
              f"{s['depth_for_strong']}x [{sr[0]}..{sr[2]}]")
    if g_full < FEASIBLE_G:
        print(f"  NOTE: full-depth G={g_full:g} < {FEASIBLE_G} -> no WGS depth "
              f"reaches the discovery-feasible band; the binding constraint is "
              f"marker density / capture design, not coverage.")
    print("\n# assumptions are planning choices; see caveats in the JSON output.")

    if getattr(args, "out", None):
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nwrote {outp}")
    return 0
