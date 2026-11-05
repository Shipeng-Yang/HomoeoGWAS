"""Figures for canonical group omniB results (``homoeogwas interact``).

Every figure is drawn from the result JSON, plus gene positions taken from the
genotype ``.bim`` files and the SNP-to-gene maps named in the run config:

* ``interaction_scan`` — group −log10 P along the genome, drawn at the position
  of every copy; groups significant at the family-wise level are highlighted and
  their copies joined.
* ``interaction_evidence`` — for each reported group, the P value of every
  encoding on every edge, with the edge omniB P above.
* ``interaction_calibration`` — the bootstrap distribution of the family
  minimum P with the observed minimum and the experiment-wide threshold, and a
  Q–Q plot of the group P values.
* ``interaction_null_variance`` — smooth_pc4 runs only: the per-sample variance
  weights against the first two genomic eigenvectors.

No statistic is computed here; the figures only redraw recorded values.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from .plots import (
    _PUB_RC,
    DEFAULT_FORMATS,
    _import_mpl,
    _offset_spines,
    save_figure,
    subgenome_palette,
    write_figure_index,
)

SIGNAL = "#A3285A"
THRESHOLD = "#D62728"
NEUTRAL = "#B9B4AE"
NEUTRAL_DARK = "#6E6862"
ENCODING_LABELS = {"minor_burden": "burden", "pc1": "PC1 × PC1",
                   "kernel_hadamard": "kernel"}
MAX_EVIDENCE_GROUPS = 12
CAPTIONS = (
    ("interaction_scan", "Group −log10 P at the position of every gene copy; magenta, groups "
     "significant at the family-wise level with their copies joined; red dashed line, "
     "experiment-wide threshold."),
    ("interaction_evidence", "Reported groups: −log10 P of each encoding (rows) on each edge "
     "(columns), with the edge omniB −log10 P above."),
    ("interaction_calibration", "Left, bootstrap distribution of the family minimum P (grey) "
     "with the experiment-wide threshold (red dashed) and the observed minimum (magenta). "
     "Right, Q–Q plot of group P values with the pointwise 95% band; magenta, significant groups."),
    ("interaction_null_variance", "smooth_pc4 null: per-sample variance weights against the "
     "first two genomic eigenvectors; dashed line, weight 1."),
)


# ----------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------

def load_interact_result(path: str | Path) -> tuple[dict, dict]:
    """Return ``(payload, primary)`` where ``primary`` is the analysed transform."""
    payload = json.loads(Path(path).read_text())
    results = payload.get("results") or {}
    if not results:
        raise ValueError(f"{path}: no results block")
    primary = results.get("INT") or next(iter(results.values()))
    return payload, primary


def _natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(text))]


def gene_positions(genotype: Mapping[str, str],
                   snp_to_gene: Mapping[str, str]) -> pd.DataFrame:
    """Chromosome and midpoint of each gene from its assigned SNPs.

    Uses the ``.bim`` of each subgenome and the ``snp_idx`` arrays of the
    SNP-to-gene map, so it needs no annotation file.
    """
    rows = []
    for sub, prefix in genotype.items():
        bim = pd.read_csv(f"{prefix}.bim", sep=r"\s+", header=None,
                          usecols=[0, 3], names=["chrom", "pos"],
                          dtype={"chrom": str, "pos": np.int64})
        mapping = np.load(snp_to_gene[sub], allow_pickle=True)
        chrom = bim["chrom"].to_numpy()
        pos = bim["pos"].to_numpy()
        for gene, index in zip(mapping["gene_ids"], mapping["snp_idx"], strict=True):
            index = np.asarray(index, int)
            if index.size == 0:
                continue
            values, counts = np.unique(chrom[index], return_counts=True)
            rows.append((str(gene), sub, str(values[np.argmax(counts)]),
                         float(np.median(pos[index]))))
    return pd.DataFrame(rows, columns=["gene_id", "subgenome", "chrom", "pos"])


def positions_from_config(config_path: str | Path) -> pd.DataFrame:
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    block = cfg.get("interact", cfg)
    return gene_positions(block["genotype"], block["snp_to_gene"])


def _groups_table(primary: dict) -> pd.DataFrame:
    """One row per group with its genes, observed and adjusted P and decision."""
    fwer = (primary.get("model_diagnostics") or {}).get("bootstrap_fwer") or {}
    ids = fwer.get("hypothesis_ids") or []
    observed = fwer.get("observed_p") or []
    adjusted = fwer.get("adjusted_p") or [None] * len(ids)
    rejected = set(fwer.get("rejected_hypothesis_ids") or [])
    rows = []
    for hid, p, padj in zip(ids, observed, adjusted, strict=True):
        group = hid.split(":", 1)[1] if ":" in hid else hid
        rows.append((hid, group, group.split("|"), p, padj, hid in rejected))
    return pd.DataFrame(rows, columns=["hypothesis_id", "group_id", "genes",
                                       "p", "p_adjusted", "significant"])


# ----------------------------------------------------------------------
# genome layout
# ----------------------------------------------------------------------

def _chromosome_order(chroms: Sequence[str]) -> list[str]:
    """Order chromosomes by homoeologous number, then subgenome letter."""
    def key(name):
        digits = re.findall(r"\d+", name)
        letters = re.sub(r"(?i)^chr", "", name)
        letters = re.sub(r"\d+", "", letters)
        return (int(digits[-1]) if digits else 10 ** 6, letters.lower(), _natural_key(name))
    return sorted(set(chroms), key=key)


def _layout(positions: pd.DataFrame, gap_fraction: float = 0.01):
    order = _chromosome_order(positions["chrom"].astype(str))
    lengths = positions.groupby("chrom")["pos"].max().reindex(order).fillna(0.0)
    gap = float(lengths.sum()) * gap_fraction
    offsets, cursor = {}, 0.0
    for chrom in order:
        offsets[chrom] = cursor
        cursor += float(lengths[chrom]) + gap
    centres = {c: offsets[c] + float(lengths[c]) / 2 for c in order}
    return order, offsets, centres, cursor


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------

def plot_interaction_scan(primary: dict, positions: pd.DataFrame,
                          subgenomes: Sequence[str], out_dir: Path,
                          stem: str = "interaction_scan",
                          formats: Sequence[str] = DEFAULT_FORMATS) -> list[str]:
    plt, mpl = _import_mpl()
    groups = _groups_table(primary)
    pos = positions.set_index("gene_id")
    order, offsets, centres, total = _layout(positions)
    sub_of_chrom = positions.groupby("chrom")["subgenome"].agg(
        lambda s: s.mode().iat[0]).to_dict()
    palette = subgenome_palette(list(subgenomes))
    threshold = (primary.get("model_diagnostics") or {}).get(
        "bootstrap_fwer", {}).get("threshold")

    points = []
    for row in groups.itertuples():
        if row.p is None or not np.isfinite(row.p):
            continue
        for gene in row.genes:
            if gene in pos.index:
                record = pos.loc[gene]
                points.append((offsets[record["chrom"]] + record["pos"],
                               -np.log10(row.p), row.significant, row.group_id))
    frame = pd.DataFrame(points, columns=["x", "y", "significant", "group_id"])

    with mpl.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(7.2, 2.6))
        shade = {sub: c for sub, c in palette.items()}
        for i, chrom in enumerate(order):
            if i % 2 == 0:
                ax.axvspan(offsets[chrom], offsets[chrom] + (centres[chrom] - offsets[chrom]) * 2,
                           color="#F3F1EE", lw=0, zorder=0)
        background = frame[~frame["significant"]]
        ax.scatter(background["x"], background["y"], s=4, color=NEUTRAL,
                   lw=0, zorder=2, rasterized=True)
        hits = frame[frame["significant"]]
        for _, block in hits.groupby("group_id"):
            xs = np.sort(block["x"].to_numpy())
            ax.plot(xs, block["y"].to_numpy()[np.argsort(block["x"].to_numpy())],
                    color=SIGNAL, lw=0.7, alpha=0.7, zorder=3)
        ax.scatter(hits["x"], hits["y"], s=14, color=SIGNAL, lw=0, zorder=4)
        if threshold:
            ax.axhline(-np.log10(threshold), color=THRESHOLD, lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.set_xlim(0, total)
        ax.set_ylim(0, max(frame["y"].max() * 1.08, -np.log10(threshold) * 1.15 if threshold else 1))
        ax.set_xticks([centres[c] for c in order])
        ax.set_xticklabels([re.sub(r"(?i)^chr", "", c) for c in order], fontsize=6)
        ax.tick_params(axis="x", length=0, pad=6)
        strip = ax.get_xaxis_transform()
        for chrom in order:
            width = (centres[chrom] - offsets[chrom]) * 2
            ax.add_patch(mpl.patches.Rectangle(
                (offsets[chrom], -0.035), width, 0.025, transform=strip, clip_on=False,
                color=shade.get(sub_of_chrom.get(chrom), NEUTRAL_DARK), lw=0))
        ax.set_ylabel(r"group $-\log_{10} P$")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        fig.tight_layout()
        written = save_figure(fig, out_dir, stem, formats)
        plt.close(fig)
    return written


def _reported_groups(primary: dict) -> list[dict]:
    fwer = (primary.get("model_diagnostics") or {}).get("bootstrap_fwer") or {}
    sig = list(fwer.get("sig") or primary.get("sig") or [])
    chosen = sig if sig else list(primary.get("top") or [])[:3]
    return [g for g in chosen if g.get("edge_components")][:MAX_EVIDENCE_GROUPS]


def plot_interaction_evidence(primary: dict, out_dir: Path,
                              stem: str = "interaction_evidence",
                              formats: Sequence[str] = DEFAULT_FORMATS) -> list[str]:
    groups = _reported_groups(primary)
    if not groups:
        return []
    plt, mpl = _import_mpl()
    encodings = list(ENCODING_LABELS)
    n = len(groups)
    ncol = min(n, 4)
    nrow = int(np.ceil(n / ncol))
    values = [
        -np.log10(np.clip([[e["component_p"].get(k) if e["component_p"].get(k) is not None
                            else np.nan for e in g["edge_components"]]
                           for k in encodings], 1e-300, 1.0))
        for g in groups]
    vmax = max(np.nanmax(v) for v in values)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("signal", ["#FFFFFF", SIGNAL])
    with mpl.rc_context(_PUB_RC):
        fig = plt.figure(figsize=(1.75 * ncol + 0.6, 1.9 * nrow))
        grid = fig.add_gridspec(nrow * 2, ncol, height_ratios=[0.35, 1] * nrow,
                                hspace=0.15 * 2, wspace=0.55)
        image = None
        for i, (group, matrix) in enumerate(zip(groups, values, strict=True)):
            r, c = divmod(i, ncol)
            top = fig.add_subplot(grid[2 * r, c])
            ax = fig.add_subplot(grid[2 * r + 1, c], sharex=top)
            edges = [e["direction"] for e in group["edge_components"]]
            edge_p = {e["edge_id"]: e.get("p_omnib") for e in group.get("edge_localization", [])}
            bars = [-np.log10(max(edge_p.get(e["edge_id"]) or 1.0, 1e-300))
                    for e in group["edge_components"]]
            top.bar(range(len(edges)), bars, color=SIGNAL, width=0.6)
            top.set_ylim(0, max(bars) * 1.15 if max(bars) > 0 else 1)
            top.tick_params(labelbottom=False, labelsize=5.5)
            top.set_ylabel(r"edge $-\log_{10} P$", fontsize=5.5)
            for side in ("top", "right"):
                top.spines[side].set_visible(False)
            image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                              interpolation="nearest")
            ax.set_xticks(range(len(edges)))
            ax.set_xticklabels(edges, fontsize=6)
            ax.set_yticks(range(len(encodings)))
            ax.set_yticklabels([ENCODING_LABELS[k] for k in encodings] if c == 0 else [],
                               fontsize=6)
            ax.tick_params(length=0)
            for side in ax.spines.values():
                side.set_visible(False)
            genes = group.get("ordered_genes") or group.get("group_id", "").split("|")
            top.set_title("\n".join(genes), fontsize=5.5, loc="left")
        cax = fig.add_axes([0.93, 0.25, 0.012, 0.5])
        bar = fig.colorbar(image, cax=cax)
        bar.set_label(r"encoding $-\log_{10} P$", fontsize=6)
        bar.ax.tick_params(labelsize=5.5)
        written = save_figure(fig, out_dir, stem, formats)
        plt.close(fig)
    return written


def plot_interaction_calibration(primary: dict, out_dir: Path,
                                 stem: str = "interaction_calibration",
                                 formats: Sequence[str] = DEFAULT_FORMATS) -> list[str]:
    fwer = (primary.get("model_diagnostics") or {}).get("bootstrap_fwer") or {}
    minima = np.asarray([v for v in fwer.get("null_minima") or [] if v is not None], float)
    observed = np.asarray([v for v in fwer.get("observed_p") or [] if v is not None], float)
    if observed.size == 0:
        return []
    plt, mpl = _import_mpl()
    threshold = fwer.get("threshold")
    rejected = set(fwer.get("rejected_hypothesis_ids") or [])
    flags = np.asarray([hid in rejected for hid, p in zip(
        fwer.get("hypothesis_ids") or [], fwer.get("observed_p") or [], strict=True)
        if p is not None])
    with mpl.rc_context(_PUB_RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.0, 2.4),
                                 gridspec_kw={"width_ratios": [1.2, 1]})
        ax = axes[0]
        if minima.size:
            logs = -np.log10(np.clip(minima, 1e-300, 1.0))
            ax.hist(logs, bins=40, color=NEUTRAL, edgecolor="white", linewidth=0.3)
            ax.set_xlabel(r"$-\log_{10}$ family minimum P (bootstrap null)")
            ax.set_ylabel("replicates")
        if threshold:
            ax.axvline(-np.log10(threshold), color=THRESHOLD, lw=0.9, ls=(0, (4, 2)))
        ax.axvline(-np.log10(observed.min()), color=SIGNAL, lw=1.2)
        _offset_spines(ax, 4)

        ax = axes[1]
        order = np.argsort(observed)
        n = observed.size
        expected = -np.log10((np.arange(1, n + 1) - 0.5) / n)
        from scipy.stats import beta
        k = np.arange(1, n + 1)
        lo = -np.log10(beta.ppf(0.975, k, n - k + 1))
        hi = -np.log10(beta.ppf(0.025, k, n - k + 1))
        ax.fill_between(expected, lo, hi, color="#ECEAE7", lw=0)
        obs_sorted = -np.log10(np.clip(observed[order], 1e-300, 1.0))
        significant = flags[order] if flags.size else np.zeros(n, bool)
        ax.scatter(expected[~significant], obs_sorted[~significant], s=5, color=NEUTRAL_DARK,
                   lw=0, rasterized=True, zorder=2)
        ax.scatter(expected[significant], obs_sorted[significant], s=14, color=SIGNAL,
                   lw=0, zorder=3)
        ax.plot([0, expected.max()], [0, expected.max()], color="#8C8781", lw=0.6)
        if threshold:
            ax.axhline(-np.log10(threshold), color=THRESHOLD, lw=0.8, ls=(0, (4, 2)))
        ax.set_xlabel(r"expected $-\log_{10} P$")
        ax.set_ylabel(r"observed group $-\log_{10} P$")
        _offset_spines(ax, 4)
        fig.tight_layout()
        written = save_figure(fig, out_dir, stem, formats)
        plt.close(fig)
    return written


def plot_null_variance(primary: dict, out_dir: Path,
                       stem: str = "interaction_null_variance",
                       formats: Sequence[str] = DEFAULT_FORMATS) -> list[str]:
    profile = (primary.get("model_diagnostics") or {}).get("null_variance_profile")
    if not profile:
        return []
    plt, mpl = _import_mpl()
    weights = np.asarray(profile["weights"], float)
    with mpl.rc_context(_PUB_RC):
        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.3), sharey=True)
        for ax, key, label in zip(axes, ("pc1", "pc2"),
                                  ("genomic eigenvector 1", "genomic eigenvector 2"),
                                  strict=True):
            x = np.asarray(profile[key], float)
            ax.scatter(x, weights, s=5, color=SIGNAL, alpha=0.55, lw=0, rasterized=True)
            ax.axhline(1.0, color="#8C8781", lw=0.6, ls=(0, (2, 2)))
            ax.set_xlabel(label)
            _offset_spines(ax, 4)
        axes[0].set_ylabel("null variance weight")
        fig.tight_layout()
        written = save_figure(fig, out_dir, stem, formats)
        plt.close(fig)
    return written


# ----------------------------------------------------------------------
# orchestration
# ----------------------------------------------------------------------

def make_interact_figures(result_path: str | Path,
                          out_dir: str | Path | None = None,
                          config_path: str | Path | None = None,
                          formats: Sequence[str] = DEFAULT_FORMATS) -> list[str]:
    """Write every applicable figure for one ``interact_<trait>.json``."""
    result_path = Path(result_path)
    payload, primary = load_interact_result(result_path)
    out_dir = Path(out_dir) if out_dir else result_path.parent / "figures"
    config_path = config_path or (payload.get("provenance") or {}).get("config_path")
    written: list[str] = []
    if config_path and Path(config_path).is_file():
        try:
            positions = positions_from_config(config_path)
        except (KeyError, FileNotFoundError, ValueError) as exc:
            print(f"[interact] scan figure skipped: gene positions unavailable ({exc})")
        else:
            written += plot_interaction_scan(
                primary, positions, payload.get("subgenomes") or [], out_dir, formats=formats)
    written += plot_interaction_evidence(primary, out_dir, formats=formats)
    written += plot_interaction_calibration(primary, out_dir, formats=formats)
    written += plot_null_variance(primary, out_dir, formats=formats)
    if "png" in formats:
        index = write_figure_index(
            out_dir, f"HomoeoGWAS interaction results: {payload.get('trait', '')}", CAPTIONS)
        if index:
            written.append(index)
    return written
