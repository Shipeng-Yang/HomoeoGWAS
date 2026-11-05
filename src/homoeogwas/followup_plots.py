"""Figures written by ``homoeogwas follow-up`` for the significant groups.

* ``followup_allele_combinations`` — for the driving edge of each group, the
  mean phenotype of the four combinations of PC1-defined genotype classes of the
  two copies, relative to the class with both copies low, with the additive
  expectation. Raw means: descriptive, not adjusted for relatedness.
* ``followup_robustness`` — the nominal P of each group in every
  accession-deletion fold (and environment-deletion reanalysis when run), with
  the full-data P.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .interact_plots import NEUTRAL, NEUTRAL_DARK, SIGNAL, THRESHOLD
from .plots import _PUB_RC, DEFAULT_FORMATS, _import_mpl, _offset_spines, save_figure

COMBINATIONS = (("low", "low"), ("high", "low"), ("low", "high"), ("high", "high"))
COMBINATION_LABELS = ("–/–", "+/–", "–/+", "+/+")
MIN_CLASS_N = 10


def two_means_split(values: np.ndarray) -> np.ndarray:
    """``True`` for values above the cut that minimises the within-class sum of
    squares of a one-dimensional two-class split."""
    x = np.asarray(values, float)
    order = np.sort(x)
    n = order.size
    csum, csum2 = np.cumsum(order), np.cumsum(order ** 2)
    best, cut = np.inf, None
    for k in range(1, n):
        if order[k] <= order[k - 1]:
            continue
        left = csum2[k - 1] - csum[k - 1] ** 2 / k
        right = (csum2[-1] - csum2[k - 1]) - (csum[-1] - csum[k - 1]) ** 2 / (n - k)
        if left + right < best:
            best, cut = left + right, (order[k - 1] + order[k]) / 2
    if cut is None:
        return np.zeros(n, bool)
    return x > cut


def combination_table(pc_x: np.ndarray, pc_y: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Mean phenotype, 95% t interval and n for the 2 × 2 genotype classes."""
    high_x, high_y = two_means_split(pc_x), two_means_split(pc_y)
    rows = []
    for (cx, cy), label in zip(COMBINATIONS, COMBINATION_LABELS, strict=True):
        mask = (high_x == (cx == "high")) & (high_y == (cy == "high"))
        values = np.asarray(y, float)[mask]
        n = int(values.size)
        mean = float(values.mean()) if n else np.nan
        half = (float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))
                if n > 1 else np.nan)
        rows.append((label, n, mean, mean - half, mean + half))
    frame = pd.DataFrame(rows, columns=["combination", "n", "mean", "low", "high"])
    base = frame["mean"].iat[0]
    for column in ("mean", "low", "high"):
        frame[f"{column}_shift"] = frame[column] - base
    return frame


def _driving_copies(edge: str) -> tuple[str, str, str, str] | None:
    parts = str(edge).split(":")
    if len(parts) != 3 or len(parts[0]) != 2:
        return None
    return parts[0][0], parts[1], parts[0][1], parts[2]


def plot_allele_combinations(prepared, out_dir: Path,
                             stem: str = "followup_allele_combinations",
                             formats: Sequence[str] = DEFAULT_FORMATS) -> list[str]:
    hits = prepared.inputs.formal_hits
    panels = []
    for row in hits.itertuples():
        copies = _driving_copies(getattr(row, "driving_edge", None))
        if copies is None:
            continue
        sub_x, gene_x, sub_y, gene_y = copies
        fx = prepared.scores.feature_cache.get((sub_x, gene_x))
        fy = prepared.scores.feature_cache.get((sub_y, gene_y))
        if fx is None or fy is None:
            continue
        pc_x = np.asarray(fx[1], float).reshape(-1)
        pc_y = np.asarray(fy[1], float).reshape(-1)
        table = combination_table(pc_x, pc_y, prepared.y_raw)
        if table["n"].iat[0] < 2:
            continue
        panels.append((f"{sub_x}{sub_y}", gene_x, gene_y, table))
    if not panels:
        return []
    plt, mpl = _import_mpl()
    ncol = min(len(panels), 4)
    nrow = int(np.ceil(len(panels) / ncol))
    with mpl.rc_context(_PUB_RC):
        fig, axes = plt.subplots(nrow, ncol, figsize=(1.9 * ncol + 0.4, 2.2 * nrow),
                                 squeeze=False)
        for ax in axes.flat[len(panels):]:
            ax.set_visible(False)
        for ax, (edge, gene_x, gene_y, table) in zip(axes.flat, panels, strict=False):
            colours = [NEUTRAL] * 3 + [SIGNAL]
            small = (table["n"] < MIN_CLASS_N).to_numpy()
            bars = ax.bar(range(4), table["mean_shift"].fillna(0.0), color=colours, width=0.68)
            for bar, few in zip(bars, small, strict=True):
                if few:
                    bar.set_alpha(0.35)
                    bar.set_hatch("////")
            ax.vlines(range(4), table["low_shift"], table["high_shift"],
                      color=NEUTRAL_DARK, lw=0.8)
            expectation = table["mean_shift"].iat[1] + table["mean_shift"].iat[2]
            ax.hlines(expectation, 2.62, 3.38, color="#222222", lw=0.8, ls=(0, (3, 2)))
            ax.axhline(0, color="#8C8781", lw=0.5)
            ax.set_xticks(range(4))
            ax.set_xticklabels([f"{label}\nn={n}" for label, n in
                                zip(COMBINATION_LABELS, table["n"], strict=True)], fontsize=6)
            ax.set_title(f"{gene_x}\n{gene_y}", fontsize=5.5, loc="left")
            ax.set_xlabel(f"{edge[0]} / {edge[1]} PC1 class", fontsize=6.5)
            _offset_spines(ax, 3)
        for r in range(nrow):
            axes[r, 0].set_ylabel("phenotype shift from –/–")
        fig.tight_layout()
        written = save_figure(fig, out_dir, stem, formats)
        plt.close(fig)
    return written


def plot_robustness(stability: pd.DataFrame, material: pd.DataFrame,
                    environment: pd.DataFrame | None, out_dir: Path,
                    stem: str = "followup_robustness",
                    formats: Sequence[str] = DEFAULT_FORMATS) -> list[str]:
    if stability is None or stability.empty or material is None or material.empty:
        return []
    plt, mpl = _import_mpl()
    units = stability.sort_values("rank").reset_index(drop=True)
    gene_cols = [c for c in units.columns if c.startswith("gene_")]
    if gene_cols:
        names = [" × ".join(str(units.at[i, c]) for c in gene_cols) for i in range(len(units))]
    else:
        names = [str(h).split(":", 1)[-1].replace("|", " × ") for h in units["hypothesis_id"]]
    rng = np.random.default_rng(0)
    with mpl.rc_context(_PUB_RC):
        fig, ax = plt.subplots(figsize=(4.4, 0.42 * len(units) + 0.9))
        xmax = 0.0
        for i, unit in units.iterrows():
            y = len(units) - 1 - i
            folds = material.loc[material["rank"] == unit["rank"], "p_interaction"].dropna()
            logs = -np.log10(np.clip(folds.to_numpy(float), 1e-300, 1.0))
            ax.scatter(logs, y + rng.uniform(-0.13, 0.13, logs.size), s=6,
                       color=NEUTRAL, lw=0, zorder=2)
            n_total, n_support = logs.size, int((folds < 0.05).sum())
            if environment is not None and not environment.empty:
                env = environment.loc[environment["rank"] == unit["rank"], "p_interaction"].dropna()
                env_logs = -np.log10(np.clip(env.to_numpy(float), 1e-300, 1.0))
                ax.scatter(env_logs, np.full(env_logs.size, y + 0.25), marker="^", s=12,
                           facecolor="none", edgecolor=NEUTRAL_DARK, lw=0.6, zorder=2)
                n_total += env_logs.size
                n_support += int((env < 0.05).sum())
                logs = np.concatenate([logs, env_logs])
            full = -np.log10(max(float(unit["formal_p"]), 1e-300))
            ax.scatter([full], [y], marker="D", s=26, color=SIGNAL, edgecolor="white",
                       lw=0.5, zorder=3)
            xmax = max(xmax, full, logs.max() if logs.size else 0)
            ax.text(1.02, y, f"{n_support}/{n_total}", transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=6.5)
        ax.axvline(-np.log10(0.05), color=THRESHOLD, lw=0.8, ls=(0, (4, 2)))
        ax.set_yticks(range(len(units)))
        ax.set_yticklabels(list(reversed(names)), fontsize=5.5)
        ax.set_xlim(0, xmax * 1.08)
        ax.set_xlabel(r"$-\log_{10} P$ (reanalysis)")
        _offset_spines(ax, 4)
        fig.tight_layout()
        written = save_figure(fig, out_dir, stem, formats)
        plt.close(fig)
    return written
