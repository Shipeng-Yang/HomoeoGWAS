"""Interaction figures drawn from a canonical group omniB result."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from homoeogwas import cli
from homoeogwas import interact_plots as P

pytest.importorskip("matplotlib")

SUBS = ["A", "B"]


def _inputs(tmp_path, n_genes=12):
    genotype, snp_to_gene, groups = {}, {}, []
    for sub in SUBS:
        rows, gene_ids, snp_idx = [], [], []
        for g in range(n_genes):
            chrom = f"{g % 3 + 1}{sub}"
            idx = []
            for k in range(4):
                idx.append(len(rows))
                rows.append(f"{chrom}\t{sub}_{g}_{k}\t0\t{1000 * (g + 1) + 10 * k}\tA\tG")
            gene_ids.append(f"{sub}g{g}")
            snp_idx.append(np.asarray(idx))
        prefix = tmp_path / f"geno_{sub}"
        (tmp_path / f"geno_{sub}.bim").write_text("\n".join(rows) + "\n")
        npz = tmp_path / f"snp_to_gene_{sub}.npz"
        holder = np.empty(len(snp_idx), dtype=object)
        holder[:] = snp_idx
        np.savez(npz, gene_ids=np.asarray(gene_ids, dtype=object), snp_idx=holder)
        genotype[sub], snp_to_gene[sub] = str(prefix), str(npz)
    config = tmp_path / "interact.yaml"
    config.write_text(yaml.safe_dump({"interact": {"genotype": genotype, "snp_to_gene": snp_to_gene}}))
    for g in range(n_genes):
        groups.append(f"Ag{g}|Bg{g}")
    return config, groups


def _result(groups, *, smooth=True, components=True):
    rng = np.random.default_rng(1)
    observed = rng.uniform(0.01, 1, len(groups))
    observed[0], observed[1] = 1e-7, 3e-6
    ids = [f"group:{g}" for g in groups]
    minima = sorted(rng.uniform(1e-6, 0.05, 99))
    threshold = minima[4]
    rejected = [h for h, p in zip(ids, observed, strict=True) if p < threshold]

    def record(i):
        a, b = groups[i].split("|")
        edge = f"AB:{a}:{b}"
        out = {"hypothesis_id": ids[i], "group_id": groups[i], "ordered_genes": [a, b],
               "edge_localization": [{"edge_id": edge, "direction": "AB", "p_omnib": observed[i]}]}
        if components:
            out["edge_components"] = [{"edge_id": edge, "direction": "AB", "component_p": {
                "minor_burden": observed[i] * 3, "pc1": 0.2, "kernel_hadamard": observed[i] * 10}}]
        return out
    sig = [record(i) for i in range(len(groups)) if ids[i] in rejected]
    diagnostics = {"bootstrap_fwer": {
        "hypothesis_ids": ids, "observed_p": list(observed),
        "adjusted_p": [min(1.0, p * 50) for p in observed], "threshold": threshold,
        "rejected_hypothesis_ids": rejected, "null_minima": minima, "sig": sig}}
    if smooth:
        n = 50
        diagnostics["null_variance_profile"] = {
            "weights": list(np.exp(rng.normal(0, 0.2, n)) / 1.0), "pc1": list(rng.normal(size=n)),
            "pc2": list(rng.normal(size=n)), "squared_residuals": list(rng.chisquare(1, n))}
    return {"tool": "homoeogwas", "command": "interact", "subgenomes": SUBS,
            "results": {"INT": {"model_diagnostics": diagnostics, "sig": sig,
                                "top": [record(i) for i in range(3)]}}}


def _write(tmp_path, payload, config):
    payload["provenance"] = {"config_path": str(config)}
    path = tmp_path / "interact_trait.json"
    path.write_text(json.dumps(payload))
    return path


def test_gene_positions_use_the_bim_and_snp_to_gene_map(tmp_path):
    config, _ = _inputs(tmp_path)
    positions = P.positions_from_config(config)
    row = positions.set_index("gene_id").loc["Bg4"]
    assert row["chrom"] == "2B" and row["pos"] == 5015.0 and row["subgenome"] == "B"


def test_all_figures_are_written(tmp_path):
    config, groups = _inputs(tmp_path)
    result = _write(tmp_path, _result(groups), config)
    written = P.make_interact_figures(result)
    stems = {p.rsplit("/", 1)[1].rsplit(".", 1)[0] for p in written}
    assert stems == {"interaction_scan", "interaction_evidence",
                     "interaction_calibration", "interaction_null_variance", "figures"}
    assert all(p.endswith((".png", ".pdf", ".svg", ".html")) for p in written)
    assert (tmp_path / "figures" / "interaction_scan.png").stat().st_size > 0
    page = (tmp_path / "figures" / "figures.html").read_text()
    assert page.count("<img ") == 4 and 'src="interaction_scan.png"' in page


def test_optional_figures_are_skipped_without_their_data(tmp_path):
    config, groups = _inputs(tmp_path)
    result = _write(tmp_path, _result(groups, smooth=False, components=False), config)
    stems = {p.rsplit("/", 1)[1].rsplit(".", 1)[0] for p in P.make_interact_figures(result)}
    assert stems == {"interaction_scan", "interaction_calibration", "figures"}


def test_plot_command_recognises_an_interaction_run(tmp_path, capsys):
    config, groups = _inputs(tmp_path)
    _write(tmp_path, _result(groups), config)
    assert cli.main(["plot", str(tmp_path), "--formats", "png"]) == 0
    assert (tmp_path / "figures" / "interaction_calibration.png").exists()
    assert not (tmp_path / "figures" / "interaction_scan.pdf").exists()
