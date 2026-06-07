#!/usr/bin/env python3
"""OAT Step 5: morning-review roll-up. Reads deploy_oat_{flank,body}.json + attrition/ETL
summaries and emits a single oat_master_summary.json + console digest: manifest, attrition,
genome-wide hit table (INT flank primary) with raw/body concordance, top candidates per
trait, perm calibration table (lambda_perm), and an honest positioning verdict."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("/mnt/7302share/fast_ysp/U7_GWAS/results/phase7/bio_oat")


def _load(p):
    fp = OUT / p
    return json.loads(fp.read_text()) if fp.exists() else None


def _pair_p(dep, trait, transform, sub_pair, pair):
    """Look up the analytic p for a specific (sub_pair, gene_x, gene_y) in a deploy json."""
    if not dep or trait not in dep["results"]:
        return None
    r = dep["results"][trait]
    key = "INT_primary" if transform == "INT" else "raw_sensitivity"
    if key not in r:
        return None
    fam = r[key]["per_family"].get(sub_pair, {})
    for h in fam.get("top", []):
        if list(h["pair"]) == list(pair):
            return h["p"]
    return None


def main():
    etl = _load("etl_summary.json")
    attr = _load("triad_attrition.json")
    flank = _load("deploy_oat_flank.json")
    body = _load("deploy_oat_body.json")

    summary = dict(species="Avena_sativa_AACCDD", panel="OLD_rahman2025_737",
                   manifest=dict(etl=etl, triad_attrition=attr,
                                 callable_pairs_flank=flank["callable_pairs"] if flank else None,
                                 callable_pairs_body=body["callable_pairs"] if body else None,
                                 G_total_flank=flank["G_total"] if flank else None,
                                 alpha_genomewide_flank=flank["alpha_genomewide"] if flank else None,
                                 n_traits=flank["n_traits"] if flank else None))

    # ---- genome-wide hit table (INT flank primary) with concordance ----
    hits = []
    cand = []  # top candidate per trait (min analytic INT-flank p across families)
    cal_rows = []
    if flank:
        for trait, r in flank["results"].items():
            if "INT_primary" not in r:
                continue
            ip = r["INT_primary"]
            # GW hits
            for h in ip.get("hits_gw", []):
                sp, pair, p = h["sub_pair"], h["pair"], h["p"]
                hits.append(dict(trait=trait, sub_pair=sp, pair=pair, p_INT_flank=p,
                                 p_raw_flank=_pair_p(flank, trait, "raw", sp, pair),
                                 p_INT_body=_pair_p(body, trait, "INT", sp, pair),
                                 p_raw_body=_pair_p(body, trait, "raw", sp, pair)))
            # top candidate (min p over families' top entries)
            best = None
            for sp, fam in ip["per_family"].items():
                for h in fam.get("top", []):
                    if best is None or h["p"] < best["p"]:
                        best = dict(sub_pair=sp, pair=h["pair"], p=h["p"])
            if best:
                cand.append(dict(trait=trait, **best))
            # calibration rows
            cal = ip.get("calibration", {})
            for sp, c in cal.items():
                cal_rows.append(dict(trait=trait, sub_pair=sp,
                                     lambda_perm=c.get("lambda_perm_median"),
                                     acat_emp=c.get("acat_emp")))

    cand.sort(key=lambda x: x["p"])
    summary["genome_wide_hits_INT_flank"] = hits
    summary["top_candidates"] = cand[:15]
    lam = [c["lambda_perm"] for c in cal_rows if c["lambda_perm"] is not None]
    summary["calibration"] = dict(
        n_perm_families=len(cal_rows),
        lambda_perm_median=float(np.median(lam)) if lam else None,
        lambda_perm_iqr=[float(np.percentile(lam, 25)), float(np.percentile(lam, 75))] if lam else None,
        rows=cal_rows)

    # ---- verdict ----
    G = flank["G_total"] if flank else 0
    n_hit = len(hits)
    if n_hit > 0:
        verdict = (f"OAT POSITIVE candidate(s): {n_hit} INT-flank genome-wide hit(s) "
                   f"(alpha={flank['alpha_genomewide']:.2e}). Check raw/body concordance + "
                   f"calibration before any claim; needs conditional sanity (VIF) + single-gene "
                   f"marginal check (run follow-up). Framework extends to 3rd allopolyploid AACCDD.")
    else:
        verdict = (f"OAT NULL under sparse-GBS pairwise scan (G_total={G} callable pairs, "
                   f"alpha={flank['alpha_genomewide'] if flank else float('nan'):.2e}). "
                   f"Honest positioning: UNDERPOWERED exploratory scan (GBS coverage ~4-6%/subgenome, "
                   f"strict 3-way triads ~0 callable), NOT 'oat genome-wide null'. If lambda_perm ~1, "
                   f"value = calibration holds on a 3rd allopolyploid (4th species) + 2nd hexaploid "
                   f"under de-novo triads; with rapeseed forms boundary set (dense-no-pheno vs "
                   f"sparse-pheno-limited-callable).")
    summary["verdict"] = verdict

    (OUT / "oat_master_summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # console digest
    print("=" * 70)
    print("OAT MASTER SUMMARY")
    print("=" * 70)
    if flank:
        print(f"callable pairs (flank): {flank['callable_pairs']} G_total={G}")
        print(f"genome-wide alpha (INT flank): {flank['alpha_genomewide']:.2e} "
              f"= 0.05/({G}x{flank['n_traits']})")
    print(f"strict 3-way triads (reported only): {attr['strict_triads_all'] if attr else '?'} "
          f"(callable ~0 under GBS -> pairwise pivot)")
    print(f"\nGENOME-WIDE HITS (INT flank primary): {n_hit}")
    for h in hits:
        print(f"  {h['trait']} {h['sub_pair']} {h['pair']} "
              f"p_INT_flank={h['p_INT_flank']:.2e} p_raw_flank={h['p_raw_flank']} "
              f"p_INT_body={h['p_INT_body']}")
    print(f"\nTOP 8 CANDIDATES (min INT-flank p):")
    for c in cand[:8]:
        print(f"  {c['trait']:10s} {c['sub_pair']} p={c['p']:.2e} {c['pair']}")
    print(f"\nCALIBRATION lambda_perm: median={summary['calibration']['lambda_perm_median']} "
          f"IQR={summary['calibration']['lambda_perm_iqr']}")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {OUT/'oat_master_summary.json'}")


if __name__ == "__main__":
    main()
