"""Check a wheat Scan T run against the frozen fingerprint in expected_lead_results.yaml."""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
EXPECTED = Path(__file__).with_name("expected_lead_results.yaml")


def check(result_json: Path) -> int:
    r = json.loads(result_json.read_text())["results"]["INT"]
    exp = yaml.safe_load(EXPECTED.read_text())["wheat_watkins_scanT"]["days_to_emerg"]
    tol = float(exp["tolerance"]["relative"])
    bad = []

    def cmp(name, got, want, exact=False):
        ok = got == want if exact else abs(got / want - 1) <= tol
        if not ok:
            bad.append(f"{name}: got {got!r}, want {want!r}")

    for k, v in exp["families"].items():
        cmp(f"families.{k}.n_tests", r["families"][k]["n_tests"], v["n_tests"], True)
        cmp(f"families.{k}.per_test_alpha", r["families"][k]["per_test_alpha"],
            v["per_test_alpha"], True)
    for k in ("n_planned", "n_valid", "n_unestimable", "by_contrast"):
        cmp(f"estimability.{k}", r["estimability"][k], exp["estimability"][k], True)
    g, ge = r["group_omnibus"], exp["group_omnibus"]
    cmp("group_omnibus.min_p", g["min_p"], ge["min_p"])
    cmp("group_omnibus.min_p_adjusted_bonferroni", g["min_p_adjusted_bonferroni"],
        ge["min_p_adjusted_bonferroni"])
    for k in ("n_sig", "n_partial"):
        cmp(f"group_omnibus.{k}", g[k], ge[k], True)
    cmp("group_omnibus.top_triad", list(g["top"][0]["triad"]), ge["top_triad"], True)
    for tag, pe in exp["pairwise"].items():
        pw = r["pairwise"][tag]
        cmp(f"{tag}.min_p", pw["min_p"], pe["min_p"])
        cmp(f"{tag}.n_sig", pw["n_sig"], pe["n_sig"], True)
        cmp(f"{tag}.lambda_gc_obs", pw["lambda_gc_obs"], pe["lambda_gc_obs"])
        cmp(f"{tag}.acat", pw["acat"], pe["acat"])
        cmp(f"{tag}.top_triad", list(pw["top"][0]["triad"]), pe["top_triad"], True)
    cmp("triad_acat_omnibus", r["triad_acat_omnibus"], exp["triad_acat_omnibus"])
    for s in ("A", "B", "D", "e"):
        cmp(f"sigma_hat.{s}", r["sigma_hat"][s], exp["sigma_hat"][s])
    mc = exp["manuscript_cross_check"]
    cmp("chr1_BD vs manuscript", r["pairwise"]["BD"]["min_p"], mc["chr1_BD_paper_value"])
    cmp("chr5_AD vs manuscript", r["pairwise"]["AD"]["min_p"], mc["chr5_AD_paper_value"])

    if bad:
        print(f"FAIL ({len(bad)}):")
        for b in bad:
            print("  " + b)
        return 1
    print("OK: wheat Scan T matches the frozen fingerprint")
    return 0


if __name__ == "__main__":
    default = ROOT / "results/paper_leads/wheat_watkins_scanT/days_to_emerg/interact_days_to_emerg.json"
    sys.exit(check(Path(sys.argv[1]) if len(sys.argv) > 1 else default))
