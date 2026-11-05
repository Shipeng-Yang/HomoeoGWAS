import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from homoeogwas.audit import audit_result
from homoeogwas.input_qc import duplicate_message, duplicated_genes, screen_phenotype
from homoeogwas.workflow import check_phenotype_inputs
from tests.test_audit import _canonical_group_omnib_payload, _write
from tests.test_prep import _fixture


def _values(n=300, outlier_sd=None, seed=7):
    y = np.random.default_rng(seed).normal(50.0, 2.0, n)
    if outlier_sd is not None:
        rest = np.delete(y, 0)
        y[0] = rest.mean() + outlier_sd * rest.std(ddof=1) * 1.05
    return y


def _ids(n=300):
    return [f"S{i:04d}" for i in range(n)]


def test_screen_phenotype_levels():
    ok = screen_phenotype(_ids(), _values())
    assert ok["status"] == "ok" and ok["flagged"] == []
    warn = screen_phenotype(_ids(), _values(outlier_sd=7))
    assert warn["status"] == "warn"
    assert warn["flagged"][0]["sample"] == "S0000"
    review = screen_phenotype(_ids(), _values(outlier_sd=15))
    assert review["status"] == "review"
    assert review["flagged"][0]["sample"] == "S0000"
    assert review["warn_sd"] == 5 and review["review_sd"] == 10
    assert review["max_abs_z"] > 10


def test_screen_phenotype_ignores_missing_and_constant():
    y = _values()
    y[5] = np.nan
    assert screen_phenotype(_ids(), y)["n"] == 299
    flat = screen_phenotype(_ids(10), np.ones(10))
    assert flat["status"] == "ok" and flat["max_abs_z"] is None


def test_check_phenotype_inputs_reads_whole_file(tmp_path):
    y = _values(outlier_sd=15)
    order = list(range(1, 300)) + [0]
    pheno = tmp_path / "pheno.tsv"
    pd.DataFrame({"sample": [_ids()[i] for i in order],
                  "trait": y[order]}).to_csv(pheno, sep="\t", index=False)
    out = check_phenotype_inputs(str(pheno), "sample", "trait")
    assert out["ok"] is True
    assert out["phenotype_qc"]["status"] == "review"
    assert out["phenotype_qc"]["flagged"][0]["sample"] == "S0000"
    assert "S0000" in out["advice"]


def test_duplicated_genes_per_subgenome():
    rows = [("gA1", "gD1"), ("gA2", "gD1"), ("gA3", "gD2"), ("gA3", "gD3")]
    dup = duplicated_genes(rows, ["A", "D"])
    assert dup == {"A": ["gA3"], "D": ["gD1"]}
    assert duplicated_genes([("a", "b"), ("c", "d")], ["A", "B"]) == {}


def _pair_cfg(tmp_path, rows, allow=None, waiver=None):
    from homoeogwas import prep
    gff, sgmap, bed_by_sub = _fixture(tmp_path)
    out = tmp_path / "prep"
    prep.build_snp_to_gene(str(gff), bed_by_sub, prep.load_subgenome_map(str(sgmap)),
                           out_dir=str(out))
    pairs = tmp_path / "pairs.tsv"
    pd.DataFrame(rows, columns=["gene_A", "gene_B"]).to_csv(pairs, sep="\t", index=False)
    phenotype = tmp_path / "phenotype.csv"
    pd.DataFrame({"sample": [f"s{i}" for i in range(12)],
                  "trait": np.arange(12, dtype=float)}).to_csv(phenotype, index=False)
    ic = {"mode": "pairwise", "subgenomes": ["A", "B"],
          "genotype": {s: bed_by_sub[s] for s in ("A", "B")},
          "snp_to_gene": {s: str(out / f"snp_to_gene_{s}.npz") for s in ("A", "B")},
          "pairs": str(pairs), "phenotype": str(phenotype),
          "sample_col": "sample", "trait": "trait",
          "burden": {"cap": 150, "min_snp": 1}}
    if allow is not None:
        ic["allow_duplicate_genes"] = allow
    if waiver is not None:
        ic["duplicate_gene_waiver"] = waiver(pairs) if callable(waiver) else waiver
    return {"interact": ic, "outputs": {"out_dir": str(tmp_path / "interact")}}


def test_preflight_refuses_duplicated_genes(tmp_path):
    from homoeogwas.interact import preflight_interact, validate_interact_config
    cfg = _pair_cfg(tmp_path, [("gA1", "gB1"), ("gA2", "gB1")])
    validate_interact_config(cfg)
    problems = preflight_interact(cfg)
    assert any("gB1" in p and "duplicated" in p for p in problems)


def test_duplicate_message_names_no_bypass():
    msg = duplicate_message({"B": ["gB1"]})
    assert "nothing was run" in msg
    assert "allow_duplicate_genes" not in msg and "waiver" not in msg and "true" not in msg


@pytest.mark.parametrize("value", [True, False, "yes"])
def test_validate_rejects_removed_allow_duplicate_genes(tmp_path, value):
    from homoeogwas.interact import validate_interact_config
    cfg = _pair_cfg(tmp_path, [("gA1", "gB1")], allow=value)
    with pytest.raises(SystemExit, match="allow_duplicate_genes was removed"):
        validate_interact_config(cfg)


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


GOOD_REASON = "paralog pair confirmed by synteny; analysed on purpose for a sensitivity check"


def test_waiver_with_matching_table_hash_allows_duplicates(tmp_path):
    from homoeogwas.interact import preflight_interact, validate_interact_config
    cfg = _pair_cfg(tmp_path, [("gA1", "gB1"), ("gA2", "gB1")],
                    waiver=lambda t: {"reason": GOOD_REASON, "approver": "data owner",
                                      "table_sha256": _sha(t)})
    validate_interact_config(cfg)
    assert preflight_interact(cfg) == []


def test_waiver_for_another_table_is_refused(tmp_path):
    from homoeogwas.interact import preflight_interact, validate_interact_config
    cfg = _pair_cfg(tmp_path, [("gA1", "gB1"), ("gA2", "gB1")],
                    waiver={"reason": GOOD_REASON, "approver": "data owner", "table_sha256": "0" * 64})
    validate_interact_config(cfg)
    problems = preflight_interact(cfg)
    assert any("does not match" in p for p in problems)
    assert any("gB1" in p for p in problems)


@pytest.mark.parametrize("waiver, match", [
    ({"reason": "too short", "approver": "x", "table_sha256": "0" * 64}, "reason"),
    ({"reason": GOOD_REASON, "approver": "", "table_sha256": "0" * 64}, "approver"),
    ({"reason": GOOD_REASON, "approver": "x", "table_sha256": "abc"}, "table_sha256"),
    ({"reason": GOOD_REASON, "approver": "x"}, "table_sha256"),
    ("yes", "mapping"),
])
def test_validate_rejects_incomplete_waiver(tmp_path, waiver, match):
    from homoeogwas.interact import validate_interact_config
    cfg = _pair_cfg(tmp_path, [("gA1", "gB1")], waiver=waiver)
    with pytest.raises(SystemExit, match=match):
        validate_interact_config(cfg)


def _fit_summary(qc):
    return {"trait": "height", "n_samples": 500,
            "reml": {"pve": {"A": 0.2, "B": 0.3, "D": 0.1, "e": 0.4},
                     "pve_ci": {"A": [0.1, 0.3]}},
            "lambda_gc": {"all": 1.0}, "acceptance_all_passed": True, "acceptance": [],
            "input_qc": {"phenotype": qc}}


def test_audit_fit_phenotype_flags(tmp_path):
    clean = audit_result(_write(tmp_path / "summary_a.json", _fit_summary(
        screen_phenotype(_ids(), _values()))))
    assert clean.status == "INTERNAL_RESULT_COMPLETE"
    warn = audit_result(_write(tmp_path / "summary_b.json", _fit_summary(
        screen_phenotype(_ids(), _values(outlier_sd=7)))))
    assert warn.status == "INTERNAL_RESULT_COMPLETE"
    assert {f.code: f.severity for f in warn.flags}["PHENOTYPE_EXTREME_VALUES"] == "info"
    review = audit_result(_write(tmp_path / "summary_c.json", _fit_summary(
        screen_phenotype(_ids(), _values(outlier_sd=15)))))
    assert review.status == "INTERNAL_RESULT_REVIEW_REQUIRED"
    assert {f.code: f.severity for f in review.flags}["PHENOTYPE_OUTLIER"] == "review"


def test_audit_interact_input_flags(tmp_path):
    payload = _canonical_group_omnib_payload()
    base = audit_result(_write(tmp_path / "interact_a.json", payload))
    assert base.status == "NO_FAMILYWISE_DISCOVERY"
    payload = json.loads(json.dumps(payload))
    payload["input_qc"] = {
        "phenotype": screen_phenotype(_ids(), _values(outlier_sd=15)),
        "group_table": {"waiver": {"reason": GOOD_REASON, "approver": "data owner",
                                   "table_sha256": "a" * 64},
                        "duplicated_genes": {"D": ["gD1"]}}}
    record = audit_result(_write(tmp_path / "interact_b.json", payload))
    codes = {f.code: f.severity for f in record.flags}
    assert codes["PHENOTYPE_OUTLIER"] == "review"
    assert codes["OVERRIDDEN_INPUT_CHECK"] == "review"
    msg = {f.code: f.message for f in record.flags}["OVERRIDDEN_INPUT_CHECK"]
    assert "data owner" in msg and GOOD_REASON in msg and "gD1" in msg
    assert record.status == "NO_FAMILYWISE_DISCOVERY_REVIEW_REQUIRED"


def test_duplicated_genes_ignores_missing_cells():
    rows = [("gA1", float("nan")), ("gA2", float("nan")), ("gA3", None), ("gA4", " ")]
    assert duplicated_genes(rows, ["A", "B"]) == {}


def test_check_phenotype_inputs_averages_repeated_rows(tmp_path):
    y = _values(outlier_sd=15)
    ids = _ids()
    pheno = tmp_path / "pheno.tsv"
    pd.DataFrame({"sample": ids + ["S0000"], "trait": list(y) + [y[1]]}).to_csv(
        pheno, sep="\t", index=False)
    qc = check_phenotype_inputs(str(pheno), "sample", "trait")["phenotype_qc"]
    assert qc["n"] == 300
    assert "scope" in qc


def test_audit_multitrait_phenotype_flags(tmp_path):
    payload = _canonical_group_omnib_payload()
    payload["input_qc"] = {"phenotype_by_trait": {
        "t1": screen_phenotype(_ids(), _values()),
        "t2": screen_phenotype(_ids(), _values(outlier_sd=15))}}
    record = audit_result(_write(tmp_path / "interact_m.json", payload))
    flags = [f for f in record.flags if f.code == "PHENOTYPE_OUTLIER"]
    assert len(flags) == 1 and flags[0].message.startswith("t2: ")


# --- sample join record, normalisable ID mismatch refusal, join-loss warning ---

def test_sample_join_refuses_ids_matching_only_after_normalisation():
    from homoeogwas.input_qc import sample_join
    geno = [f"{i:04d}" for i in range(1000, 1030)] + [f"{i:04d}" for i in range(10)]
    pheno = [str(i) for i in range(1000, 1030)] + [str(i) for i in range(10)]   # leading zeros lost
    j = sample_join(pheno, geno, [g for g in geno if g in set(pheno)])
    assert j["status"] == "refuse" and j["normalisable_pairs"] == 10
    assert j["n_phenotyped"] == 40 and j["n_genotyped"] == 40 and j["n_analysis"] == 30
    assert ("1", "0001") in [tuple(x) for x in j["pair_examples"]]


@pytest.mark.parametrize("variant", ["S0", " s0 ", "s0.0"])
def test_sample_join_normalises_case_space_and_float_suffix(variant):
    from homoeogwas.input_qc import sample_join
    geno = [f"s{i}" for i in range(40)]
    pheno = [variant] + [f"s{i}" for i in range(1, 40)]
    assert sample_join(pheno, geno, geno[1:])["status"] == "refuse"


def test_sample_join_warns_on_unexplained_loss_and_passes_clean():
    from homoeogwas.input_qc import sample_join, sample_join_message
    geno = [f"g{i}" for i in range(100)]
    pheno = geno[:90] + [f"x{i}" for i in range(10)]                     # 10% unmatched, no pairing
    j = sample_join(pheno, geno, geno[:90])
    assert j["status"] == "warn" and j["n_dropped"] == 10 and abs(j["dropped_fraction"] - 0.1) < 1e-12
    assert "10 of 100" in sample_join_message(j)
    ok = sample_join(geno[:98] + ["x1"], geno, geno[:98])              # 1% loss
    assert ok["status"] == "ok" and sample_join_message(ok) is None


def test_interact_refuses_case_mismatched_sample_ids(tmp_path, capsys):
    import argparse

    from homoeogwas.interact import cmd_interact
    cfg = _pair_cfg(tmp_path, [("gA1", "gB1"), ("gA2", "gB2")])
    ph = tmp_path / "phenotype.csv"
    pd.DataFrame({"sample": [f"S{i}" for i in range(12)],
                  "trait": np.arange(12, dtype=float)}).to_csv(ph, index=False)
    cfgp = tmp_path / "cfg.yaml"
    import yaml
    cfgp.write_text(yaml.safe_dump(cfg))
    rc = cmd_interact(argparse.Namespace(config=str(cfgp), out_dir=None, n_jobs=1))
    out = capsys.readouterr().out
    assert rc == 1 and "match only after normalisation" in out and "S0" in out


def test_audit_flags_sample_join_loss(tmp_path):
    from homoeogwas.input_qc import sample_join
    payload = _canonical_group_omnib_payload()
    geno = [f"g{i}" for i in range(100)]
    payload["input_qc"] = {"sample_join": sample_join(geno[:90] + [f"x{i}" for i in range(10)], geno, geno[:90])}
    record = audit_result(_write(tmp_path / "interact_j.json", payload))
    assert {f.code: f.severity for f in record.flags}["SAMPLE_JOIN_LOSS"] == "review"


# --- trait computed as the raw average of trials on different scales ---

def _trials(shift=(0.0, 8.0, 18.0), n=80, seed=3):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=n)
    df = pd.DataFrame({"sample": [f"S{i}" for i in range(n)]})
    for k, sh in enumerate(shift):
        df[f"trial{k + 1}"] = base + sh + rng.normal(scale=0.5, size=n)
    df["yield"] = df[[f"trial{k + 1}" for k in range(len(shift))]].mean(axis=1)
    df["height"] = rng.normal(size=n)
    return df


def test_raw_trial_average_flagged_when_trial_means_differ():
    from homoeogwas.input_qc import raw_trial_average
    r = raw_trial_average(_trials(), "sample", "yield")
    assert r is not None and r["columns"] == ["trial1", "trial2", "trial3"]
    assert r["mean_range_sd"] > 3


def test_raw_trial_average_not_flagged_for_similar_trials_or_other_traits():
    from homoeogwas.input_qc import raw_trial_average
    assert raw_trial_average(_trials(shift=(0.0, 0.3, 0.6)), "sample", "yield") is None
    assert raw_trial_average(_trials(), "sample", "height") is None


def test_check_phenotype_inputs_warns_raw_trial_average(tmp_path):
    p = tmp_path / "pheno.tsv"
    _trials().to_csv(p, sep="\t", index=False)
    out = check_phenotype_inputs(str(p), "sample", "yield")
    assert out["raw_trial_average"]["columns"] == ["trial1", "trial2", "trial3"]
    assert "raw average" in out["advice"]


def test_audit_flags_raw_trial_average(tmp_path):
    from homoeogwas.input_qc import raw_trial_average
    payload = _fit_summary(screen_phenotype(_ids(), _values()))
    payload["input_qc"]["raw_trial_average"] = raw_trial_average(_trials(), "sample", "yield")
    record = audit_result(_write(tmp_path / "summary_t.json", payload))
    assert {f.code: f.severity for f in record.flags}["PHENOTYPE_RAW_TRIAL_AVERAGE"] == "review"


# --- input checks: edge cases ---

def test_raw_trial_average_finds_trials_beyond_twelfth_column_quickly():
    import time

    from homoeogwas.input_qc import raw_trial_average
    df = _trials(n=5000)
    rng = np.random.default_rng(9)
    for j in range(20):
        df.insert(1, f"other{j}", rng.normal(size=len(df)))     # trials now sit after 20 other columns
    t = time.time()
    r = raw_trial_average(df, "sample", "yield")
    assert r is not None and r["columns"] == ["trial1", "trial2", "trial3"]
    assert time.time() - t < 2.0
    assert raw_trial_average(df, "sample", "height") is None


def test_raw_trial_average_constant_trials_and_audit_do_not_crash(tmp_path):
    from homoeogwas.input_qc import raw_trial_average
    df = pd.DataFrame({"sample": [f"S{i}" for i in range(40)], "t1": 1.0, "t2": 5.0})
    df["avg"] = df[["t1", "t2"]].mean(axis=1)
    r = raw_trial_average(df, "sample", "avg")
    assert r is None or np.isfinite(r["mean_range_sd"]) or r["mean_range_sd"] is None
    payload = _fit_summary(screen_phenotype(_ids(), _values()))
    payload["input_qc"]["raw_trial_average"] = {"columns": ["t1", "t2"], "mean_range_sd": None}
    record = audit_result(_write(tmp_path / "summary_c.json", payload))
    assert "PHENOTYPE_RAW_TRIAL_AVERAGE" in {f.code for f in record.flags}
