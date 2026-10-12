import json

import numpy as np
import pandas as pd
import pytest

from homoeogwas.audit import audit_result
from homoeogwas.input_qc import duplicated_genes, screen_phenotype
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


def _pair_cfg(tmp_path, rows, allow=None):
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
    return {"interact": ic, "outputs": {"out_dir": str(tmp_path / "interact")}}


def test_preflight_refuses_duplicated_genes(tmp_path):
    from homoeogwas.interact import preflight_interact, validate_interact_config
    cfg = _pair_cfg(tmp_path, [("gA1", "gB1"), ("gA2", "gB1")])
    validate_interact_config(cfg)
    problems = preflight_interact(cfg)
    assert any("gB1" in p and "duplicated" in p for p in problems)


def test_preflight_allows_declared_duplicates(tmp_path):
    from homoeogwas.interact import preflight_interact, validate_interact_config
    cfg = _pair_cfg(tmp_path, [("gA1", "gB1"), ("gA2", "gB1")], allow=True)
    validate_interact_config(cfg)
    assert preflight_interact(cfg) == []


def test_validate_rejects_non_bool_allow_duplicate_genes(tmp_path):
    from homoeogwas.interact import validate_interact_config
    cfg = _pair_cfg(tmp_path, [("gA1", "gB1")], allow="yes")
    with pytest.raises(SystemExit, match="allow_duplicate_genes"):
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
        "group_table": {"allow_duplicate_genes": True,
                        "duplicated_genes": {"D": ["gD1"]}}}
    record = audit_result(_write(tmp_path / "interact_b.json", payload))
    codes = {f.code: f.severity for f in record.flags}
    assert codes["PHENOTYPE_OUTLIER"] == "review"
    assert codes["DUPLICATED_GROUP_GENES"] == "review"
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
