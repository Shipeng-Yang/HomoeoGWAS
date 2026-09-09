"""Executable marker-QC and binding contracts for the v5 benchmark."""

import hashlib

import numpy as np
import pytest

import homoeogwas.interact as I
import homoeogwas.omnib_family as F
from homoeogwas.group_family import MasterGroupFamily
from homoeogwas.interact import SubgenomeData


def _mask_fixture():
    rng = np.random.default_rng(930)
    n = 40
    subdata = {}
    for sub in ("A", "D"):
        X = rng.integers(0, 3, size=(n, 6)).astype(float)
        subdata[sub] = SubgenomeData(
            X=X,
            gene_snp={"g0": np.arange(6)},
            samples=[f"s{i}" for i in range(n)],
            chunk=None,
        )
    family = MasterGroupFamily(("A", "D"), ("g",), (("g0", "g0"),))
    phenotype = rng.normal(size=n)
    masks = {
        "A": np.array([True, True, True, False, False, False]),
        "D": np.array([True, True, True, False, False, False]),
    }
    return subdata, family, phenotype, np.arange(n), masks


def _mask_sha256(mask):
    return hashlib.sha256(
        np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    ).hexdigest()


def test_retained_variant_mask_uses_inclusive_call_maf_and_mac_boundaries():
    n = 250
    X = np.zeros((n, 5), float)
    X[:5, 0] = 1.0
    X[224:, 1] = np.nan
    X[:5, 1] = 1.0
    X[225:, 2] = np.nan
    X[:5, 2] = 1.0
    X[:4, 3] = 1.0
    X[:5, 4] = 1.0

    mask, provenance = I.build_retained_variant_mask(
        X, call_rate_min=0.90, maf_min=0.01, mac_min=5,
    )

    np.testing.assert_array_equal(mask, [True, False, True, False, True])
    assert provenance["call_rate_boundary"] == "inclusive_greater_than_or_equal"
    assert provenance["maf_boundary"] == "inclusive_greater_than_or_equal"
    assert provenance["mac_boundary"] == "inclusive_greater_than_or_equal"
    assert provenance["n_variants_input"] == 5
    assert provenance["n_variants_retained"] == 3
    assert provenance["retained_variant_mask_sha256"] == _mask_sha256(mask)


def test_retained_variant_mask_rejects_non_hard_call_dosages():
    with pytest.raises(ValueError, match="hard-call A1 dosages"):
        I.build_retained_variant_mask(
            np.array([[0.0, 0.5], [1.0, 2.0]]),
            call_rate_min=0.90,
            maf_min=0.01,
            mac_min=1,
        )


def test_retained_variant_mask_chunked_sample_index_matches_literal_contract():
    X = np.array(
        [
            [1.0, 0.0, np.nan, 0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0, 0.0, 1.0, np.nan],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [2.0, 1.0, 2.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            [0.0, np.nan, np.nan, 0.0, 0.0, 0.0],
        ]
    )
    sample_idx = np.array([5, 0, 3, 1])

    mask, provenance = I.build_retained_variant_mask(
        X,
        sample_idx=sample_idx,
        column_chunk_size=2,
        call_rate_min=0.75,
        maf_min=0.125,
        mac_min=1,
    )

    expected = np.array([True, True, False, False, True, False])
    np.testing.assert_array_equal(mask, expected)
    assert provenance["n_samples"] == 4
    assert provenance["n_variants_input"] == 6
    assert provenance["n_variants_retained"] == 3
    assert provenance["retained_variant_mask_sha256"] == _mask_sha256(expected)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda masks: masks | {"A": np.ones(5, bool)}, "length"),
        (lambda masks: masks | {"A": np.ones(6, np.uint8)}, "boolean"),
        (lambda masks: {"A": masks["A"]}, "missing subgenomes"),
        (
            lambda masks: masks | {
                "A": {"mask": masks["A"], "sha256": "0" * 64}
            },
            "hash mismatch",
        ),
    ],
)
def test_formal_retained_variant_masks_fail_closed(mutator, message):
    subdata, family, phenotype, sample_idx, masks = _mask_fixture()
    with pytest.raises(ValueError, match=message):
        F.score_omnib_family(
            subdata,
            family,
            phenotype,
            sample_idx,
            feature_seed=17,
            retained_variant_masks=mutator(masks),
            bootstrap_B=0,
            n_jobs=1,
            grm_method="grm_from_X",
            min_snp=3,
        )


def test_supplied_mask_governs_grm_and_gene_features():
    subdata, family, phenotype, sample_idx, masks = _mask_fixture()
    scores, _ = F.score_omnib_family(
        subdata,
        family,
        phenotype,
        sample_idx,
        feature_seed=17,
        retained_variant_masks=masks,
        bootstrap_B=0,
        n_jobs=1,
        grm_method="grm_from_X",
        min_snp=3,
    )

    for sub in family.subgenomes:
        provenance = scores.grm_provenance[sub]
        assert provenance["filter_policy"] == "explicit_retained_variant_mask"
        assert provenance["n_variants_used"] == 3
        assert provenance["retained_variant_mask_sha256"] == _mask_sha256(
            masks[sub]
        )
        assert scores.feature_identity[(sub, "g0")][
            "retained_global_variant_indices"
        ] == [0, 1, 2]


def test_legacy_omitted_mask_is_labelled_maf_only():
    subdata, family, phenotype, sample_idx, _masks = _mask_fixture()
    scores, _ = F.score_omnib_family(
        subdata,
        family,
        phenotype,
        sample_idx,
        feature_seed=17,
        bootstrap_B=0,
        n_jobs=1,
        grm_method="grm_from_X",
        min_snp=3,
    )
    assert {
        provenance["filter_policy"]
        for provenance in scores.grm_provenance.values()
    } == {"legacy_maf_only"}


def test_benchmark_evidence_role_requires_explicit_seed_and_masks():
    subdata, family, phenotype, sample_idx, masks = _mask_fixture()
    with pytest.raises(ValueError, match="explicit feature_seed"):
        F.run_group_scan_omnib(
            subdata,
            family,
            phenotype,
            sample_idx,
            evidence_role="benchmark_qa",
            retained_variant_masks=masks,
            bootstrap_B=1,
            n_jobs=1,
            grm_method="grm_from_X",
            min_snp=3,
        )
    with pytest.raises(ValueError, match="retained_variant_masks"):
        F.run_group_scan_omnib(
            subdata,
            family,
            phenotype,
            sample_idx,
            evidence_role="benchmark_qa",
            feature_seed=17,
            bootstrap_B=1,
            n_jobs=1,
            grm_method="grm_from_X",
            min_snp=3,
        )


def test_bound_mask_record_is_retained_in_prepared_design_identity():
    subdata, family, phenotype, sample_idx, masks = _mask_fixture()
    records = {}
    sample_hash = F._array_identity(sample_idx)["sha256"]
    for sub, mask in masks.items():
        records[sub] = {
            "mask": mask,
            "sha256": _mask_sha256(mask),
            "panel_id": "REALG.TEST",
            "sample_context": "full",
            "subgenome": sub,
            "ordered_sample_index_sha256": sample_hash,
            "source_BIM_sha256": ("a" if sub == "A" else "d") * 64,
            "input_variant_count": 6,
            "thresholds": {"call_rate_min": 0.90, "maf_min": 0.01, "mac_min": 5},
            "retained_variant_count": 3,
        }

    scores, _ = F.prepare_omnib_design(
        subdata,
        family,
        phenotype,
        sample_idx,
        feature_seed=17,
        retained_variant_masks=records,
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        cap=150,
        n_pc=3,
        transform="INT",
    )

    provenance = scores.retained_variant_mask_identity
    assert provenance["A"]["panel_id"] == "REALG.TEST"
    assert provenance["A"]["thresholds"] == {
        "call_rate_min": 0.90,
        "maf_min": 0.01,
        "mac_min": 5,
    }
    assert provenance["D"]["source_BIM_sha256"] == "d" * 64
    assert "mask" not in provenance["A"]
    assert scores.prepared_design_identity["retained_variant_masks"]["A"] == {
        "input_variant_count": 6,
        "retained_variant_count": 3,
        "retained_variant_mask_encoding": "uint8_input_variant_order",
        "retained_variant_mask_sha256": _mask_sha256(masks["A"]),
    }


def test_declarative_mask_labels_do_not_change_executable_prepared_digest():
    subdata, family, phenotype, sample_idx, masks = _mask_fixture()
    records = {}
    for sub, mask in masks.items():
        records[sub] = {
            "mask": mask,
            "sha256": _mask_sha256(mask),
            "source": {"label": "cli", "nested": ["original"]},
        }
    first, _ = F.prepare_omnib_design(
        subdata,
        family,
        phenotype,
        sample_idx,
        feature_seed=17,
        retained_variant_masks=records,
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        cap=150,
        n_pc=3,
        transform="INT",
    )
    records["A"]["source"]["label"] = "benchmark"
    second, _ = F.prepare_omnib_design(
        subdata,
        family,
        phenotype,
        sample_idx,
        feature_seed=17,
        retained_variant_masks=records,
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        cap=150,
        n_pc=3,
        transform="INT",
    )

    assert first.prepared_design_sha256 == second.prepared_design_sha256
    assert first.retained_variant_mask_identity["A"]["source"]["label"] == "cli"


def test_cli_benchmark_mask_records_use_chunked_sample_index_and_bind_identity(
    tmp_path, monkeypatch,
):
    subdata, family, _phenotype, sample_idx, _masks = _mask_fixture()
    prefixes = {}
    for sub in family.subgenomes:
        prefix = tmp_path / f"panel-{sub}"
        (tmp_path / f"panel-{sub}.bim").write_text(
            "".join(
                f"{sub}\tv{index}\t0\t{index + 1}\tA\tC\n"
                for index in range(6)
            ),
            encoding="utf-8",
        )
        prefixes[sub] = str(prefix)
    config = {
        "subgenomes": list(family.subgenomes),
        "genotype": prefixes,
        "benchmark_identity": {
            "panel_id": "REALG.TEST",
            "sample_context": "full",
            "feature_seed": 17,
        },
    }

    original = I.build_retained_variant_mask
    calls = []

    def capture(X, **kwargs):
        calls.append((X, kwargs.copy()))
        return original(X, **kwargs)

    monkeypatch.setattr(I, "build_retained_variant_mask", capture)
    records = I._build_benchmark_mask_records(config, subdata, sample_idx)

    expected_sample_hash = F._array_identity(sample_idx)["sha256"]
    assert len(calls) == len(family.subgenomes)
    for (X, kwargs), sub in zip(calls, family.subgenomes, strict=True):
        assert X is subdata[sub].X
        np.testing.assert_array_equal(kwargs["sample_idx"], sample_idx)
        assert kwargs["column_chunk_size"] > 0
    for sub, record in records.items():
        assert record["panel_id"] == "REALG.TEST"
        assert record["subgenome"] == sub
        assert record["ordered_sample_index_sha256"] == expected_sample_hash
        assert record["thresholds"] == {
            "call_rate_min": 0.90,
            "maf_min": 0.01,
            "mac_min": 5,
        }
        assert len(record["source_BIM_sha256"]) == 64
        assert record["sha256"] == record["retained_variant_mask_sha256"]
