from __future__ import annotations

import csv

import numpy as np
import pytest
from bm_native_qa_harness import materialize

from scripts.benchmarks.v201.track_omnib import build_synthetic_omnib_context


def test_materialization_gate_rejects_before_numerical_adapter_is_called() -> None:
    called = False

    def adapter() -> None:
        nonlocal called
        called = True

    with pytest.raises(
        RuntimeError,
        match="response_materialization_authorized=false",
    ):
        materialize.materialize_one(
            {"response_materialization_authorized": False},
            adapter,
        )

    assert called is False


def test_materialization_gate_rejects_bare_true_without_review_bindings() -> None:
    with pytest.raises(RuntimeError, match="materialization evidence is incomplete"):
        materialize.materialize_one(
            {"response_materialization_authorized": True},
            lambda: "must-not-run",
        )


def test_materialization_gate_accepts_only_complete_reviewed_authority() -> None:
    authority = {
        "response_materialization_authorized": True,
        "amendment_status": "accepted_for_response_materialization",
        "amendment_sha256": "a" * 64,
        "runner_review_verdict": "ACCEPT",
        "runner_review_sha256": "b" * 64,
        "prospective_design_hash": "c" * 64,
        "prospective_inventory_sha256": "d" * 64,
        "source_input_reverification_sha256": "e" * 64,
    }

    assert materialize.materialize_one(authority, lambda: "called") == "called"


def test_prepare_anchor_requires_finite_symmetric_positive_definite_vhat() -> None:
    context = build_synthetic_omnib_context(n=24, groups=2, copies=2, seed=7)

    prepared = materialize.prepare_anchor(
        context,
        design_hash="0" * 64,
        scenario_id="qa_test.synthetic.pc1",
        anchor_seed=11,
    )

    assert prepared.anchor.shape == (24,)
    assert np.isfinite(prepared.v_hat).all()
    assert np.array_equal(prepared.v_hat, prepared.v_hat.T)
    assert np.linalg.eigvalsh(prepared.v_hat).min() > 0.0
    assert np.allclose(prepared.root_v @ prepared.root_v.T, prepared.v_hat)


@pytest.mark.parametrize(
    "bad_covariance",
    (
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        np.array([[1.0, 0.0], [0.0, 0.0]]),
        np.array([[1.0, 0.25], [0.0, 1.0]]),
    ),
)
def test_bad_vhat_fails_before_root_helper_is_called(
    monkeypatch: pytest.MonkeyPatch,
    bad_covariance: np.ndarray,
) -> None:
    called = False

    def forbidden_root(_covariance: np.ndarray) -> np.ndarray:
        nonlocal called
        called = True
        raise AssertionError("root helper must not consume invalid V_hat")

    monkeypatch.setattr(materialize, "_root_from_covariance", forbidden_root)

    with pytest.raises(RuntimeError, match="fitted V_hat"):
        materialize.validated_covariance_root(bad_covariance)

    assert called is False


def test_generate_gaussian_and_mixed_sign_responses_from_fitted_vhat() -> None:
    context = build_synthetic_omnib_context(n=24, groups=2, copies=2, seed=7)
    prepared = materialize.prepare_anchor(
        context,
        design_hash="0" * 64,
        scenario_id="qa_test.synthetic.pc1",
        anchor_seed=11,
    )

    null = materialize.generate_response(
        prepared,
        response_id="qa_test.synthetic.gaussian_null",
        truth_id="gaussian_null",
        response_seed=12,
    )
    mixed = materialize.generate_response(
        prepared,
        response_id="qa_test.synthetic.mixed_sign_diagnostic_pve0p03",
        truth_id="mixed_sign_diagnostic_pve0p03",
        response_seed=13,
    )

    assert null.signal is None
    assert "signal" not in null.metadata
    assert null.metadata["generator_scale_interaction_pve"] == 0.0
    assert mixed.metadata["causal_group_id"] == "group_0"
    assert mixed.metadata["causal_pair_edge"] == ["A", "B"]
    assert mixed.metadata["generator_scale_interaction_pve"] == pytest.approx(
        0.03, abs=1e-12
    )
    diagnostic = mixed.metadata["post_int_interaction_diagnostic"]
    assert diagnostic["status"] == "descriptive_noninferential_only"
    assert "exact_pve" not in diagnostic
    assert 0.0 <= diagnostic["squared_correlation"] <= 1.0


def test_write_response_round_trips_little_endian_npy_and_17_digit_tsv(
    tmp_path,
) -> None:
    context = build_synthetic_omnib_context(n=24, groups=2, copies=2, seed=7)
    prepared = materialize.prepare_anchor(
        context,
        design_hash="0" * 64,
        scenario_id="qa_test.synthetic.pc1",
        anchor_seed=11,
    )
    response = materialize.generate_response(
        prepared,
        response_id="qa_test.synthetic.gaussian_null",
        truth_id="gaussian_null",
        response_seed=12,
    )
    sample_ids = tuple(f"sample_{index}" for index in range(24))
    npy_path = tmp_path / "response.npy"
    tsv_path = tmp_path / "phenotype.tsv"

    record = materialize.write_roundtrip_response(
        response,
        sample_ids=sample_ids,
        npy_path=npy_path,
        tsv_path=tsv_path,
    )

    loaded_npy = np.load(npy_path, allow_pickle=False)
    assert loaded_npy.dtype.str == "<f8"
    assert np.array_equal(
        loaded_npy.view(np.uint64), response.values.view(np.uint64)
    )
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert [row["sample_id"] for row in rows] == list(sample_ids)
    loaded_tsv = np.asarray([float(row["qa_trait"]) for row in rows])
    assert np.array_equal(loaded_tsv.view(np.uint64), response.values.view(np.uint64))
    assert len(record["npy_sha256"]) == 64
    assert len(record["tsv_sha256"]) == 64


def test_write_response_rejects_any_existing_target_before_writing(tmp_path) -> None:
    existing_tsv = tmp_path / "phenotype.tsv"
    existing_tsv.write_text("reserved\n", encoding="utf-8")
    response = materialize.MaterializedResponse(
        response_id="qa_test.only",
        truth_id="gaussian_null",
        response_seed=1,
        values=np.array([0.25, -0.25]),
        post_int_values=np.array([1.0, -1.0]),
        signal=None,
        metadata={},
    )
    npy_path = tmp_path / "response.npy"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        materialize.write_roundtrip_response(
            response,
            sample_ids=("a", "b"),
            npy_path=npy_path,
            tsv_path=existing_tsv,
        )

    assert not npy_path.exists()
