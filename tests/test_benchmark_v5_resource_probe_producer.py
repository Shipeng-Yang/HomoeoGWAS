from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from homoeogwas.group_family import MasterGroupFamily, expand_pair_edges
from homoeogwas.omnib_family import OmniBFamilyScores
from scripts.benchmarks.v201 import resource_probe
from scripts.benchmarks.v201.contracts import canonical_json, sha256_payload


def _prepared_fixture():
    rng = np.random.default_rng(9271)
    sample_count = 40
    group_ids = tuple(f"g{index:02d}" for index in range(80))
    family = MasterGroupFamily(
        ("A", "D"),
        group_ids,
        tuple(("geneA", "geneD") for _ in group_ids),
    )
    expanded = expand_pair_edges(family)
    blocks = {
        ("A", "geneA"): rng.binomial(2, 0.31, size=(sample_count, 1)).astype(float),
        ("D", "geneD"): rng.binomial(2, 0.39, size=(sample_count, 1)).astype(float),
    }
    scores = OmniBFamilyScores(
        edge_p=np.empty((len(expanded.edges), 0)),
        group_p=np.empty((len(group_ids), 0)),
        edge_components_obs=np.empty((len(expanded.edges), 3)),
        edge_estimable=np.ones(len(expanded.edges), dtype=bool),
        group_estimable=np.ones(len(group_ids), dtype=bool),
        W=np.eye(sample_count),
        y=np.zeros(sample_count),
        covariance_components={"e": 1.0},
        null_design=np.ones((sample_count, 1)),
        gated_snp={
            ("A", "geneA"): np.asarray((7,), dtype=int),
            ("D", "geneD"): np.asarray((9,), dtype=int),
        },
        prepared_design_sha256="c" * 64,
    )
    context = SimpleNamespace(
        panel_id="REALG.CGVD1245",
        sample_context="full",
        family=family,
    )
    return (
        SimpleNamespace(
            context=context,
            scores=scores,
            expanded=expanded,
            gene_blocks=blocks,
        ),
        np.ascontiguousarray(rng.normal(size=(sample_count, 20)), dtype=np.float64),
    )


def _array_sha256(array):
    serialized = (
        str(array.dtype).encode("ascii")
        + repr(array.shape).encode("ascii")
        + array.tobytes(order="C")
    )
    return hashlib.sha256(serialized).hexdigest()


def _authorization(response_bank, response_ids, *, enabled: bool = True):
    bank_identity = {
        "schema": "homoeogwas-snpxsnp-response-bank-v1",
        "dtype": "float64",
        "shape": list(response_bank.shape),
        "ordered_response_ids": list(response_ids),
        "array_sha256": _array_sha256(response_bank),
    }
    payload = {
        "schema": "homoeogwas-snpxsnp-resource-probe-authorization-v1",
        "authorization_id": "unit-test-only",
        "implementation_commit": "1" * 40,
        "matched_comparator_contract_sha256": "2" * 64,
        "panel_context": {
            "panel_id": "REALG.CGVD1245",
            "design_hash": "a" * 64,
            "context_fingerprint": "b" * 64,
            "prepared_design_sha256": "c" * 64,
            "response_bank_sha256": sha256_payload(bank_identity),
            "response_ids_sha256": sha256_payload({
                "ordered_response_ids": list(response_ids),
            }),
        },
        "response_widths": [1, 5, 20],
        "resource_probe_authorized": enabled,
        "formal_execution_authorized": False,
    }
    return payload, sha256_payload(payload)


def _produce(monkeypatch, width: int = 1):
    prepared, responses = _prepared_fixture()
    monkeypatch.setattr(
        resource_probe, "_context_fingerprint", lambda _context: "b" * 64
    )
    response_ids = [f"response-{index:02d}" for index in range(20)]
    authorization, authorization_sha256 = _authorization(responses, response_ids)
    return resource_probe.produce_snpxsnp_resource_probe(
        prepared,
        responses,
        response_ids,
        response_width=width,
        design_hash="a" * 64,
        context_fingerprint="b" * 64,
        authorization_payload=authorization,
        authorization_sha256=authorization_sha256,
        implementation_commit="1" * 40,
        matched_comparator_contract_sha256="2" * 64,
    )


def _witness():
    # Rebuild the independent caller inputs; never trust fields in the artifact.
    _prepared, response_bank = _prepared_fixture()
    response_ids = [f"response-{index:02d}" for index in range(20)]
    authorization, authorization_sha256 = _authorization(response_bank, response_ids)
    return {
        "authorization_payload": authorization,
        "authorization_sha256": authorization_sha256,
        "response_bank": response_bank,
        "response_ids": response_ids,
    }


def _refresh_artifact_hashes(artifact):
    evidence = artifact["score_evidence"]
    record = artifact["record"]
    record["response_bank_sha256"] = sha256_payload(
        evidence["response_bank_identity"]
    )
    evidence["response_prefix_identity"]["parent_response_bank_sha256"] = record[
        "response_bank_sha256"
    ]
    record["response_prefix_sha256"] = sha256_payload(
        evidence["response_prefix_identity"]
    )
    encoded = canonical_json(evidence).encode("utf-8")
    record["score_evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    record["output_bytes"] = len(encoded)


def test_artifact_validation_requires_external_witness(monkeypatch):
    artifact = _produce(monkeypatch)
    with pytest.raises((TypeError, ValueError)):
        resource_probe.validate_snpxsnp_resource_probe_artifact(artifact)


def test_artifact_audit_requires_external_witness(monkeypatch):
    from scripts.benchmarks.v201.audit import (
        BenchmarkAuditError,
        _audit_comparator_probe_artifact,
    )

    artifact = _produce(monkeypatch)
    with pytest.raises((TypeError, BenchmarkAuditError)):
        _audit_comparator_probe_artifact(artifact)


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing_field", "extra_field", "bool_count", "key_type"],
)
def test_artifact_validator_rejects_malformed_raw_input_bindings(monkeypatch, mutation):
    artifact = _produce(monkeypatch)
    raw = artifact["score_evidence"]["raw_score_evidence"]
    bindings = raw["input_block_bindings"]
    if mutation == "duplicate":
        bindings.append(copy.deepcopy(bindings[0]))
    elif mutation == "missing_field":
        del bindings[0]["gene_id"]
    elif mutation == "extra_field":
        bindings[0]["extra"] = "unbound"
    elif mutation == "bool_count":
        bindings[0]["variant_count"] = True
    else:
        bindings[0]["subgenome"] = 1
    for binding in bindings:
        binding["binding_sha256"] = sha256_payload({
            key: value for key, value in binding.items() if key != "binding_sha256"
        })
    raw["input_family_sha256"] = sha256_payload({
        "schema": "homoeogwas-snpxsnp-input-family-v1", "blocks": bindings,
    })
    artifact["record"]["input_family_sha256"] = raw["input_family_sha256"]
    _refresh_artifact_hashes(artifact)
    with pytest.raises(ValueError, match="input.block|marker count"):
        resource_probe.validate_snpxsnp_resource_probe_artifact(artifact, **_witness())


@pytest.mark.parametrize("field", ["prefix", "bank", "authorization"])
@pytest.mark.parametrize("width", [1, 5, 20])
def test_standalone_audit_rejects_rehashed_response_identity_forgery(
    monkeypatch, field, width,
):
    from scripts.benchmarks.v201.audit import (
        BenchmarkAuditError,
        _audit_comparator_probe_artifact,
    )

    artifact = _produce(monkeypatch, width)
    if field == "authorization":
        artifact["record"]["probe_authorization_sha256"] = "f" * 64
    else:
        artifact["score_evidence"][f"response_{field}_identity"][
            "array_sha256"
        ] = "f" * 64
    _refresh_artifact_hashes(artifact)

    with pytest.raises(BenchmarkAuditError, match="resource probe artifact"):
        _audit_comparator_probe_artifact(artifact, **_witness())


@pytest.mark.parametrize("mutation", ["count", "key", "missing"])
@pytest.mark.parametrize("independent", [False, True])
def test_standalone_audit_rejects_marker_counts_detached_from_raw_inputs(
    monkeypatch, mutation, independent,
):
    from scripts.benchmarks.v201.audit import (
        BenchmarkAuditError,
        _audit_comparator_probe_artifact,
    )

    artifact = _produce(monkeypatch)
    counts = artifact["record"]["gated_marker_count_by_gene"]
    if mutation == "count":
        counts["A|geneA"] = 2
    elif mutation == "key":
        counts["A|other"] = counts.pop("A|geneA")
    else:
        del counts["A|geneA"]
    _refresh_artifact_hashes(artifact)
    if independent:
        # Exercise the audit's own result-to-record join even if the producer
        # validator regresses to accepting detached counts.
        monkeypatch.setattr(
            resource_probe, "validate_snpxsnp_resource_probe_artifact",
            lambda payload, **_kwargs: payload,
        )
    else:
        with pytest.raises(ValueError, match="marker count"):
            resource_probe.validate_snpxsnp_resource_probe_artifact(
                artifact, **_witness(),
            )
    with pytest.raises(BenchmarkAuditError, match="resource probe artifact"):
        _audit_comparator_probe_artifact(artifact, **_witness())


@pytest.mark.parametrize("width", [1, 5, 20])
def test_producer_scores_an_exact_prefix_and_emits_auditable_v2_artifact(monkeypatch, width):
    artifact = _produce(monkeypatch, width)

    validated = resource_probe.validate_snpxsnp_resource_probe_artifact(
        artifact, **_witness(),
    )
    record = validated["record"]
    assert record["schema"] == "homoeogwas-snpxsnp-resource-probe-v2"
    assert record["response_width"] == width
    assert record["response_ids"] == [
        f"response-{index:02d}" for index in range(20)
    ]
    assert record["sampled_pids"] == record["worker_pids"] == [record["root_pid"]]
    assert record["peak_aggregate_pss_bytes"] >= record["peak_parent_rss_bytes"]
    assert record["formal_execution_authorized"] is False
    assert len(validated["score_evidence"]["group_p"]) == 80
    assert all(len(row) == width for row in validated["score_evidence"]["group_p"])
    bank = _witness()["response_bank"]
    assert validated["score_evidence"]["response_prefix_identity"][
        "array_sha256"
    ] == _array_sha256(bank[:, :width])


def test_artifact_audit_rejects_score_evidence_changed_after_measurement(monkeypatch):
    artifact = _produce(monkeypatch, 1)
    tampered = copy.deepcopy(artifact)
    tampered["score_evidence"]["group_p"][0][0] = 0.5

    with pytest.raises(ValueError, match="score evidence"):
        resource_probe.validate_snpxsnp_resource_probe_artifact(tampered, **_witness())


def test_widths_bind_one_bank_and_literal_nested_prefixes(monkeypatch):
    width1 = _produce(monkeypatch, 1)
    width5 = _produce(monkeypatch, 5)

    record1 = width1["record"]
    record5 = width5["record"]
    assert record1["response_bank_sha256"] == record5["response_bank_sha256"]
    assert record1["response_prefix_sha256"] != record5["response_prefix_sha256"]
    assert width1["score_evidence"]["response_ids"] == record1["response_ids"][:1]
    assert width5["score_evidence"]["response_ids"] == record5["response_ids"][:5]


def test_audit_wrapper_rejects_a_detached_score_artifact(monkeypatch):
    from scripts.benchmarks.v201.audit import (
        BenchmarkAuditError,
        _audit_comparator_probe_artifact,
    )

    artifact = _produce(monkeypatch, 1)
    assert _audit_comparator_probe_artifact(
        artifact, **_witness(),
    )["record"] == artifact["record"]
    tampered = copy.deepcopy(artifact)
    tampered["score_evidence"]["raw_score_evidence"]["tested_pair_count"] = 2
    with pytest.raises(BenchmarkAuditError, match="resource probe artifact"):
        _audit_comparator_probe_artifact(tampered, **_witness())


def test_audit_independently_reconstructs_the_raw_member_family(monkeypatch):
    from scripts.benchmarks.v201.audit import (
        BenchmarkAuditError,
        _audit_comparator_probe_artifact,
    )

    artifact = _produce(monkeypatch, 1)
    tampered = copy.deepcopy(artifact)
    raw = tampered["score_evidence"]["raw_score_evidence"]
    raw["member_ids"][0] = raw["member_ids"][0] + ".changed"
    encoded = canonical_json(tampered["score_evidence"]).encode("utf-8")
    tampered["record"]["score_evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    tampered["record"]["output_bytes"] = len(encoded)

    resource_probe.validate_snpxsnp_resource_probe_artifact(tampered, **_witness())
    with pytest.raises(BenchmarkAuditError, match="streaming evidence"):
        _audit_comparator_probe_artifact(tampered, **_witness())


@pytest.mark.parametrize(
    "field",
    [
        "implementation_commit", "matched_comparator_contract_sha256",
        "panel_id", "design_hash", "context_fingerprint", "prepared_design_sha256",
        "response_bank_sha256", "response_ids_sha256", "response_widths",
        "resource_probe_authorized", "formal_execution_authorized",
    ],
)
def test_audit_checks_all_external_authorization_bindings(monkeypatch, field):
    from scripts.benchmarks.v201.audit import (
        BenchmarkAuditError,
        _audit_comparator_probe_artifact,
    )

    artifact = _produce(monkeypatch)
    witness = _witness()
    authorization = witness["authorization_payload"]
    if field in authorization["panel_context"]:
        authorization["panel_context"][field] = (
            "REALG.WATKINS_F2143" if field == "panel_id" else "f" * 64
        )
    elif field == "response_widths":
        authorization[field] = [1, 5]
    elif field.endswith("authorized"):
        authorization[field] = not authorization[field]
    else:
        authorization[field] = "f" * (40 if field == "implementation_commit" else 64)
    witness["authorization_sha256"] = sha256_payload(authorization)
    artifact["record"]["probe_authorization_sha256"] = witness[
        "authorization_sha256"
    ]
    with pytest.raises(BenchmarkAuditError, match="resource probe artifact"):
        _audit_comparator_probe_artifact(artifact, **witness)


@pytest.mark.parametrize(
    "mutation", ["bytes", "float32", "fortran", "nonfinite", "width", "ids"],
)
def test_audit_requires_the_exact_external_response_bank(monkeypatch, mutation):
    from scripts.benchmarks.v201.audit import (
        BenchmarkAuditError,
        _audit_comparator_probe_artifact,
    )

    artifact = _produce(monkeypatch)
    witness = _witness()
    bank = witness["response_bank"]
    if mutation == "bytes":
        bank[0, -1] += 1.0  # Also bind columns outside the width-1 prefix.
    elif mutation == "float32":
        witness["response_bank"] = bank.astype(np.float32)
    elif mutation == "fortran":
        witness["response_bank"] = np.asfortranarray(bank)
    elif mutation == "nonfinite":
        bank[0, 0] = np.nan
    elif mutation == "width":
        witness["response_bank"] = np.ascontiguousarray(bank[:, :5])
    else:
        witness["response_ids"] = witness["response_ids"][::-1]
    with pytest.raises(BenchmarkAuditError, match="resource probe artifact"):
        _audit_comparator_probe_artifact(artifact, **witness)


def _artifact_series(monkeypatch):
    artifacts = [_produce(monkeypatch, width) for width in (1, 5, 20)]
    # This fixture tests audit/projection logic, not noisy process measurements.
    for artifact in artifacts:
        record = artifact["record"]
        width = record["response_width"]
        record["scorer_wall_seconds"] = float(width)
        record["scorer_cpu_seconds"] = float(2 * width)
        record["peak_parent_rss_bytes"] = 1024 ** 3
        record["peak_aggregate_pss_bytes"] = 2 * 1024 ** 3
    return artifacts


def test_grounded_artifact_series_validates_all_widths_and_projects(monkeypatch):
    artifacts = _artifact_series(monkeypatch)
    projection = resource_probe.validate_snpxsnp_resource_probe_artifact_series(
        artifacts, **_witness(),
    )
    assert projection["response_widths"] == [1, 5, 20]
    assert projection["projected"]["elapsed_seconds"] == 4000.0
    assert projection["projected"]["peak_parent_rss_bytes"] == 2 * 1024 ** 3
    assert projection["accepted"] is True


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("mutation", ["prefix", "replacement_bank", "raw_member"])
def test_grounded_series_rejects_any_replaced_artifact(monkeypatch, index, mutation):
    from scripts.benchmarks.v201.audit import BenchmarkAuditError

    artifacts = _artifact_series(monkeypatch)
    artifact = artifacts[index]
    evidence = artifact["score_evidence"]
    if mutation == "prefix":
        evidence["response_prefix_identity"]["array_sha256"] = "f" * 64
    elif mutation == "replacement_bank":
        witness = _witness()
        bank = witness["response_bank"].copy()
        bank[0, 0] += 1.0
        bank_identity, prefix_identity = resource_probe._response_identities(
            bank, tuple(witness["response_ids"]), artifact["record"]["response_width"],
        )
        evidence["response_bank_identity"] = bank_identity
        evidence["response_prefix_identity"] = prefix_identity
        _authorization_payload, authorization_sha256 = _authorization(
            bank, witness["response_ids"],
        )
        artifact["record"]["probe_authorization_sha256"] = authorization_sha256
    else:
        evidence["raw_score_evidence"]["member_ids"][0] += ".changed"
    _refresh_artifact_hashes(artifact)
    with pytest.raises(BenchmarkAuditError):
        resource_probe.validate_snpxsnp_resource_probe_artifact_series(
            artifacts, **_witness(),
        )


@pytest.mark.parametrize("mutation", ["missing_width", "slow", "memory", "input_family"])
def test_grounded_series_retains_structural_and_projection_gates(monkeypatch, mutation):
    artifacts = _artifact_series(monkeypatch)
    if mutation == "missing_width":
        artifacts.pop()
    elif mutation == "slow":
        artifacts[-1]["record"]["scorer_wall_seconds"] = 25.0
    elif mutation == "memory":
        for artifact in artifacts:
            artifact["record"]["peak_parent_rss_bytes"] = 20 * 1024 ** 3
            artifact["record"]["peak_aggregate_pss_bytes"] = 24 * 1024 ** 3
    else:
        artifact = artifacts[-1]
        raw = artifact["score_evidence"]["raw_score_evidence"]
        binding = raw["input_block_bindings"][0]
        binding["dosage_sha256"] = "f" * 64
        binding["binding_sha256"] = sha256_payload({
            key: value for key, value in binding.items() if key != "binding_sha256"
        })
        raw["input_family_sha256"] = sha256_payload({
            "schema": "homoeogwas-snpxsnp-input-family-v1",
            "blocks": raw["input_block_bindings"],
        })
        artifact["record"]["input_family_sha256"] = raw["input_family_sha256"]
        _refresh_artifact_hashes(artifact)
    with pytest.raises(ValueError):
        resource_probe.validate_snpxsnp_resource_probe_artifact_series(
            artifacts, **_witness(),
        )


def test_producer_rejects_missing_authorization_before_calling_the_scorer(
    monkeypatch,
):
    prepared, responses = _prepared_fixture()
    response_ids = [f"response-{index:02d}" for index in range(20)]
    authorization, authorization_sha256 = _authorization(
        responses, response_ids, enabled=False,
    )
    monkeypatch.setattr(
        resource_probe, "_context_fingerprint", lambda _context: "b" * 64
    )
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("scorer must not be called")

    monkeypatch.setattr(resource_probe, "score_snpxsnp_family", forbidden)
    with pytest.raises(ValueError, match="not authorized"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared,
            responses,
            response_ids,
            response_width=1,
            design_hash="a" * 64,
            context_fingerprint="b" * 64,
            authorization_payload=authorization,
            authorization_sha256=authorization_sha256,
            implementation_commit="1" * 40,
            matched_comparator_contract_sha256="2" * 64,
        )
    assert called is False


def test_authorization_cannot_be_reused_after_response_bank_changes(monkeypatch):
    prepared, responses = _prepared_fixture()
    response_ids = [f"response-{index:02d}" for index in range(20)]
    authorization, authorization_sha256 = _authorization(responses, response_ids)
    changed = responses.copy()
    changed[0, 0] += 1.0
    monkeypatch.setattr(
        resource_probe, "_context_fingerprint", lambda _context: "b" * 64
    )
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("scorer must not be called")

    monkeypatch.setattr(resource_probe, "score_snpxsnp_family", forbidden)
    with pytest.raises(ValueError, match="authorization identity or scope"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared,
            changed,
            response_ids,
            response_width=1,
            design_hash="a" * 64,
            context_fingerprint="b" * 64,
            authorization_payload=authorization,
            authorization_sha256=authorization_sha256,
            implementation_commit="1" * 40,
            matched_comparator_contract_sha256="2" * 64,
        )
    assert called is False


def test_producer_recomputes_the_context_fingerprint_before_scoring(monkeypatch):
    prepared, responses = _prepared_fixture()
    response_ids = [f"response-{index:02d}" for index in range(20)]
    authorization, authorization_sha256 = _authorization(responses, response_ids)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _context: "f" * 64)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("scorer must not be called")

    monkeypatch.setattr(resource_probe, "score_snpxsnp_family", forbidden)
    with pytest.raises(ValueError, match="context fingerprint is detached"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared,
            responses,
            response_ids,
            response_width=1,
            design_hash="a" * 64,
            context_fingerprint="b" * 64,
            authorization_payload=authorization,
            authorization_sha256=authorization_sha256,
            implementation_commit="1" * 40,
            matched_comparator_contract_sha256="2" * 64,
        )
    assert called is False
