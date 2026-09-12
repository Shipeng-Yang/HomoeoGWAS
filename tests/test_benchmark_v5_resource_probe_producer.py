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


def _authorization(response_bank, response_ids, *, enabled: bool = True):
    bank_identity, _prefix_identity = resource_probe._response_identities(
        response_bank, tuple(response_ids), 1,
    )
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


def test_producer_scores_an_exact_prefix_and_emits_auditable_v2_artifact(monkeypatch):
    artifact = _produce(monkeypatch, 1)

    validated = resource_probe.validate_snpxsnp_resource_probe_artifact(artifact)
    record = validated["record"]
    assert record["schema"] == "homoeogwas-snpxsnp-resource-probe-v2"
    assert record["response_width"] == 1
    assert record["response_ids"] == [
        f"response-{index:02d}" for index in range(20)
    ]
    assert record["sampled_pids"] == record["worker_pids"] == [record["root_pid"]]
    assert record["peak_aggregate_pss_bytes"] >= record["peak_parent_rss_bytes"]
    assert record["formal_execution_authorized"] is False
    assert len(validated["score_evidence"]["group_p"]) == 80
    assert all(len(row) == 1 for row in validated["score_evidence"]["group_p"])


def test_artifact_audit_rejects_score_evidence_changed_after_measurement(monkeypatch):
    artifact = _produce(monkeypatch, 1)
    tampered = copy.deepcopy(artifact)
    tampered["score_evidence"]["group_p"][0][0] = 0.5

    with pytest.raises(ValueError, match="score evidence"):
        resource_probe.validate_snpxsnp_resource_probe_artifact(tampered)


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
    assert _audit_comparator_probe_artifact(artifact)["record"] == artifact["record"]
    tampered = copy.deepcopy(artifact)
    tampered["score_evidence"]["raw_score_evidence"]["tested_pair_count"] = 2
    with pytest.raises(BenchmarkAuditError, match="resource probe artifact"):
        _audit_comparator_probe_artifact(tampered)


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

    resource_probe.validate_snpxsnp_resource_probe_artifact(tampered)
    with pytest.raises(BenchmarkAuditError, match="streaming evidence"):
        _audit_comparator_probe_artifact(tampered)


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
