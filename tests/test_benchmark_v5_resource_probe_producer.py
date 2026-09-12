from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from homoeogwas.group_family import MasterGroupFamily, expand_pair_edges
from homoeogwas.omnib_family import (
    OmniBFamilyScores,
    _array_identity,
    _text_identity,
)
from scripts.benchmarks.v201 import comparators as comparator_module
from scripts.benchmarks.v201 import resource_probe
from scripts.benchmarks.v201.contracts import canonical_json, sha256_payload


class _IdentityAccessorAttack(dict):
    """Serialize stored identity values but return attacker values via ``get``."""

    def __init__(self, stored, accessor_values):
        super().__init__(stored)
        self._accessor_values = dict(accessor_values)

    def get(self, key, default=None):
        return self._accessor_values.get(key, super().get(key, default))


class _AlwaysEqualStr(str):
    __hash__ = str.__hash__

    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


class _AlwaysEqualMapping(dict):
    def __eq__(self, _other):
        return True

    def __ne__(self, _other):
        return False


class _AuthorizationPanelAccessorAttack(dict):
    def __init__(self, stored, panel_context):
        super().__init__(stored)
        self._panel_context = panel_context

    def __getitem__(self, key):
        if key == "panel_context":
            return self._panel_context
        return super().__getitem__(key)


class _ArtifactAccessorAttack(dict):
    def __init__(self, stored, record):
        super().__init__(stored)
        self._record = record

    def __getitem__(self, key):
        if key == "record":
            return self._record
        return super().__getitem__(key)


class _ArtifactEvidenceAccessorAttack(dict):
    def __init__(self, stored, score_evidence):
        super().__init__(stored)
        self._score_evidence = score_evidence

    def __getitem__(self, key):
        if key == "score_evidence":
            return self._score_evidence
        return super().__getitem__(key)


class _SingleReadPrepared:
    def __init__(self, prepared, guarded_field):
        self._prepared = prepared
        self._guarded_field = guarded_field
        self.reads = {"scores": 0, "expanded": 0}

    @property
    def context(self):
        return self._prepared.context

    @property
    def scores(self):
        return self._read("scores")

    @property
    def expanded(self):
        return self._read("expanded")

    @property
    def gene_blocks(self):
        return self._prepared.gene_blocks

    def _read(self, field):
        self.reads[field] += 1
        if field == self._guarded_field and self.reads[field] > 1:
            raise AssertionError(f"prepared.{field} was read more than once")
        return getattr(self._prepared, field)


class _PreparedAccessForbidden:
    def __getattr__(self, field):
        raise AssertionError(f"invalid producer input accessed prepared.{field}")


class _ContextReadOnce:
    def __init__(self, context):
        self._context = context
        self.reads = {"family": 0, "panel_id": 0, "sample_context": 0}

    def _read(self, field):
        self.reads[field] += 1
        if self.reads[field] > 1:
            raise AssertionError(f"context.{field} was read more than once")
        return getattr(self._context, field)

    @property
    def family(self):
        return self._read("family")

    @property
    def panel_id(self):
        return self._read("panel_id")

    @property
    def sample_context(self):
        return self._read("sample_context")

    def __getattr__(self, field):
        return getattr(self._context, field)


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
    W = np.eye(sample_count)
    null_design = np.ones((sample_count, 1))
    null_fit_identity = {
        "W": _array_identity(W),
        "design": _array_identity(null_design),
    }
    null_fit_sha256 = _text_identity(null_fit_identity)
    prepared_design_identity = {"null_fit_sha256": null_fit_sha256}
    scores = OmniBFamilyScores(
        edge_p=np.empty((len(expanded.edges), 0)),
        group_p=np.empty((len(group_ids), 0)),
        edge_components_obs=np.empty((len(expanded.edges), 3)),
        edge_estimable=np.ones(len(expanded.edges), dtype=bool),
        group_estimable=np.ones(len(group_ids), dtype=bool),
        W=W,
        y=np.zeros(sample_count),
        covariance_components={"e": 1.0},
        null_design=null_design,
        gated_snp={
            ("A", "geneA"): np.asarray((7,), dtype=int),
            ("D", "geneD"): np.asarray((9,), dtype=int),
        },
        null_fit_identity=null_fit_identity,
        null_fit_sha256=null_fit_sha256,
        prepared_design_identity=prepared_design_identity,
        prepared_design_sha256=_text_identity(prepared_design_identity),
    )
    context = SimpleNamespace(
        panel_id="REALG.CGVD1245",
        sample_context="full",
        family=family,
        subdata={
            "A": SimpleNamespace(
                X=np.zeros((sample_count, 1)),
                samples=np.asarray([f"s{index}" for index in range(sample_count)]),
                gene_snp={"geneA": np.asarray([0])},
            ),
            "D": SimpleNamespace(
                X=np.zeros((sample_count, 1)),
                samples=np.asarray([f"s{index}" for index in range(sample_count)]),
                gene_snp={"geneD": np.asarray([0])},
            ),
        },
        feature_seed=1,
        marker_mask_identity={"A": {}, "D": {}},
        marker_mask_sha256="d" * 64,
        sample_idx=np.arange(sample_count),
        phenotype=np.zeros(sample_count),
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


def _input_family_sha256(prepared):
    bindings = []
    for (subgenome, gene_id), values in sorted(prepared.gene_blocks.items()):
        binding = {
            "subgenome": subgenome,
            "gene_id": gene_id,
            "sample_count": values.shape[0],
            "variant_count": values.shape[1],
            "source_column_indices_encoding": "little_endian_int64_c_order",
            "source_column_indices_sha256": hashlib.sha256(
                prepared.scores.gated_snp[(subgenome, gene_id)].astype("<i8").tobytes()
            ).hexdigest(),
            "dosage_encoding": "little_endian_float64_c_order",
            "dosage_sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
        }
        binding["binding_sha256"] = sha256_payload(binding)
        bindings.append(binding)
    return sha256_payload({
        "schema": "homoeogwas-snpxsnp-input-family-v1", "blocks": bindings,
    })


def _authorization(response_bank, response_ids, *, enabled: bool = True):
    prepared, _responses = _prepared_fixture()
    bank_identity = {
        "schema": "homoeogwas-snpxsnp-response-bank-v1",
        "dtype": "float64",
        "shape": list(response_bank.shape),
        "ordered_response_ids": list(response_ids),
        "array_sha256": _array_sha256(response_bank),
    }
    payload = {
        "schema": "homoeogwas-snpxsnp-resource-probe-authorization-v2",
        "authorization_id": "unit-test-only",
        "implementation_commit": "1" * 40,
        "matched_comparator_contract_sha256": "2" * 64,
        "panel_context": {
            "panel_id": "REALG.CGVD1245",
            "design_hash": "a" * 64,
            "context_fingerprint": "b" * 64,
            "prepared_design_sha256": prepared.scores.prepared_design_sha256,
            "input_family_sha256": _input_family_sha256(prepared),
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


def _alternate_input_result(width):
    prepared, responses = _prepared_fixture()
    prepared.gene_blocks[("A", "geneA")] = np.column_stack((
        prepared.gene_blocks[("A", "geneA")],
        np.random.default_rng(118).binomial(2, 0.4, size=40),
    ))
    prepared.scores = replace(prepared.scores, gated_snp={
        ("A", "geneA"): np.array([7, 11]),
        ("D", "geneD"): np.array([9]),
    })
    return comparator_module.score_snpxsnp_family(
        prepared.scores, prepared.context.family, prepared.expanded,
        prepared.gene_blocks, np.ascontiguousarray(responses[:, :width]),
        max_offered_pairs=10000,
    )


def test_audit_rejects_whole_input_family_substitution_under_unchanged_authorization(
    monkeypatch,
):
    from scripts.benchmarks.v201.audit import (
        BenchmarkAuditError,
        _audit_comparator_probe_artifact,
    )

    artifact = _produce(monkeypatch)
    witness = _witness()
    original_authorization = copy.deepcopy(witness["authorization_payload"])
    result = _alternate_input_result(1)
    raw = result.evidence_payload()
    artifact["score_evidence"]["raw_score_evidence"] = raw
    artifact["score_evidence"]["group_p"] = result.group_p.tolist()
    for field in (
        "input_family_sha256", "member_family_sha256", "offered_pair_count",
        "design_nonestimable_pair_count", "tested_pair_count",
        "nonfinite_pair_score_count", "failed_response_indices",
    ):
        artifact["record"][field] = raw[field]
    artifact["record"]["gated_marker_count_by_gene"] = {"A|geneA": 2, "D|geneD": 1}
    _refresh_artifact_hashes(artifact)
    assert witness["authorization_payload"] == original_authorization
    with pytest.raises(BenchmarkAuditError, match="resource probe artifact"):
        _audit_comparator_probe_artifact(artifact, **witness)


def test_artifact_validator_rehashes_raw_inputs_against_authorization(monkeypatch):
    artifact = _produce(monkeypatch)
    binding = artifact["score_evidence"]["raw_score_evidence"]["input_block_bindings"][0]
    binding["dosage_sha256"] = "f" * 64
    binding["binding_sha256"] = sha256_payload({
        key: value for key, value in binding.items() if key != "binding_sha256"
    })
    # Keep the advertised family SHA equal to the authorized value, but detach
    # its block contents. Rehash only the artifact envelope, not that family SHA.
    _refresh_artifact_hashes(artifact)
    with pytest.raises(ValueError, match="input.family|authorization"):
        resource_probe.validate_snpxsnp_resource_probe_artifact(artifact, **_witness())


def test_producer_rejects_scorer_input_family_not_in_authorization(monkeypatch):
    # Change only dosage values, preserving the producer's prepared marker counts.
    prepared, responses = _prepared_fixture()
    prepared.gene_blocks[("A", "geneA")][0, 0] = (
        prepared.gene_blocks[("A", "geneA")][0, 0] + 1.0
    ) % 3.0
    result = comparator_module.score_snpxsnp_family(
        prepared.scores, prepared.context.family, prepared.expanded,
        prepared.gene_blocks, np.ascontiguousarray(responses[:, :1]),
        max_offered_pairs=10000,
    )
    assert result.input_family_sha256 != _input_family_sha256(_prepared_fixture()[0])
    monkeypatch.setattr(
        resource_probe, "_score_snpxsnp_bound_inputs", lambda *_a, **_k: result,
    )
    with pytest.raises(ValueError, match="input.family|authorization"):
        _produce(monkeypatch)


@pytest.mark.parametrize("mutation", ["dosage", "source_columns", "both"])
def test_producer_checks_live_input_family_before_measurement(monkeypatch, mutation):
    from scripts.benchmarks.v201 import cli

    prepared, bank = _prepared_fixture()
    witness = _witness()
    if mutation in {"dosage", "both"}:
        values = prepared.gene_blocks[("A", "geneA")]
        values[0, 0] = (values[0, 0] + 1.0) % 3.0
    if mutation in {"source_columns", "both"}:
        prepared.scores = replace(prepared.scores, gated_snp={
            ("A", "geneA"): np.array([11]),
            ("D", "geneD"): np.array([9]),
        })
    assert _input_family_sha256(prepared) != witness["authorization_payload"][
        "panel_context"
    ]["input_family_sha256"]
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    events = []
    scorer = resource_probe._score_snpxsnp_bound_inputs
    measure = cli._measure_comparator_operation

    def record_score(*args, **kwargs):
        events.append("scorer")
        return scorer(*args, **kwargs)

    def record_measurement(*args, **kwargs):
        events.append("measurement")
        return measure(*args, **kwargs)

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", record_score)
    monkeypatch.setattr(cli, "_measure_comparator_operation", record_measurement)
    with pytest.raises(ValueError, match="input.family|authorization"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared, bank, witness["response_ids"], response_width=1,
            design_hash="a" * 64, context_fingerprint="b" * 64,
            authorization_payload=witness["authorization_payload"],
            authorization_sha256=witness["authorization_sha256"],
            implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
        )
    assert events == [], f"unauthorized live input reached {events}"


@pytest.mark.parametrize("mutation", ["dosage", "source_columns", "replace_mappings"])
@pytest.mark.parametrize("width", [1, 20])
def test_producer_scores_authorized_genotype_despite_mutate_restore(
    monkeypatch, mutation, width,
):
    from scripts.benchmarks.v201 import cli

    prepared, bank = _prepared_fixture()
    witness = _witness()
    expected = comparator_module.score_snpxsnp_family(
        prepared.scores, prepared.context.family, prepared.expanded,
        prepared.gene_blocks, np.ascontiguousarray(bank[:, :width]),
        max_offered_pairs=10000,
    )
    blocks = prepared.gene_blocks
    old_block = blocks[("A", "geneA")].copy()
    columns = prepared.scores.gated_snp[("A", "geneA")]
    old_columns = columns.copy()
    measure = cli._measure_comparator_operation

    def mutate_measure_restore(*args, **kwargs):
        try:
            if mutation == "dosage":
                blocks[("A", "geneA")][0, 0] = (old_block[0, 0] + 1.0) % 3.0
            elif mutation == "source_columns":
                columns[:] = 11
            else:
                prepared.gene_blocks = {}
            return measure(*args, **kwargs)
        finally:
            blocks[("A", "geneA")][:] = old_block
            columns[:] = old_columns
            prepared.gene_blocks = blocks

    monkeypatch.setattr(cli, "_measure_comparator_operation", mutate_measure_restore)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    artifact = resource_probe.produce_snpxsnp_resource_probe(
        prepared, bank, witness["response_ids"], response_width=width,
        design_hash="a" * 64, context_fingerprint="b" * 64,
        authorization_payload=witness["authorization_payload"],
        authorization_sha256=witness["authorization_sha256"],
        implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
    )
    np.testing.assert_array_equal(artifact["score_evidence"]["group_p"], expected.group_p)
    assert artifact["score_evidence"]["raw_score_evidence"] == expected.evidence_payload()


@pytest.mark.parametrize("mutation", ["W", "null_design"])
def test_producer_scores_private_context_despite_mutate_restore(monkeypatch, mutation):
    from scripts.benchmarks.v201 import cli

    prepared, bank = _prepared_fixture()
    witness = _witness()
    expected = comparator_module.score_snpxsnp_family(
        prepared.scores,
        prepared.context.family,
        prepared.expanded,
        prepared.gene_blocks,
        np.ascontiguousarray(bank[:, :20]),
        max_offered_pairs=10000,
    )
    old_W = prepared.scores.W.copy()
    old_design = prepared.scores.null_design.copy()
    measure = cli._measure_comparator_operation

    def mutate_measure_restore(*args, **kwargs):
        try:
            if mutation == "W":
                prepared.scores.W[:] = np.diag(np.linspace(0.4, 1.6, old_W.shape[0]))
            else:
                prepared.scores.null_design[:, 0] = np.linspace(-1.0, 1.0, old_design.shape[0])
            return measure(*args, **kwargs)
        finally:
            prepared.scores.W[:] = old_W
            prepared.scores.null_design[:] = old_design

    monkeypatch.setattr(cli, "_measure_comparator_operation", mutate_measure_restore)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    artifact = resource_probe.produce_snpxsnp_resource_probe(
        prepared, bank, witness["response_ids"], response_width=20,
        design_hash="a" * 64, context_fingerprint="b" * 64,
        authorization_payload=witness["authorization_payload"],
        authorization_sha256=witness["authorization_sha256"],
        implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
    )
    np.testing.assert_array_equal(artifact["score_evidence"]["group_p"], expected.group_p)
    assert artifact["score_evidence"]["raw_score_evidence"] == expected.evidence_payload()


@pytest.mark.parametrize(
    "mutation",
    [
        "live_W",
        "live_null_design",
        "prepared_design_identity",
        "prepared_null_fit_sha256",
        "null_fit_identity",
    ],
)
def test_producer_rejects_detached_score_context_before_measurement(
    monkeypatch, mutation,
):
    from scripts.benchmarks.v201 import cli

    prepared, bank = _prepared_fixture()
    witness = _witness()
    if mutation == "live_W":
        prepared.scores.W[0, 0] += 0.1
    elif mutation == "live_null_design":
        prepared.scores.null_design[0, 0] += 0.1
    elif mutation == "prepared_design_identity":
        prepared.scores.prepared_design_identity = {"null_fit_sha256": "f" * 64}
    elif mutation == "prepared_null_fit_sha256":
        prepared.scores.null_fit_sha256 = "f" * 64
    else:
        prepared.scores.null_fit_identity = {
            **prepared.scores.null_fit_identity,
            "detached": True,
        }
    events = []
    scorer = resource_probe._score_snpxsnp_bound_inputs
    measure = cli._measure_comparator_operation

    def record_score(*args, **kwargs):
        events.append("scorer")
        return scorer(*args, **kwargs)

    def record_measurement(*args, **kwargs):
        events.append("measurement")
        return measure(*args, **kwargs)

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", record_score)
    monkeypatch.setattr(cli, "_measure_comparator_operation", record_measurement)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    with pytest.raises(ValueError, match="score context identity is detached"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared, bank, witness["response_ids"], response_width=1,
            design_hash="a" * 64, context_fingerprint="b" * 64,
            authorization_payload=witness["authorization_payload"],
            authorization_sha256=witness["authorization_sha256"],
            implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
        )
    assert events == [], f"detached context reached {events}"


@pytest.mark.parametrize(
    "attack",
    ["prepared_null_fit_sha256", "null_fit_W", "null_fit_design"],
)
def test_producer_rejects_identity_accessor_attacks_before_measurement(
    monkeypatch, attack,
):
    from scripts.benchmarks.v201 import cli

    prepared, bank = _prepared_fixture()
    witness = _witness()
    if attack == "prepared_null_fit_sha256":
        prepared.scores.W[:] = np.diag(np.linspace(0.4, 1.6, prepared.scores.W.shape[0]))
        new_null_fit_identity = {
            "W": _array_identity(prepared.scores.W),
            "design": _array_identity(prepared.scores.null_design),
        }
        new_null_fit_sha256 = _text_identity(new_null_fit_identity)
        prepared.scores.null_fit_identity = new_null_fit_identity
        prepared.scores.null_fit_sha256 = new_null_fit_sha256
        prepared.scores.prepared_design_identity = _IdentityAccessorAttack(
            prepared.scores.prepared_design_identity,
            {"null_fit_sha256": new_null_fit_sha256},
        )
    elif attack == "null_fit_W":
        prepared.scores.W[:] = np.diag(np.linspace(0.4, 1.6, prepared.scores.W.shape[0]))
        prepared.scores.null_fit_identity = _IdentityAccessorAttack(
            prepared.scores.null_fit_identity,
            {"W": _array_identity(prepared.scores.W)},
        )
    else:
        prepared.scores.null_design[:, 0] = np.linspace(
            -1.0, 1.0, prepared.scores.null_design.shape[0],
        )
        prepared.scores.null_fit_identity = _IdentityAccessorAttack(
            prepared.scores.null_fit_identity,
            {"design": _array_identity(prepared.scores.null_design)},
        )
    events = []
    scorer = resource_probe._score_snpxsnp_bound_inputs
    measure = cli._measure_comparator_operation

    def record_score(*args, **kwargs):
        events.append("scorer")
        return scorer(*args, **kwargs)

    def record_measurement(*args, **kwargs):
        events.append("measurement")
        return measure(*args, **kwargs)

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", record_score)
    monkeypatch.setattr(cli, "_measure_comparator_operation", record_measurement)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    with pytest.raises(ValueError, match="score context identity is detached"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared, bank, witness["response_ids"], response_width=1,
            design_hash="a" * 64, context_fingerprint="b" * 64,
            authorization_payload=witness["authorization_payload"],
            authorization_sha256=witness["authorization_sha256"],
            implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
        )
    assert events == [], f"identity accessor attack reached {events}"


@pytest.mark.parametrize(
    "attack",
    [
        "prepared_design_sha256",
        "authorization_panel_id",
        "authorization_sha256",
        "implementation_commit",
    ],
)
def test_producer_rejects_scalar_subclass_attacks_before_measurement(
    monkeypatch, attack,
):
    from scripts.benchmarks.v201 import cli

    prepared, bank = _prepared_fixture()
    witness = _witness()
    authorization = witness["authorization_payload"]
    authorization_sha256 = witness["authorization_sha256"]
    implementation_commit = "1" * 40
    if attack == "prepared_design_sha256":
        prepared.scores.prepared_design_sha256 = _AlwaysEqualStr("f" * 64)
    elif attack == "authorization_panel_id":
        authorization["panel_context"]["panel_id"] = _AlwaysEqualStr("ATTACK")
        authorization_sha256 = sha256_payload(authorization)
    elif attack == "authorization_sha256":
        authorization_sha256 = _AlwaysEqualStr("f" * 64)
    else:
        implementation_commit = _AlwaysEqualStr("f" * 40)
    events = []
    scorer = resource_probe._score_snpxsnp_bound_inputs
    measure = cli._measure_comparator_operation

    def record_score(*args, **kwargs):
        events.append("scorer")
        return scorer(*args, **kwargs)

    def record_measurement(*args, **kwargs):
        events.append("measurement")
        return measure(*args, **kwargs)

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", record_score)
    monkeypatch.setattr(cli, "_measure_comparator_operation", record_measurement)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    with pytest.raises(ValueError, match="authorization|score context"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared, bank, witness["response_ids"], response_width=1,
            design_hash="a" * 64, context_fingerprint="b" * 64,
            authorization_payload=authorization,
            authorization_sha256=authorization_sha256,
            implementation_commit=implementation_commit,
            matched_comparator_contract_sha256="2" * 64,
        )
    assert events == [], f"scalar subclass attack reached {events}"


def test_producer_rejects_authorization_accessor_attack_before_measurement(
    monkeypatch,
):
    from scripts.benchmarks.v201 import cli

    prepared, bank = _prepared_fixture()
    witness = _witness()
    stored = copy.deepcopy(witness["authorization_payload"])
    stored["panel_context"]["panel_id"] = "ATTACK"
    authorization = _AuthorizationPanelAccessorAttack(
        stored,
        _AlwaysEqualMapping(witness["authorization_payload"]["panel_context"]),
    )
    events = []
    scorer = resource_probe._score_snpxsnp_bound_inputs
    measure = cli._measure_comparator_operation

    def record_score(*args, **kwargs):
        events.append("scorer")
        return scorer(*args, **kwargs)

    def record_measurement(*args, **kwargs):
        events.append("measurement")
        return measure(*args, **kwargs)

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", record_score)
    monkeypatch.setattr(cli, "_measure_comparator_operation", record_measurement)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    with pytest.raises(ValueError, match="authorization"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared, bank, witness["response_ids"], response_width=1,
            design_hash="a" * 64, context_fingerprint="b" * 64,
            authorization_payload=authorization,
            authorization_sha256=sha256_payload(authorization),
            implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
        )
    assert events == [], f"authorization accessor attack reached {events}"


def test_producer_rejects_response_id_subclass_before_measurement(monkeypatch):
    from scripts.benchmarks.v201 import cli

    prepared, bank = _prepared_fixture()
    witness = _witness()
    response_ids = [_AlwaysEqualStr(value) for value in witness["response_ids"]]
    events = []
    scorer = resource_probe._score_snpxsnp_bound_inputs
    measure = cli._measure_comparator_operation

    def record_score(*args, **kwargs):
        events.append("scorer")
        return scorer(*args, **kwargs)

    def record_measurement(*args, **kwargs):
        events.append("measurement")
        return measure(*args, **kwargs)

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", record_score)
    monkeypatch.setattr(cli, "_measure_comparator_operation", record_measurement)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    with pytest.raises(ValueError, match="response IDs"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared, bank, response_ids, response_width=1,
            design_hash="a" * 64, context_fingerprint="b" * 64,
            authorization_payload=witness["authorization_payload"],
            authorization_sha256=witness["authorization_sha256"],
            implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
        )
    assert events == [], f"response ID subclass reached {events}"


@pytest.mark.parametrize("guarded_field", ["scores", "expanded"])
def test_producer_reads_prepared_scoring_fields_once(monkeypatch, guarded_field):
    prepared, bank = _prepared_fixture()
    witness = _witness()
    proxy = _SingleReadPrepared(prepared, guarded_field)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    artifact = resource_probe.produce_snpxsnp_resource_probe(
        proxy, bank, witness["response_ids"], response_width=1,
        design_hash="a" * 64, context_fingerprint="b" * 64,
        authorization_payload=witness["authorization_payload"],
        authorization_sha256=witness["authorization_sha256"],
        implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
    )
    assert artifact["record"]["response_width"] == 1
    assert proxy.reads[guarded_field] == 1


@pytest.mark.parametrize("invalid", ["response_width", "response_bank"])
def test_invalid_response_inputs_precede_prepared_access_and_pair_ceiling(
    monkeypatch, invalid,
):
    _prepared, bank = _prepared_fixture()
    resource_limit_calls = []

    def forbidden_resource_limit(*_args, **_kwargs):
        resource_limit_calls.append("resource_limit")
        raise AssertionError("invalid response input reached resource limit")

    monkeypatch.setattr(
        resource_probe, "comparator_resource_limit", forbidden_resource_limit,
    )
    response_width = 2 if invalid == "response_width" else 1
    response_bank = bank if invalid == "response_width" else bank.astype(np.float32)
    expected = "width" if invalid == "response_width" else "response bank"
    with pytest.raises(ValueError, match=expected):
        resource_probe.produce_snpxsnp_resource_probe(
            _PreparedAccessForbidden(),
            response_bank,
            [f"response-{index:02d}" for index in range(20)],
            response_width=response_width,
            design_hash="a" * 64,
            context_fingerprint="b" * 64,
            authorization_payload={},
            authorization_sha256="0" * 64,
            implementation_commit="1" * 40,
            matched_comparator_contract_sha256="2" * 64,
        )
    assert resource_limit_calls == []


def test_producer_pins_context_identity_for_unmocked_fingerprint():
    prepared, bank = _prepared_fixture()
    witness = _witness()
    fingerprint = resource_probe._context_fingerprint(prepared.context)
    witness["authorization_payload"]["panel_context"]["context_fingerprint"] = fingerprint
    witness["authorization_sha256"] = sha256_payload(witness["authorization_payload"])
    proxy = _ContextReadOnce(prepared.context)
    prepared.context = proxy
    artifact = resource_probe.produce_snpxsnp_resource_probe(
        prepared, bank, witness["response_ids"], response_width=1,
        design_hash="a" * 64, context_fingerprint=fingerprint,
        authorization_payload=witness["authorization_payload"],
        authorization_sha256=witness["authorization_sha256"],
        implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
    )
    assert artifact["record"]["context_fingerprint"] == fingerprint
    assert proxy.reads == {"family": 1, "panel_id": 1, "sample_context": 1}


def test_producer_uses_local_resource_limit_not_context_pair_helper(monkeypatch):
    prepared, bank = _prepared_fixture()
    witness = _witness()
    original_limit = resource_probe.comparator_resource_limit
    original_score = resource_probe._score_snpxsnp_bound_inputs
    observed = []

    def one_pair_limit(*args, **kwargs):
        return replace(original_limit(*args, **kwargs), max_offered_pairs=1)

    def record_score(*args, **kwargs):
        observed.append(kwargs["max_offered_pairs"])
        return original_score(*args, **kwargs)

    monkeypatch.setattr(resource_probe, "comparator_resource_limit", one_pair_limit)
    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", record_score)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    resource_probe.produce_snpxsnp_resource_probe(
        prepared, bank, witness["response_ids"], response_width=1,
        design_hash="a" * 64, context_fingerprint="b" * 64,
        authorization_payload=witness["authorization_payload"],
        authorization_sha256=witness["authorization_sha256"],
        implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
    )
    assert observed == [1]


@pytest.mark.parametrize(
    "field",
    [
        "panel_id", "expected_sha256", "implementation_commit",
        "matched_comparator_contract_sha256", "design_hash", "context_fingerprint",
        "prepared_design_sha256", "input_family_sha256", "response_bank_sha256",
        "response_ids_sha256",
    ],
)
def test_validate_authorization_requires_exact_expected_identifiers(field):
    prepared, bank = _prepared_fixture()
    witness = _witness()
    values = {
        "expected_sha256": witness["authorization_sha256"],
        "panel_id": "REALG.CGVD1245",
        "implementation_commit": "1" * 40,
        "matched_comparator_contract_sha256": "2" * 64,
        "design_hash": "a" * 64,
        "context_fingerprint": "b" * 64,
        "prepared_design_sha256": prepared.scores.prepared_design_sha256,
        "input_family_sha256": _input_family_sha256(prepared),
        "response_bank_sha256": witness["authorization_payload"]["panel_context"]["response_bank_sha256"],
        "response_ids_sha256": witness["authorization_payload"]["panel_context"]["response_ids_sha256"],
    }
    values[field] = _AlwaysEqualStr("f" * len(values[field]))
    with pytest.raises(ValueError, match="authorization"):
        resource_probe._validate_authorization(
            witness["authorization_payload"], response_width=1, **values,
        )


def test_context_fingerprint_subclass_rejected_before_fingerprinting(monkeypatch):
    prepared, bank = _prepared_fixture()
    witness = _witness()

    def forbidden_fingerprint(*_args, **_kwargs):
        raise AssertionError("subclass context fingerprint reached fingerprinting")

    monkeypatch.setattr(resource_probe, "_context_fingerprint", forbidden_fingerprint)
    with pytest.raises(ValueError, match="design identity"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared, bank, witness["response_ids"], response_width=1,
            design_hash="a" * 64, context_fingerprint=_AlwaysEqualStr("b" * 64),
            authorization_payload=witness["authorization_payload"],
            authorization_sha256=witness["authorization_sha256"],
            implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
        )


def test_artifact_validator_rejects_accessor_split_view(monkeypatch):
    artifact = _produce(monkeypatch)
    stored = copy.deepcopy(artifact)
    stored["record"]["panel_id"] = "ATTACK"
    attack = _ArtifactAccessorAttack(stored, artifact["record"])
    with pytest.raises(ValueError):
        resource_probe.validate_snpxsnp_resource_probe_artifact(attack, **_witness())


def test_artifact_validator_rejects_score_evidence_accessor_split_view(monkeypatch):
    artifact = _produce(monkeypatch)
    stored = copy.deepcopy(artifact)
    stored["score_evidence"]["panel_id"] = "ATTACK"
    attack = _ArtifactEvidenceAccessorAttack(stored, artifact["score_evidence"])
    with pytest.raises(ValueError):
        resource_probe.validate_snpxsnp_resource_probe_artifact(attack, **_witness())


def test_artifact_validator_rejects_str_subclass_fields(monkeypatch):
    artifact = _produce(monkeypatch)
    attack = copy.deepcopy(artifact)
    attack["record"]["score_evidence_sha256"] = _AlwaysEqualStr("f" * 64)
    with pytest.raises(ValueError):
        resource_probe.validate_snpxsnp_resource_probe_artifact(attack, **_witness())


def test_producer_private_core_never_receives_live_scores(monkeypatch):
    prepared, bank = _prepared_fixture()
    witness = _witness()
    scorer = resource_probe._score_snpxsnp_bound_inputs

    def assert_private_core(*args, **kwargs):
        assert all(argument is not prepared.scores for argument in args)
        return scorer(*args, **kwargs)

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", assert_private_core)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    resource_probe.produce_snpxsnp_resource_probe(
        prepared, bank, witness["response_ids"], response_width=1,
        design_hash="a" * 64, context_fingerprint="b" * 64,
        authorization_payload=witness["authorization_payload"],
        authorization_sha256=witness["authorization_sha256"],
        implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
    )


def test_authorization_v1_is_rejected_before_scoring(monkeypatch):
    prepared, bank = _prepared_fixture()
    witness = _witness()
    witness["authorization_payload"]["schema"] = (
        "homoeogwas-snpxsnp-resource-probe-authorization-v1"
    )
    witness["authorization_sha256"] = sha256_payload(witness["authorization_payload"])
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("authorization-v1 reached the scorer")

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", forbidden)
    with pytest.raises(ValueError, match="authorization"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared, bank, witness["response_ids"], response_width=1,
            design_hash="a" * 64, context_fingerprint="b" * 64,
            authorization_payload=witness["authorization_payload"],
            authorization_sha256=witness["authorization_sha256"],
            implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
        )
    assert called is False


def test_authorization_v2_admits_the_exact_live_input_family(monkeypatch):
    prepared, bank = _prepared_fixture()
    witness = _witness()
    witness["authorization_payload"]["schema"] = (
        "homoeogwas-snpxsnp-resource-probe-authorization-v2"
    )
    witness["authorization_sha256"] = sha256_payload(witness["authorization_payload"])
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    artifact = resource_probe.produce_snpxsnp_resource_probe(
        prepared, bank, witness["response_ids"], response_width=1,
        design_hash="a" * 64, context_fingerprint="b" * 64,
        authorization_payload=witness["authorization_payload"],
        authorization_sha256=witness["authorization_sha256"],
        implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
    )
    assert artifact["record"]["input_family_sha256"] == _input_family_sha256(prepared)


@pytest.mark.parametrize("invalid", ["missing", None, "F" * 64, 123])
def test_producer_requires_authorized_input_family_before_scoring(monkeypatch, invalid):
    prepared, bank = _prepared_fixture()
    witness = _witness()
    authorization = witness["authorization_payload"]
    if invalid == "missing":
        del authorization["panel_context"]["input_family_sha256"]
    else:
        authorization["panel_context"]["input_family_sha256"] = invalid
    witness["authorization_sha256"] = sha256_payload(authorization)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("scoring started before authorization validation")

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", forbidden)
    with pytest.raises(ValueError, match="authorization"):
        resource_probe.produce_snpxsnp_resource_probe(
            prepared, bank, witness["response_ids"], response_width=1,
            design_hash="a" * 64, context_fingerprint="b" * 64,
            authorization_payload=authorization,
            authorization_sha256=witness["authorization_sha256"],
            implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
        )
    assert called is False


def test_producer_scores_private_bank_despite_caller_mutate_restore(monkeypatch):
    prepared, bank = _prepared_fixture()
    original = bank.copy()
    witness = _witness()
    scorer = resource_probe._score_snpxsnp_bound_inputs
    expected = comparator_module.score_snpxsnp_family(
        prepared.scores, prepared.context.family, prepared.expanded,
        prepared.gene_blocks, original, max_offered_pairs=10000,
    )

    def mutate_score_restore(*args, **kwargs):
        try:
            bank[:] = np.random.default_rng(552).normal(size=bank.shape)
            return scorer(*args, **kwargs)
        finally:
            bank[:] = original

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", mutate_score_restore)
    monkeypatch.setattr(resource_probe, "_context_fingerprint", lambda _c: "b" * 64)
    artifact = resource_probe.produce_snpxsnp_resource_probe(
        prepared, bank, witness["response_ids"], response_width=20,
        design_hash="a" * 64, context_fingerprint="b" * 64,
        authorization_payload=witness["authorization_payload"],
        authorization_sha256=witness["authorization_sha256"],
        implementation_commit="1" * 40, matched_comparator_contract_sha256="2" * 64,
    )
    np.testing.assert_array_equal(bank, original)
    assert artifact["score_evidence"]["response_bank_identity"][
        "array_sha256"
    ] == _array_sha256(original)
    np.testing.assert_array_equal(artifact["score_evidence"]["group_p"], expected.group_p)


def test_artifact_validator_freezes_external_bank_before_identity_checks(monkeypatch):
    artifact = _produce(monkeypatch, 20)
    witness = _witness()
    bank = witness["response_bank"]
    original = bank.copy()
    identify = resource_probe._response_identities

    def mutate_identify_restore(*args, **kwargs):
        try:
            bank[:] += 1.0
            return identify(*args, **kwargs)
        finally:
            bank[:] = original

    monkeypatch.setattr(resource_probe, "_response_identities", mutate_identify_restore)
    validated = resource_probe.validate_snpxsnp_resource_probe_artifact(artifact, **witness)
    assert validated["record"] == artifact["record"]


def test_artifact_series_freezes_one_bank_before_auditing_any_width(monkeypatch):
    from scripts.benchmarks.v201 import audit

    artifacts = _artifact_series(monkeypatch)
    witness = _witness()
    bank = witness["response_bank"]
    original = bank.copy()
    reconstruct = audit._snpxsnp_result_from_evidence

    def reconstruct_then_mutate(*args, **kwargs):
        result = reconstruct(*args, **kwargs)
        bank[:] += 1.0
        return result

    monkeypatch.setattr(audit, "_snpxsnp_result_from_evidence", reconstruct_then_mutate)
    try:
        projection = resource_probe.validate_snpxsnp_resource_probe_artifact_series(
            artifacts, **witness,
        )
    finally:
        bank[:] = original
    assert projection["accepted"] is True


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
    witness = _witness()
    witness["authorization_payload"]["panel_context"]["input_family_sha256"] = raw[
        "input_family_sha256"
    ]
    witness["authorization_sha256"] = sha256_payload(witness["authorization_payload"])
    artifact["record"]["probe_authorization_sha256"] = witness["authorization_sha256"]
    with pytest.raises(ValueError, match="input.block|marker count"):
        resource_probe.validate_snpxsnp_resource_probe_artifact(artifact, **witness)


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
            resource_probe, "_validate_snpxsnp_resource_probe_artifact_snapshot",
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
        "input_family_sha256", "response_bank_sha256", "response_ids_sha256", "response_widths",
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
    from scripts.benchmarks.v201.audit import BenchmarkAuditError

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
    expected_error = BenchmarkAuditError if mutation == "input_family" else ValueError
    with pytest.raises(expected_error):
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

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", forbidden)
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

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", forbidden)
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

    monkeypatch.setattr(resource_probe, "_score_snpxsnp_bound_inputs", forbidden)
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
