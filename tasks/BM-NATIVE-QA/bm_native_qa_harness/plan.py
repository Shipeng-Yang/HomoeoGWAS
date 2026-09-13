from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


class PlanError(RuntimeError):
    """The prospective plan violates its accepted execution boundary."""


@dataclass(frozen=True)
class ContextSpec:
    key: str
    panel_id: str
    sample_context: str
    feature_seed: int
    groups_path: Path
    groups_sha256: str
    samples_path: Path
    samples_sha256: str
    expected_calibrated_groups: int
    subgenomes: tuple[str, ...] = ()
    bed_prefixes: tuple[tuple[str, str], ...] = ()
    snp_to_gene: tuple[tuple[str, str], ...] = ()
    input_file_sha256s: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ResponseSpec:
    response_id: str
    context_key: str
    truth_id: str
    generator_scale_interaction_pve: float


@dataclass(frozen=True)
class InvocationSpec:
    invocation_id: str
    response_id: str
    panel_id: str
    sample_context: str
    truth_id: str
    jobs: int


@dataclass(frozen=True)
class ProspectiveInventory:
    contexts: tuple[ContextSpec, ...]
    responses: tuple[ResponseSpec, ...]
    invocations: tuple[InvocationSpec, ...]


def bind_context_inputs(
    inventory: ProspectiveInventory,
    context_evidence: dict[str, Any],
    input_file_sha256s: dict[str, str] | None = None,
) -> ProspectiveInventory:
    evidence_contexts = context_evidence["contexts"]
    contexts: list[ContextSpec] = []
    for context in inventory.contexts:
        copies = evidence_contexts[context.key]["copies"]
        subgenomes = tuple(str(label) for label in copies)
        declared_paths = declared_context_input_paths(context, context_evidence)
        bound_hashes: tuple[tuple[str, str], ...] = ()
        if input_file_sha256s is not None:
            missing = [path for path in declared_paths if path not in input_file_sha256s]
            if missing:
                raise PlanError(f"missing prospective input hash: {missing[0]}")
            bound_hashes = tuple(
                (path, str(input_file_sha256s[path])) for path in declared_paths
            )
            hashes = dict(bound_hashes)
            if hashes[str(context.groups_path)] != context.groups_sha256:
                raise PlanError("prospective groups hash differs from amendment")
            if hashes[str(context.samples_path)] != context.samples_sha256:
                raise PlanError("prospective samples hash differs from amendment")
        contexts.append(
            replace(
                context,
                subgenomes=subgenomes,
                bed_prefixes=tuple(
                    (label, str(Path(copies[label]["bed"]).with_suffix("")))
                    for label in subgenomes
                ),
                snp_to_gene=tuple(
                    (label, str(copies[label]["mapping"])) for label in subgenomes
                ),
                input_file_sha256s=bound_hashes,
            )
        )
    return replace(inventory, contexts=tuple(contexts))


def declared_context_input_paths(
    context: ContextSpec,
    context_evidence: dict[str, Any],
) -> tuple[str, ...]:
    copies = context_evidence["contexts"][context.key]["copies"]
    paths = [str(context.groups_path), str(context.samples_path)]
    for label in copies:
        bed = Path(copies[label]["bed"])
        if bed.suffix != ".bed":
            raise PlanError(f"context BED path lacks .bed suffix: {bed}")
        paths.extend(
            (
                str(bed),
                str(bed.with_suffix(".bim")),
                str(bed.with_suffix(".fam")),
                str(copies[label]["mapping"]),
            )
        )
    if len(paths) != len(set(paths)):
        raise PlanError(f"duplicate declared input path within context {context.key}")
    return tuple(paths)


def _context_spec(key: str, record: dict[str, Any]) -> ContextSpec:
    return ContextSpec(
        key=key,
        panel_id=str(record["panel_id"]),
        sample_context=key.split(".", 1)[1],
        feature_seed=int(record["feature_seed"]),
        groups_path=Path(record["groups_path"]),
        groups_sha256=str(record["groups_sha256"]),
        samples_path=Path(record["samples_path"]),
        samples_sha256=str(record["samples_sha256"]),
        expected_calibrated_groups=int(record["expected_calibrated_groups"]),
    )


def _response_id(panel_id: str, sample_context: str, truth_id: str) -> str:
    return f"qa_real80_v4.{panel_id}.{sample_context}.{truth_id}"


def build_inventory(amendment: dict[str, Any]) -> ProspectiveInventory:
    if amendment.get("execution_authorized") is not False:
        raise PlanError("execution_authorized must remain false during runner TDD")
    if amendment.get("response_materialization_authorized") is not False:
        raise PlanError(
            "response_materialization_authorized must remain false during runner TDD"
        )
    matrix = amendment["matrix"]
    context_jobs = tuple(
        (str(row["sample_context"]), tuple(int(job) for job in row["jobs"]))
        for row in matrix["context_job_rows"]
    )
    if context_jobs != (
        ("pc1_spread_192", (1, 4)),
        ("seeded_random_192", (4,)),
        ("holdout_192", (4,)),
    ):
        raise PlanError("context/jobs differ from the frozen context/job matrix")
    if tuple(amendment["worker_invariance_pair"]["contexts"]) != (
        "cotton.pc1_spread_192",
        "wheat.pc1_spread_192",
    ):
        raise PlanError("worker-invariance contexts differ from the frozen pair scope")
    raw_contexts = amendment["context_bindings"]["contexts"]
    contexts_by_identity = {
        (record["panel_id"], key.split(".", 1)[1]): _context_spec(key, record)
        for key, record in raw_contexts.items()
    }
    contexts = tuple(
        contexts_by_identity[(panel_id, row["sample_context"])]
        for panel_id in matrix["panels"]
        for row in matrix["context_job_rows"]
    )
    responses = tuple(
        ResponseSpec(
            response_id=_response_id(context.panel_id, context.sample_context, truth["id"]),
            context_key=context.key,
            truth_id=str(truth["id"]),
            generator_scale_interaction_pve=float(
                truth["generator_scale_interaction_pve"]
            ),
        )
        for context in contexts
        for truth in matrix["truths"]
    )
    invocations = tuple(
        InvocationSpec(
            invocation_id=(
                f"{_response_id(panel_id, row['sample_context'], truth['id'])}.jobs{jobs}"
            ),
            response_id=_response_id(panel_id, row["sample_context"], truth["id"]),
            panel_id=str(panel_id),
            sample_context=str(row["sample_context"]),
            truth_id=str(truth["id"]),
            jobs=int(jobs),
        )
        for panel_id in matrix["panels"]
        for truth in matrix["truths"]
        for row in matrix["context_job_rows"]
        for jobs in row["jobs"]
    )
    response_ids = [response.response_id for response in responses]
    invocation_ids = [invocation.invocation_id for invocation in invocations]
    if (
        len(set(response_ids)) != len(response_ids)
        or len(set(invocation_ids)) != len(invocation_ids)
    ):
        raise PlanError("duplicate prospective IDs are forbidden")
    if (
        len(contexts) != int(matrix["unique_contexts"])
        or len(responses) != int(matrix["unique_responses"])
        or len(invocations) != int(matrix["native_invocations"])
    ):
        raise PlanError("expanded inventory differs from declared matrix counts")
    return ProspectiveInventory(
        contexts=contexts,
        responses=responses,
        invocations=invocations,
    )


def build_successor_inventory(
    amendment: dict[str, Any],
    successor: dict[str, Any],
) -> ProspectiveInventory:
    """Replace only the obsolete worker-count invocation layer."""

    base = build_inventory(amendment)
    if successor.get("schema") != (
        "homoeogwas-bm-native-qa-njobs128-successor-design-v1"
    ):
        raise PlanError("successor design schema is invalid")
    if successor.get("execution_authorized") is not False:
        raise PlanError("successor execution_authorized must remain false")
    override = successor.get("invocation_override")
    identity = successor.get("successor_identity")
    if not isinstance(override, dict) or not isinstance(identity, dict):
        raise PlanError("successor invocation or identity block is missing")
    jobs = override.get("requested_jobs")
    if isinstance(jobs, bool) or jobs != 128:
        raise PlanError("successor requested_jobs must equal 128")
    rows = tuple(
        (
            str(row["sample_context"]),
            tuple(str(replica) for replica in row["replicas"]),
        )
        for row in override.get("context_rows", ())
    )
    expected_rows = (
        ("pc1_spread_192", ("replica_a", "replica_b")),
        ("seeded_random_192", ("primary",)),
        ("holdout_192", ("primary",)),
    )
    if rows != expected_rows:
        raise PlanError("successor context/replica rows differ from the reviewed design")
    namespace = identity.get("run_namespace")
    if namespace != "qa_real80_njobs128_v1":
        raise PlanError("successor run namespace differs from the reviewed design")

    contexts_by_identity = {
        (context.panel_id, context.sample_context): context for context in base.contexts
    }
    panels = tuple(str(panel) for panel in amendment["matrix"]["panels"])
    truths = tuple(amendment["matrix"]["truths"])
    contexts = tuple(
        contexts_by_identity[(panel_id, sample_context)]
        for panel_id in panels
        for sample_context, _replicas in rows
    )
    responses = tuple(
        ResponseSpec(
            response_id=f"{namespace}.{context.panel_id}.{context.sample_context}.{truth['id']}",
            context_key=context.key,
            truth_id=str(truth["id"]),
            generator_scale_interaction_pve=float(
                truth["generator_scale_interaction_pve"]
            ),
        )
        for context in contexts
        for truth in truths
    )
    response_by_key = {
        (response.context_key, response.truth_id): response for response in responses
    }
    invocations = tuple(
        InvocationSpec(
            invocation_id=(
                f"{response_by_key[(context.key, str(truth['id']))].response_id}"
                f".workers128.{replica}"
            ),
            response_id=response_by_key[
                (context.key, str(truth["id"]))
            ].response_id,
            panel_id=panel_id,
            sample_context=sample_context,
            truth_id=str(truth["id"]),
            jobs=128,
        )
        for panel_id in panels
        for truth in truths
        for sample_context, replicas in rows
        for context in (contexts_by_identity[(panel_id, sample_context)],)
        for replica in replicas
    )
    if (
        len(contexts) != 6
        or len(responses) != 12
        or len(invocations) != 16
        or len({row.invocation_id for row in invocations}) != 16
        or {row.jobs for row in invocations} != {128}
    ):
        raise PlanError("successor inventory differs from reviewed 6/12/16 workers128 design")
    return ProspectiveInventory(
        contexts=contexts,
        responses=responses,
        invocations=invocations,
    )
