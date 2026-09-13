from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class CLIError(RuntimeError):
    """The requested orchestration stage is not prospectively authorized."""


SUCCESSOR_DESIGN_SHA256 = (
    "f2dfacc5181c504b420fa09835e32cea4c4229e28188a8d87533d4c778dda586"
)
WORKER_DECISION_SHA256 = (
    "ade451d2eaef49530011d1558d042bbce3fc2c2d3f6e9cea78c2bcb244c4e0e0"
)
MATERIALIZATION_AUTHORITY_PATH = Path(
    "/mnt/7302share/fast_ysp/U7_GWAS/tasks/BM-NATIVE-QA/"
    "QA-NJOBS128-RESPONSE-MATERIALIZATION-AUTHORIZATION-v1-20260913.yaml"
)


def _ensure_repo_import_root() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bm-native-qa")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--amendment", type=Path, required=True)
    plan.add_argument("--context-evidence", type=Path, required=True)
    plan.add_argument("--fixture-manifest", type=Path, required=True)
    plan.add_argument("--successor-design", type=Path)
    plan.add_argument("--out", type=Path, required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--authority", type=Path)
    materialize.add_argument("--amendment", type=Path)
    materialize.add_argument("--out", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--amendment", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CLIError(f"expected a YAML mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CLIError(f"expected a JSON object: {path}")
    return value


def _runner_test_sha256s(task_root: Path) -> dict[str, str]:
    from .authority import enumerate_runner_test_sources
    from .identity import sha256_file

    members = enumerate_runner_test_sources(task_root)
    return {
        relative: sha256_file(members[relative]) for relative in sorted(members)
    }


def _seed_derivations(
    inventory,
    *,
    run_namespace: str = "qa_real80_v4",
) -> list[dict[str, object]]:
    derivations: list[dict[str, object]] = []
    responses_by_context = {
        context.key: tuple(
            response
            for response in inventory.responses
            if response.context_key == context.key
        )
        for context in inventory.contexts
    }
    for context in inventory.contexts:
        base = f"{run_namespace}.{context.panel_id}.{context.sample_context}"
        derivations.append(
            {
                "purpose": "anchor",
                "context_key": context.key,
                "truth_id": None,
                "track": "omnib",
                "scenario_id": base,
                "replicate_index": 0,
                "stage_or_role": "pilot:qa_anchor",
            }
        )
        for response in responses_by_context[context.key]:
            for purpose, stage_or_role in (
                ("observed", "pilot:qa_observed"),
                ("bootstrap", "pilot:native_bootstrap"),
            ):
                derivations.append(
                    {
                        "purpose": purpose,
                        "context_key": context.key,
                        "truth_id": response.truth_id,
                        "track": "omnib",
                        "scenario_id": response.response_id,
                        "replicate_index": 0,
                        "stage_or_role": stage_or_role,
                    }
                )
    return derivations


def _semantic_invocations(
    inventory,
    design_payload: dict[str, object],
) -> list[dict[str, object]]:
    layout = design_payload["future_artifacts"]
    assert isinstance(layout, dict)
    invocation_paths = {
        row["invocation_id"]: row for row in layout["invocations"]
    }
    response_paths = {row["response_id"]: row for row in layout["responses"]}
    contexts = {context.key: context for context in inventory.contexts}
    responses = {response.response_id: response for response in inventory.responses}
    rows = []
    for invocation in inventory.invocations:
        response = responses[invocation.response_id]
        context = contexts[response.context_key]
        rows.append(
            {
                "invocation_id": invocation.invocation_id,
                "response_id": response.response_id,
                "context_key": context.key,
                "panel_id": context.panel_id,
                "sample_context": context.sample_context,
                "truth_id": response.truth_id,
                "jobs": invocation.jobs,
                "feature_seed": context.feature_seed,
                "subgenomes": list(context.subgenomes),
                "groups": str(context.groups_path),
                "genotype": dict(context.bed_prefixes),
                "snp_to_gene": dict(context.snp_to_gene),
                "phenotype_tsv": response_paths[response.response_id][
                    "phenotype_tsv"
                ],
                "config_yaml": invocation_paths[invocation.invocation_id][
                    "config_yaml"
                ],
                "checkpoint_root": invocation_paths[invocation.invocation_id][
                    "checkpoint_root"
                ],
                "result_dir": invocation_paths[invocation.invocation_id]["result_dir"],
                "audit_dir": invocation_paths[invocation.invocation_id]["audit_dir"],
                "canonical_interact": design_payload["canonical_interact"],
            }
        )
    return rows


def _validate_amendment_acceptance(amendment: dict[str, Any], path: Path) -> None:
    from .identity import sha256_file

    if amendment.get("status") != "accepted_for_runner_config_TDD_only":
        raise CLIError("amendment status is not accepted for runner/config TDD")
    review = amendment.get("independent_review")
    if not isinstance(review, dict) or review.get("status") != (
        "ACCEPT_FOR_RUNNER_CONFIG_TDD_ONLY"
    ):
        raise CLIError("amendment independent review is not accepted")
    staged = review.get("staged_materialization_delta")
    if not isinstance(staged, dict) or staged.get("verdict") != "ACCEPT":
        raise CLIError("amendment staged-materialization review is not accepted")
    if path.parent.parent.name != "tasks":
        raise CLIError("accepted amendment must reside below the project tasks directory")
    project_root = path.parent.parent.parent
    for label, binding in (
        ("independent review", review.get("report")),
        ("staged-materialization review", staged),
    ):
        if not isinstance(binding, dict):
            raise CLIError(f"{label} binding is missing")
        report_path = Path(str(binding.get("path", "")))
        if not report_path.is_absolute():
            report_path = project_root / report_path
        if sha256_file(report_path) != binding.get("sha256"):
            raise CLIError(f"{label} report differs from its accepted binding")


def _prospective_input_hashes(
    inventory,
    context_evidence: dict[str, Any],
    project_root: Path,
) -> dict[str, str]:
    from .identity import sha256_file
    from .plan import declared_context_input_paths

    hashes: dict[str, str] = {}
    for context in inventory.contexts:
        for declared in declared_context_input_paths(context, context_evidence):
            path = Path(declared)
            resolved = path if path.is_absolute() else project_root / path
            observed = sha256_file(resolved)
            previous = hashes.setdefault(declared, observed)
            if previous != observed:
                raise CLIError(f"prospective input changed while hashing: {declared}")
    return hashes


def _write_plan(args: argparse.Namespace) -> None:
    if args.successor_design is None:
        _ensure_repo_import_root()
    from .identity import freeze_identity, freeze_successor_identity, sha256_file
    from .plan import bind_context_inputs, build_inventory, build_successor_inventory

    amendment = _load_yaml(args.amendment)
    _validate_amendment_acceptance(amendment, args.amendment)
    context_evidence = _load_json(args.context_evidence)
    fixture_manifest = _load_json(args.fixture_manifest)
    expected_context_hash = amendment["normative_bindings"]["context_evidence"][
        "sha256"
    ]
    expected_fixture = amendment["normative_bindings"]["accepted_fixture"]
    context_hash = sha256_file(args.context_evidence)
    fixture_hash = sha256_file(args.fixture_manifest)
    if context_hash != expected_context_hash:
        raise CLIError("context evidence differs from the accepted amendment binding")
    if fixture_hash != expected_fixture["sha256"]:
        raise CLIError("fixture manifest differs from the accepted amendment binding")
    if (
        fixture_manifest.get("source_commit") != expected_fixture["source_commit"]
        or fixture_manifest.get("source_tree") != expected_fixture["source_tree"]
    ):
        raise CLIError("fixture source identity differs from the accepted binding")
    if context_evidence.get("phenotype_values_read") is not False:
        raise CLIError("context evidence is not phenotype-blind")

    task_root = Path(__file__).resolve().parents[1]
    runner_test_sha256s = _runner_test_sha256s(task_root)
    project_root = args.amendment.parent.parent.parent
    successor = None
    successor_hash = None
    decision_hash = None
    run_namespace = "qa_real80_v4"
    if args.successor_design is None:
        inventory = build_inventory(amendment)
    else:
        successor_hash = sha256_file(args.successor_design)
        if successor_hash != SUCCESSOR_DESIGN_SHA256:
            raise CLIError("successor design differs from the independently accepted bytes")
        successor = _load_yaml(args.successor_design)
        superseded = successor["supersedes_for_future_materialization"]
        base_binding = superseded["base_amendment"]
        if sha256_file(args.amendment) != base_binding.get("sha256"):
            raise CLIError("amendment differs from the successor design binding")
        decision = successor["authority"]["global_worker_override"]
        decision_path = Path(str(decision["decision_path"]))
        if not decision_path.is_absolute():
            decision_path = project_root / decision_path
        decision_hash = sha256_file(decision_path)
        if (
            decision_hash != WORKER_DECISION_SHA256
            or decision_hash != decision.get("decision_sha256")
        ):
            raise CLIError("worker decision differs from the successor design binding")
        run_namespace = str(successor["successor_identity"]["run_namespace"])
        inventory = build_successor_inventory(amendment, successor)
    input_file_sha256s = _prospective_input_hashes(
        inventory,
        context_evidence,
        project_root,
    )
    inventory = bind_context_inputs(
        inventory,
        context_evidence,
        input_file_sha256s,
    )
    if successor is None:
        frozen = freeze_identity(
            inventory,
            fixture_manifest_sha256=fixture_hash,
            amendment_sha256=sha256_file(args.amendment),
            runner_test_sha256s=runner_test_sha256s,
        )
        schema = "homoeogwas-bm-native-qa-prospective-inventory-v1"
    else:
        artifact_root = Path(str(successor["successor_identity"]["artifact_root"]))
        frozen = freeze_successor_identity(
            inventory,
            fixture_manifest_sha256=fixture_hash,
            amendment_sha256=sha256_file(args.amendment),
            successor_design_sha256=str(successor_hash),
            worker_decision_sha256=str(decision_hash),
            runner_test_sha256s=runner_test_sha256s,
            artifact_root=artifact_root,
        )
        schema = str(successor["successor_identity"]["prospective_inventory_schema"])
    design_payload = dict(frozen.design_payload)
    future_artifacts = design_payload["future_artifacts"]
    assert isinstance(future_artifacts, dict)
    payload = {
        "schema": schema,
        "status": "prospective_only_no_generated_values",
        "response_materialization_authorized": False,
        "execution_authorized": False,
        "qa_design_hash": frozen.qa_design_hash,
        "counts": {
            "anchors": len(future_artifacts["anchors"]),
            "contexts": len(inventory.contexts),
            "responses": len(inventory.responses),
            "invocations": len(inventory.invocations),
        },
        "source_bindings": {
            "amendment_sha256": sha256_file(args.amendment),
            "context_evidence_sha256": context_hash,
            "fixture_manifest_sha256": fixture_hash,
            "fixture_source_commit": fixture_manifest["source_commit"],
            "fixture_source_tree": fixture_manifest["source_tree"],
        },
        "runner_test_sha256s": runner_test_sha256s,
        "design_payload": design_payload,
        "seed_derivations": _seed_derivations(
            inventory,
            run_namespace=run_namespace,
        ),
        "semantic_invocations": _semantic_invocations(inventory, design_payload),
    }
    if successor_hash is not None and decision_hash is not None:
        payload["source_bindings"].update(
            {
                "successor_design_sha256": successor_hash,
                "worker_decision_sha256": decision_hash,
            }
        )
    text = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    try:
        with args.out.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise CLIError(f"prospective inventory target already exists: {args.out}") from exc


def _reject_closed_stage(args: argparse.Namespace) -> None:
    if args.amendment is None:
        raise CLIError("materialize requires the exact reviewed authority file")
    amendment = _load_yaml(args.amendment)
    flag = (
        "response_materialization_authorized"
        if args.command == "materialize"
        else "execution_authorized"
    )
    if amendment.get(flag) is not True:
        raise CLIError(f"{flag}=false")
    raise CLIError(f"{args.command} stage lacks a separately reviewed activation adapter")


def _verify_materialization_cli(args: argparse.Namespace):
    from .authority import AuthorityBlocked, verify_materialization_authority

    if args.authority is None:
        _reject_closed_stage(args)
    try:
        return verify_materialization_authority(
            args.authority,
            expected_authority_path=MATERIALIZATION_AUTHORITY_PATH,
            task_root=Path(__file__).resolve().parents[1],
            expected_successor_design_sha256=SUCCESSOR_DESIGN_SHA256,
            expected_worker_decision_sha256=WORKER_DECISION_SHA256,
        )
    except AuthorityBlocked as exc:
        raise CLIError(str(exc)) from exc


def _require_rehydrated_identity(
    frozen,
    *,
    qa_design_hash: str,
    inventory_payload: object,
) -> None:
    from .authority import _mapping_sha256

    if (
        frozen.qa_design_hash != qa_design_hash
        or not isinstance(inventory_payload, dict)
        or _mapping_sha256(frozen.design_payload)
        != _mapping_sha256(inventory_payload)
    ):
        raise CLIError("rehydrated successor identity differs from frozen inventory")


def _activate_materialization(args: argparse.Namespace, verified) -> None:
    """Import numerical code only after the complete authority rehash passes."""

    from .authority import verify_loaded_numerical_origins
    from .bundle import make_real_context_loader, materialize_bundle
    from .identity import freeze_successor_identity, sha256_file
    from .plan import bind_context_inputs, build_successor_inventory

    verify_loaded_numerical_origins(verified)

    project_root = MATERIALIZATION_AUTHORITY_PATH.parent.parent.parent
    amendment_path = (
        project_root
        / "tasks/BM-NATIVE-QA/QA-EXECUTION-AMENDMENT-v1-20260910.yaml"
    )
    context_evidence_path = (
        project_root
        / "tasks/BM-INPUTS/staging/real-core-v4-primary80-estimability/manifest.json"
    )
    fixture_manifest_path = (
        project_root / "tasks/BM-FIXTURE-V3/manifest.20260910-v5-r3.json"
    )
    successor_path = verified.paths["successor_design"]
    amendment = _load_yaml(amendment_path)
    _validate_amendment_acceptance(amendment, amendment_path)
    successor = _load_yaml(successor_path)
    context_evidence = _load_json(context_evidence_path)
    fixture_manifest = _load_json(fixture_manifest_path)
    inventory = build_successor_inventory(amendment, successor)
    input_hashes = _prospective_input_hashes(
        inventory,
        context_evidence,
        project_root,
    )
    inventory = bind_context_inputs(inventory, context_evidence, input_hashes)
    frozen = freeze_successor_identity(
        inventory,
        fixture_manifest_sha256=sha256_file(fixture_manifest_path),
        amendment_sha256=sha256_file(amendment_path),
        successor_design_sha256=SUCCESSOR_DESIGN_SHA256,
        worker_decision_sha256=WORKER_DECISION_SHA256,
        runner_test_sha256s=verified.inventory["runner_test_sha256s"],
        artifact_root=verified.artifact_root,
    )
    _require_rehydrated_identity(
        frozen,
        qa_design_hash=verified.qa_design_hash,
        inventory_payload=verified.inventory.get("design_payload"),
    )
    if args.out.resolve() != verified.artifact_root.resolve():
        raise CLIError("materialization --out differs from the reviewed artifact root")
    expected_fixture = verified.inventory["source_bindings"]
    if (
        fixture_manifest.get("source_commit")
        != expected_fixture.get("fixture_source_commit")
        or fixture_manifest.get("source_tree")
        != expected_fixture.get("fixture_source_tree")
    ):
        raise CLIError("fixture source identity differs during materialization")
    loader = make_real_context_loader(
        project_root=project_root,
        context_evidence=context_evidence,
    )
    materialize_bundle(
        verified,
        inventory=inventory,
        seeds=frozen.seeds,
        context_loader=loader,
        context_evidence=context_evidence,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            _write_plan(args)
        elif args.command == "materialize" and args.authority is not None:
            verified = _verify_materialization_cli(args)
            from .authority import verify_runtime_import_isolation

            verify_runtime_import_isolation(
                verified,
                task_root=Path(__file__).resolve().parents[1],
            )
            _activate_materialization(args, verified)
        else:
            _reject_closed_stage(args)
    except (CLIError, FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        print(f"ERROR: bm-native-qa: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
