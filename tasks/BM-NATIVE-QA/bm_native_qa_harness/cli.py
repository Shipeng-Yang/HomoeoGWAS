from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


class CLIError(RuntimeError):
    """The requested orchestration stage is not prospectively authorized."""


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
    plan.add_argument("--out", type=Path, required=True)
    for name in ("materialize", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--amendment", type=Path, required=True)
        command.add_argument("--out", type=Path, required=True)
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
    from .identity import sha256_file

    members = sorted(
        (*task_root.joinpath("bm_native_qa_harness").rglob("*.py"),)
        + (*task_root.joinpath("tests").rglob("*.py"),),
        key=lambda path: path.relative_to(task_root).as_posix(),
    )
    if not members:
        raise CLIError("runner/test source inventory is empty")
    return {
        path.relative_to(task_root).as_posix(): sha256_file(path) for path in members
    }


def _seed_derivations(inventory) -> list[dict[str, object]]:
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
        base = f"qa_real80_v4.{context.panel_id}.{context.sample_context}"
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
    _ensure_repo_import_root()
    from .identity import freeze_identity, sha256_file
    from .plan import bind_context_inputs, build_inventory

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
    inventory = build_inventory(amendment)
    project_root = args.amendment.parent.parent.parent
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
    frozen = freeze_identity(
        inventory,
        fixture_manifest_sha256=fixture_hash,
        amendment_sha256=sha256_file(args.amendment),
        runner_test_sha256s=runner_test_sha256s,
    )
    design_payload = dict(frozen.design_payload)
    future_artifacts = design_payload["future_artifacts"]
    assert isinstance(future_artifacts, dict)
    payload = {
        "schema": "homoeogwas-bm-native-qa-prospective-inventory-v1",
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
        "seed_derivations": _seed_derivations(inventory),
        "semantic_invocations": _semantic_invocations(inventory, design_payload),
    }
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
    amendment = _load_yaml(args.amendment)
    flag = (
        "response_materialization_authorized"
        if args.command == "materialize"
        else "execution_authorized"
    )
    if amendment.get(flag) is not True:
        raise CLIError(f"{flag}=false")
    raise CLIError(f"{args.command} stage lacks a separately reviewed activation adapter")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "plan":
            _write_plan(args)
        else:
            _reject_closed_stage(args)
    except (CLIError, FileNotFoundError, KeyError, ValueError) as exc:
        print(f"ERROR: bm-native-qa: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
