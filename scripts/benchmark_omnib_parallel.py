#!/usr/bin/env python3
"""Deterministic release benchmark for canonical omniB fork workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from homoeogwas_launcher import NUMERIC_THREAD_ENV


def evaluate_runs(runs: dict[int, dict]) -> dict:
    """Apply the frozen identity, process, CPU, speed and memory gates."""
    normalized = {int(jobs): dict(values) for jobs, values in runs.items()}
    if 1 not in normalized:
        raise ValueError("benchmark requires a one-job baseline")
    baseline = normalized[1]
    baseline_wall = float(baseline["wall_seconds"])
    baseline_memory = int(baseline["peak_rss_bytes"])
    baseline_hash = str(baseline["result_sha256"])
    if baseline_wall <= 0 or baseline_memory <= 0 or len(baseline_hash) != 64:
        raise ValueError("one-job benchmark baseline is invalid")

    evaluated = {}
    eligible = []
    for jobs in sorted(normalized):
        run = normalized[jobs]
        effective = int(run["effective_jobs"])
        pids = {int(pid) for pid in run["worker_pids"]}
        wall = float(run["wall_seconds"])
        memory = int(run["peak_rss_bytes"])
        result_identity = str(run["result_sha256"]) == baseline_hash
        worker_pid_gate = len(pids) == effective
        cpu_percent = float(run["cpu_percent"])
        cpu_gate = jobs == 1 or cpu_percent > 200.0
        oversubscription_gate = (
            jobs == 1 or cpu_percent <= (effective + 1) * 110.0)
        speed_gate = jobs == 1 or wall < baseline_wall
        memory_ratio = memory / baseline_memory
        memory_gate = memory_ratio <= 1.35
        accepted = (
            jobs > 1
            and result_identity
            and worker_pid_gate
            and cpu_gate
            and oversubscription_gate
            and speed_gate
            and memory_gate
        )
        enriched = dict(run)
        enriched.update({
            "result_identity": result_identity,
            "worker_pid_gate": worker_pid_gate,
            "cpu_gate": cpu_gate,
            "oversubscription_gate": oversubscription_gate,
            "speed_gate": speed_gate,
            "memory_ratio_vs_serial": memory_ratio,
            "memory_gate": memory_gate,
            "speedup_vs_serial": baseline_wall / wall,
            "accepted_candidate": accepted,
        })
        evaluated[str(jobs)] = enriched
        if accepted:
            eligible.append((wall, jobs))

    four_job_gate = bool(evaluated.get("4", {}).get("accepted_candidate"))
    selected = min(eligible)[1] if four_job_gate and eligible else None
    return {
        "schema": "homoeogwas-omnib-process-benchmark-v1",
        "accepted": four_job_gate,
        "selected_jobs": selected,
        "required_four_job_gate": four_job_gate,
        "memory_metric": "peak aggregate proportional resident bytes (PSS)",
        "runs": evaluated,
    }


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        array = np.ascontiguousarray(values)
        digest.update(array.dtype.str.encode())
        digest.update(b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _cpu_seconds() -> float:
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


def _child_run(
        jobs: int, *, n: int, groups: int, responses: int,
        marker: Path | None = None) -> dict:
    from homoeogwas.group_family import MasterGroupFamily
    from homoeogwas.interact import SubgenomeData
    from homoeogwas.omnib_family import (
        _prepare_checkpoint_omnib,
        score_omnib_null_indices,
    )

    rng = np.random.default_rng(20260828)
    snps_per_gene = 8
    subgenomes = ("A", "B", "D")
    subdata = {}
    for subgenome in subgenomes:
        dosage = rng.integers(
            0, 3, size=(n, groups * snps_per_gene), dtype=np.int8
        ).astype(float)
        mapping = {
            f"g{index}": np.arange(
                index * snps_per_gene, (index + 1) * snps_per_gene)
            for index in range(groups)
        }
        subdata[subgenome] = SubgenomeData(
            X=dosage,
            gene_snp=mapping,
            samples=[f"sample_{index}" for index in range(n)],
            chunk=None,
        )
    family = MasterGroupFamily(
        subgenomes=subgenomes,
        group_ids=tuple(f"group_{index}" for index in range(groups)),
        genes=tuple((f"g{index}",) * 3 for index in range(groups)),
    )
    phenotype = rng.normal(size=n)
    scores, expanded = _prepare_checkpoint_omnib(
        subdata,
        family,
        phenotype,
        np.arange(n),
        cap=150,
        n_pc=3,
        transform="INT",
        bootstrap_seed=2026,
        n_jobs=1,
        grm_method="grm_from_X",
        maf_min=0.01,
        burden_maf=0.01,
        min_snp=3,
        covariates=None,
    )

    cpu_start = _cpu_seconds()
    wall_start = time.perf_counter()
    with threadpool_limits(limits=1):
        if marker is not None:
            marker.write_text("score_started\n")
        edge_p, group_p, components = score_omnib_null_indices(
            scores,
            family,
            expanded,
            range(responses),
            base_seed=2026,
            n_jobs=jobs,
            return_components=True,
        )
        if marker is not None:
            marker.write_text("score_finished\n")
    wall = time.perf_counter() - wall_start
    cpu = _cpu_seconds() - cpu_start
    execution = dict(scores.parallel_execution)
    return {
        "jobs": jobs,
        "effective_jobs": int(execution["effective_jobs"]),
        "backend": execution["backend"],
        "worker_pids": [int(pid) for pid in execution["worker_pids"]],
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "cpu_percent": 100.0 * cpu / wall,
        "result_sha256": _array_hash(edge_p, group_p, components),
        "fixture": {
            "seed": 20260828,
            "n": n,
            "groups": groups,
            "unique_edges": len(expanded.edges),
            "responses": responses,
        },
    }


def _process_tree(root_pid: int) -> set[int]:
    found = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in found or not Path(f"/proc/{pid}").exists():
            continue
        found.add(pid)
        children = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            pending.extend(int(value) for value in children.read_text().split())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    return found


def _process_memory(pid: int) -> tuple[int, int]:
    rss = 0
    pss = 0
    try:
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            if line.startswith("Rss:"):
                rss = int(line.split()[1]) * 1024
            elif line.startswith("Pss:"):
                pss = int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return rss, pss


def _process_cpu_ticks(pid: int) -> int | None:
    try:
        values = Path(f"/proc/{pid}/stat").read_text().split()
        return int(values[13]) + int(values[14])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None


def _process_thread_count(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Threads:"):
                return int(line.split()[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        pass
    return None


def _run_subprocess(jobs: int, args) -> dict:
    temporary = tempfile.TemporaryDirectory(prefix="homoeogwas-omnib-bench-")
    marker = Path(temporary.name) / "phase.txt"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--child-jobs", str(jobs),
        "--n", str(args.n),
        "--groups", str(args.groups),
        "--responses", str(args.responses),
        "--marker", str(marker),
    ]
    environment = os.environ.copy()
    for name in NUMERIC_THREAD_ENV:
        environment[name] = "1"
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
    )
    peak_rss = 0
    peak_pss = 0
    sampled_cpu_ticks = 0
    sampled_cpu_ticks_by_pid = {}
    previous_ticks = {}
    max_threads_by_pid = {}
    observed_start = None
    observed_wall = None
    while process.poll() is None:
        process_ids = _process_tree(process.pid)
        memory = [_process_memory(pid) for pid in process_ids]
        peak_rss = max(peak_rss, sum(value[0] for value in memory))
        peak_pss = max(peak_pss, sum(value[1] for value in memory))
        phase = marker.read_text().strip() if marker.exists() else ""
        if phase == "score_started" and observed_start is None:
            observed_start = time.perf_counter()
            previous_ticks = {
                pid: ticks for pid in process_ids
                if (ticks := _process_cpu_ticks(pid)) is not None
            }
        if observed_start is not None and observed_wall is None:
            for pid in process_ids:
                thread_count = _process_thread_count(pid)
                if thread_count is not None:
                    max_threads_by_pid[pid] = max(
                        max_threads_by_pid.get(pid, 0), thread_count)
                ticks = _process_cpu_ticks(pid)
                if ticks is None:
                    continue
                previous = previous_ticks.get(pid, ticks)
                if ticks >= previous:
                    delta = ticks - previous
                    sampled_cpu_ticks += delta
                    sampled_cpu_ticks_by_pid[pid] = (
                        sampled_cpu_ticks_by_pid.get(pid, 0) + delta)
                previous_ticks[pid] = ticks
            if phase == "score_finished":
                observed_wall = time.perf_counter() - observed_start
        time.sleep(0.05)
    stdout, stderr = process.communicate()
    temporary.cleanup()
    if process.returncode != 0:
        raise RuntimeError(
            f"benchmark child jobs={jobs} failed ({process.returncode}):\n{stderr}")
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"benchmark child jobs={jobs} emitted no JSON")
    if observed_start is None or observed_wall is None:
        raise RuntimeError(f"benchmark child jobs={jobs} omitted score phase markers")
    record = json.loads(lines[-1])
    record["resource_cpu_seconds"] = record["cpu_seconds"]
    record["resource_cpu_percent"] = record["cpu_percent"]
    record["cpu_seconds"] = sampled_cpu_ticks / os.sysconf("SC_CLK_TCK")
    record["cpu_percent"] = 100.0 * record["cpu_seconds"] / observed_wall
    record["observed_process_tree_wall_seconds"] = observed_wall
    record["cpu_percent_by_pid"] = {
        str(pid): 100.0 * ticks / os.sysconf("SC_CLK_TCK") / observed_wall
        for pid, ticks in sorted(sampled_cpu_ticks_by_pid.items())
    }
    record["max_threads_by_pid"] = {
        str(pid): count for pid, count in sorted(max_threads_by_pid.items())
    }
    record["peak_aggregate_rss_bytes"] = peak_rss
    record["peak_rss_bytes"] = peak_pss
    record["memory_metric"] = "aggregate_pss_bytes"
    record["command"] = command
    return record


def _parse_jobs(value: str) -> tuple[int, ...]:
    jobs = tuple(int(item) for item in value.split(","))
    if jobs != (1, 4, 8):
        raise argparse.ArgumentTypeError("release benchmark jobs must be exactly 1,4,8")
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=_parse_jobs, default=(1, 4, 8))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--n", type=int, default=192)
    parser.add_argument("--groups", type=int, default=192)
    parser.add_argument("--responses", type=int, default=199)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--child-jobs", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--marker", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.child:
        print(json.dumps(_child_run(
            args.child_jobs, n=args.n, groups=args.groups,
            responses=args.responses, marker=args.marker), sort_keys=True))
        return 0
    if args.out is None:
        raise SystemExit("--out is required")
    runs = {jobs: _run_subprocess(jobs, args) for jobs in args.jobs}
    report = evaluate_runs(runs)
    report["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["python"] = sys.version
    report["platform"] = sys.platform
    report["logical_cpus"] = os.cpu_count()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "accepted": report["accepted"],
        "selected_jobs": report["selected_jobs"],
        "out": str(args.out),
    }, sort_keys=True))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
