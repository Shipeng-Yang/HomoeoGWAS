"""Run many single-response interact cases of one panel in one resident process.

Each case keeps its own config, output directory, checkpoint manifest and audit;
only phenotype-independent panel preparation is shared (InteractPanelState).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path

from .interact import InteractPanelState, cmd_interact

EXECUTION_RECORD_SCHEMA = "homoeogwas-interact-batch-execution-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _memory_kib() -> dict:
    values = {}
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            name, _, rest = line.partition(":")
            if name in {"Rss", "Pss"}:
                values[name.lower() + "_kib"] = int(rest.split()[0])
    except OSError:
        pass
    return values


@contextlib.contextmanager
def _capture_fds(path: Path):
    """Send fd 1 and 2 (including forked workers) to one exclusive log file."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with path.open("x") as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)
        try:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                yield handle
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def _attempt_log(base: Path) -> Path:
    attempt = 1
    while True:
        path = base.with_name(f"{base.name}.attempt{attempt}")
        if not path.exists():
            return path
        attempt += 1


def _write_exclusive(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write(json.dumps(payload, indent=1, sort_keys=True))


def _load_cases(path: Path) -> list[dict]:
    cases = json.loads(Path(path).read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("batch case list must be a non-empty JSON list")
    seen = set()
    for case in cases:
        missing = {"case_id", "config", "execution_record", "log"} - set(case)
        if missing:
            raise ValueError(f"batch case lacks fields: {sorted(missing)}")
        if case["case_id"] in seen:
            raise ValueError(f"duplicate batch case_id: {case['case_id']}")
        seen.add(case["case_id"])
    return cases


def run_batch(cases_path, *, n_jobs: int, batch_id: str | None = None) -> int:
    cases = _load_cases(Path(cases_path))
    batch_id = batch_id or uuid.uuid4().hex
    state = InteractPanelState()
    failures = 0
    for case in cases:
        record_path = Path(case["execution_record"])
        if record_path.exists():
            failures += json.loads(record_path.read_text())["exit_code"] != 0
            continue
        config = Path(case["config"])
        record = {
            "schema": EXECUTION_RECORD_SCHEMA,
            "batch_id": batch_id,
            "batch_cases_sha256": _sha256(Path(cases_path)),
            "case_id": case["case_id"],
            "config": str(config),
            "config_sha256": _sha256(config),
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "n_jobs": int(n_jobs),
            "start_epoch": time.time(),
        }
        args = argparse.Namespace(config=str(config), out_dir=None, n_jobs=int(n_jobs))
        base_log = Path(case["log"])
        base_log.parent.mkdir(parents=True, exist_ok=True)
        log = _attempt_log(base_log)
        with _capture_fds(log):
            try:
                exit_code = int(cmd_interact(args, panel_state=state))
            except Exception:  # noqa: BLE001 - one case failure must not hide the others
                traceback.print_exc()
                exit_code = 1
        record |= {
            "end_epoch": time.time(),
            "exit_code": exit_code,
            "config_sha256_after": _sha256(config),
            "log": str(log),
            "log_sha256": _sha256(log),
            "memory_after": _memory_kib(),
        }
        _write_exclusive(record_path, record)
        failures += exit_code != 0
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m homoeogwas.interact_batch")
    ap.add_argument("--cases", required=True, help="JSON list of batch cases")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--batch-id", default=None)
    args = ap.parse_args(argv)
    return run_batch(args.cases, n_jobs=args.n_jobs, batch_id=args.batch_id)


if __name__ == "__main__":
    sys.exit(main())
