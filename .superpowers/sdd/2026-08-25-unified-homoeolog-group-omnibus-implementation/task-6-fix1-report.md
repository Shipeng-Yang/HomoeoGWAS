# Task 6 Review Fix Round 1 Report

Status: COMPLETE

## Implemented

- Audits now reject formal discovery or rejection authority in every
  non-primary transform while accepting only explicitly noninferential QA
  diagnostics.
- Canonical family counts and bootstrap sizes are validated as real serialized
  integers (never booleans, floats, or strings). Hypothesis IDs must be unique
  strings, and observed/adjusted probability vectors must contain only null or
  finite values in `[0, 1]` before the shared FWER consistency checker runs.
- Primary QA-only serialization remains valid: all formal authority fields stay
  null and engineering quantities are checked under
  `qa_diagnostics.role=noninferential_do_not_threshold`.
- Breeder-facing interaction summaries now expose group/edge counts and hashes,
  authoritative calibrated hits with adjusted p-values, lambda-GC, evidence
  drivers, the complete ranking path, and audit status. Descriptive top rows are
  explicitly labeled and kept separate from discoveries.
- Canonical audit reports resolve `hypothesis_id`, `group_id`, and `edge_id`, and
  render group `driving_component` or edge `smallest_component` without changing
  the pairwise-omnibus/no-higher-order evidence boundary.
- `prep-homoeologs --mode group` now builds generic wide 2+-copy tables. Curated
  long/wide tables retain their group IDs; generalized DIAMOND grouping requires
  one consistent reciprocal-best-hit clique across every subgenome pair. Legacy
  pair and triad output schemas remain unchanged. The MCP wrapper forwards table
  format and its existing four-copy `mode=group` command is now executable.

## TDD evidence

The initial focused RED run failed 34 new corruption, summary, rendering, CLI,
and all-pair-clique cases. Later dedicated primary-QA and MCP-command
regressions also failed before their fixes were added. All new tests then passed.

Shared-tree affected suites:

```text
tests/test_audit.py tests/test_workflow.py tests/test_prep.py: passed
3 optional skips (MCP and real DIAMOND unavailable)
```

Broader shared-tree regressions covering canonical calibration, Task 5 fixes,
interaction execution, and checkpoint/resume also passed.

## Exact-index verification

The candidate was reconstructed from `HEAD` plus only the staged binary diff in
a fresh `/tmp` checkout. The following exact-index suites passed:

```text
263 passed, 3 skipped; 266 collected
tests/test_audit.py
tests/test_workflow.py
tests/test_prep.py
tests/test_interaction_config.py
tests/test_omnib_family_calibration.py
tests/test_task5_review_fixes.py
tests/test_interact.py
tests/test_resampling_checkpoint.py
```

`python -m compileall -q src/homoeogwas` passed. Selected-file Ruff and
`git diff --cached --check` passed on the final exact-index candidate.

## Dirty-tree handling

Only Task 6 fix hunks were staged. Pre-existing marker/PVE work in
`workflow.py`, `prep.py`, and `tests/test_audit.py` remains unstaged. The
committed test file removes an already-unused `validate_config` import so the
exact-index selected-file lint is clean; the user's unstaged PVE test and its
working-tree import remain preserved.
