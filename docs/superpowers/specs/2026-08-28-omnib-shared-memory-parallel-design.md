# omniB Shared-Memory Process Parallelism Design

## Purpose

Make `homoeogwas interact --n-jobs N` use real multi-core execution for the
canonical group omniB scorer without changing any statistical result or
replicate stream. The current joblib `threading` blocks run in one Python
process and remain near one effective CPU core because the edge loop is
Python-heavy. The release must also state the actual execution model so that a
user does not mistake one process for N workers.

After the engine passes correctness, scaling, and memory gates, the same build
will run the canonical two-copy group analyses for cotton and peanut.

## Scope

The change covers the canonical prepared-response omniB paths used by
`interact.mode: group`:

- initial observed/bootstrap edge scoring;
- indexed/checkpointed bootstrap scoring;
- deletion-sensitivity scoring that reuses prepared features.

Legacy burden/permutation and experimental triad3 paths are not migrated in
this change. Their public behavior remains unchanged. Statistical definitions,
family construction, INT transformation, bootstrap draws, ACAT combination,
minP calibration, hypothesis counts, and output ordering are frozen.

## Public contract

`--n-jobs 1` remains serial. On supported POSIX/Linux systems,
`--n-jobs N` for canonical group omniB means at most `N` worker processes,
bounded by the number of score blocks and available logical CPUs.

The command prints an execution-plan record before scoring and serializes the
same record in the interaction result:

- requested jobs;
- effective jobs;
- backend (`serial` or `fork_shared_memory`);
- process model (`processes`);
- inner native-library thread limit (`1`);
- parent and observed worker process identifiers;
- fallback reason, if parallel execution is unavailable.

This metadata is descriptive and must not enter any analysis/checkpoint hash.
Changing `n_jobs` therefore cannot change the declared statistical experiment.

## Architecture

### Parallel runner

Add a focused internal module, `src/homoeogwas/parallel.py`, with a small
process runner and an immutable execution record. The runner accepts block
identifiers plus module-level initializer and worker callables. It uses
`multiprocessing.get_context("fork")` on POSIX/Linux.

Large arrays and prepared feature dictionaries are assigned to a module-level,
read-only worker context immediately before the pool is forked. Children
inherit the underlying NumPy buffers through operating-system copy-on-write;
the task queue contains only small block bounds. Workers never mutate shared
arrays and return only block indices and score matrices. The parent clears its
worker context after pool shutdown, including exceptional shutdown.

The serial path calls the same worker function in the parent. This prevents
the parallel and serial modes from drifting into separate numerical
implementations.

### Native thread control

The installed `homoeogwas` launcher sets OpenBLAS, OpenMP, MKL and NumExpr to
one thread before importing NumPy/SciPy for `interact`. Each worker also runs
under `threadpoolctl.threadpool_limits(limits=1)` as a second layer. Canonical
parallel omniB refuses a direct/bypassed invocation if an oversized native pool
has already initialized, with a repair instruction to use the installed
launcher or `--n-jobs 1`. `threadpoolctl` is an explicit runtime dependency.

### Determinism and failures

Blocks are assembled by their original indices, never by completion order.
RNG is consumed only in the parent before the pool starts, preserving the
existing bootstrap response matrix and feature stream exactly. A worker
exception terminates the pool and is re-raised with the block identity; no
partial formal output is written.

If `fork` is unavailable, canonical scoring falls back to serial execution and
records the reason. It must not silently fall back to the ineffective thread
backend. The package already declares POSIX/Linux support, so this fallback is
mainly defensive.

## Configuration and compatibility

No new biological YAML field is required. Existing configs and
`--n-jobs` commands remain valid. The canonical group runner receives one
internal `parallel_execution` record and includes it in the result JSON and
audit-readable metadata. Checkpoint identity deliberately excludes this
record, allowing a run to resume with a different safe worker count.

No raw genotype, phenotype, group, or SNP-to-gene data is copied or rewritten.

## Tests

Implementation follows red-green TDD.

1. A process-runner test requests two jobs, executes enough controlled blocks,
   and asserts that more than one worker PID is observed. It fails against the
   current thread-only implementation.
2. A canonical omniB regression compares `n_jobs=1` and `n_jobs=2` with fixed
   input and seed using exact array equality for edge p-values, group p-values,
   component p-values, estimability masks, rankings, and calibration fields.
3. A failure test proves that a worker exception is propagated and no result is
   accepted.
4. A metadata test proves requested/effective jobs and backend are emitted but
   do not alter the checkpoint manifest hash.
5. Existing interaction, family, checkpoint, CLI, workflow, and audit suites
   run unchanged.

## Performance acceptance gates

A release benchmark uses one fixed prepared fixture and identical response
columns for `n_jobs=1`, `4`, and `8`.

- Numerical arrays and ranking hashes must be identical across worker counts.
- The observed worker-PID count must equal the effective process count when
  enough blocks exist.
- Aggregate CPU usage must exceed 200% for the 4-process run during its scoring
  phase.
- Four processes must be faster than one on the fixed benchmark; speedup is
  reported, not assumed.
- Peak resident memory for four processes must not exceed 1.35 times the serial
  peak. If it does, the engine is not accepted for formal runs.

The benchmark writes JSON containing commands, versions, timing, CPU, RSS,
hashes, and pass/fail gates.

## Cotton and peanut execution

Only after all correctness and performance gates pass:

1. Generate deterministic canonical `mode: group`, `statistic: omniB`,
   `hypothesis_unit: edge`, `subset_order: 2`, `family_scope: primary_only`,
   `INT`, bootstrap-minP `B=2000` configs for cotton FibLen/FibElo and peanut
   hundred-seed-weight/seed-length.
2. Bind every SNP-to-gene NPZ to its analysis BED fingerprint and validate each
   config before computation.
3. Run with the highest worker count admitted by the benchmark and current
   memory headroom, not blindly by logical CPU count.
4. Run `homoeogwas audit` for every output.
5. Compare each two-copy canonical result against its legacy pairwise result.
   Hypothesis counts and raw p-values must be exactly equal; any discrepancy
   stops publication reporting and triggers diagnosis.
6. Summarize significant units, adjusted p-values, component drivers, family
   hashes/counts, audit status, and stability evidence without treating
   negative results as failures.

## Release documentation

CLI help and interaction documentation will say “worker processes” for the
canonical group omniB engine. Release notes will state that the change improves
execution and observability only and does not alter the statistical estimand or
multiplicity family.
