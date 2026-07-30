# Triad3 Stage 1c-D3 confirmatory calibration design

Date: 2026-07-30
Status: user-approved design; implementation not authorized
Scope: phenotype-sealed statistical confirmation and runner-v2 engineering
Claude: not called
Real phenotype: sealed

## 1. Purpose

D1 completed correctly but stopped because the classical-F oracle count was
65/1,000. D2 found no statistical implementation error. A same-stream,
raw-Gaussian exchangeable control changed that count to 60/1,000, so the
one-time rank INT mismatch was not sufficient to explain the stop. The leading
explanation is the low-probability outer Monte Carlo stop that the D1 boundary
allowed by construction.

D3 is a minimal confirmatory study. It separates four questions that D1 partly
combined:

1. Does each statistic calibrate when outer and inner responses are exactly
   exchangeable under a known Gaussian covariance?
2. Does each statistic calibrate under the current one-time observed rank-INT
   and conditional transformed-scale bootstrap semantics?
3. Does the production REML-refit path calibrate in the balanced profile?
4. Does it calibrate in the low-heritability boundary profile?

D3 may eliminate a candidate method for candidate-specific calibration
failure. It cannot rank surviving methods, establish power, validate the
2,143-triad production family, or authorize biological analysis.

## 2. Non-goals

D3 will not:

- alter, top up, overwrite or reinterpret D1;
- read, hash or inspect real phenotype values;
- call Claude;
- change the 24-triad stress family;
- compare planted-signal power;
- run independent holdout or full-family replay;
- choose a seed by trying alternatives;
- modify the manifest-bound D1 runner;
- implement a new rank-INT estimand such as re-ranking bootstrap draws.

## 3. Frozen statistical architecture

### 3.1 Family and methods

- Ordered family: the same frozen 24-triad `gate_balanced` stress subset used
  in D1.
- Candidate methods: `classical_f` and `hc3_t`, evaluated separately.
- Inner bootstrap replicates: 999.
- Outer replicates: 2,000 for every cell.
- Primary level: 0.05.
- Observed and inner degeneracy remain failures, never non-rejections.

### 3.2 Four cells

| Cell | Observed response | Covariance/whitener | Inner response |
|---|---|---|---|
| `raw_oracle_balanced` | raw balanced Gaussian; no rank INT | known balanced generating covariance and oracle whitener | fitted-intercept Gaussian draws under the same known covariance; no rank INT |
| `int_oracle_balanced` | the same raw response, rank-INT once | known balanced generating covariance and oracle whitener | conditional transformed-scale Gaussian draws; no second rank INT |
| `int_reml_balanced` | the same balanced raw response, rank-INT once | production REML refit on the analyzed scale | fitted-null conditional transformed-scale draws; no second rank INT |
| `int_reml_low_h2` | low-h2 response from the paired outer innovation, rank-INT once | production REML refit on the analyzed scale | fitted-null conditional transformed-scale draws; no second rank INT |

The raw oracle fitted intercept lies in every nuisance model. Its numerical
value therefore cannot affect the conditional interaction statistics. The raw
observed and inner columns are exactly covariance-matched for this
implementation control.

The true component profiles remain:

- balanced: A/B/D/e = 0.20/0.20/0.20/0.40;
- low-h2: A/B/D/e = 0.05/0.05/0.00/0.90.

### 3.3 Exact decision boundaries

There are eight evaluations: four cells for each of two methods. Both error
directions use a conservative union bound across all eight evaluations, with
per-evaluation alpha 0.05/8 = 0.00625.

For X rejections among 2,000 outer replicates:

- acceptable: X <= 120;
- inconclusive: 121 <= X <= 125;
- unacceptable: X >= 126.

Exact binomial operating characteristics are:

| True FWER | Pr(X <= 120) | Pr(X >= 126) |
|---:|---:|---:|
| 0.050 | 0.98008 | 0.00561 |
| 0.075 | 0.00502 | 0.98314 |

Consequently:

- when all eight true FWER values are at most 0.05, the union-bound
  probability of at least one false `unacceptable` call is at most about
  0.0449;
- when an evaluation has true FWER at least 0.075, its false `acceptable`
  probability is at most about 0.00502;
- the union-bound probability of any false `acceptable` call across eight
  evaluations at true FWER 0.075 is at most about 0.0402.

Pairing or dependence between evaluations does not invalidate these union
bounds. There is no optional top-up, seed replacement or pooled count.

### 3.4 Candidate-specific interpretation

Each method follows its own four-gate chain:

1. `raw_oracle_balanced` must be acceptable;
2. `int_oracle_balanced` must be acceptable;
3. `int_reml_balanced` must be acceptable;
4. `int_reml_low_h2` must be acceptable.

A method is eligible for the later, separately frozen Plan S stage only if all
four counts are at most 120.

- If any count is at least 126, that method is eliminated.
- If no count is unacceptable but at least one is 121--125, that method is
  inconclusive and cannot advance.
- If both methods are eligible, both advance to Plan S; D3 does not rank them.
- If exactly one method is eligible, only that candidate advances.
- If neither is eligible, development remains stopped.

Execution validity is global. A corrupt manifest, missing/duplicate identity,
failed unit, REML optimizer non-success or observed/inner degeneracy invalidates
the whole D3 run rather than only one candidate.

## 4. Random-stream contract

### 4.1 Master seed

No numeric formal seed is chosen during design. After this specification is
reviewed, locked and hashed, the formal 128-bit master seed is derived
mechanically as the first 16 bytes of:

`SHA256("triad3-stage1c-d3-confirmatory-v1\0" || spec_sha256)`

interpreted little-endian. Any change to the locked specification changes the
seed and requires a new lock record. QA uses a distinct
`triad3-stage1c-d3-qa-v1` domain and never consumes the formal stream.

### 4.2 Namespaces and pairing

For outer index `i`:

- outer innovation: `(d3, outer, i)`, shared across covariance profiles;
- balanced inner base normals: `(d3, inner, balanced, i)`, shared before
  cell-specific covariance transforms by the three balanced cells;
- low-h2 inner base normals: `(d3, inner, low_h2, i)`;
- balanced REML optimizer: `(d3, reml, balanced, i)`;
- low-h2 REML optimizer: `(d3, reml, low_h2, i)`.

The same balanced raw response feeds the raw oracle, INT oracle and balanced
REML cells. The low-h2 response applies the low-h2 covariance root to the same
outer standard-normal innovation.

Method identity never enters a random namespace. Classical F and HC3 always
receive the same observed response and inner draws.

The implementation uses SHA-256, `numpy.random.SeedSequence` and PCG64.
Runtime `hash()` is forbidden. Checkpoints store seed IDs and SHA-256 hashes
of all outer and inner base-normal arrays.

### 4.3 No-peeking rule

All 8,000 cell-by-outer identities must reach a terminal checkpoint before any
empirical P value, rejection indicator, rejection count or method decision is
inspected. Runtime progress is restricted to completed, pending and failure
unit counts.

## 5. Runner-v2 architecture

### 5.1 Isolation

D3 uses a new dedicated runner and a new output directory. The D1 runner and
D1 outputs remain byte-unchanged.

The runner has five bounded components:

1. strict config validation and path resolution;
2. outcome-free input preparation and immutable manifest construction;
3. deterministic per-outer bundle computation;
4. concurrent scheduling plus atomic checkpoint persistence;
5. non-inferential completion ledger generation.

The independent post-run auditor is a separate executable and does not import
runner aggregation or summary code.

### 5.2 Scheduling unit

One scheduling task represents an `outer_index bundle`. It reconstructs the
shared outer innovation and evaluates the missing subset of the four cells.
The complete run therefore schedules 2,000 bundles but produces 8,000 atomic
cell checkpoints and 16,000 method rows.

Scheduling batch size is completely separate from checkpoint persistence.
`checkpoint_every` is not a scheduler input.

The scheduler:

- maintains up to `2 * outer_workers` in-flight bundles;
- writes each returned cell checkpoint immediately and atomically;
- uses completion-order collection, not fixed serial batches;
- never overwrites an accepted checkpoint;
- records actual worker count, maximum observed concurrency, elapsed time and
  peak RSS.

### 5.3 Parallelism

Every outer worker uses one BLAS thread:

- `OPENBLAS_NUM_THREADS=1`;
- `OMP_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`;
- `NUMEXPR_NUM_THREADS=1`.

QA benchmarks 32, 64 and 96 outer workers on the same 192 QA bundles, with two
clean repeats per worker count. The formal worker count is the smallest value
whose median wall time is within 10% of the fastest observed median, provided
that:

- all checkpoint bytes are identical across worker counts;
- maximum active concurrency is greater than one and reaches the expected
  scheduler range;
- no unit fails;
- peak RSS and checkpoint I/O remain within the measured reservation.

The expected formal setting is 64 workers, but the benchmark rule, not this
expectation, makes the final choice.

The formal config records the selected value. Resume may use only a
pre-benchmarked value from {32, 64, 96}; any override is logged and remains
outside RNG identity.

### 5.4 Checkpoint and resume

Checkpoint identity is `(manifest_id, cell, outer_index)`. Each checkpoint
contains:

- schema and manifest versions;
- cell, profile and outer index;
- success/failure terminal status;
- outer/inner seed IDs and array hashes;
- raw and analyzed response hashes;
- fitted covariance and whitener diagnostics;
- REML result, boundary components and optimizer record where applicable;
- per-method observed family minimum, inner comparison count, plus-one
  empirical P, rejection indicator and inner-minima hash;
- explicit degeneracy fields.

Writes use a same-directory temporary file, flush, fsync, atomic rename and
directory fsync.

On resume, every existing checkpoint is schema-validated and fingerprint-bound
before acceptance. Unexpected, duplicate or out-of-range checkpoint paths are
fatal. A partially completed bundle regenerates the same shared streams and
computes only missing cells; existing checkpoints are not rewritten.

## 6. Manifest and sealed inputs

Before any formal stream is consumed, `--prepare-only` creates a canonical
manifest binding:

- generated D3 config;
- this approved specification and its lock record;
- D0, D1 and D2 decision records;
- runner, independent auditor and transitive statistical code;
- Python, NumPy, SciPy, BLAS and machine environment;
- relevant git commit and worktree state;
- ordered 24-triad IDs and all rebuilt design-array hashes;
- genotype, mapping, kernel and sample-ID fingerprints;
- balanced and low-h2 covariance/root hashes;
- RNG domains and derived formal seed ID;
- phenotype path metadata with `trait_column_read=false` and
  `file_bytes_hashed=false`.

The manifest ID is the SHA-256 of its canonical body; the body never contains
its own ID. Formal execution requires an exact manifest match.

## 7. Testing and QA gates

Implementation cannot enter formal preparation until all of the following
pass:

1. strict config-schema and exact-boundary tests;
2. seed-namespace, method-exclusion and paired-stream tests;
3. raw-oracle exchangeability and four-cell formula regression tests;
4. failure, degeneracy and REML boundary-semantics tests;
5. checkpoint schema, atomic-write and unexpected-file tests;
6. interruption/resume equivalence tests;
7. a scheduler regression proving maximum concurrent active tasks exceeds one
   when multiple workers are requested;
8. 1/32/64/96-worker byte-identity tests;
9. fixed replay of outer indices 0, 1, 999 and 1999 across four cells:
   16/16 checkpoints must be byte-identical;
10. 32/64/96 two-repeat compute benchmark on 192 QA bundles;
11. relevant lint, compile and complete repository tests;
12. independent pre-run rehash of every manifest-bound file and array.

The final pre-run ledger must be all PASS and SHA-256 attested. Technical
readiness does not itself authorize formal execution.

## 8. Formal execution and independent post-run audit

Formal execution begins only after a separate explicit user authorization.
The runner emits no inferential summary during execution.

After all 8,000 checkpoints complete, the independent auditor:

- rehashes the manifest and all bound artifacts;
- verifies exact planned identity coverage and zero duplicate/extra records;
- verifies outer and inner stream pairing;
- checks failure, optimizer, boundary and degeneracy semantics;
- recomputes plus-one empirical P values and all eight rejection counts from
  raw checkpoint comparison counts;
- independently recomputes the 120/126 binomial operating characteristics;
- applies the candidate-specific four-gate decision rule;
- compares checkpoint-derived results with the 16,000-row replicate TSV;
- performs the frozen 16-checkpoint replay;
- reports exact Clopper--Pearson intervals descriptively;
- confirms no biological or 2,143-family claim was made.

A separate third-path reparser, importing neither runner aggregation nor the
primary auditor, must reproduce all eight counts and decisions before the D3
decision is signed.

## 9. Outputs

The new D3 output directory contains:

- `configs/d3.generated.yaml`;
- `manifest.frozen.json` and SHA-256 sidecar;
- atomic checkpoints under one directory per cell;
- `d3_replicates.tsv`;
- non-inferential `run_complete.json`;
- independent audit JSON and logs;
- 16-checkpoint replay evidence;
- primary-artifact SHA-256 ledger;
- final Codex-only D3 decision report.

No real phenotype values, biological rankings or rolling rejection summaries
are written.

## 10. Advancement rule

D3 success means only that at least one candidate passed its four
candidate-specific 24-triad calibration gates. A surviving method may then
enter the already separate Plan S matched-power and independent-holdout
process.

Real phenotype remains sealed until the complete later chain succeeds,
including method handling under Plan S and the selected method's frozen
2,143-triad integrated replay.

## 11. Approved design decisions

The user approved:

1. the four-cell, eight-evaluation architecture with outer B=2,000, inner
   B=999 and exact 120/126 boundaries;
2. shared paired streams with candidate-specific four-gate interpretation;
3. a new runner-v2 with true outer concurrency, 32/64/96 benchmarking,
   atomic checkpoints, 16-checkpoint byte replay and independent post-run
   recomputation.

These approvals authorize this design specification only. They do not
authorize implementation, formal seed consumption or simulation.
