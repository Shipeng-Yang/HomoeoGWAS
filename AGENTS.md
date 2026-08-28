# HomoeoGWAS — Agent Workflow Specification

This file is the **single source of truth** for driving HomoeoGWAS end-to-end.
It is written to be read by any LLM/agent (Claude Code, Cursor, Cline, …) **and**
by humans. The Claude skill (`.claude/skills/homoeogwas/`) and the MCP server
(`homoeogwas-mcp`) both defer to this document; the executable implementation of
these rules lives in `src/homoeogwas/workflow.py`.

> **Audience: bioinformaticians and plant breeders, not programmers.** An agent
> using this tool must collect a few *biological* inputs and **generate the YAML
> configs itself** — never ask the user to hand-write a config.

HomoeoGWAS runs **subgenome-stratified linear-mixed-model GWAS** in allopolyploids
(wheat AABBDD, cotton AADD, rapeseed AACC, hexaploid AABBCC, octoploid AABBCCDD,
and diploids). Its signature output is the per-subgenome variance partition; it
also offers one homoeolog-group omniB interaction engine for 2+ subgenomes.

---

## 1. Minimal user inputs

Collect only these. Everything else has a sensible default.

**GWAS from existing per-subgenome PLINK BEDs:**
- `bed_prefixes`: map of subgenome → PLINK prefix (expects `.bed/.bim/.fam`)
- `phenotype`: TSV/CSV path
- `sample_col`: the sample-id column name
- `trait`: the trait column name
- `subgenomes`: labels, e.g. `[A, B, D]`
- `out_dir`

**GWAS from one raw VCF** (adds a split step):
- `vcf` + `species_yaml` (or a `subgenome_map` TSV) — used to split into
  per-subgenome BEDs first
- then as above (phenotype/sample_col/trait/subgenomes/out_dir)

**Homoeolog interaction** (adds prep steps):
- per-subgenome `bed_prefixes`, phenotype/sample_col/trait/subgenomes/out_dir
- **either** prebuilt `snp_to_gene` NPZs + one wide homoeolog `groups` TSV
  (`pairs`/`triads` TSVs remain accepted as legacy aliases),
  **or** a `gff` + `subgenome_map` (to build snp_to_gene) + proteins/`diamond`
  (to build homoeolog pairs). See `docs/interact_inputs.md`.

---

## 2. Decision tree

- Have per-subgenome BED prefixes → **skip split**.
- Have one VCF → **run `split` first**.
- Only GWAS requested → `fit` → `plot`.
- Interaction requested and prep inputs missing → `prep-snps` and/or
  `prep-homoeologs` first, then `interact`.
- Interaction → canonical `mode: group`, with pair edges as the common
  primitive. Two copies yield 1 edge, three yield 3, and four yield 6. Never
  fit or claim a direct four-way coefficient.
- Always `validate` a generated config before a long run.

---

## 3. Canonical pipeline order

```
VCF  → split → [build fit.yaml] → validate → fit → plot → audit → summarize
BED  →         [build fit.yaml] → validate → fit → plot → audit → summarize
interaction → prep-snps? → prep-homoeologs? → [build interact.yaml] → validate → interact → audit → summarize
```

---

## 4. Config generation rules

The agent maps high-level inputs into the existing YAML schemas and writes them
under `<out_dir>/configs/` with deterministic names (`fit.generated.yaml`,
`interact.generated.group.omnib.yaml`).

**fit YAML** (the GWAS):
```yaml
fit_version: 1
panel:
  name: <panel>           # any label
  subgenomes: [A, B, D]   # from user
phenotype:
  path: <phenotype>
  sample_col: <sample_col>
  trait: <trait>
genotype:
  scan_bed_prefix_template: <dir>/{subgenome}/all   # if prefixes follow a template
  grm:
    source: bed
    bed_prefix_template: <dir>/{subgenome}/all       # defaults to scan template
    maf_min: 0.05
kernels:
  normalize: trace
  include_hadamard: false   # true adds the homoeolog Hadamard kernel
  hadamard_name: hom
reml: {n_starts: 10, seed: 2026}
scan:
  mode: memory              # memory | stream | auto
  backend: cpu
  maf_min: 0.05
  call_rate_min: 0.9
  loco: {enabled: false}    # set true (+ fallback: error) for leave-one-chrom-out
plots: {enabled: true}
outputs: {out_dir: <out_dir>, prefix: <trait>}
```
If the per-subgenome prefixes do **not** share a `{subgenome}` template, the
agent should symlink/copy them into a templated layout (`<out_dir>/geno/<S>/all`)
and point the template there.

**interact YAML**:
```yaml
interact:
  mode: group
  subgenomes: [A, B, D]
  groups: <groups.tsv>        # group_id,gene_A,gene_B,gene_D
  statistic: omniB            # encoding-robust primary; legacy burden must be explicit
  hypothesis_unit: group      # edge for pair discovery; group for group omnibus
  subset_order: 2             # all unique pair edges
  family_scope: primary_only  # or joint: one calibration over edge+group union
  primary_transform: INT
  primary_multiplicity: bootstrap_minp
  genotype:    {A: <prefixA>, B: <prefixB>, D: <prefixD>}
  snp_to_gene: {A: <npzA>, B: <npzB>, D: <npzD>}
  phenotype: <phenotype>
  sample_col: <sample_col>
  trait: <trait>
  burden: {cap: 150, min_snp: 3, maf_min: 0.01, n_pc: 3}
  grm: {method: grm_from_X, maf_min: 0.01, scope: all_subgenomes}
  calibration: {method: bootstrap, B: 2000, seed: 2026}
outputs: {out_dir: <out_dir>, full_ranking: true}
```

`omniB` combines minor-allele burden, PC1 and kernel-Hadamard interaction
evidence. The combined p is primary; component p-values localize the evidence
but are not separate discoveries. A frozen legacy analysis may instead declare
`statistic: burden` and `calibration: {method: permutation, perm_B: 2000}`.

**subgenome map TSV** (for split / prep): columns `chrom, subgenome[, base_group]`,
where `chrom` is the name as it appears in **both** the GFF and the `.bim`.

---

## 5. Gotcha rules (mandatory)

1. **Sample IDs are strings.** Read the phenotype sample column as strings and
   stringify `.fam` IIDs before joining; the built-in fit/interact readers do
   this automatically. If leading zeros were already lost upstream, stop and
   repair the source identifiers before running.
2. **GFF and `.bim` chromosome names must match exactly** (for `prep-snps`). If
   they differ (e.g. an NCBI accession in the GFF vs `1A` in the `.bim`), rename
   one side first; stop and say so.
3. **subgenome map** columns are `chrom, subgenome[, base_group]`.
4. **DIAMOND 2.2.0 deadlocks** on `makedb`; use a 2.1.x binary via `--diamond`.
   Protein FASTAs must have `.`/`*` stop/gap characters stripped.
5. **Never run a 4-way interaction.** A four-copy group omniB aggregates its
   six pair edges. Never run AB/AD/BD (or other directions) as independent
   alpha families and then multiply adjusted p-values after the run.
6. **Validate before expensive runs.** Record every generated config under
   `<out_dir>/configs/`.
7. **Bind interaction mappings to their BED.** Current `prep-snps` NPZs record
   the source BIM SHA-256 and variant count. Refuse legacy/unverified NPZs or a
   BIM fingerprint mismatch; rerun `prep-snps` with the analysis BED.

---

## 6. Command templates

```bash
# split a VCF into per-subgenome BEDs
homoeogwas split --species-yaml {species_yaml} --vcf {vcf} \
  -o {outdir} --threads {threads}

# GWAS
homoeogwas validate -c {outdir}/configs/fit.generated.yaml
homoeogwas fit      -c {outdir}/configs/fit.generated.yaml
homoeogwas plot     {outdir}            # regenerate figures, no recompute
homoeogwas audit    {outdir}             # validity/evidence/replication report

# interaction inputs
homoeogwas prep-snps --gff {gff} --subgenome-map {sgmap} \
  --bed A={prefixA} --bed B={prefixB} --bed C={prefixC} \
  --feature gene --id-attr ID --flank-bp 2000 --min-snp 2 --out-dir {prepdir}
homoeogwas prep-homoeologs --mode triad --subgenomes A,B,C \
  --genes {prepdir}/genes_{S}.tsv --method diamond-rbh \
  --proteins A={faaA} --proteins B={faaB} --proteins C={faaC} \
  --restrict-base-group --subgenome-map {sgmap} --diamond {diamond} \
  --out {prepdir}/triads.tsv

# interaction
homoeogwas interact -c {outdir}/configs/interact.generated.group.omnib.yaml --n-jobs {jobs}
homoeogwas audit    {outdir}

# install self-test
homoeogwas demo
```

For canonical group omniB, `--n-jobs` means POSIX worker processes sharing
read-only prepared arrays through fork copy-on-write; each worker uses one
native-library thread. The result records requested/effective jobs, backend and
observed worker PIDs. If fork is unavailable the engine records a serial
fallback rather than presenting Python threads as independent workers. Always
launch parallel interaction through the installed `homoeogwas interact`
command: its lightweight entry point sets OpenBLAS/OpenMP/MKL/NumExpr limits
before NumPy/SciPy import. A bypassed parallel invocation with an already
oversubscribed native pool is refused instead of silently multiplying threads.

---

## 7. Outputs and how to summarize them

After `fit`, in `<out_dir>`:
- `summary_<trait>.json` — REML `sigma2`/`pve` per subgenome (the headline
  variance partition), `lambda_gc`, acceptance checks.
- `sumstats_<trait>.tsv` — per-SNP `beta/se/chi2/p/maf` with `subgenome`/`chrom`/`pos`.
- `variance_components_<trait>.{png,pdf,svg}` (signature figure),
  `manhattan_…`, `qq_…`, `lambda_gc_…`.

Report to the user: the **per-subgenome PVE** (which subgenomes carry the
heritability), genome-wide `λ_GC` (calibration), the strongest hits from
sumstats, the figure paths — plus any warnings and the next recommended step.

After `interact`: name the declared primary unit (`edge` or `group`), the one
experiment-wide bootstrap-minP family, adjusted p-values, significant units,
the evidence-driving omniB component, complete ranking, family hashes/counts,
and audit status. Edge primary permits “encoding-robust omnibus interaction
evidence for a homoeolog pair”; group primary permits “encoding-robust omnibus
pairwise interaction evidence within a homoeolog group”. Neither is a
third-/fourth-order causal or physical mechanism.

---

## 8. Agent behavior contract

- Ask only for **high-level biological/file inputs** (§1), never raw YAML fields.
- **Generate** configs, **validate**, then run; record configs under `<out_dir>`.
- Enforce the §5 gotcha rules *before* long runs; on a validation failure return
  a concrete repair instruction (sample-id coercion, chromosome rename, missing
  PLINK file, DIAMOND version), not a stack trace.
- Generate canonical group omniB configs for 2+ subgenomes; refuse direct
  four-way statistics and post-run direction-wise multiplicity patches.
- Run `audit` on finished outputs and summarize biologically (§7), not just paths.
