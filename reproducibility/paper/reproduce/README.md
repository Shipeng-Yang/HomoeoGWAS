# Reproducing the whole-genome interaction leads

Each `run.sh` regenerates the gitignored derived artifacts from the public
panels (retrieved via the accessions in the manuscript Data availability) and
runs the committed configs. Compare the `INT` block of each output JSON against
`expected_lead_results.yaml`.

```
DIAMOND=/path/to/diamond PLINK2=/path/to/plink2 bash cotton_cottongvd1245/run.sh
bash rapeseed_bnagvd/run.sh
```

| lead | n | G | top pair | min_p |
|---|---|---|---|---|
| cotton FibElo | 1245 | 3757 | Gh_A01G025100 / Gh_D01G022200 | 1.227544e-05 |
| cotton FibLen | 1245 | 3757 | Gh_A05G118800 / Gh_D05G131200 | 5.985552e-06 |
| rapeseed flowering | 926 | 17404 | BnaA02g19610D / BnaC02g22960D | 3.268177e-07 |

Use DIAMOND 2.1.x (2.2.0 deadlocks on these proteomes). The cotton driver
cleans and subgenome-splits the CRI proteome inline; the rapeseed driver needs
`rapeseed_bnagvd/pep_{A,C}.faa` (Darmor-bzh proteome split by subgenome, not
tracked in Git).
