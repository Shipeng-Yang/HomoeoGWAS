# HomoeoGWAS

A GWAS framework for **allopolyploid crops** built around three architectural pillars:

1. **Subgenome-partitioned LMM** — `y = Xβ + u_A + u_B [+ u_D ...] + ε`, with per-subgenome GRM and LOCO option per logical chromosome
2. **Optional homoeolog Hadamard kernel** `K_hom = K_A ⊙ K_B [⊙ K_D]` as scope-conditional epistasis term — see Algorithm section for the 5-panel negative finding and Tier 1/2 revival paths
3. **Zero-shot DL prior re-ranking** — PlantCaduceus + AgroNT log-likelihood fused with GWAS p-value over suggestive hits + LD blocks; panel-size-dependent recall lift +0.20–0.70

→ **[Getting Started](getting_started.md)** for install + first run
→ **[Algorithm](algorithm.md)** for the model
→ **[API Reference](api.md)** for module-level docs
→ **[Project Charter](00_charter.md)** for the frozen claim hierarchy
