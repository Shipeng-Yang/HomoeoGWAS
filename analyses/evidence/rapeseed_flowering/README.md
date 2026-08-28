# Rapeseed flowering candidate evidence fixture

These four rows freeze the source-grounded evidence collected for the two formal FWER-significant
A-C edges in the Wu-2019 flowering panel. They are input fixtures for the generic evidence
adapter, not species-specific software rules.

- Protein annotations are local BLASTP matches to TAIR10 and support orthology/domain wording
  only.
- Expression summaries are derived from the ORDER apex/leaf records reported by Jones et al.
  (2020), DOI `10.1186/s12870-020-02509-x`. Only the two transporter-like candidates meet the
  predeclared matched-tissue-support flag.
- QTL distances use the project-curated flowering-QTL table and its 500-kb window. None of the
  four genes overlaps that window.

The software records SHA-256 hashes for all three TSV sources. These tables do not prove causal
flowering function, physical interaction or independent replication.
