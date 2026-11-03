# Data-owner waivers (not for automated agents)

HomoeoGWAS refuses some inputs at preflight. One refusal can be waived by the
person responsible for the data, after they have looked at the problem; an
automated agent (AI assistant, pipeline script) must never write a waiver or
name itself or the user as approver without that person's explicit decision.

## Duplicated homoeolog genes

A gene that occurs in more than one group of the same subgenome normally means
paralog contamination. To analyse such a table on purpose (e.g. a sensitivity
analysis), the data owner adds to the `interact` block:

```yaml
duplicate_gene_waiver:
  reason: "<why this table is analysed as is; at least 30 characters>"
  approver: "<name of the person who decided>"
  table_sha256: "<sha256 of the exact table file>"
```

The waiver is valid only for that exact file (any edit changes the hash). It is
copied into `input_qc.group_table.waiver` of the result JSON, the run prints
`input check OVERRIDDEN`, `audit` flags `OVERRIDDEN_INPUT_CHECK` (review) and the
report card lists the statement every report of this run must carry.
