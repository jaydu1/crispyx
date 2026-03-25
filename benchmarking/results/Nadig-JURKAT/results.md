## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 7 (63.6% success rate)
- **Did not succeed:** 4
  - Timeouts: 1
  - Errors: 3
- **Average runtime:** 4514.536s
- **Notable issues:**
  - Other errors recorded:
    - Could not calculate statistics for groups NUP155 since they only contain one sample.
    - Exceeded time limit of 21600 seconds
    - Process exited with code -9

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 36.219 | 1483.477 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 27.793 | 1224.773 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 31.6 | 26901.379 | 259086.0 | 8882.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 55.795 | 35684.352 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 11258.18 | 6629.15 | 2393.0 | 8882.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 84.201 | 1824.98 | 2393.0 | 8882.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 771.815 | 14568.074 | 2393.0 | 8882.0 | de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 21605.109 |  |
| pertpy_de_pydeseq2 | error | 15593.188 | 18283.238 |
| scanpy_de_t_test | error | 49.698 | 8078.621 |
| scanpy_de_wilcoxon | error | 146.304 | 9911.797 |

