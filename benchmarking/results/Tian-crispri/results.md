## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 9 (81.8% success rate)
- **Did not succeed:** 2
  - Timeouts: 1
  - Errors: 1
- **Average runtime:** 588.164s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 3600 seconds
    - Process exited with code -9

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 28.32 | 4088.863 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 25.01 | 3040.664 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 37.519 | 13856.027 | 32255.0 | 23451.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 11.785 | 4561.699 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 404.754 | 2243.95 | 184.0 | 33538.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 8.006 | 920.422 | 184.0 | 33538.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 43.568 | 2236.125 | 184.0 | 33538.0 | de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 3605.105 |  |
| pertpy_de_pydeseq2 | error | 2006.881 | 5470.621 |
| scanpy_de_t_test | success | 35.702 | 2593.883 |
| scanpy_de_wilcoxon | success | 263.159 | 3632.422 |

