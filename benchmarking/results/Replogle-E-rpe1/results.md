## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 8 (72.7% success rate)
- **Did not succeed:** 3
  - Timeouts: 2
  - Errors: 1
- **Average runtime:** 6807.269s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds
    - Process exited with code -9

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 33.331 | 1527.25 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 25.503 | 1383.469 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 29.064 | 25133.246 | 244412.0 | 8749.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 51.789 | 33284.422 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 12599.408 | 6521.04 | 2393.0 | 8749.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 77.036 | 1874.094 | 2393.0 | 8749.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 738.461 | 13829.727 | 2393.0 | 8749.0 | de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 21605.037 |  |
| pertpy_de_pydeseq2 | error | 17931.276 | 31149.645 |
| scanpy_de_t_test | success | 183.952 | 11730.883 |
| scanpy_de_wilcoxon | timeout | 21605.101 |  |

