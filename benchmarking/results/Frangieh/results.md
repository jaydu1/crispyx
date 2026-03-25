## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 7 (63.6% success rate)
- **Did not succeed:** 4
  - Timeouts: 2
  - Memory limit exceeded: 1
  - Errors: 1
- **Average runtime:** 4747.713s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds
    - MemoryError: Unable to allocate 10.2 GiB for an array with shape (57605, 23712) and data type float64
    - Process exited with code 2

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 291.886 | 8070.164 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 267.409 | 7328.941 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 71.992 | 17438.68 | 218023.0 | 23710.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 103.028 | 23088.996 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | memory_limit | 5655.446 | 117756.695 |  |  |  |
| crispyx_de_t_test | t-test differential expression test | success | 47.293 | 860.289 | 248.0 | 23712.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 1131.662 | 8482.113 | 248.0 | 23712.0 | de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | error | 1366.925 |  |
| pertpy_de_pydeseq2 | timeout | 21605.154 |  |
| scanpy_de_t_test | success | 78.994 | 9631.305 |
| scanpy_de_wilcoxon | timeout | 21605.051 |  |

