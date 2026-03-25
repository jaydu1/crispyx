## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 4 (36.4% success rate)
- **Did not succeed:** 7
  - Memory limit exceeded: 5
  - Errors: 2
- **Average runtime:** 4294.649s
- **Notable issues:**
  - Other errors recorded:
    - MemoryError: Unable to allocate 146. GiB for an array with shape (535083, 36518) and data type float64
    - MemoryError: Unable to allocate 158. GiB for an array with shape (1161864, 36518) and data type float32
    - Output dtype not compatible with inputs.
    - Process exited with code -9

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 3057.635 | 4674.375 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 2612.103 | 3532.863 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 8505.394 | 46701.621 | 1161864.0 | 33165.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 16.734 | 345.625 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | memory_limit | 27050.187 | 76968.152 |  |  |  |
| crispyx_de_t_test | t-test differential expression test | success | 320.439 | 1825.574 | 444.0 | 36518.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | error | 5582.078 | 102774.695 |  |  |  |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | memory_limit | 11.143 |  |
| pertpy_de_pydeseq2 | memory_limit | 16.886 | 365.551 |
| scanpy_de_t_test | error | 65.231 | 46666.398 |
| scanpy_de_wilcoxon | memory_limit | 3.311 | 352.758 |

