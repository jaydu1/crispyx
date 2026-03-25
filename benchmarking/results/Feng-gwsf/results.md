## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 4 (36.4% success rate)
- **Did not succeed:** 7
  - Memory limit exceeded: 3
  - Errors: 4
- **Average runtime:** 429.129s
- **Notable issues:**
  - Other errors recorded:
    - Could not calculate statistics for groups PSMA3, RPL26, DDX10, RPL35A, CDCA8 since they only contain one sample.
    - Could not calculate statistics for groups RPL35A, PSMA3, DDX10, CDCA8, RPL26 since they only contain one sample.
    - MemoryError: Unable to allocate 43.9 GiB for an array with shape (322746, 36518) and data type float32
    - MemoryError: Unable to allocate 59.5 GiB for an array with shape (218652, 36518) and data type float64
    - MemoryError: Unable to allocate 59.5 GiB for an array with shape (218701, 36518) and data type int64
    - Process exited with code -9
    - Process exited with code 2

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 376.638 | 5694.734 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 100.131 | 4788.109 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 277.989 | 123087.883 | 320706.0 | 31772.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 409.215 | 90223.477 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | memory_limit | 1255.188 | 99443.262 |  |  |  |
| crispyx_de_t_test | t-test differential expression test | success | 227.467 | 5912.848 | 2254.0 | 36518.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | error | 1001.107 | 22385.484 |  |  |  |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | error | 593.664 |  |
| pertpy_de_pydeseq2 | memory_limit | 76.869 | 106261.266 |
| scanpy_de_t_test | error | 252.543 | 15896.562 |
| scanpy_de_wilcoxon | error | 149.609 | 46025.59 |

