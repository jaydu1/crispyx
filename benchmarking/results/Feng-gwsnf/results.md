## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 4 (36.4% success rate)
- **Did not succeed:** 7
  - Memory limit exceeded: 3
  - Errors: 4
- **Average runtime:** 358.824s
- **Notable issues:**
  - Other errors recorded:
    - Could not calculate statistics for groups BFAR, ITM2A, RAD1, ZNF830 since they only contain one sample.
    - Could not calculate statistics for groups BFAR, ZNF830, RAD1, ITM2A since they only contain one sample.
    - MemoryError: Unable to allocate 38.3 GiB for an array with shape (281380, 36518) and data type float32
    - MemoryError: Unable to allocate 53.9 GiB for an array with shape (396458, 36518) and data type float32
    - MemoryError: Unable to allocate 76.5 GiB for an array with shape (281346, 36518) and data type float64
    - Process exited with code 2
    - Worker process crashed (exitcode=-9)

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 420.167 | 9475.867 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 142.245 | 8573.059 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 400.023 | 17834.848 | 393465.0 | 32373.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 304.178 | 110887.223 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | memory_limit | 541.207 | 57205.656 |  |  |  |
| crispyx_de_t_test | t-test differential expression test | success | 276.635 | 12399.695 | 4955.0 | 36518.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | error | 1055.229 | 47244.781 |  |  |  |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | error | 547.62 |  |
| pertpy_de_pydeseq2 | memory_limit | 52.624 | 94747.809 |
| scanpy_de_t_test | error | 92.249 | 19590.02 |
| scanpy_de_wilcoxon | error | 114.881 | 57444.816 |

