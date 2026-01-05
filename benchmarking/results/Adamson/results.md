## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 10 (90.9% success rate)
  - Recovered (completed after timeout): 1
- **Did not succeed:** 1
  - Timeouts: 1
- **Average runtime:** 1012.875s
- **Notable issues:**
  - Other errors recorded:
    - Container timed out after 3600s

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 22.761 | 1937.43 |  |  |  |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 16.887 | 1426.871 |  |  |  |
| crispyx_qc_filtered | Streaming quality control filters | success | 23.39 | 1898.352 | 65282.0 | 19568.0 | benchmarking/results/Adamson/preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 16.248 | 7354.891 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 2866.062 | 3397.34 | 91.0 | 32738.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 7.162 | 943.426 | 91.0 | 32738.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 81.003 | 3799.453 | 91.0 | 32738.0 | benchmarking/results/Adamson/de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 3660.105 |  |
| pertpy_de_pydeseq2 | recovered | 3660.08 |  |
| scanpy_de_t_test | success | 22.94 | 3046.578 |
| scanpy_de_wilcoxon | success | 764.992 | 3735.445 |

