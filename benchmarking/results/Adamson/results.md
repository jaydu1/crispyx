## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 10 (90.9% success rate)
- **Did not succeed:** 1
  - Timeouts: 1
- **Average runtime:** 1442.787s
- **Notable issues:**
  - Other errors recorded:
    - Container timed out after 7200s

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 22.801 | 1966.621 |  |  |  |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 16.598 | 1455.793 |  |  |  |
| crispyx_qc_filtered | Streaming quality control filters | success | 22.241 | 2708.719 | 65282.0 | 19568.0 | benchmarking/results/Adamson/preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 16.005 | 7356.805 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 1699.803 | 3433.71 | 91.0 | 32738.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 6.664 | 349.391 | 91.0 | 32738.0 | benchmarking/results/Adamson/de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 83.483 | 3841.77 | 91.0 | 32738.0 | benchmarking/results/Adamson/de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 7260.107 |  |
| pertpy_de_pydeseq2 | success | 5944.332 | 45839.973 |
| scanpy_de_t_test | success | 22.738 | 3049.934 |
| scanpy_de_wilcoxon | success | 775.891 | 3736.016 |

