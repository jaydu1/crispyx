## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 10 (90.9% success rate)
- **Did not succeed:** 1
  - Timeouts: 1
- **Average runtime:** 467.629s
- **Notable issues:**
  - Other errors recorded:
    - Container timed out after 3600s

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 29.083 | 1963.141 |  |  |  |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 27.199 | 1437.91 |  |  |  |
| crispyx_qc_filtered | Streaming quality control filters | success | 13.47 | 2146.699 | 21071.0 | 22040.0 | benchmarking/results/Tian-crispra/preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 8.315 | 2317.141 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 344.237 | 1465.13 | 100.0 | 33538.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 5.158 | 355.082 | 100.0 | 33538.0 | benchmarking/results/Tian-crispra/de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 19.745 | 1420.98 | 100.0 | 33538.0 | benchmarking/results/Tian-crispra/de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 3660.105 |  |
| pertpy_de_pydeseq2 | success | 959.826 | 5224.406 |
| scanpy_de_t_test | success | 20.313 | 1259.094 |
| scanpy_de_wilcoxon | success | 56.465 | 1778.312 |

