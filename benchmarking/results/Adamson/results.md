## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 10 (90.9% success rate)
- **Did not succeed:** 1
  - Timeouts: 1
- **Average runtime:** 3016.779s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 23.423 | 3860.125 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 16.86 | 2838.652 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 8.643 | 5702.973 | 65282.0 | 19568.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 16.241 | 7439.895 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 2448.905 | 3492.4 | 91.0 | 32738.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 19.489 | 696.688 | 91.0 | 32738.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 136.682 | 2887.094 | 91.0 | 32738.0 | de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 21605.14 |  |
| pertpy_de_pydeseq2 | success | 6710.994 | 46285.703 |
| scanpy_de_t_test | success | 46.112 | 3355.434 |
| scanpy_de_wilcoxon | success | 2152.085 | 3897.691 |

