## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 10 (90.9% success rate)
- **Did not succeed:** 1
  - Timeouts: 1
- **Average runtime:** 470.341s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 3600 seconds

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 17.527 | 3793.977 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 15.019 | 2745.715 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 21.457 | 8143.57 | 21071.0 | 22040.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 7.632 | 2434.047 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 220.45 | 1591.27 | 100.0 | 33538.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 5.476 | 647.711 | 100.0 | 33538.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 24.483 | 1388.465 | 100.0 | 33538.0 | de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 3605.056 |  |
| pertpy_de_pydeseq2 | success | 1100.903 | 5427.602 |
| scanpy_de_t_test | success | 20.261 | 1595.48 |
| scanpy_de_wilcoxon | success | 135.481 | 2019.652 |

