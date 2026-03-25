## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 9 (81.8% success rate)
- **Did not succeed:** 2
  - Timeouts: 1
  - Errors: 1
- **Average runtime:** 6218.818s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds
    - Process exited with code -9

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 138.901 | 1581.109 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 30.393 | 1373.109 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 39.738 | 30583.453 | 304114.0 | 8563.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 83.774 | 40468.148 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 7311.273 | 5692.43 | 2057.0 | 8563.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 82.697 | 1708.094 | 2057.0 | 8563.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 622.276 | 17029.656 | 2057.0 | 8563.0 | de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 21605.372 |  |
| pertpy_de_pydeseq2 | error | 17780.002 | 28907.996 |
| scanpy_de_t_test | success | 181.832 | 14844.59 |
| scanpy_de_wilcoxon | success | 20530.736 | 23824.699 |

