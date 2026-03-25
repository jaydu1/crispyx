## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 9 (81.8% success rate)
- **Did not succeed:** 2
  - Timeouts: 1
  - Errors: 1
- **Average runtime:** 4673.530s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 21600 seconds
    - Process exited with code -9

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 21.482 | 1563.496 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 16.402 | 1408.371 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 19.543 | 16163.219 | 142281.0 | 9624.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 31.962 | 21360.129 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 5071.153 | 6564.95 | 2393.0 | 9624.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 68.482 | 1964.664 | 2393.0 | 9624.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 560.233 | 9738.324 | 2393.0 | 9624.0 | de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 21605.106 |  |
| pertpy_de_pydeseq2 | error | 12291.075 | 9856.316 |
| scanpy_de_t_test | success | 181.598 | 9376.875 |
| scanpy_de_wilcoxon | success | 11541.795 | 12721.445 |

