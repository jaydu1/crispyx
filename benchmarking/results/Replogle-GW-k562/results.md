## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 0 (0.0% success rate)
- **Did not succeed:** 11
  - Timeouts: 1
  - Memory limit exceeded: 2
  - Errors: 8
- **Average runtime:** 8131.373s
- **Notable issues:**
  - Other errors recorded:
    - Exceeded time limit of 86400 seconds
    - MemoryError: Unable to allocate 61.1 GiB for an array with shape (1989578, 8248) and data type float32
    - Output dtype not compatible with inputs.
    - Process exited with code 2
    - compute_average_log_expression() got an unexpected keyword argument 'path'
    - compute_pseudobulk_expression() got an unexpected keyword argument 'path'
    - nb_glm_test() got an unexpected keyword argument 'path'
    - quality_control_summary() got an unexpected keyword argument 'path'
    - t_test() got an unexpected keyword argument 'path'
    - wilcoxon_test() got an unexpected keyword argument 'path'

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds |
| --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | error | 0.55 |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | error | 0.517 |
| crispyx_qc_filtered | Streaming quality control filters | error | 0.904 |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 1108.37 | 126242.984 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | error | 5.615 | 322.746 |
| crispyx_de_t_test | t-test differential expression test | error | 2.576 | 325.82 |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | error | 2.541 | 320.285 |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | error | 702.594 |  |
| pertpy_de_pydeseq2 | timeout | 86405.101 |  |
| scanpy_de_t_test | error | 436.009 | 67600.281 |
| scanpy_de_wilcoxon | memory_limit | 780.321 | 97825.332 |

