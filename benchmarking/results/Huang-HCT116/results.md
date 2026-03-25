## Benchmark summary

- **Methods executed:** 8
- **Succeeded:** 3 (37.5% success rate)
- **Did not succeed:** 5
  - Memory limit exceeded: 4
  - Errors: 1
- **Average runtime:** 1537.313s
- **Notable issues:**
  - Other errors recorded:
    - MemoryError: Unable to allocate 130. GiB for an array with shape (17414414157,) and data type int64
    - MemoryError: Unable to allocate 3.25 MiB for an array with shape (3409169,) and data type bool
    - MemoryError: Unable to allocate 64.9 GiB for an array with shape (17414414157,) and data type int32
    - [Errno 12] Cannot allocate memory

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | result_path |
| --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 3193.915 | 29440.82 | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 2888.485 | 29040.273 | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | error | 1383.072 |  |  |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 171.846 | 67281.477 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_t_test | t-test differential expression test | success | 2469.267 | 46927.543 | 18293.0 | 38606.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | memory_limit | 1474.387 | 91401.383 |  |  |  |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_de_t_test | memory_limit | 544.97 | 67022.062 |
| scanpy_de_wilcoxon | memory_limit | 172.562 | 67278.918 |

