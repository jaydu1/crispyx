## Benchmark summary

- **Methods executed:** 8
- **Succeeded:** 3 (37.5% success rate)
- **Did not succeed:** 5
  - Memory limit exceeded: 4
  - Errors: 1
- **Average runtime:** 2440.531s
- **Notable issues:**
  - Other errors recorded:
    - MemoryError: Unable to allocate 109. GiB for an array with shape (29136391388,) and data type int32
    - MemoryError: Unable to allocate 217. GiB for an array with shape (29136391388,) and data type int64
    - MemoryError: Unable to allocate 4.32 MiB for an array with shape (4534299,) and data type bool
    - [Errno 12] Cannot allocate memory

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | result_path |
| --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 4836.866 | 29081.855 | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 4091.182 | 29049.242 | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | error | 3394.655 |  |  |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 743.524 | 112466.902 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_t_test | t-test differential expression test | success | 3432.236 | 47251.062 | 18311.0 | 38606.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | memory_limit | 1489.853 | 123132.027 |  |  |  |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_de_t_test | memory_limit | 1042.252 | 111890.281 |
| scanpy_de_wilcoxon | memory_limit | 493.68 | 112429.215 |

