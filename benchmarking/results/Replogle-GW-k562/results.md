## Benchmark summary

- **Methods executed:** 3
- **Succeeded:** 2 (66.7% success rate)
- **Did not succeed:** 1
  - Memory limit exceeded: 1
- **Average runtime:** 2771.089s
- **Notable issues:**
  - Other errors recorded:
    - MemoryError: Unable to allocate 61.1 GiB for an array with shape (1989578, 8248) and data type float32

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered | Streaming quality control filters | success | 1352.964 | 314959.508 | 1971608.0 | 8248.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | memory_limit | 65.336 | 79083.5 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 6894.968 | 26493.15 | 9866.0 | 8248.0 | de/crispyx_de_nb_glm.h5ad |

