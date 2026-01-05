## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 10 (90.9% success rate)
- **Did not succeed:** 1
  - Timeouts: 1
- **Average runtime:** 462.760s
- **Notable issues:**
  - Other errors recorded:
    - Container timed out after 3600s

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 29.635 | 1961.574 |  |  |  |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 27.407 | 1437.668 |  |  |  |
| crispyx_qc_filtered | Streaming quality control filters | success | 16.361 | 1577.879 | 21071.0 | 22040.0 | benchmarking/results/Tian-crispra/preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 8.638 | 2315.57 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 247.535 | 1463.82 | 100.0 | 33538.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 7.808 | 812.73 | 100.0 | 33538.0 | benchmarking/results/Tian-crispra/de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 23.269 | 1413.09 | 100.0 | 33538.0 | benchmarking/results/Tian-crispra/de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | timeout | 3660.105 |  |
| pertpy_de_pydeseq2 | success | 999.44 | 5222.949 |
| scanpy_de_t_test | success | 20.228 | 1256.223 |
| scanpy_de_wilcoxon | success | 49.934 | 1775.773 |

