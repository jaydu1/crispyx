## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 9 (81.8% success rate)
- **Did not succeed:** 2
  - Timeouts: 1
  - Memory limit exceeded: 1
- **Average runtime:** 2536.822s
- **Notable issues:**
  - Other errors recorded:
    - Container killed (OOM): DEBUG: Importing module 'benchmarking.tools.run_benchmarks' -> 'benchmarking.tools.run_benchmarks' function 'run_edger_de'
DEBUG: sys.path[:5] = ['/app', '/app', '/workspace', '/usr/local/lib/python311.zip', '/usr/local/lib/python3.11']
R callback write-console: Loading required package: limma
  

    - Container timed out after 10800s

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 34.74 | 947.57 |  |  |  |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 27.888 | 872.973 |  |  |  |
| crispyx_qc_filtered | Streaming quality control filters | success | 222.404 | 8662.855 | 304114.0 | 8563.0 | benchmarking/results/Replogle-E-k562/preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 68.33 | 40198.453 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 8628.816 | 33064.367 | 2057.0 | 8563.0 | benchmarking/results/Replogle-E-k562/de/crispyx_de_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 52.491 | 1525.504 | 2057.0 | 8563.0 | benchmarking/results/Replogle-E-k562/de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 526.475 | 23009.711 | 2057.0 | 8563.0 | benchmarking/results/Replogle-E-k562/de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | memory_limit | 2437.111 |  |
| pertpy_de_pydeseq2 | timeout | 10860.106 |  |
| scanpy_de_t_test | success | 136.609 | 14699.109 |
| scanpy_de_wilcoxon | success | 4910.077 | 23561.613 |

