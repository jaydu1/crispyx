# Benchmark Results

## Performance

| method | status | elapsed_seconds | peak_memory_mb | avg_memory_mb | cells_kept | genes_kept | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | success | 71.72 | 66.23 | 24.018 |  |  | 2.0 |
| crispyx_de_t_test | success | 1.034 | 89.762 | 38.082 |  |  | 2.0 |
| crispyx_de_wilcoxon | success | 4.528 | 81.602 | 69.632 |  |  | 2.0 |
| crispyx_pb_avg_log | success | 3.157 | 483.074 | 230.352 |  |  |  |
| crispyx_pb_pseudobulk | success | 2.137 | 330.625 | 136.009 |  |  |  |
| crispyx_qc_filtered | success | 2.985 | 274.855 | 103.261 | 1716.0 | 10500.0 |  |
| edger_de_glm | error | 0.186 |  |  |  |  |  |
| pertpy_de_pydeseq2 | error | 3.449 |  |  |  |  |  |
| scanpy_de_t_test | error | 3.207 |  |  |  |  |  |
| scanpy_de_wilcoxon | error | 0.464 |  |  |  |  |  |
| scanpy_qc_filtered | success | 1.627 | 178.504 | 85.64 | 1716.0 | 10500.0 |  |


## Performance Comparison

| comparison | crispyx_time_s | other_time_s | time_diff_s | time_pct | crispyx_mem_mb | other_mem_mb | mem_diff_mb | mem_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 2.985 | 1.627 | 1.358 | 183.419 | 274.855 | 178.504 | 96.352 | 153.977 |


## Accuracy

| comparison | cells_diff | genes_diff |
| --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.0 | 0.0 |

