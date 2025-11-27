# Benchmark Results

## Performance

| method | status | elapsed_seconds | peak_memory_mb | avg_memory_mb | cells_kept | genes_kept | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | success | 76.713 | 64.746 | 21.417 |  |  | 2.0 |
| crispyx_de_t_test | success | 1.168 | 89.59 | 40.241 |  |  | 2.0 |
| crispyx_de_wilcoxon | success | 4.458 | 67.613 | 58.297 |  |  | 2.0 |
| crispyx_pb_avg_log | success | 3.467 | 483.082 | 228.696 |  |  |  |
| crispyx_pb_pseudobulk | success | 2.966 | 330.633 | 164.617 |  |  |  |
| crispyx_qc_filtered | success | 3.3 | 273.684 | 108.45 | 1716.0 | 10500.0 |  |
| edger_de_glm | error | 0.182 |  |  |  |  |  |
| pertpy_de_pydeseq2 | error | 3.919 |  |  |  |  |  |
| scanpy_de_t_test | error | 3.424 |  |  |  |  |  |
| scanpy_de_wilcoxon | error | 0.52 |  |  |  |  |  |
| scanpy_qc_filtered | success | 1.74 | 178.484 | 86.982 | 1716.0 | 10500.0 |  |


## Performance Comparison

| comparison | crispyx_time_s | other_time_s | time_diff_s | time_pct | crispyx_mem_mb | other_mem_mb | mem_diff_mb | mem_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 3.3 | 1.74 | 1.56 | 189.617 | 273.684 | 178.484 | 95.199 | 153.338 |


## Accuracy

| comparison | cells_diff | genes_diff |
| --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.0 | 0.0 |

