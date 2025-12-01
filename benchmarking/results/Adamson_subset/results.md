# Benchmark Results

## Performance

| method | status | elapsed_seconds | peak_memory_mb | avg_memory_mb | cells_kept | genes_kept | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | success | 55.259 | 67.051 | 25.273 |  |  | 2.0 |
| crispyx_de_t_test | success | 1.02 | 89.977 | 35.841 |  |  | 2.0 |
| crispyx_de_wilcoxon | success | 8.079 | 59.199 | 54.326 |  |  | 2.0 |
| crispyx_pb_avg_log | success | 3.051 | 483.539 | 212.489 |  |  |  |
| crispyx_pb_pseudobulk | success | 2.445 | 331.09 | 149.056 |  |  |  |
| crispyx_qc_filtered | success | 2.977 | 272.617 | 104.674 | 1716.0 | 10500.0 |  |
| edger_de_glm | error | 0.188 |  |  |  |  |  |
| pertpy_de_pydeseq2 | error | 3.107 |  |  |  |  |  |
| scanpy_de_t_test | success | 6.189 | 0.0 | 0.0 |  |  | 2.0 |
| scanpy_de_wilcoxon | success | 14.35 | 184.469 | 0.0 |  |  | 2.0 |
| scanpy_qc_filtered | success | 4.298 | 0.0 | 0.0 | 1716.0 | 10500.0 |  |


## Performance Comparison

| comparison | crispyx_time_s | other_time_s | time_diff_s | time_pct | crispyx_mem_mb | other_mem_mb | mem_diff_mb | mem_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 2.977 | 4.298 | -1.321 | 69.26 | 272.617 | 0.0 | 272.617 |  |
| crispyx_de_t_test vs scanpy_de_t_test | 1.02 | 6.189 | -5.169 | 16.488 | 89.977 | 0.0 | 89.977 |  |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon | 8.079 | 14.35 | -6.271 | 56.297 | 59.199 | 184.469 | -125.27 | 32.092 |


## Accuracy

| comparison | cells_diff | genes_diff | effect_max_abs_diff | effect_pearson_corr | effect_spearman_corr | effect_top_k_overlap | statistic_max_abs_diff | statistic_pearson_corr | statistic_spearman_corr | statistic_top_k_overlap | pvalue_max_abs_diff | pvalue_pearson_corr | pvalue_spearman_corr | pvalue_top_k_overlap | pvalue_stream_auroc | pvalue_reference_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| crispyx_de_t_test vs scanpy_de_t_test |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 0.001 | 1.0 | 1.0 | 1.0 | 0.001 | 1.0 | 1.0 | 0.98 |  |  |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 4.658 | 0.975 | 0.979 | 0.94 | 0.758 | 0.886 | 0.932 | 0.94 |  |  |

