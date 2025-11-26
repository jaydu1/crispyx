# Benchmark Results

## Performance

| method | status | elapsed_seconds | peak_memory_mb | avg_memory_mb | cells_kept | genes_kept | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | success | 40.39 | 102.965 | 56.694 |  |  | 2.0 |
| crispyx_de_t_test | success | 0.222 | 124.941 | 64.346 |  |  | 2.0 |
| crispyx_de_wilcoxon | success | 3.734 | 100.324 | 66.65 |  |  | 2.0 |
| crispyx_pb_avg_log | success | 0.444 | 506.23 | 209.938 |  |  |  |
| crispyx_pb_pseudobulk | success | 0.345 | 353.707 | 124.382 |  |  |  |
| crispyx_qc_filtered | success | 0.844 | 322.547 | 118.014 | 1716.0 | 10500.0 |  |
| edger_de_glm | success | 97.318 | 1962.645 | 1473.276 |  |  | 2.0 |
| pertpy_de_pydeseq2 | success | 622.177 | 2465.062 | 1064.904 |  |  | 2.0 |
| scanpy_de_t_test | success | 0.62 | 109.676 | 33.673 |  |  | 2.0 |
| scanpy_de_wilcoxon | success | 0.879 | 155.844 | 70.514 |  |  | 2.0 |
| scanpy_qc_filtered | success | 0.424 | 199.781 | 114.348 | 1716.0 | 10500.0 |  |


## Performance Comparison

| comparison | crispyx_time_s | other_time_s | time_diff_s | time_pct | crispyx_mem_mb | other_mem_mb | mem_diff_mb | mem_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.844 | 0.424 | 0.419 | 198.768 | 322.547 | 199.781 | 122.766 | 161.45 |
| crispyx_de_nb_glm vs edger_de_glm | 40.39 | 97.318 | -56.928 | 41.503 | 102.965 | 1962.645 | -1859.68 | 5.246 |
| crispyx_de_nb_glm vs pertpy_de_pydeseq2 | 40.39 | 622.177 | -581.786 | 6.492 | 102.965 | 2465.062 | -2362.098 | 4.177 |
| crispyx_de_t_test vs scanpy_de_t_test | 0.222 | 0.62 | -0.398 | 35.849 | 124.941 | 109.676 | 15.266 | 113.919 |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon | 3.734 | 0.879 | 2.855 | 424.828 | 100.324 | 155.844 | -55.52 | 64.375 |


## Accuracy

| comparison | cells_diff | genes_diff | effect_max_abs_diff | effect_pearson_corr | effect_spearman_corr | effect_top_k_overlap | statistic_max_abs_diff | statistic_pearson_corr | statistic_spearman_corr | statistic_top_k_overlap | pvalue_max_abs_diff | pvalue_pearson_corr | pvalue_spearman_corr | pvalue_top_k_overlap | pvalue_stream_auroc | pvalue_reference_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| crispyx_de_nb_glm vs edger_de_glm |  |  | 2.465 | 0.791 | 0.911 | 0.32 | 729.928 | -0.112 | -0.317 | 0.6 | 1.0 | 0.82 | 0.87 | 0.64 |  |  |
| crispyx_de_nb_glm vs pertpy_de_pydeseq2 |  |  | 2.646 | 0.56 | 0.639 | 0.18 | 8.881 | 0.911 | 0.802 | 0.78 | 0.996 | 0.575 | 0.622 | 0.78 |  |  |
| crispyx_de_t_test vs scanpy_de_t_test |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 0.001 | 1.0 | 1.0 | 1.0 | 0.001 | 1.0 | 1.0 | 0.98 |  |  |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon |  |  | 24.029 | 0.346 | 0.848 | 0.0 | 4.658 | 0.975 | 0.979 | 0.94 | 0.758 | 0.886 | 0.932 | 0.94 |  |  |

