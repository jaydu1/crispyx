# Benchmark Results

## Performance

| method | status | elapsed_seconds | peak_memory_mb | avg_memory_mb | cells_kept | genes_kept | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | success | 33.281 | 578.898 | 440.719 |  |  | 2.0 |
| crispyx_de_t_test | success | 2.071 | 185.598 | 117.057 |  |  | 2.0 |
| crispyx_de_wilcoxon | success | 4.011 | 301.422 | 224.529 |  |  | 2.0 |
| crispyx_pb_avg_log | success | 1.867 | 563.492 | 288.844 |  |  |  |
| crispyx_pb_pseudobulk | success | 1.703 | 402.367 | 164.363 |  |  |  |
| crispyx_qc_filtered | success | 1.829 | 349.188 | 128.493 | 1716.0 | 10500.0 |  |
| edger_de_glm | success | 69.179 | 3541.758 | 2230.327 |  |  | 2.0 |
| pertpy_de_pydeseq2 | success | 318.437 | 3407.516 | 1814.625 |  |  | 2.0 |
| scanpy_de_t_test | success | 3.505 | 314.801 | 182.823 |  |  | 2.0 |
| scanpy_de_wilcoxon | success | 6.92 | 655.109 | 379.183 |  |  | 2.0 |
| scanpy_qc_filtered | success | 2.29 | 345.363 | 101.002 | 1716.0 | 10500.0 |  |


## Performance Comparison

| comparison | crispyx_time_s | other_time_s | time_diff_s | time_pct | crispyx_mem_mb | other_mem_mb | mem_diff_mb | mem_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 1.829 | 2.29 | -0.461 | 79.874 | 349.188 | 345.363 | 3.824 | 101.107 |
| crispyx_de_nb_glm vs edger_de_glm | 33.281 | 69.179 | -35.898 | 48.109 | 578.898 | 3541.758 | -2962.859 | 16.345 |
| crispyx_de_nb_glm vs pertpy_de_pydeseq2 | 33.281 | 318.437 | -285.155 | 10.451 | 578.898 | 3407.516 | -2828.617 | 16.989 |
| crispyx_de_t_test vs scanpy_de_t_test | 2.071 | 3.505 | -1.434 | 59.085 | 185.598 | 314.801 | -129.203 | 58.957 |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon | 4.011 | 6.92 | -2.909 | 57.961 | 301.422 | 655.109 | -353.688 | 46.011 |


## Accuracy

| comparison | cells_diff | genes_diff | effect_max_abs_diff | effect_pearson_corr | effect_spearman_corr | effect_top_k_overlap | statistic_max_abs_diff | statistic_pearson_corr | statistic_spearman_corr | statistic_top_k_overlap | pvalue_max_abs_diff | pvalue_pearson_corr | pvalue_spearman_corr | pvalue_top_k_overlap | pvalue_stream_auroc | pvalue_reference_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| crispyx_de_nb_glm vs edger_de_glm |  |  | 2.465 | 0.791 | 0.911 | 0.32 | 729.928 | -0.112 | -0.317 | 0.6 | 1.0 | 0.82 | 0.87 | 0.64 |  |  |
| crispyx_de_nb_glm vs pertpy_de_pydeseq2 |  |  | 2.646 | 0.56 | 0.639 | 0.18 | 8.881 | 0.911 | 0.802 | 0.78 | 0.996 | 0.575 | 0.622 | 0.78 |  |  |
| crispyx_de_t_test vs scanpy_de_t_test |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 0.001 | 1.0 | 1.0 | 1.0 | 0.001 | 1.0 | 1.0 | 0.98 |  |  |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 4.658 | 0.975 | 0.979 | 0.94 | 0.758 | 0.886 | 0.932 | 0.94 |  |  |

