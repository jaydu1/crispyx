# Benchmark Results

## Performance

| method | status | elapsed_seconds | peak_memory_mb | avg_memory_mb | cells_kept | genes_kept | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | success | 28.896 | 234.562 | 224.363 |  |  | 2.0 |
| crispyx_de_t_test | success | 2.202 | 191.922 | 92.422 |  |  | 2.0 |
| crispyx_de_wilcoxon | success | 3.699 | 167.953 | 138.519 |  |  | 2.0 |
| crispyx_pb_avg_log | success | 2.31 | 552.516 | 299.406 |  |  |  |
| crispyx_pb_pseudobulk | success | 2.086 | 403.312 | 0.0 |  |  |  |
| crispyx_qc_filtered | success | 2.397 | 355.344 | 198.855 | 1716.0 | 10500.0 |  |
| edger_de_glm | success | 67.106 | 3485.828 | 2573.597 |  |  | 2.0 |
| pertpy_de_pydeseq2 | success | 620.906 | 3272.672 | 1565.686 |  |  | 2.0 |
| scanpy_de_t_test | success | 3.486 | 248.719 | 97.158 |  |  | 2.0 |
| scanpy_de_wilcoxon | success | 3.139 | 507.297 | 201.0 |  |  | 2.0 |
| scanpy_qc_filtered | success | 2.767 | 359.406 | 99.455 | 1716.0 | 10500.0 |  |


## Performance Comparison

| comparison | crispyx_time_s | other_time_s | time_diff_s | time_pct | crispyx_mem_mb | other_mem_mb | mem_diff_mb | mem_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 2.397 | 2.767 | -0.371 | 86.609 | 355.344 | 359.406 | -4.062 | 98.87 |
| crispyx_de_nb_glm vs edger_de_glm | 28.896 | 67.106 | -38.21 | 43.06 | 234.562 | 3485.828 | -3251.266 | 6.729 |
| crispyx_de_nb_glm vs pertpy_de_pydeseq2 | 28.896 | 620.906 | -592.01 | 4.654 | 234.562 | 3272.672 | -3038.109 | 7.167 |
| crispyx_de_t_test vs scanpy_de_t_test | 2.202 | 3.486 | -1.284 | 63.175 | 191.922 | 248.719 | -56.797 | 77.164 |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon | 3.699 | 3.139 | 0.56 | 117.854 | 167.953 | 507.297 | -339.344 | 33.107 |


## Accuracy

| comparison | cells_diff | genes_diff | effect_max_abs_diff | effect_pearson_corr | effect_spearman_corr | effect_top_k_overlap | statistic_max_abs_diff | statistic_pearson_corr | statistic_spearman_corr | statistic_top_k_overlap | pvalue_max_abs_diff | pvalue_pearson_corr | pvalue_spearman_corr | pvalue_top_k_overlap | pvalue_stream_auroc | pvalue_reference_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| crispyx_de_nb_glm vs edger_de_glm |  |  | 2.465 | 0.791 | 0.911 | 0.32 | 600.159 | -0.168 | -0.331 | 0.64 | 1.0 | 0.822 | 0.87 | 0.66 |  |  |
| crispyx_de_nb_glm vs pertpy_de_pydeseq2 |  |  | 2.646 | 0.56 | 0.639 | 0.18 | 8.881 | 0.911 | 0.802 | 0.78 | 0.996 | 0.575 | 0.622 | 0.78 |  |  |
| crispyx_de_t_test vs scanpy_de_t_test |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 0.001 | 1.0 | 1.0 | 1.0 | 0.001 | 1.0 | 1.0 | 0.98 |  |  |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 4.658 | 0.975 | 0.979 | 0.94 | 0.758 | 0.886 | 0.932 | 0.94 |  |  |

