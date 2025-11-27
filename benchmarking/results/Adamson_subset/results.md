# Benchmark Results

## Performance

| method | status | elapsed_seconds | peak_memory_mb | avg_memory_mb | cells_kept | genes_kept | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | success | 70.694 | 60.652 | 14.772 |  |  | 2.0 |
| crispyx_de_t_test | success | 0.997 | 83.426 | 33.404 |  |  | 2.0 |
| crispyx_de_wilcoxon | success | 4.511 | 84.121 | 68.306 |  |  | 2.0 |
| crispyx_pb_avg_log | success | 2.865 | 461.52 | 180.438 |  |  |  |
| crispyx_pb_pseudobulk | success | 2.224 | 309.066 | 124.458 |  |  |  |
| crispyx_qc_filtered | success | 2.34 | 278.133 | 90.54 | 1716.0 | 10500.0 |  |
| scanpy_de_t_test | success | 4.505 | 0.0 | 0.0 |  |  | 2.0 |
| scanpy_de_wilcoxon | success | 15.013 | 234.242 | 36.629 |  |  | 2.0 |
| scanpy_qc_filtered | success | 4.483 | 36.969 | 0.0 | 1716.0 | 10500.0 |  |


## Performance Comparison

| comparison | crispyx_time_s | other_time_s | time_diff_s | time_pct | crispyx_mem_mb | other_mem_mb | mem_diff_mb | mem_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 2.34 | 4.483 | -2.143 | 52.189 | 278.133 | 36.969 | 241.164 | 752.346 |
| crispyx_de_t_test vs scanpy_de_t_test | 0.997 | 4.505 | -3.508 | 22.133 | 83.426 | 0.0 | 83.426 |  |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon | 4.511 | 15.013 | -10.502 | 30.046 | 84.121 | 234.242 | -150.121 | 35.912 |


## Accuracy

| comparison | cells_diff | genes_diff | effect_max_abs_diff | effect_pearson_corr | effect_spearman_corr | effect_top_k_overlap | statistic_max_abs_diff | statistic_pearson_corr | statistic_spearman_corr | statistic_top_k_overlap | pvalue_max_abs_diff | pvalue_pearson_corr | pvalue_spearman_corr | pvalue_top_k_overlap | pvalue_stream_auroc | pvalue_reference_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| crispyx_de_t_test vs scanpy_de_t_test |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 0.001 | 1.0 | 1.0 | 1.0 | 0.001 | 1.0 | 1.0 | 0.98 |  |  |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 4.658 | 0.975 | 0.979 | 0.94 | 0.758 | 0.886 | 0.932 | 0.94 |  |  |

