# Benchmark Results

## Performance

| method | status | elapsed_seconds | peak_memory_mb | avg_memory_mb | cells_kept | genes_kept | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | success | 92.849 | 59.23 | 13.902 |  |  | 2.0 |
| crispyx_de_t_test | success | 5.671 | 156.414 | 122.954 |  |  | 2.0 |
| crispyx_de_wilcoxon | success | 14.003 | 57.934 | 27.427 |  |  | 2.0 |
| crispyx_pb_avg_log | success | 4.555 | 483.895 | 218.709 |  |  |  |
| crispyx_pb_pseudobulk | success | 3.224 | 331.445 | 143.667 |  |  |  |
| crispyx_qc_filtered | success | 4.456 | 274.199 | 106.394 | 1716.0 | 10500.0 |  |
| edger_de_glm | error | 2.36 |  |  |  |  |  |
| pertpy_de_pydeseq2 | error | 5.079 |  |  |  |  |  |
| scanpy_de_t_test | success | 7.207 | 235.734 | 157.652 |  |  | 2.0 |
| scanpy_de_wilcoxon | success | 18.332 | 456.762 | 265.878 |  |  | 2.0 |
| scanpy_qc_filtered | success | 2.202 | 179.254 | 80.838 | 1716.0 | 10500.0 |  |


## Performance Comparison

| comparison | crispyx_time_s | other_time_s | time_diff_s | time_pct | crispyx_mem_mb | other_mem_mb | mem_diff_mb | mem_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 4.456 | 2.202 | 2.254 | 202.395 | 274.199 | 179.254 | 94.945 | 152.967 |
| crispyx_de_t_test vs scanpy_de_t_test | 5.671 | 7.207 | -1.537 | 78.68 | 156.414 | 235.734 | -79.32 | 66.352 |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon | 14.003 | 18.332 | -4.328 | 76.388 | 57.934 | 456.762 | -398.828 | 12.684 |


## Accuracy

| comparison | cells_diff | genes_diff | effect_max_abs_diff | effect_pearson_corr | effect_spearman_corr | effect_top_k_overlap | statistic_max_abs_diff | statistic_pearson_corr | statistic_spearman_corr | statistic_top_k_overlap | pvalue_max_abs_diff | pvalue_pearson_corr | pvalue_spearman_corr | pvalue_top_k_overlap | pvalue_stream_auroc | pvalue_reference_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| crispyx_de_t_test vs scanpy_de_t_test |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.001 | 1.0 | 1.0 | 0.98 |  |  |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon |  |  | 24.029 | 0.346 | 0.848 | 0.0 | 4.658 | 0.975 | 0.979 | 0.94 | 0.758 | 0.886 | 0.932 | 0.94 |  |  |

