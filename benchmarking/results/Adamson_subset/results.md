# Benchmark Results

## Performance

| method | status | elapsed_seconds | peak_memory_mb | avg_memory_mb | cells_kept | genes_kept | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | success | 67.531 | 59.207 | 14.585 |  |  | 2.0 |
| crispyx_de_t_test | success | 0.885 | 82.422 | 34.215 |  |  | 2.0 |
| crispyx_de_wilcoxon | success | 6.34 | 81.934 | 50.623 |  |  | 2.0 |
| crispyx_pb_avg_log | success | 2.969 | 482.934 | 227.536 |  |  |  |
| crispyx_pb_pseudobulk | success | 2.122 | 330.672 | 136.693 |  |  |  |
| crispyx_qc_filtered | success | 3.067 | 272.711 | 108.731 | 1716.0 | 10500.0 |  |
| edger_de_glm | error | 1.523 |  |  |  |  |  |
| pertpy_de_pydeseq2 | error | 3.221 |  |  |  |  |  |
| scanpy_de_t_test | success | 5.044 | 236.113 | 156.865 |  |  | 2.0 |
| scanpy_de_wilcoxon | success | 12.298 | 456.914 | 265.195 |  |  | 2.0 |
| scanpy_qc_filtered | success | 1.616 | 178.637 | 88.017 | 1716.0 | 10500.0 |  |


## Performance Comparison

| comparison | crispyx_time_s | other_time_s | time_diff_s | time_pct | crispyx_mem_mb | other_mem_mb | mem_diff_mb | mem_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 3.067 | 1.616 | 1.451 | 189.769 | 272.711 | 178.637 | 94.074 | 152.662 |
| crispyx_de_t_test vs scanpy_de_t_test | 0.885 | 5.044 | -4.158 | 17.551 | 82.422 | 236.113 | -153.691 | 34.908 |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon | 6.34 | 12.298 | -5.958 | 51.551 | 81.934 | 456.914 | -374.98 | 17.932 |


## Accuracy

| comparison | cells_diff | genes_diff | effect_max_abs_diff | effect_pearson_corr | effect_spearman_corr | effect_top_k_overlap | statistic_max_abs_diff | statistic_pearson_corr | statistic_spearman_corr | statistic_top_k_overlap | pvalue_max_abs_diff | pvalue_pearson_corr | pvalue_spearman_corr | pvalue_top_k_overlap | pvalue_stream_auroc | pvalue_reference_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| crispyx_de_t_test vs scanpy_de_t_test |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 0.001 | 1.0 | 1.0 | 1.0 | 0.001 | 1.0 | 1.0 | 0.98 |  |  |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 4.658 | 0.975 | 0.979 | 0.94 | 0.758 | 0.886 | 0.932 | 0.94 |  |  |

