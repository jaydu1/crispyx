# Benchmark Results

## Performance

| method | status | elapsed_seconds | spawn_overhead_seconds | import_seconds | load_seconds | process_seconds | de_seconds | convert_seconds | save_seconds | peak_memory_mb | avg_memory_mb | cells_kept | genes_kept | groups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | success | 31.763 |  |  |  |  |  |  |  | 291.875 | 247.51 |  |  | 2.0 |
| crispyx_de_t_test | success | 0.828 |  |  |  |  |  |  |  | 328.156 | 285.664 |  |  | 2.0 |
| crispyx_de_wilcoxon | success | 3.539 |  |  |  |  |  |  |  | 319.73 | 315.293 |  |  | 2.0 |
| crispyx_pb_avg_log | success | 1.047 |  |  |  |  |  |  |  | 704.602 | 435.78 |  |  |  |
| crispyx_pb_pseudobulk | success | 0.842 |  |  |  |  |  |  |  | 552.078 | 334.284 |  |  |  |
| crispyx_qc_filtered | success | 1.242 |  |  |  |  |  |  |  | 426.039 | 309.118 | 1716.0 | 10500.0 |  |
| edger_de_glm | success | 100.458 |  | 0.562 | 0.079 | 98.958 |  |  | 0.118 | 2607.199 | 1815.683 |  |  | 2.0 |
| pertpy_de_pydeseq2 | success | 23.638 | 1.6 | 1.669 | 0.055 | 19.949 | 19.948 | 0.001 | 0.124 | 2553.434 | 1436.582 |  |  | 2.0 |
| scanpy_de_t_test | success | 2.406 | 1.484 | 0.454 | 0.054 | 0.267 |  |  | 0.075 | 375.348 | 232.978 |  |  | 2.0 |
| scanpy_de_wilcoxon | success | 3.007 | 1.479 | 0.451 | 0.053 | 0.86 |  |  | 0.075 | 412.102 | 223.129 |  |  | 2.0 |
| scanpy_qc_filtered | success | 2.306 | 1.521 | 0.455 | 0.054 | 0.141 |  |  | 0.061 | 475.84 | 224.487 | 1716.0 | 10500.0 |  |


## Performance Comparison

| comparison | crispyx_time_s | other_time_s | time_diff_s | time_pct | crispyx_mem_mb | other_mem_mb | mem_diff_mb | mem_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 1.242 | 2.306 | -1.064 | 53.873 | 426.039 | 475.84 | -49.801 | 89.534 |
| crispyx_de_nb_glm vs edger_de_glm | 31.763 | 100.458 | -68.694 | 31.618 | 291.875 | 2607.199 | -2315.324 | 11.195 |
| crispyx_de_nb_glm vs pertpy_de_pydeseq2 | 31.763 | 23.638 | 8.125 | 134.371 | 291.875 | 2553.434 | -2261.559 | 11.431 |
| crispyx_de_t_test vs scanpy_de_t_test | 0.828 | 2.406 | -1.578 | 34.406 | 328.156 | 375.348 | -47.191 | 87.427 |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon | 3.539 | 3.007 | 0.533 | 117.713 | 319.73 | 412.102 | -92.371 | 77.585 |


## Accuracy

| comparison | cells_diff | genes_diff | effect_max_abs_diff | effect_pearson_corr | effect_spearman_corr | effect_top_k_overlap | statistic_max_abs_diff | statistic_pearson_corr | statistic_spearman_corr | statistic_top_k_overlap | pvalue_max_abs_diff | pvalue_pearson_corr | pvalue_spearman_corr | pvalue_top_k_overlap | pvalue_stream_auroc | pvalue_reference_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| crispyx_de_nb_glm vs edger_de_glm |  |  | 2.652 | 0.949 | 0.957 | 0.5 | 731.197 | -0.226 | -0.317 | 0.56 | 0.993 | 0.84 | 0.892 | 0.68 |  |  |
| crispyx_de_nb_glm vs pertpy_de_pydeseq2 |  |  | 2.406 | 0.721 | 0.574 | 0.5 | 10.487 | 0.818 | 0.589 | 0.64 | 1.0 | 0.304 | 0.4 | 0.64 |  |  |
| crispyx_de_t_test vs scanpy_de_t_test |  |  | 0.0 | 1.0 | 1.0 | 1.0 | 0.001 | 1.0 | 1.0 | 1.0 | 0.001 | 1.0 | 1.0 | 0.98 |  |  |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon |  |  | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |  |  |

