# Benchmark Results

## Performance

| method | status | elapsed_seconds | peak_memory_mb | avg_memory_mb | cells_kept | genes_kept | groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | success | 22.563 | 339.312 | 247.182 |  |  | 2.0 |
| crispyx_de_t_test | success | 1.551 | 171.246 | 84.861 |  |  | 2.0 |
| crispyx_de_wilcoxon | success | 3.357 | 156.23 | 128.308 |  |  | 2.0 |
| crispyx_pb_avg_log | success | 1.712 | 558.473 | 236.029 |  |  |  |
| crispyx_pb_pseudobulk | success | 1.606 | 403.055 | 166.045 |  |  |  |
| crispyx_qc_filtered | success | 1.857 | 362.289 | 141.472 | 1716.0 | 10500.0 |  |
| edger_de_glm | success | 66.306 | 3526.723 | 2289.605 |  |  | 2.0 |
| pertpy_de_pydeseq2 | success | 617.51 | 3425.059 | 1709.589 |  |  | 2.0 |
| scanpy_de_t_test | success | 2.84 | 275.785 | 153.797 |  |  | 2.0 |
| scanpy_de_wilcoxon | success | 3.8 | 590.785 | 284.983 |  |  | 2.0 |
| scanpy_qc_filtered | success | 2.181 | 350.184 | 73.421 | 1716.0 | 10500.0 |  |


## Performance Comparison

| comparison | crispyx_time_s | other_time_s | time_diff_s | time_pct | crispyx_mem_mb | other_mem_mb | mem_diff_mb | mem_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 1.857 | 2.181 | -0.324 | 85.145 | 362.289 | 350.184 | 12.105 | 103.457 |
| crispyx_de_nb_glm vs edger_de_glm | 22.563 | 66.306 | -43.742 | 34.03 | 339.312 | 3526.723 | -3187.41 | 9.621 |
| crispyx_de_nb_glm vs pertpy_de_pydeseq2 | 22.563 | 617.51 | -594.946 | 3.654 | 339.312 | 3425.059 | -3085.746 | 9.907 |
| crispyx_de_t_test vs scanpy_de_t_test | 1.551 | 2.84 | -1.289 | 54.609 | 171.246 | 275.785 | -104.539 | 62.094 |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon | 3.357 | 3.8 | -0.443 | 88.341 | 156.23 | 590.785 | -434.555 | 26.445 |


## Accuracy

| comparison | cells_diff | genes_diff | effect_max_abs_diff | effect_pearson_corr | effect_spearman_corr | effect_top_k_overlap | statistic_max_abs_diff | statistic_pearson_corr | statistic_spearman_corr | statistic_top_k_overlap | pvalue_max_abs_diff | pvalue_pearson_corr | pvalue_spearman_corr | pvalue_top_k_overlap | pvalue_stream_auroc | pvalue_reference_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_qc_filtered vs scanpy_qc_filtered | 0.0 | 0.0 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| crispyx_de_nb_glm vs edger_de_glm |  |  | 2.652 | 0.949 | 0.957 | 0.5 | 731.197 | -0.226 | -0.317 | 0.56 | 0.993 | 0.84 | 0.892 | 0.68 |  |  |
| crispyx_de_nb_glm vs pertpy_de_pydeseq2 |  |  | 2.406 | 0.721 | 0.574 | 0.5 | 10.487 | 0.818 | 0.589 | 0.64 | 1.0 | 0.304 | 0.4 | 0.64 |  |  |
| crispyx_de_t_test vs scanpy_de_t_test |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 0.001 | 1.0 | 1.0 | 1.0 | 0.001 | 1.0 | 1.0 | 0.98 |  |  |
| crispyx_de_wilcoxon vs scanpy_de_wilcoxon |  |  | 24.025 | 0.362 | 0.852 | 0.0 | 4.658 | 0.975 | 0.979 | 0.94 | 0.758 | 0.886 | 0.932 | 0.94 |  |  |

