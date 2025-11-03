## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 11 (100.0% success rate)
- **Average runtime:** 6.300s
- **Average runtime by category:**
  - Streaming pipeline: 0.380s across 3 method(s) (success=3)
  - Differential expression: 1.370s across 2 method(s) (success=2)
  - Reference: Scanpy: 4.226s across 3 method(s) (success=3)
  - Reference: Pertpy: 17.580s across 3 method(s) (success=3)
- **Notable issues:**
  - Other errors recorded:
    - pertpy.tools.differential_expression module unavailable

### Streaming pipeline

| method | description | status | runtime_seconds | peak_memory_mb | cells_total | cells_kept | cells_kept_pct | cells_removed | genes_total | genes_kept | genes_kept_pct | genes_removed | rows | columns | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| average_log_expression | Average log-normalised expression per perturbation | success | 0.272 | 861392.0 |  |  |  |  |  |  |  |  | 2.0 | 11630.0 | benchmark_avg_log_avg_log_effects.h5ad |
| pseudobulk_expression | Pseudo-bulk log fold-change per perturbation | success | 0.12 | 705216.0 |  |  |  |  |  |  |  |  | 2.0 | 11630.0 | benchmark_pseudobulk_pseudobulk_effects.h5ad |
| quality_control | Streaming quality control filters | success | 0.749 | 650208.0 | 1716.0 | 1716.0 | 100.0 | 0.0 | 11630.0 | 11626.0 | 99.966 | 4.0 |  |  | benchmark_filtered.h5ad |

### Differential expression

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wald_test | Wald differential expression test | success | 0.346 | 995120.0 | 2.0 | 11630.0 | benchmark_wald_wald_de.h5ad |
| wilcoxon_test | Wilcoxon rank-sum differential expression | success | 2.395 | 559744.0 | 2.0 | 11630.0 | benchmark_wilcoxon_wilcoxon_de.h5ad |

### Reference: Scanpy

| method | description | status | runtime_seconds | peak_memory_mb | comparison_category | test_type | reference_tool | effect_max_abs_diff | statistic_max_abs_diff | pvalue_max_abs_diff | effect_pearson_corr | effect_spearman_corr | effect_top_k_overlap | statistic_pearson_corr | statistic_spearman_corr | statistic_top_k_overlap | pvalue_pearson_corr | pvalue_spearman_corr | pvalue_top_k_overlap | stream_total_seconds | reference_total_seconds | stream_peak_memory_mb | reference_peak_memory_mb | stream_timing_breakdown | reference_timing_breakdown | streaming_result_path | reference_result_path | normalization_max_abs_diff | log1p_max_abs_diff | avg_log_effect_max_abs_diff | pseudobulk_effect_max_abs_diff | streamlined_cell_count | reference_cell_count | streamlined_gene_count | reference_gene_count | wald_effect_max_abs_diff | wald_effect_pearson_corr | wald_effect_spearman_corr | wald_effect_top_k_overlap | wald_statistic_max_abs_diff | wald_statistic_pearson_corr | wald_statistic_spearman_corr | wald_statistic_top_k_overlap | wald_pvalue_max_abs_diff | wald_pvalue_pearson_corr | wald_pvalue_spearman_corr | wald_pvalue_top_k_overlap | wilcoxon_effect_max_abs_diff | wilcoxon_effect_pearson_corr | wilcoxon_effect_spearman_corr | wilcoxon_effect_top_k_overlap | wilcoxon_statistic_max_abs_diff | wilcoxon_statistic_pearson_corr | wilcoxon_statistic_spearman_corr | wilcoxon_statistic_top_k_overlap | wilcoxon_pvalue_max_abs_diff | wilcoxon_pvalue_pearson_corr | wilcoxon_pvalue_spearman_corr | wilcoxon_pvalue_top_k_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scanpy_quality_control_comparison | Quality control comparison against Scanpy | success | 6.433 | 1507328.0 | quality_control_preprocessing |  | scanpy |  |  |  |  |  |  |  |  |  |  |  |  | 3.222 | 2.434 | 1172.469 | 2675.609 | average_log_expression=0.290s; normalize_total+log1p=0.384s; pseudobulk_expression=0.127s; quality_control=0.404s; wald_test=0.327s; wilcoxon_test=1.689s | filter_cells=0.019s; filter_genes=0.040s; filtered_log1p=0.147s; filtered_normalize_total=0.032s; log1p=0.146s; normalize_total=0.031s; wald_test=0.149s; wilcoxon_test=1.869s |  |  | 0.0 | 0.0 | 0.0 | 0.0 | 1716.0 | 1716.0 | 11626.0 | 11626.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| scanpy_wald_comparison | Wald/t-test comparison against Scanpy | success | 2.269 | 1215328.0 | differential_expression | wald | scanpy_t-test | 24.029 | 0.632 | 0.252 | 0.362 | 0.848 | 0.02 | 1.0 | 0.999 | 0.96 | 0.995 | 0.997 | 0.98 |  |  |  |  |  |  | benchmark_scanpy_wald_wald_de.h5ad | benchmark_scanpy_reference_t-test_scanpy_de.csv |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| scanpy_wilcoxon_comparison | Wilcoxon comparison against Scanpy | success | 3.974 | 964400.0 | differential_expression | wilcoxon | scanpy_wilcoxon | 24.028 | 4.658 | 0.758 | 0.395 | 0.836 | 0.0 | 0.975 | 0.979 | 0.94 | 0.886 | 0.932 | 0.94 |  |  |  |  |  |  | benchmark_scanpy_wilcoxon_wilcoxon_de.h5ad | benchmark_scanpy_reference_wilcoxon_scanpy_de.csv |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Reference: Pertpy

| method | description | status | runtime_seconds | peak_memory_mb | comparison_category | test_type | reference_tool | streaming_result_path | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pertpy_edger_comparison | GLM comparison against edgeR via Pertpy | success | 37.82 | 1480656.0 | differential_expression | glm | pertpy_edger | benchmark_edger_wald_wald_de.h5ad | pertpy.tools.differential_expression module unavailable |
| pertpy_pydeseq2_comparison | GLM comparison against PyDESeq2 via Pertpy | success | 7.337 | 1399664.0 | differential_expression | glm | pertpy_pydeseq2 | benchmark_pydeseq2_wald_wald_de.h5ad | pertpy.tools.differential_expression module unavailable |
| pertpy_statsmodels_comparison | GLM comparison against statsmodels via Pertpy | success | 7.583 | 1396496.0 | differential_expression | glm | pertpy_statsmodels | benchmark_statsmodels_wald_wald_de.h5ad | pertpy.tools.differential_expression module unavailable |
