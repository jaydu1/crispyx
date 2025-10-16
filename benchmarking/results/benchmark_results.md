## Benchmark summary

- **Methods executed:** 11
- **Succeeded:** 10 (90.9% success rate)
- **Did not succeed:** 1
  - Errors: 1
- **Average runtime:** 0.560s
- **Average runtime by category:**
  - Streaming pipeline: 0.188s across 3 method(s) (success=3)
  - Differential expression: 0.231s across 2 method(s) (success=2)
  - Reference: Scanpy: 1.540s across 3 method(s) (success=2, error=1)
  - Reference: Pertpy: 0.174s across 3 method(s) (success=3)
- **Notable issues:**
  - Dependency errors detected:
    - No module named 'pertpy'
  - Other errors recorded:
    - zero-size array to reduction operation maximum which has no identity

### Streaming pipeline

| method | description | status | runtime_seconds | peak_memory_mb | cells_total | cells_kept | cells_kept_pct | cells_removed | genes_total | genes_kept | genes_kept_pct | genes_removed | rows | columns | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| average_log_expression | Average log-normalised expression per perturbation | success | 0.152 | 114.641 |  |  |  |  |  |  |  |  | 5.0 | 100.0 |  |
| pseudobulk_expression | Pseudo-bulk log fold-change per perturbation | success | 0.145 | 114.395 |  |  |  |  |  |  |  |  | 5.0 | 100.0 |  |
| quality_control | Streaming quality control filters | success | 0.266 | 116.32 | 400.0 | 400.0 | 100.0 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 |  |  | benchmark_filtered.h5ad |

### Differential expression

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| wald_test | Wald differential expression test | success | 0.179 | 115.352 | 5.0 | 100.0 | benchmark_wald_wald_de.h5ad |
| wilcoxon_test | Wilcoxon rank-sum differential expression | success | 0.282 | 118.523 | 5.0 | 100.0 | benchmark_wilcoxon_wilcoxon_de.h5ad |

### Reference: Scanpy

| method | description | status | runtime_seconds | peak_memory_mb | comparison_category | test_type | reference_tool | streaming_result_path | reference_result_path | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scanpy_quality_control_comparison | Quality control comparison against Scanpy | error | 0.814 |  |  |  |  |  |  | zero-size array to reduction operation maximum which has no identity |
| scanpy_wald_comparison | Wald/t-test comparison against Scanpy | success | 1.834 | 275.484 | differential_expression | wald | scanpy_t-test | benchmark_scanpy_wald_wald_de.h5ad | benchmark_scanpy_reference_t-test_scanpy_de.csv |  |
| scanpy_wilcoxon_comparison | Wilcoxon comparison against Scanpy | success | 1.971 | 278.766 | differential_expression | wilcoxon | scanpy_wilcoxon | benchmark_scanpy_wilcoxon_wilcoxon_de.h5ad | benchmark_scanpy_reference_wilcoxon_scanpy_de.csv |  |

### Reference: Pertpy

| method | description | status | runtime_seconds | peak_memory_mb | comparison_category | test_type | reference_tool | streaming_result_path | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pertpy_edger_comparison | GLM comparison against edgeR via Pertpy | success | 0.178 | 115.352 | differential_expression | glm | pertpy_edger | benchmark_edger_wald_wald_de.h5ad | No module named 'pertpy' |
| pertpy_pydeseq2_comparison | GLM comparison against PyDESeq2 via Pertpy | success | 0.18 | 115.352 | differential_expression | glm | pertpy_pydeseq2 | benchmark_pydeseq2_wald_wald_de.h5ad | No module named 'pertpy' |
| pertpy_statsmodels_comparison | GLM comparison against statsmodels via Pertpy | success | 0.164 | 115.355 | differential_expression | glm | pertpy_statsmodels | benchmark_statsmodels_wald_wald_de.h5ad | No module named 'pertpy' |
