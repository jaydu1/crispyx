| method | description | status | elapsed_seconds | max_memory_mb | rows | columns | total_cells | kept_cells | cells_removed | total_genes | kept_genes | genes_removed | output_path | groups | genes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| average_log_expression | Average log-normalised expression per perturbation | success | 0.191 | 122.031 | 5.0 | 100.0 |  |  |  |  |  |  |  |  |  |
| pseudobulk_expression | Pseudo-bulk log fold-change per perturbation | success | 0.155 | 122.051 | 5.0 | 100.0 |  |  |  |  |  |  |  |  |  |
| quality_control | Streaming quality control filters | success | 0.282 | 122.664 |  |  | 400.0 | 400.0 | 0.0 | 100.0 | 100.0 | 0.0 | /workspace/Streamlining-CRISPR-Screen-Analysis/benchmarking/results/benchmark_filtered.h5ad |  |  |
| wald_test | Wald differential expression test | success | 0.163 | 122.543 |  |  |  |  |  |  |  |  | /workspace/Streamlining-CRISPR-Screen-Analysis/benchmarking/results/benchmark_wald_wald_de.h5ad | 5.0 | 100.0 |
| wilcoxon_test | Wilcoxon rank-sum differential expression | success | 0.28 | 125.918 |  |  |  |  |  |  |  |  | /workspace/Streamlining-CRISPR-Screen-Analysis/benchmarking/results/benchmark_wilcoxon_wilcoxon_de.h5ad | 5.0 | 100.0 |
