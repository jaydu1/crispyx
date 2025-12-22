## Benchmark summary

- **Methods executed:** 12
- **Succeeded:** 12 (100.0% success rate)
- **Average runtime:** 23.526s
- **Notable issues:** None

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 3.392 | 747.352 |  |  |  |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 3.321 | 593.004 |  |  |  |
| crispyx_qc_filtered | Streaming quality control filters | success | 3.234 | 515.582 | 1716.0 | 10500.0 | benchmarking/results/Adamson_subset/preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 3.901 | 521.734 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM (independent) with base + apeGLM shrinkage | success | 49.368 | 1664.051 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_joint | NB-GLM (joint) with base + apeGLM shrinkage | success | 51.25 | 1257.801 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_nb_glm_joint_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 2.81 | 397.438 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 8.052 | 396.488 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | success | 93.603 | 2254.668 |
| pertpy_de_pydeseq2 | success | 34.513 | 2796.656 |
| scanpy_de_t_test | success | 5.555 | 446.27 |
| scanpy_de_wilcoxon | success | 23.311 | 702.145 |

