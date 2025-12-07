## Benchmark summary

- **Methods executed:** 12
- **Succeeded:** 12 (100.0% success rate)
- **Average runtime:** 31.664s
- **Notable issues:** None

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 2.982 | 748.867 |  |  |  |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 2.862 | 596.648 |  |  |  |
| crispyx_qc_filtered | Streaming quality control filters | success | 3.597 | 518.875 | 1716.0 | 10500.0 | benchmarking/results/Adamson_subset/preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 3.812 | 520.188 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM (independent) with base + apeGLM shrinkage | success | 91.994 | 3550.289 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_joint | NB-GLM (joint) with base + apeGLM shrinkage | success | 107.872 | 5167.59 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_nb_glm_joint_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 2.529 | 398.898 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 7.78 | 395.629 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | success | 95.134 | 2255.574 |
| pertpy_de_pydeseq2 | success | 33.523 | 2807.008 |
| scanpy_de_t_test | success | 5.31 | 445.988 |
| scanpy_de_wilcoxon | success | 22.574 | 705.867 |

