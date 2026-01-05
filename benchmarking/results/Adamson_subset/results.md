## Benchmark summary

- **Methods executed:** 15
- **Succeeded:** 15 (100.0% success rate)
- **Average runtime:** 17.545s
- **Notable issues:** None

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 3.289 | 460.301 |  |  |  |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 3.034 | 309.254 |  |  |  |
| crispyx_qc_filtered | Streaming quality control filters | success | 3.476 | 228.07 | 1716.0 | 10500.0 | benchmarking/results/Adamson_subset/preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 3.715 | 234.941 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 19.858 | 1399.703 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_pydeseq2 | NB-GLM with PyDESeq2-parity settings (fisher SE, per-comparison SF/disp) | success | 37.724 | 1523.047 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_nb_glm_pydeseq2_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 2.932 | 110.129 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 3.697 | 191.586 | 2.0 | 11630.0 | benchmarking/results/Adamson_subset/de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | success | 95.311 | 1969.246 |
| pertpy_de_pydeseq2 | success | 27.318 | 2518.965 |
| scanpy_de_t_test | success | 5.216 | 156.938 |
| scanpy_de_wilcoxon | success | 22.823 | 419.25 |

