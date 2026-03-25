## Benchmark summary

- **Methods executed:** 15
- **Succeeded:** 13 (86.7% success rate)
- **Did not succeed:** 2
  - Errors: 2
- **Average runtime:** 7.046s
- **Notable issues:**
  - Dependency errors detected:
    - No module named 'rpy2'
  - Other errors recorded:
    - Output dtype not compatible with inputs.

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | cells_kept | genes_kept | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 0.887 | 684.391 |  |  | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 0.814 | 532.133 |  |  | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | success | 1.016 | 329.012 | 1716.0 | 10500.0 | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| scanpy_qc_filtered | success | 4.099 | 544.238 |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | NB-GLM base fitting (no shrinkage) | success | 15.257 | 426.33 | 2.0 | 11630.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_pydeseq2 | NB-GLM with PyDESeq2-parity settings (fisher SE, per-comparison SF/disp) | success | 42.588 | 607.54 | 2.0 | 11630.0 | de/crispyx_de_nb_glm_pydeseq2_nb_glm.h5ad |
| crispyx_de_t_test | t-test differential expression test | success | 1.873 | 407.512 | 2.0 | 11630.0 | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | success | 8.718 | 505.375 | 2.0 | 11630.0 | de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | error | 0.93 |  |
| pertpy_de_pydeseq2 | success | 20.759 | 2744.395 |
| scanpy_de_t_test | error | 2.159 | 402.883 |
| scanpy_de_wilcoxon | success | 5.15 | 570.395 |

