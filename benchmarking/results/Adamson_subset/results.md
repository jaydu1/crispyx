## Benchmark summary

- **Methods executed:** 15
- **Succeeded:** 7 (46.7% success rate)
- **Did not succeed:** 8
- **Average runtime:** 35.515s
- **Notable issues:** None

## Preprocessing

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | result_path |
| --- | --- | --- | --- | --- | --- |
| crispyx_pb_avg_log | Average log-normalised expression per perturbation | success | 0.838 | 695.203 | preprocessing/crispyx_pb_avg_log_avg_log_effects.h5ad |
| crispyx_pb_pseudobulk | Pseudo-bulk log fold-change per perturbation | success | 0.84 | 542.68 | preprocessing/crispyx_pb_pseudobulk_pseudobulk_effects.h5ad |
| crispyx_qc_filtered | Streaming quality control filters | skipped_existing |  |  | preprocessing/crispyx_qc_filtered.h5ad |


### Reference Comparisons

| method | status |
| --- | --- |
| scanpy_qc_filtered | skipped_existing |


## Differential Expression

### crispyx Methods

| method | description | status | runtime_seconds | peak_memory_mb | groups | genes | result_path |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crispyx_de_nb_glm | Negative binomial GLM differential expression (independent, no shrinkage) | success | 96.953 | 3825.449 | 2.0 | 11630.0 | de/crispyx_de_nb_glm.h5ad |
| crispyx_de_nb_glm_joint | Negative binomial GLM differential expression (joint model, no shrinkage) | success | 141.826 | 5067.625 | 2.0 | 11630.0 | de/crispyx_de_nb_glm_joint_nb_glm.h5ad |
| crispyx_de_nb_glm_joint_shrunk | LFC shrinkage on joint NB-GLM results (apeglm) | success | 0.628 | 226.574 | 2.0 | 11630.0 | de/crispyx_de_nb_glm_joint_shrunk.h5ad |
| crispyx_de_nb_glm_shrunk | LFC shrinkage on independent NB-GLM results (apeglm) | skipped_existing |  |  |  |  | de/crispyx_de_nb_glm_shrunk.h5ad |
| crispyx_de_t_test | t-test differential expression test | skipped_existing |  |  |  |  | de/crispyx_de_t_test.h5ad |
| crispyx_de_wilcoxon | Wilcoxon rank-sum differential expression | skipped_existing |  |  |  |  | de/crispyx_de_wilcoxon.h5ad |


### Reference Comparisons

| method | status | runtime_seconds | peak_memory_mb |
| --- | --- | --- | --- |
| edger_de_glm | skipped_existing |  |  |
| pertpy_de_pydeseq2 | skipped_existing |  |  |
| pertpy_de_pydeseq2_shrunk | skipped_existing |  |  |
| scanpy_de_t_test | success | 3.507 | 420.605 |
| scanpy_de_wilcoxon | success | 4.011 | 433.746 |

