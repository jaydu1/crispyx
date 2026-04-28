# crispyx Benchmark Summary

Benchmarked across 12 public CRISPR and Perturb-seq datasets ranging from
21,000 to 1,970,000 cells. All experiments were run in a Docker/Singularity
container with a 128 GB memory limit. Timing reported in seconds; memory
in GB (peak RSS).

Full per-dataset CSV data: `figures/aggregated_performance.csv`  
Full accuracy CSV data: `figures/aggregated_accuracy.csv`

---

## Completion rates

| Method | Datasets completed (out of 12) |
|---|---|
| **crispyx (t-test, Wilcoxon, NB-GLM)** | **12 / 12** |
| Scanpy (t-test) | 12 / 12 |
| Scanpy (Wilcoxon) | ~8 / 12 (timeout or OOM on large screens) |
| Pertpy/PyDESeq2 (NB-GLM) | ~3 / 12 (memory limit on genome-wide screens) |
| edgeR (GLM) | ~2 / 12 (timeout or error on most screens) |

---

## Runtime comparison

### t-test

| Dataset | Cells | crispyx (s) | Scanpy (s) | Speedup |
|---|---|---|---|---|
| Adamson | 65,282 | 7.3 | 18.6 | **2.6×** |
| Feng-gwsf | 320,706 | 172.8 | 590.3 | **3.4×** |
| Feng-gwsnf | ~1,970,000 | 411.0 | timeout | — |
| Tian-crispra | — | fast | fast | ~1–2× |

### Wilcoxon

| Dataset | Cells | crispyx (s) | Scanpy (s) | Speedup |
|---|---|---|---|---|
| Adamson | 65,282 | 122.8 | 1,304.8 | **10.6×** |
| Feng-gwsf | 320,706 | 10,761 | timeout (>32,400 s) | — |
| Feng-gwsnf | ~1,970,000 | 665 | timeout | — |

### NB-GLM

| Dataset | Cells | crispyx (s) | Pertpy/PyDESeq2 (s) | Speedup |
|---|---|---|---|---|
| Adamson | 65,282 | 2,472 | 5,318 | **2.2×** |
| Feng-gwsf | 320,706 | 5,416 | memory limit (>124 GB) | — |
| Feng-gwsnf | ~1,970,000 | 7,729 | memory limit | — |

---

## Memory comparison

### QC

| Dataset | crispyx peak (GB) | Scanpy peak (GB) |
|---|---|---|
| Adamson | 5.7 | 7.4 |
| Feng-gwsf | 15.2 | memory limit (>90 GB) |

### NB-GLM

| Dataset | crispyx peak (GB) | Pertpy peak (GB) |
|---|---|---|
| Adamson | 3.5 | 33.8 |
| Feng-gwsf | 21.1 | 124.4 (memory limit) |

---

## Accuracy

Results match reference implementations closely.

### t-test vs Scanpy (LFC Pearson *r*, mean across perturbations)

| Dataset | Pearson *r* (effect) | Top-100 gene overlap |
|---|---|---|
| Adamson | > 0.9999 | 1.00 |
| Feng-gwsf | 0.9998 | 0.9994 |
| Replogle-GW-k562 | > 0.9999 | 1.00 |
| Tian-crispra | > 0.9999 | 1.00 |

### NB-GLM vs Pertpy/PyDESeq2 (LFC Pearson *r*)

| Dataset | Pearson *r* (LFC) | Top-100 gene overlap |
|---|---|---|
| Adamson | 0.994 | 0.73 |
| Tian-crispra | 0.972 | 0.46 |
| Tian-crispri | 0.980 | 0.58 |

NB-GLM overlap at top-100 is lower than t-test because NB-GLM and PyDESeq2
use different shrinkage approaches (apeGLM vs DESeq2's MLE), which affects
gene ranking for low-effect perturbations. LFC magnitude agreement (Pearson
*r*) remains high.

---

## Figures

Benchmark figures are in `figures/`:

- `benchmark_figure.png` — combined overview panel
- `figure_a_runtime.png` — runtime comparison (all methods)
- `figure_b_memory.png` — peak memory comparison
- `figure_c_scalability.png` — runtime vs dataset size
- `figure_d_success_heatmap.png` — completion status by dataset × method
- `figure_e_accuracy.png` — accuracy metrics vs reference methods
- `figure_f_speedup.png` — per-dataset speedup factors

---

## Reproduction

```bash
cd benchmarking
./run_benchmark.sh config/Adamson.yaml          # single dataset
./run_benchmark.sh config/*.yaml                 # all datasets
cd figures
python aggregate_results.py
python generate_benchmark_figure.py
```

See `README.md` for dataset download instructions and configuration options.
