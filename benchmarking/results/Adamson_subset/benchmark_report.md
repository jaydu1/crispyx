# Benchmark Results

## 1. Performance

### Preprocessing / QC

| Package | Method | Status | Total (s) | Memory (MB) | Cells | Genes |
| --- | --- | --- | --- | --- | --- | --- |
| scanpy | QC filter | success | 4.1 | 544.24 | 1716.0 | 10500.0 |
| crispyx | QC filter | success | 1.02 | 329.01 | 1716.0 | 10500.0 |
| crispyx | pseudobulk (avg log) | success | 0.89 | 684.39 |  |  |
| crispyx | pseudobulk | success | 0.81 | 532.13 |  |  |


### DE: t-test

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | t-test | success | 1.87 | 407.51 | 2.0 |
| scanpy | t-test | error | 2.16 | 402.88 |  |


### DE: Wilcoxon

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| scanpy | Wilcoxon | success | 5.15 | 570.39 | 2 |
| crispyx | Wilcoxon | success | 8.72 | 505.38 | 2 |


### DE: NB GLM

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | NB-GLM (pydeseq2) | success | 42.59 | 607.54 | 2.0 |
| crispyx | NB-GLM | success | 15.26 | 426.33 | 2.0 |
| edgeR | NB-GLM | error | 0.93 |  |  |
| pertpy | NB-GLM | success | 20.76 | 2744.39 | 2.0 |


### Other

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | lfcShrink (pydeseq2) | success | 0.03 | 324.38 | 2 |
| crispyx | lfcShrink | success | 0.02 | 324.38 | 2 |
| pertpy | lfcShrink | success | 1.38 | 2700.62 | 2 |


## 2. Performance Comparison

### crispyx vs Reference Tools

_crispyx as baseline. Negative values = crispyx is faster/uses less memory._

#### Preprocessing / QC

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QC filter | scanpy QC filter | -3.1s | 24.8% | ✅ | -215.2 MB | 60.5% | ✅ |


#### DE: Wilcoxon

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | +3.6s | 169.3% | ❌ | -65.0 MB | 88.6% | ✅ |


#### DE: NB GLM

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | pertpy NB-GLM | -5.5s | 73.5% | ✅ | -2318.1 MB | 15.5% | ✅ |


#### Other

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lfcShrink | crispyx lfcShrink (pydeseq2) | -0.0s | 92.6% | ⚠️ | 0.0 MB | 100.0% | ⚠️ |
| lfcShrink | pertpy lfcShrink | -1.4s | 1.8% | ✅ | -2376.2 MB | 12.0% | ✅ |
| lfcShrink (pydeseq2) | pertpy lfcShrink | -1.4s | 2.0% | ✅ | -2376.2 MB | 12.0% | ✅ |


## 3. Accuracy

_Correlation metrics between crispyx and reference methods. ✅ >0.95, ⚠️ 0.8-0.95, ❌ <0.8_

### Preprocessing / QC

| crispyx method | compared to | Cells Δ |  | Genes Δ |   |
| --- | --- | --- | --- | --- | --- |
| QC filter | scanpy QC filter | +0 | ✅ | +0 | ✅ |


### DE: Wilcoxon

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ |


### DE: NB GLM

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | pertpy NB-GLM | 0.999<br><small>±0.001</small> | ✅ | 0.999<br><small>±0.000</small> | ✅ | 0.981<br><small>±0.002</small> | ✅ | 0.915<br><small>±0.056</small> | ⚠️ | 0.940<br><small>±0.001</small> | ⚠️ | 0.860<br><small>±0.079</small> | ⚠️ |


### Other

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lfcShrink | crispyx lfcShrink (pydeseq2) | 0.971<br><small>±0.019</small> | ✅ | 0.889<br><small>±0.043</small> | ⚠️ | 0.986<br><small>±0.004</small> | ✅ | 0.925<br><small>±0.051</small> | ⚠️ | 0.956<br><small>±0.017</small> | ✅ | 0.884<br><small>±0.072</small> | ⚠️ |
| lfcShrink | pertpy lfcShrink | 0.904<br><small>±0.075</small> | ⚠️ | 0.931<br><small>±0.036</small> | ⚠️ | 0.981<br><small>±0.002</small> | ✅ | 0.915<br><small>±0.056</small> | ⚠️ | 0.940<br><small>±0.001</small> | ⚠️ | 0.860<br><small>±0.079</small> | ⚠️ |
| lfcShrink (pydeseq2) | pertpy lfcShrink | 0.906<br><small>±0.081</small> | ⚠️ | 0.909<br><small>±0.066</small> | ⚠️ | 0.985<br><small>±0.001</small> | ✅ | 0.995<br><small>±0.000</small> | ✅ | 0.946<br><small>±0.013</small> | ⚠️ | 0.989<br><small>±0.002</small> | ✅ |


## 4. Gene Set Overlap

_Overlap ratio of top-k DE genes between methods. ✅ >0.7, ⚠️ 0.5-0.7, ❌ <0.5_

### Effect Size Overlap

| crispyx method | compared to | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | 0.970 | ✅ | 0.985 | ✅ | 0.969 | ✅ |
| NB-GLM | pertpy NB-GLM | 0.960 | ✅ | 0.960 | ✅ | 0.964 | ✅ |
| lfcShrink | crispyx lfcShrink (pydeseq2) | 0.930 | ✅ | 0.975 | ✅ | 0.935 | ✅ |
| lfcShrink | pertpy lfcShrink | 0.530 | ⚠️ | 0.570 | ⚠️ | 0.726 | ✅ |
| lfcShrink (pydeseq2) | pertpy lfcShrink | 0.530 | ⚠️ | 0.560 | ⚠️ | 0.691 | ⚠️ |


### P-value Overlap

| crispyx method | compared to | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| NB-GLM | pertpy NB-GLM | 0.750 | ✅ | 0.825 | ✅ | 0.920 | ✅ |


_Note: Some methods are missing due to errors:_
- NB-GLM vs edgeR NB-GLM: _missing output: edger_de_glm (no output file)_
- t-test vs scanpy t-test: _missing output: scanpy_de_t_test (no output file)_

### Overlap Heatmaps (Top-100)

#### Effect Size

![Effect Size Top-100 Overlap](benchmark_effect_top_100_overlap.png)

#### P-value

![P-value Top-100 Overlap](benchmark_pvalue_top_100_overlap.png)

## 5. Shrinkage Quality

_Shrinkage should reduce LFC magnitude toward zero. Genes with |shrunk| > |raw| indicate aberrant shrinkage._

_✅ <1% inflated | ⚠️ 1-10% inflated | ❌ >10% inflated_

| Method | % Inflated |  | Max Inflation | Median Ratio | Genes |
| --- | --- | --- | --- | --- | --- |
| crispyx lfcShrink | 0.0% | ✅ | 1.00× | 0.547 | 15,618 |
| crispyx lfcShrink (pydeseq2) | 0.0% | ✅ | 1.00× | 0.460 | 15,464 |
| pertpy lfcShrink | 10.7% | ❌ | 16.79× | 0.407 | 15,578 |

_Note: A proper apeGLM implementation should have 0% inflated genes. The 'Median Ratio' shows how much shrinkage is applied on average (lower = more shrinkage)._

---

**Legend:**
- **Performance:** ✅ >10% better | ⚠️ within ±10% | ❌ >10% worse
- **Accuracy:** ✅ ρ≥0.95 | ⚠️ 0.8≤ρ<0.95 | ❌ ρ<0.8
- **Overlap:** ✅ ≥0.7 | ⚠️ 0.5-0.7 | ❌ <0.5
- **Shrinkage:** ✅ <1% inflated | ⚠️ 1-10% inflated | ❌ >10% inflated

**Abbreviations:**
- ρ = Pearson correlation, ρₛ = Spearman correlation
- log-Pval = correlations on -log₁₀(p) transformed values
- sf=per = per-comparison size factor estimation (matches PyDESeq2)

**Notes:**
- Correlation and overlap values shown as mean±std across perturbations
- crispyx lfcShrink uses `method='stats'` (Gaussian approximation) which is numerically stable and ~35× faster than `method='full'`.
- P-value overlap excludes lfcShrink methods since shrinkage only affects effect sizes, not p-values.
- **Warning:** PyDESeq2 may produce aberrant shrinkage when dispersion trend fitting fails. crispyx shrinkage is more robust.
