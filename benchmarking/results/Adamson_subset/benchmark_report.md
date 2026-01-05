# Benchmark Results

## 1. Performance

### Preprocessing / QC

| Package | Method | Status | Total (s) | Memory (MB) | Cells | Genes |
| --- | --- | --- | --- | --- | --- | --- |
| scanpy | QC filter | success | 3.71 | 234.94 | 1716.0 | 10500.0 |
| crispyx | QC filter | success | 3.48 | 228.07 | 1716.0 | 10500.0 |
| crispyx | pseudobulk (avg log) | success | 3.29 | 460.3 |  |  |
| crispyx | pseudobulk | success | 3.03 | 309.25 |  |  |


### DE: t-test

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | t-test | success | 2.93 | 110.13 | 2 |
| scanpy | t-test | success | 5.22 | 156.94 | 2 |


### DE: Wilcoxon

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| scanpy | Wilcoxon | success | 22.82 | 419.25 | 2 |
| crispyx | Wilcoxon | success | 3.7 | 191.59 | 2 |


### DE: NB GLM

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | NB-GLM (pydeseq2) | success | 37.72 | 1523.05 | 2 |
| crispyx | NB-GLM | success | 19.86 | 1399.7 | 2 |
| edgeR | NB-GLM | success | 95.31 | 1969.25 | 2 |
| pertpy | NB-GLM | success | 27.32 | 2518.96 | 2 |


### Other

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | lfcShrink (pydeseq2) | success | 2.54 | 16.88 | 2 |
| crispyx | lfcShrink | success | 2.54 | 15.0 | 2 |
| pertpy | lfcShrink | success | 29.71 | 2512.84 | 2 |


## 2. Performance Comparison

### crispyx vs Reference Tools

_crispyx as baseline. Negative values = crispyx is faster/uses less memory._

#### Preprocessing / QC

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QC filter | scanpy QC filter | -0.2s | 93.6% | ⚠️ | -6.9 MB | 97.1% | ⚠️ |


#### DE: t-test

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | -2.3s | 56.2% | ✅ | -46.8 MB | 70.2% | ✅ |


#### DE: Wilcoxon

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | -19.1s | 16.2% | ✅ | -227.7 MB | 45.7% | ✅ |


#### DE: NB GLM

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | edgeR NB-GLM | -75.5s | 20.8% | ✅ | -569.5 MB | 71.1% | ✅ |
| NB-GLM | pertpy NB-GLM | -7.5s | 72.7% | ✅ | -1119.3 MB | 55.6% | ✅ |


#### Other

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lfcShrink | crispyx lfcShrink (pydeseq2) | +0.0s | 100.1% | ⚠️ | -1.9 MB | 88.9% | ✅ |
| lfcShrink | pertpy lfcShrink | -27.2s | 8.5% | ✅ | -2497.8 MB | 0.6% | ✅ |
| lfcShrink (pydeseq2) | pertpy lfcShrink | -27.2s | 8.5% | ✅ | -2496.0 MB | 0.7% | ✅ |


### Tool Comparisons

_Comparisons between external tools._

| package A | method A | package B | method B | Time Δ (A-B) | Time % (A/B) |  | Mem Δ (A-B) | Mem % (A/B) |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | NB-GLM | pertpy | NB-GLM | +68.0s | 348.9% | ❌ | -549.7 MB | 78.2% | ✅ |


## 3. Accuracy

_Correlation metrics between crispyx and reference methods. ✅ >0.95, ⚠️ 0.8-0.95, ❌ <0.8_

### Preprocessing / QC

| crispyx method | compared to | Cells Δ |  | Genes Δ |   |
| --- | --- | --- | --- | --- | --- |
| QC filter | scanpy QC filter | +0 | ✅ | +0 | ✅ |


### DE: t-test

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ |


### DE: Wilcoxon

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ |


### DE: NB GLM

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | edgeR NB-GLM | 0.918<br><small>±0.030</small> | ⚠️ | 0.924<br><small>±0.060</small> | ⚠️ | 0.630<br><small>±0.172</small> | ❌ | 0.439<br><small>±0.002</small> | ❌ | 0.874<br><small>±0.036</small> | ⚠️ | 0.526<br><small>±0.092</small> | ❌ |
| NB-GLM | pertpy NB-GLM | 0.999<br><small>±0.001</small> | ✅ | 0.999<br><small>±0.000</small> | ✅ | 0.981<br><small>±0.002</small> | ✅ | 0.915<br><small>±0.056</small> | ⚠️ | 0.940<br><small>±0.001</small> | ⚠️ | 0.860<br><small>±0.079</small> | ⚠️ |


### Other

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lfcShrink | crispyx lfcShrink (pydeseq2) | 0.971<br><small>±0.019</small> | ✅ | 0.889<br><small>±0.043</small> | ⚠️ | 0.986<br><small>±0.004</small> | ✅ | 0.925<br><small>±0.051</small> | ⚠️ | 0.956<br><small>±0.017</small> | ✅ | 0.884<br><small>±0.072</small> | ⚠️ |
| lfcShrink | pertpy lfcShrink | 0.904<br><small>±0.075</small> | ⚠️ | 0.931<br><small>±0.036</small> | ⚠️ | 0.981<br><small>±0.002</small> | ✅ | 0.915<br><small>±0.056</small> | ⚠️ | 0.940<br><small>±0.001</small> | ⚠️ | 0.860<br><small>±0.079</small> | ⚠️ |
| lfcShrink (pydeseq2) | pertpy lfcShrink | 0.906<br><small>±0.081</small> | ⚠️ | 0.909<br><small>±0.066</small> | ⚠️ | 0.985<br><small>±0.001</small> | ✅ | 0.995<br><small>±0.000</small> | ✅ | 0.946<br><small>±0.013</small> | ⚠️ | 0.989<br><small>±0.002</small> | ✅ |


### Tool Comparisons

| package A | method A | package B | method B | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | NB-GLM | pertpy | NB-GLM | 0.918<br><small>±0.028</small> | ⚠️ | 0.924<br><small>±0.060</small> | ⚠️ | 0.595<br><small>±0.163</small> | ❌ | 0.277<br><small>±0.008</small> | ❌ | 0.887<br><small>±0.019</small> | ⚠️ | 0.488<br><small>±0.045</small> | ❌ |


## 4. Gene Set Overlap

_Overlap ratio of top-k DE genes between methods. ✅ >0.7, ⚠️ 0.5-0.7, ❌ <0.5_

### Effect Size Overlap

| crispyx method | compared to | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| Wilcoxon | scanpy Wilcoxon | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| NB-GLM | edgeR NB-GLM | 0.640 | ⚠️ | 0.695 | ⚠️ | 0.709 | ✅ |
| NB-GLM | pertpy NB-GLM | 0.960 | ✅ | 0.960 | ✅ | 0.964 | ✅ |
| lfcShrink | crispyx lfcShrink (pydeseq2) | 0.930 | ✅ | 0.975 | ✅ | 0.935 | ✅ |
| lfcShrink | pertpy lfcShrink | 0.530 | ⚠️ | 0.570 | ⚠️ | 0.726 | ✅ |
| lfcShrink (pydeseq2) | pertpy lfcShrink | 0.530 | ⚠️ | 0.560 | ⚠️ | 0.691 | ⚠️ |


### P-value Overlap

| crispyx method | compared to | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| Wilcoxon | scanpy Wilcoxon | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| NB-GLM | edgeR NB-GLM | 0.700 | ✅ | 0.705 | ✅ | 0.782 | ✅ |
| NB-GLM | pertpy NB-GLM | 0.750 | ✅ | 0.825 | ✅ | 0.920 | ✅ |


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
