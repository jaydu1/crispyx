# Benchmark Results

## 1. Performance

### Preprocessing / QC

| Package | Method | Status | Total (s) | Memory (MB) | Cells | Genes |
| --- | --- | --- | --- | --- | --- | --- |
| crispyx | QC filter | success | 1.05 | 397.5 | 1716.0 | 10500.0 |
| scanpy | QC filter | success | 3.37 | 234.43 | 1716.0 | 10500.0 |
| crispyx | pseudobulk (avg log) | success | 0.83 | 698.18 |  |  |
| crispyx | pseudobulk | success | 0.76 | 545.92 |  |  |


### DE: t-test

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| scanpy | t-test | success | 5.54 | 157.67 | 2 |
| crispyx | t-test | success | 1.99 | 422.38 | 2 |


### DE: Wilcoxon

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | Wilcoxon | success | 2.56 | 487.85 | 2 |
| scanpy | Wilcoxon | success | 21.56 | 425.23 | 2 |


### DE: NB GLM

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | NB-GLM | success | 24.14 | 382.5 | 2 |
| edgeR | NB-GLM | success | 92.33 | 1968.6 | 2 |
| pertpy | NB-GLM | success | 28.66 | 2466.91 | 2 |


### Other

| Package | Method | Status | Total (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | lfcShrink | success | 0.02 | 324.38 | 2 |
| crispyx | lfcShrink (pydeseq2) | success | 0.02 | 322.5 | 2 |
| pertpy | lfcShrink | success | 31.89 | 2504.6 | 2 |


## 2. Performance Comparison

### crispyx vs Reference Tools

_crispyx as baseline. Negative values = crispyx is faster/uses less memory._

#### Preprocessing / QC

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QC filter | scanpy QC filter | -2.3s | 31.2% | ✅ | +163.1 MB | 169.6% | ❌ |


#### DE: t-test

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | -3.6s | 35.9% | ✅ | +264.7 MB | 267.9% | ❌ |


#### DE: Wilcoxon

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | -19.0s | 11.9% | ✅ | +62.6 MB | 114.7% | ❌ |


#### DE: NB GLM

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | edgeR NB-GLM | -68.2s | 26.1% | ✅ | -1586.1 MB | 19.4% | ✅ |
| NB-GLM | pertpy NB-GLM | -4.5s | 84.2% | ✅ | -2084.4 MB | 15.5% | ✅ |


#### Other

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lfcShrink | crispyx lfcShrink (pydeseq2) | -0.0s | 68.0% | ✅ | +1.9 MB | 100.6% | ⚠️ |
| lfcShrink | pertpy lfcShrink | -31.9s | 0.1% | ✅ | -2180.2 MB | 13.0% | ✅ |
| lfcShrink (pydeseq2) | pertpy lfcShrink | -31.9s | 0.1% | ✅ | -2182.1 MB | 12.9% | ✅ |


### Tool Comparisons

_Comparisons between external tools._

| package A | method A | package B | method B | Time Δ (A-B) | Time % (A/B) |  | Mem Δ (A-B) | Mem % (A/B) |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | NB-GLM | pertpy | NB-GLM | +63.7s | 322.1% | ❌ | -498.3 MB | 79.8% | ✅ |


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
