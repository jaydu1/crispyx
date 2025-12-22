# Benchmark Results

## 1. Performance

### Preprocessing / QC

| Package | Method | Status | Time (s) | Memory (MB) | Cells | Genes |
| --- | --- | --- | --- | --- | --- | --- |
| crispyx | QC filter | success | 3.23 | 515.58 | 1716.0 | 10500.0 |
| scanpy | QC filter | success | 3.9 | 521.73 | 1716.0 | 10500.0 |
| crispyx | pseudobulk (avg log) | success | 3.39 | 747.35 |  |  |
| crispyx | pseudobulk | success | 3.32 | 593.0 |  |  |


### DE: t-test

| Package | Method | Status | Time (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| scanpy | t-test | success | 5.55 | 446.27 | 2 |
| crispyx | t-test | success | 2.81 | 397.44 | 2 |


### DE: Wilcoxon

| Package | Method | Status | Time (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | Wilcoxon | success | 8.05 | 396.49 | 2 |
| scanpy | Wilcoxon | success | 23.31 | 702.14 | 2 |


### DE: NB GLM

| Package | Method | lfcShrink | Status | Time (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- | --- |
| crispyx | NB-GLM | ✓ | success | 49.37 | 1664.05 | 2 |
| crispyx | NB-GLM (joint) | ✓ | success | 51.25 | 1257.8 | 2 |
| edgeR | NB-GLM |  | success | 93.6 | 2254.67 | 2 |
| pertpy | NB-GLM | ✓ | success | 34.51 | 2796.66 | 2 |


## 2. Performance Comparison

### crispyx vs Reference Tools

_crispyx as baseline. Negative values = crispyx is faster/uses less memory._

#### Preprocessing / QC

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QC filter | scanpy QC filter | -0.7s | 82.9% | ✅ | -6.2 MB | 98.8% | ⚠️ |


#### DE: t-test

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | -2.7s | 50.6% | ✅ | -48.8 MB | 89.1% | ✅ |


#### DE: Wilcoxon

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | -15.3s | 34.5% | ✅ | -305.7 MB | 56.5% | ✅ |


#### DE: NB GLM

| crispyx method | lfcShrink | compared to | lfcShrink (ref) | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM |  | edgeR NB-GLM |  | -47.3s | 49.5% | ✅ | -590.6 MB | 73.8% | ✅ |
| NB-GLM |  | pertpy NB-GLM |  | +20.3s | 177.7% | ❌ | -1132.6 MB | 59.5% | ✅ |
| NB-GLM | ✓ | pertpy NB-GLM | ✓ | +14.9s | 143.0% | ❌ | -1132.6 MB | 59.5% | ✅ |
| NB-GLM (joint) |  | edgeR NB-GLM |  | -44.7s | 52.3% | ✅ | -996.9 MB | 55.8% | ✅ |
| NB-GLM (joint) |  | pertpy NB-GLM |  | +22.8s | 187.6% | ❌ | -1538.9 MB | 45.0% | ✅ |
| NB-GLM (joint) | ✓ | pertpy NB-GLM | ✓ | +16.7s | 148.5% | ❌ | -1538.9 MB | 45.0% | ✅ |


### Tool Comparisons

_Comparisons between external tools._

| package A | method A | package B | method B | Time Δ (A-B) | Time % (A/B) |  | Mem Δ (A-B) | Mem % (A/B) |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | NB-GLM | pertpy | NB-GLM | +67.5s | 359.0% | ❌ | -542.0 MB | 80.6% | ✅ |


## 3. Accuracy

_Correlation metrics between crispyx and reference methods. ✅ >0.95, ⚠️ 0.8-0.95, ❌ <0.8_

### Preprocessing / QC

| crispyx method | compared to | Cells Δ |  | Genes Δ |   |
| --- | --- | --- | --- | --- | --- |
| QC filter | scanpy QC filter | +0 | ✅ | +0 | ✅ |


### DE: t-test

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 0.998<br><small>±0.003</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ |


### DE: Wilcoxon

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000 | ✅ | 1.000<br><small>±0.000</small> | ✅ |


### DE: NB GLM

| crispyx method | lfcShrink | compared to | lfcShrink (ref) | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM |  | edgeR NB-GLM |  | 0.918<br><small>±0.030</small> | ⚠️ | 0.924<br><small>±0.060</small> | ⚠️ | 0.601<br><small>±0.177</small> | ❌ | 0.320<br><small>±0.019</small> | ❌ | 0.884<br><small>±0.041</small> | ⚠️ | 0.506<br><small>±0.037</small> | ❌ |
| NB-GLM |  | pertpy NB-GLM |  | 0.999<br><small>±0.001</small> | ✅ | 0.999<br><small>±0.000</small> | ✅ | 0.994<br><small>±0.001</small> | ✅ | 0.990<br><small>±0.007</small> | ✅ | 0.984<br><small>±0.003</small> | ✅ | 0.983<br><small>±0.007</small> | ✅ |
| NB-GLM | ✓ | pertpy NB-GLM | ✓ | 0.913<br><small>±0.081</small> | ⚠️ | 0.925<br><small>±0.049</small> | ⚠️ | 0.994<br><small>±0.001</small> | ✅ | 0.990<br><small>±0.007</small> | ✅ | 0.984<br><small>±0.003</small> | ✅ | 0.983<br><small>±0.007</small> | ✅ |
| NB-GLM (joint) |  | edgeR NB-GLM |  | 0.914<br><small>±0.023</small> | ⚠️ | 0.918<br><small>±0.045</small> | ⚠️ | 0.608<br><small>±0.163</small> | ❌ | 0.420<br><small>±0.041</small> | ❌ | 0.871<br><small>±0.051</small> | ⚠️ | 0.519<br><small>±0.099</small> | ❌ |
| NB-GLM (joint) |  | pertpy NB-GLM |  | 0.994<br><small>±0.003</small> | ✅ | 0.989<br><small>±0.001</small> | ✅ | 0.974<br><small>±0.001</small> | ✅ | 0.928<br><small>±0.022</small> | ⚠️ | 0.933<br><small>±0.022</small> | ⚠️ | 0.878<br><small>±0.029</small> | ⚠️ |
| NB-GLM (joint) | ✓ | pertpy NB-GLM | ✓ | 0.911<br><small>±0.077</small> | ⚠️ | 0.930<br><small>±0.028</small> | ⚠️ | 0.974<br><small>±0.001</small> | ✅ | 0.928<br><small>±0.022</small> | ⚠️ | 0.933<br><small>±0.022</small> | ⚠️ | 0.878<br><small>±0.029</small> | ⚠️ |


### Tool Comparisons

| package A | method A | package B | method B | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | NB-GLM | pertpy | NB-GLM | 0.918<br><small>±0.028</small> | ⚠️ | 0.924<br><small>±0.060</small> | ⚠️ | 0.595<br><small>±0.163</small> | ❌ | 0.277<br><small>±0.008</small> | ❌ | 0.887<br><small>±0.019</small> | ⚠️ | 0.488<br><small>±0.045</small> | ❌ |


## 4. Gene Set Overlap

_Overlap ratio of top-k DE genes between methods. ✅ >0.7, ⚠️ 0.5-0.7, ❌ <0.5_

### Effect Size Overlap

| crispyx method | lfcShrink | compared to | lfcShrink (ref) | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t-test |  | scanpy t-test |  | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| Wilcoxon |  | scanpy Wilcoxon |  | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| NB-GLM |  | edgeR NB-GLM |  | 0.700 | ✅ | 0.695<br><small>±0.078</small> | ⚠️ | 0.709<br><small>±0.058</small> | ✅ |
| NB-GLM |  | pertpy NB-GLM |  | 0.960 | ✅ | 0.960 | ✅ | 0.964<br><small>±0.023</small> | ✅ |
| NB-GLM | ✓ | pertpy NB-GLM | ✓ | 0.420 | ❌ | 0.605<br><small>±0.205</small> | ⚠️ | 0.748<br><small>±0.184</small> | ✅ |
| NB-GLM (joint) |  | edgeR NB-GLM |  | 0.700 | ✅ | 0.685<br><small>±0.078</small> | ⚠️ | 0.705<br><small>±0.058</small> | ✅ |
| NB-GLM (joint) |  | pertpy NB-GLM |  | 0.900 | ✅ | 0.935<br><small>±0.007</small> | ✅ | 0.957<br><small>±0.021</small> | ✅ |
| NB-GLM (joint) | ✓ | pertpy NB-GLM | ✓ | 0.400 | ❌ | 0.600<br><small>±0.198</small> | ⚠️ | 0.747<br><small>±0.188</small> | ✅ |


### P-value Overlap

| crispyx method | lfcShrink | compared to | lfcShrink (ref) | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t-test |  | scanpy t-test |  | 0.980 | ✅ | 0.995<br><small>±0.007</small> | ✅ | 1.000 | ✅ |
| Wilcoxon |  | scanpy Wilcoxon |  | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| NB-GLM |  | edgeR NB-GLM |  | 0.740 | ✅ | 0.750<br><small>±0.085</small> | ✅ | 0.779<br><small>±0.089</small> | ✅ |
| NB-GLM |  | pertpy NB-GLM |  | 0.920 | ✅ | 0.910<br><small>±0.028</small> | ✅ | 0.954 | ✅ |
| NB-GLM | ✓ | pertpy NB-GLM | ✓ | 0.920 | ✅ | 0.910<br><small>±0.028</small> | ✅ | 0.954 | ✅ |
| NB-GLM (joint) |  | edgeR NB-GLM |  | 0.660 | ⚠️ | 0.650<br><small>±0.071</small> | ⚠️ | 0.756<br><small>±0.071</small> | ✅ |
| NB-GLM (joint) |  | pertpy NB-GLM |  | 0.760 | ✅ | 0.740<br><small>±0.042</small> | ✅ | 0.860<br><small>±0.006</small> | ✅ |
| NB-GLM (joint) | ✓ | pertpy NB-GLM | ✓ | 0.760 | ✅ | 0.740<br><small>±0.042</small> | ✅ | 0.860<br><small>±0.006</small> | ✅ |


### Overlap Heatmaps (Top-100)

#### Effect Size

![Effect Size Top-100 Overlap](benchmark_effect_top_100_overlap.png)

#### P-value

![P-value Top-100 Overlap](benchmark_pvalue_top_100_overlap.png)

---

**Legend:**
- Performance: ✅ >10% better | ⚠️ within ±10% | ❌ >10% worse
- Accuracy: ✅ ρ≥0.95 | ⚠️ 0.8≤ρ<0.95 | ❌ ρ<0.8
- Overlap: ✅ ≥0.7 | ⚠️ 0.5-0.7 | ❌ <0.5
- ρ = Pearson correlation, ρₛ = Spearman correlation
- Correlation and overlap values shown as mean±std across perturbations
- log-Pval: correlations computed on -log₁₀(p) transformed values
- Top-k overlap: fraction of top-k genes shared between methods
- lfcShrink column: shrinkage type used (apeglm, ashr, normal) or blank if none
