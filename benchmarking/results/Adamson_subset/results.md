# Benchmark Results

## 1. Performance

### Preprocessing / QC

| Package | Method | Status | Time (s) | Memory (MB) | Cells | Genes |
| --- | --- | --- | --- | --- | --- | --- |
| crispyx | QC filter | success | 1.25 | 405.8 | 1716.0 | 10500.0 |
| scanpy | QC filter | success | 2.41 | 507.14 | 1716.0 | 10500.0 |
| crispyx | pseudobulk (avg log) | success | 0.95 | 708.74 |  |  |
| crispyx | pseudobulk | success | 0.95 | 554.34 |  |  |


### DE: t-test

| Package | Method | Status | Time (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| scanpy | t-test | success | 2.41 | 389.06 | 2 |
| crispyx | t-test | success | 0.83 | 322.13 | 2 |


### DE: Wilcoxon

| Package | Method | Status | Time (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | Wilcoxon | success | 3.34 | 298.49 | 2 |
| scanpy | Wilcoxon | success | 3.01 | 382.53 | 2 |


### DE: NB GLM

| Package | Method | Status | Time (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- | --- |
| crispyx | NB-GLM | success | 50.27 | 2344.13 | 2 |
| crispyx | NB-GLM (joint) | success | 43.3 | 611.38 | 2 |
| edgeR | NB-GLM | success | 98.23 | 2602.58 | 2 |
| pertpy | NB-GLM | success | 26.04 | 2932.93 | 2 |


## 2. Performance Comparison

### CRISPYx vs Reference Tools

_CRISPYx as baseline. Negative values = CRISPYx is faster/uses less memory._

#### Preprocessing / QC

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QC filter | scanpy QC filter | -1.2s | 51.9% | ✅ | -101.3 MB | 80.0% | ✅ |


#### DE: NB GLM

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | edgeR NB-GLM | -48.0s | 51.2% | ✅ | -258.4 MB | 90.1% | ⚠️ |
| NB-GLM | pertpy NB-GLM | +24.2s | 193.0% | ❌ | -588.8 MB | 79.9% | ✅ |
| NB-GLM (joint) | crispyx NB-GLM | -7.0s | 86.1% | ✅ | -1732.8 MB | 26.1% | ✅ |
| NB-GLM (joint) | edgeR NB-GLM | -54.9s | 44.1% | ✅ | -1991.2 MB | 23.5% | ✅ |
| NB-GLM (joint) | pertpy NB-GLM | +17.3s | 166.3% | ❌ | -2321.5 MB | 20.8% | ✅ |


#### DE: t-test

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | -1.6s | 34.6% | ✅ | -66.9 MB | 82.8% | ✅ |


#### DE: Wilcoxon

| crispyx method | compared to | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | +0.3s | 110.8% | ❌ | -84.0 MB | 78.0% | ✅ |


### Tool Comparisons

_Comparisons between external tools._

| package A | method A | package B | method B | Time Δ (A-B) | Time % (A/B) |  | Mem Δ (A-B) | Mem % (A/B) |   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | NB-GLM | pertpy | NB-GLM | +72.2s | 377.2% | ❌ | -330.3 MB | 88.7% | ✅ |


## 3. Accuracy

_Correlation metrics between crispyx and reference methods. ✅ >0.95, ⚠️ 0.8-0.95, ❌ <0.8_

### Preprocessing / QC

| crispyx method | compared to | Cells Δ |  | Genes Δ |   |
| --- | --- | --- | --- | --- | --- |
| QC filter | scanpy QC filter | +0 | ✅ | +0 | ✅ |


### DE: NB GLM

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | edgeR NB-GLM | 0.919<br><small>±0.028</small> | ⚠️ | 0.928<br><small>±0.060</small> | ⚠️ | 0.641<br><small>±0.084</small> | ❌ | 0.608<br><small>±0.047</small> | ❌ | 0.928<br><small>±0.005</small> | ⚠️ | 0.783<br><small>±0.089</small> | ❌ |
| NB-GLM | pertpy NB-GLM | 0.997<br><small>±0.001</small> | ✅ | 0.998<br><small>±0.001</small> | ✅ | 0.925<br><small>±0.000</small> | ⚠️ | 0.820<br><small>±0.047</small> | ⚠️ | 0.856<br><small>±0.001</small> | ⚠️ | 0.570<br><small>±0.083</small> | ❌ |
| NB-GLM (joint) | crispyx NB-GLM | 0.958<br><small>±0.012</small> | ✅ | 0.979<br><small>±0.012</small> | ✅ | 0.939<br><small>±0.004</small> | ⚠️ | 0.966<br><small>±0.007</small> | ✅ | 0.876<br><small>±0.015</small> | ⚠️ | 0.836<br><small>±0.040</small> | ⚠️ |
| NB-GLM (joint) | edgeR NB-GLM | 0.969<br><small>±0.013</small> | ✅ | 0.976<br><small>±0.024</small> | ✅ | 0.723<br><small>±0.047</small> | ❌ | 0.822<br><small>±0.079</small> | ⚠️ | 0.884<br><small>±0.003</small> | ⚠️ | 0.897<br><small>±0.053</small> | ⚠️ |
| NB-GLM (joint) | pertpy NB-GLM | 0.959<br><small>±0.010</small> | ✅ | 0.977<br><small>±0.011</small> | ✅ | 0.851<br><small>±0.031</small> | ⚠️ | 0.840<br><small>±0.018</small> | ⚠️ | 0.747<br><small>±0.029</small> | ❌ | 0.371<br><small>±0.026</small> | ❌ |


### DE: t-test

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | scanpy t-test | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 0.998<br><small>±0.003</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ |


### DE: Wilcoxon

| crispyx method | compared to | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | scanpy Wilcoxon | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000<br><small>±0.000</small> | ✅ | 1.000 | ✅ | 1.000<br><small>±0.000</small> | ✅ |


### Tool Comparisons

| package A | method A | package B | method B | Eff ρ |  | Eff ρₛ |   | Stat ρ |    | Stat ρₛ |     | log-Pval ρ |      | log-Pval ρₛ |       |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | NB-GLM | pertpy | NB-GLM | 0.917<br><small>±0.028</small> | ⚠️ | 0.924<br><small>±0.060</small> | ⚠️ | 0.587<br><small>±0.168</small> | ❌ | 0.279<br><small>±0.033</small> | ❌ | 0.881<br><small>±0.024</small> | ⚠️ | 0.494<br><small>±0.015</small> | ❌ |


## 4. Gene Set Overlap

_Overlap ratio of top-k DE genes between methods. ✅ >0.7, ⚠️ 0.5-0.7, ❌ <0.5_

### Effect Size Overlap

| crispyx method | compared to | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | edgeR NB-GLM | 0.700 | ✅ | 0.675<br><small>±0.035</small> | ⚠️ | 0.673<br><small>±0.035</small> | ⚠️ |
| NB-GLM | pertpy NB-GLM | 0.660 | ⚠️ | 0.670<br><small>±0.071</small> | ⚠️ | 0.693<br><small>±0.041</small> | ⚠️ |
| NB-GLM (joint) | crispyx NB-GLM | 0.620 | ⚠️ | 0.675<br><small>±0.021</small> | ⚠️ | 0.816<br><small>±0.014</small> | ✅ |
| NB-GLM (joint) | edgeR NB-GLM | 0.780 | ✅ | 0.670<br><small>±0.141</small> | ⚠️ | 0.723<br><small>±0.049</small> | ✅ |
| NB-GLM (joint) | pertpy NB-GLM | 0.580 | ⚠️ | 0.595<br><small>±0.134</small> | ⚠️ | 0.670<br><small>±0.150</small> | ⚠️ |
| NB-GLM | pertpy NB-GLM | 0.740 | ✅ | 0.695<br><small>±0.078</small> | ⚠️ | 0.715<br><small>±0.058</small> | ✅ |
| t-test | scanpy t-test | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |
| Wilcoxon | scanpy Wilcoxon | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |


### P-value Overlap

| crispyx method | compared to | Top-50 |  | Top-100 |   | Top-500 |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | edgeR NB-GLM | 0.720 | ✅ | 0.805<br><small>±0.049</small> | ✅ | 0.821<br><small>±0.004</small> | ✅ |
| NB-GLM | pertpy NB-GLM | 0.600 | ⚠️ | 0.605<br><small>±0.007</small> | ⚠️ | 0.719<br><small>±0.075</small> | ✅ |
| NB-GLM (joint) | crispyx NB-GLM | 0.600 | ⚠️ | 0.645<br><small>±0.035</small> | ⚠️ | 0.854<br><small>±0.020</small> | ✅ |
| NB-GLM (joint) | edgeR NB-GLM | 0.660 | ⚠️ | 0.655<br><small>±0.007</small> | ⚠️ | 0.785<br><small>±0.061</small> | ✅ |
| NB-GLM (joint) | pertpy NB-GLM | 0.560 | ⚠️ | 0.530<br><small>±0.071</small> | ⚠️ | 0.668<br><small>±0.116</small> | ⚠️ |
| NB-GLM | pertpy NB-GLM | 0.720 | ✅ | 0.715<br><small>±0.078</small> | ✅ | 0.783<br><small>±0.086</small> | ✅ |
| t-test | scanpy t-test | 0.980 | ✅ | 0.995<br><small>±0.007</small> | ✅ | 1.000 | ✅ |
| Wilcoxon | scanpy Wilcoxon | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |


### Overlap Heatmaps (Top-100)

#### Effect Size

![Effect Size Top-100 Overlap](effect_top_100_overlap.png)

#### P-value

![P-value Top-100 Overlap](pvalue_top_100_overlap.png)

---

**Legend:**
- Performance: ✅ >10% better | ⚠️ within ±10% | ❌ >10% worse
- Accuracy: ✅ ρ≥0.95 | ⚠️ 0.8≤ρ<0.95 | ❌ ρ<0.8
- Overlap: ✅ ≥0.7 | ⚠️ 0.5-0.7 | ❌ <0.5
- ρ = Pearson correlation, ρₛ = Spearman correlation
- Correlation and overlap values shown as mean±std across perturbations
- log-Pval: correlations computed on -log₁₀(p) transformed values
- Top-k overlap: fraction of top-k genes shared between methods
