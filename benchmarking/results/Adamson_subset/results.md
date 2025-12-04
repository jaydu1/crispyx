# Benchmark Results

## 1. Performance

### Preprocessing / QC

| Method | Status | Time (s) | Memory (MB) | Cells | Genes |
| --- | --- | --- | --- | --- | --- |
| QC Filter | success | 1.25 | 407.53 | 1716.0 | 10500.0 |
| Scanpy QC Filter | success | 2.51 | 507.81 | 1716.0 | 10500.0 |
| Pseudobulk (Avg) log | success | 1.05 | 709.41 |  |  |
| Pseudobulk | success | 0.84 | 555.01 |  |  |


### DE: t-test

| Method | Status | Time (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- |
| Scanpy t-test | success | 2.5 | 384.2 | 2.0 |
| t-test | success | 0.83 | 322.8 | 2.0 |


### DE: Wilcoxon

| Method | Status | Time (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- |
| Wilcoxon | success | 3.24 | 302.91 | 2.0 |
| Scanpy Wilcoxon | success | 3.11 | 373.99 | 2.0 |


### DE: NB GLM

| Method | Status | Time (s) | Memory (MB) | Groups |
| --- | --- | --- | --- | --- |
| NB-GLM | success | 46.89 | 2345.98 | 2.0 |
| NB-GLM (Joint) shared disp | success | 20.69 | 2191.49 | 2.0 |
| NB-GLM (Joint) | success | 34.26 | 2352.69 | 2.0 |
| edgeR | success | 99.58 | 2601.56 | 2.0 |
| PyDESeq2 | success | 23.93 | 2930.64 | 2.0 |


## 2. Performance Comparison

### CRISPYx vs Reference Tools

_CRISPYx as baseline. Negative values = CRISPYx is faster/uses less memory._

#### Preprocessing / QC

| CRISPYx Method | Compared To | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| QC Filter | Scanpy QC Filter | -1.3s | 50.0% | ✅ | -100.3 MB | 80.3% | ✅ |


#### DE: NB GLM

| CRISPYx Method | Compared To | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | edgeR | -52.7s | 47.1% | ✅ | -255.6 MB | 90.2% | ⚠️ |
| NB-GLM | PyDESeq2 | +23.0s | 195.9% | ❌ | -584.7 MB | 80.1% | ✅ |
| NB-GLM (Joint) | NB-GLM | -12.6s | 73.1% | ✅ | +6.7 MB | 100.3% | ⚠️ |
| NB-GLM (Joint) | edgeR | -65.3s | 34.4% | ✅ | -248.9 MB | 90.4% | ⚠️ |
| NB-GLM (Joint) | PyDESeq2 | +10.3s | 143.2% | ❌ | -577.9 MB | 80.3% | ✅ |


#### DE: t-test

| CRISPYx Method | Compared To | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | Scanpy t-test | -1.7s | 33.0% | ✅ | -61.4 MB | 84.0% | ✅ |


#### DE: Wilcoxon

| CRISPYx Method | Compared To | Time Δ | Time % |  | Mem Δ | Mem % |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | Scanpy Wilcoxon | +0.1s | 104.2% | ⚠️ | -71.1 MB | 81.0% | ✅ |


### Tool Comparisons

_Comparisons between external tools._

| Method A | Method B | Time Δ (A-B) | Time % (A/B) |  | Mem Δ (A-B) | Mem % (A/B) |   |
| --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | PyDESeq2 | +75.6s | 416.1% | ❌ | -329.1 MB | 88.8% | ✅ |


## 3. Accuracy

_Correlation metrics between CRISPYx and reference methods. ✅ >0.95, ⚠️ 0.8-0.95, ❌ <0.8_

### Preprocessing / QC

| CRISPYx Method | Compared To | Cells Δ |  | Genes Δ |   |
| --- | --- | --- | --- | --- | --- |
| QC Filter | Scanpy QC Filter | +0 | ✅ | +0 | ✅ |


### DE: NB GLM

| CRISPYx Method | Compared To | Effect ρ |  | Stat ρ |   | P-val ρ |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NB-GLM | edgeR | 0.906 | ⚠️ | 0.699 | ❌ | 0.692 | ❌ |
| NB-GLM | PyDESeq2 | 0.836 | ⚠️ | 0.879 | ⚠️ | 0.572 | ❌ |
| NB-GLM (Joint) | NB-GLM | 0.452 | ❌ | 0.443 | ❌ | 0.154 | ❌ |
| NB-GLM (Joint) | edgeR | 0.353 | ❌ | 0.148 | ❌ | 0.198 | ❌ |
| NB-GLM (Joint) | PyDESeq2 | 0.407 | ❌ | 0.320 | ❌ | -0.090 | ❌ |


### DE: t-test

| CRISPYx Method | Compared To | Effect ρ |  | Stat ρ |   | P-val ρ |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t-test | Scanpy t-test | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |


### DE: Wilcoxon

| CRISPYx Method | Compared To | Effect ρ |  | Stat ρ |   | P-val ρ |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wilcoxon | Scanpy Wilcoxon | 1.000 | ✅ | 1.000 | ✅ | 1.000 | ✅ |


### Tool Comparisons

| Method A | Method B | Effect ρ |  | Stat ρ |   | P-val ρ |    |
| --- | --- | --- | --- | --- | --- | --- | --- |
| edgeR | PyDESeq2 | 0.776 | ❌ | 0.679 | ❌ | 0.451 | ❌ |


---

**Legend:**
- Performance: ✅ >10% better | ⚠️ within ±10% | ❌ >10% worse
- Accuracy: ✅ ρ≥0.95 | ⚠️ 0.8≤ρ<0.95 | ❌ ρ<0.8
