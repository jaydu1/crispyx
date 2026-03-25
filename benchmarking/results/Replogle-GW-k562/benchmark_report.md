# Benchmark Results

## 1. Performance

### Preprocessing / QC

| Package | Method | Status | Total (s) | Memory (MB) |
| --- | --- | --- | --- | --- |
| crispyx | QC filter | error | 0.9 |  |
| scanpy | QC filter | memory_limit | 1108.37 | 126242.98 |
| crispyx | pseudobulk (avg log) | error | 0.55 |  |
| crispyx | pseudobulk | error | 0.52 |  |


### DE: t-test

| Package | Method | Status | Total (s) | Memory (MB) |
| --- | --- | --- | --- | --- |
| scanpy | t-test | error | 436.01 | 67600.28 |
| crispyx | t-test | error | 2.58 | 325.82 |


### DE: Wilcoxon

| Package | Method | Status | Total (s) | Memory (MB) |
| --- | --- | --- | --- | --- |
| crispyx | Wilcoxon | error | 2.54 | 320.29 |
| scanpy | Wilcoxon | memory_limit | 780.32 | 97825.33 |


### DE: NB GLM

| Package | Method | Status | Total (s) | Memory (MB) |
| --- | --- | --- | --- | --- |
| crispyx | NB-GLM | error | 5.61 | 322.75 |
| edgeR | NB-GLM | error | 702.59 |  |
| pertpy | NB-GLM | timeout | 86405.1 |  |


## 2. Performance Comparison

## 3. Accuracy

_Correlation metrics between crispyx and reference methods. ✅ >0.95, ⚠️ 0.8-0.95, ❌ <0.8_

## 4. Gene Set Overlap

_Overlap ratio of top-k DE genes between methods. ✅ >0.7, ⚠️ 0.5-0.7, ❌ <0.5_

_No overlap data available._

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
