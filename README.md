# Streamlined CRISPR Screen Analysis

This package provides a lightweight, pure-Python toolkit for performing key steps of CRISPR screen analysis without loading the entire dataset into memory. It operates on a compact on-disk representation inspired by AnnData `.h5ad` files where the first line stores metadata (cell and gene annotations) and subsequent lines stream cell-by-gene count rows. The toolkit includes quality control filters, pseudo-bulk effect size estimation, and differential expression tests that work efficiently with streaming data.

## Features
- Quality control filters for low quality cells, perturbations, and genes.
- Pseudo-bulk aggregation for effect size estimation using both averaged log counts and pseudo-bulk counts.
- Differential expression testing with Wald and Wilcoxon tests that can skip lowly expressed genes for stability.

## Development
Run the tests with:

```bash
pytest
```
