#!/usr/bin/env python3
"""Debug script to investigate lfcShrink discrepancies between crispyx and PyDESeq2.

This script:
1. Loads the Adamson_subset benchmark results
2. Compares prior_scale estimates between crispyx and PyDESeq2
3. Plots the MLE vs shrunk LFC distributions
4. Identifies genes with largest discrepancies
5. Tests shrinkage strength at different prior_scale values

Usage:
    python -m benchmarking.tools.debug_lfcshrink
    # or from repo root:
    python benchmarking/tools/debug_lfcshrink.py
"""

import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

from crispyx.glm import _estimate_apeglm_prior_scale, shrink_lfc_apeglm_from_stats


def get_results_dir() -> Path:
    """Get the results directory path."""
    return Path(__file__).parent.parent / "results/Adamson_subset/de"


def load_results():
    """Load benchmark results from Adamson_subset."""
    results_dir = get_results_dir()
    
    # Find crispyx NB-GLM results (with and without shrinkage)
    crispyx_base = results_dir / "crispyx_de_nb_glm.h5ad"
    crispyx_shrunk = results_dir / "crispyx_de_nb_glm_shrunk.h5ad"
    
    # Find pertpy/PyDESeq2 results (CSV format)
    pertpy_base = results_dir / "pertpy_de_pydeseq2.csv"
    pertpy_shrunk = results_dir / "pertpy_de_pydeseq2_shrunk.csv"
    
    print("=== Available result files ===")
    print(f"crispyx base: {crispyx_base.name} (exists: {crispyx_base.exists()})")
    print(f"crispyx shrunk: {crispyx_shrunk.name} (exists: {crispyx_shrunk.exists()})")
    print(f"pertpy base: {pertpy_base.name} (exists: {pertpy_base.exists()})")
    print(f"pertpy shrunk: {pertpy_shrunk.name} (exists: {pertpy_shrunk.exists()})")
    
    return {
        "crispyx_base": crispyx_base if crispyx_base.exists() else None,
        "crispyx_shrunk": crispyx_shrunk if crispyx_shrunk.exists() else None,
        "pertpy_base": pertpy_base if pertpy_base.exists() else None,
        "pertpy_shrunk": pertpy_shrunk if pertpy_shrunk.exists() else None,
    }


def compare_prior_scales(crispyx_adata, pertpy_adata):
    """Compare prior scale estimation between methods."""
    print("\n=== Prior Scale Comparison ===")
    
    # Extract MLE LFCs and SEs from crispyx
    if "logfoldchange_raw" in crispyx_adata.layers:
        lfc_raw = crispyx_adata.layers["logfoldchange_raw"]
        se_raw = crispyx_adata.layers.get("standard_error", None)
        if se_raw is None:
            se_raw = crispyx_adata.layers.get("standard_error_ln", None)
    else:
        lfc_raw = crispyx_adata.layers["logfoldchange"]
        se_raw = crispyx_adata.layers["standard_error"]
    
    perturbations = crispyx_adata.obs["perturbation"].tolist()
    
    print(f"\nPerturbations: {len(perturbations)}")
    print(f"LFC shape: {lfc_raw.shape}")
    print(f"SE shape: {se_raw.shape if se_raw is not None else 'None'}")
    
    # Compute prior scale for each perturbation
    prior_scales = []
    for i, pert in enumerate(perturbations):
        mle_lfc = lfc_raw[i, :]
        mle_se = se_raw[i, :] if se_raw is not None else np.ones_like(mle_lfc) * 0.1
        
        # Remove NaN values
        valid = np.isfinite(mle_lfc) & np.isfinite(mle_se) & (mle_se > 0)
        
        prior_scale = _estimate_apeglm_prior_scale(mle_lfc[valid], mle_se[valid])
        prior_scales.append(prior_scale)
        
        if i < 3:
            print(f"\n  {pert}:")
            print(f"    Valid genes: {valid.sum()}")
            print(f"    MLE LFC range: [{np.nanmin(mle_lfc):.3f}, {np.nanmax(mle_lfc):.3f}]")
            print(f"    MLE SE range: [{np.nanmin(mle_se):.3f}, {np.nanmax(mle_se):.3f}]")
            print(f"    Prior scale: {prior_scale:.6f}")
    
    print(f"\nPrior scale statistics:")
    print(f"  Mean: {np.mean(prior_scales):.6f}")
    print(f"  Median: {np.median(prior_scales):.6f}")
    print(f"  Min: {np.min(prior_scales):.6f}")
    print(f"  Max: {np.max(prior_scales):.6f}")
    
    return prior_scales


def compare_shrunk_lfcs(crispyx_shrunk_adata, pertpy_shrunk_path, perturbation_idx=0):
    """Compare shrunk LFCs between crispyx and PyDESeq2 for a specific perturbation."""
    print(f"\n=== Shrunk LFC Comparison (perturbation idx={perturbation_idx}) ===")
    
    # Get perturbation name
    pert_name = crispyx_shrunk_adata.obs["perturbation"].iloc[perturbation_idx]
    print(f"Perturbation: {pert_name}")
    
    # Extract shrunk LFCs from crispyx
    crispyx_lfc = crispyx_shrunk_adata.layers["logfoldchange"][perturbation_idx, :]
    crispyx_genes = crispyx_shrunk_adata.var_names.to_numpy()
    
    # Load pertpy CSV
    pertpy_df = pd.read_csv(pertpy_shrunk_path)
    print(f"Pertpy CSV columns: {pertpy_df.columns.tolist()}")
    print(f"Pertpy CSV shape: {pertpy_df.shape}")
    
    # Get pertpy LFCs for this perturbation
    if "perturbation" in pertpy_df.columns:
        pertpy_pert_df = pertpy_df[pertpy_df["perturbation"] == pert_name]
    elif "group" in pertpy_df.columns:
        pertpy_pert_df = pertpy_df[pertpy_df["group"] == pert_name]
    else:
        print("Cannot identify perturbation column in pertpy results!")
        print(f"First 3 rows: {pertpy_df.head(3)}")
        return None
    
    print(f"Pertpy rows for {pert_name}: {len(pertpy_pert_df)}")
    
    # Find effect size column
    effect_col = None
    for col in ["effect_size", "log2FoldChange", "logfoldchange", "lfc", "log_fc"]:
        if col in pertpy_pert_df.columns:
            effect_col = col
            break
    
    if effect_col is None:
        print(f"Cannot find effect size column! Columns: {pertpy_pert_df.columns.tolist()}")
        return None
    
    print(f"Using effect column: {effect_col}")
    
    # Build gene -> lfc mapping for pertpy
    gene_col = "gene" if "gene" in pertpy_pert_df.columns else pertpy_pert_df.columns[0]
    pertpy_lfc_map = dict(zip(pertpy_pert_df[gene_col], pertpy_pert_df[effect_col]))
    
    # Align
    common_genes = [g for g in crispyx_genes if g in pertpy_lfc_map]
    print(f"Common genes: {len(common_genes)}")
    
    crispyx_aligned = np.array([crispyx_lfc[np.where(crispyx_genes == g)[0][0]] for g in common_genes])
    pertpy_aligned = np.array([pertpy_lfc_map[g] for g in common_genes])
    
    # Remove NaNs
    valid = np.isfinite(crispyx_aligned) & np.isfinite(pertpy_aligned)
    crispyx_valid = crispyx_aligned[valid]
    pertpy_valid = pertpy_aligned[valid]
    
    print(f"Valid genes for comparison: {len(crispyx_valid)}")
    
    # Compute correlations
    from scipy.stats import pearsonr, spearmanr
    
    pearson_r, _ = pearsonr(crispyx_valid, pertpy_valid)
    spearman_r, _ = spearmanr(crispyx_valid, pertpy_valid)
    
    print(f"\nCorrelations:")
    print(f"  Pearson r: {pearson_r:.4f}")
    print(f"  Spearman r: {spearman_r:.4f}")
    
    # Stats
    print(f"\ncrispyx shrunk LFC range: [{crispyx_valid.min():.4f}, {crispyx_valid.max():.4f}]")
    print(f"pertpy shrunk LFC range: [{pertpy_valid.min():.4f}, {pertpy_valid.max():.4f}]")
    
    # Check for log-scale mismatch
    ratio = crispyx_valid / pertpy_valid
    ratio_finite = ratio[np.isfinite(ratio) & (pertpy_valid != 0)]
    print(f"\nRatio (crispyx/pertpy) median: {np.median(ratio_finite):.4f}")
    print(f"log2/ln ratio would be: {np.log(2):.4f}")
    
    # Find genes with largest discrepancy
    abs_diff = np.abs(crispyx_valid - pertpy_valid)
    top_diff_idx = np.argsort(abs_diff)[-10:][::-1]
    
    common_genes_valid = np.array(common_genes)[valid]
    print(f"\nTop 10 genes with largest discrepancy:")
    for idx in top_diff_idx:
        print(f"  {common_genes_valid[idx]}: crispyx={crispyx_valid[idx]:.4f}, pertpy={pertpy_valid[idx]:.4f}, diff={abs_diff[idx]:.4f}")
    
    return crispyx_valid, pertpy_valid


def test_shrinkage_formula(crispyx_shrunk_adata, pertpy_shrunk_path, crispyx_base_adata, perturbation_idx=0):
    """Test if adjusting prior_scale makes crispyx match pertpy better.
    
    This tests whether the discrepancy is due to:
    1. Different prior_scale estimation
    2. Different shrinkage formula/implementation
    3. Log-scale conversion issues
    """
    print(f"\n=== Shrinkage Formula Test (perturbation idx={perturbation_idx}) ===")
    
    pert_name = crispyx_shrunk_adata.obs["perturbation"].iloc[perturbation_idx]
    print(f"Perturbation: {pert_name}")
    
    # Get MLE LFC and SE from base (for re-shrinking)
    mle_lfc = crispyx_base_adata.layers["logfoldchange_raw"][perturbation_idx, :]
    mle_se = crispyx_base_adata.layers["standard_error"][perturbation_idx, :]
    crispyx_genes = crispyx_base_adata.var_names.to_numpy()
    
    lfc_base = crispyx_base_adata.uns.get("lfc_base", "log2")
    print(f"LFC base in crispyx: {lfc_base}")
    
    # Get pertpy shrunk values for comparison
    pertpy_df = pd.read_csv(pertpy_shrunk_path)
    pertpy_pert_df = pertpy_df[pertpy_df["group"] == pert_name] if "group" in pertpy_df.columns else pertpy_df[pertpy_df["perturbation"] == pert_name]
    
    effect_col = None
    for col in ["effect_size", "log2FoldChange", "logfoldchange", "lfc", "log_fc"]:
        if col in pertpy_pert_df.columns:
            effect_col = col
            break
    
    gene_col = "gene" if "gene" in pertpy_pert_df.columns else pertpy_pert_df.columns[0]
    pertpy_lfc_map = dict(zip(pertpy_pert_df[gene_col], pertpy_pert_df[effect_col]))
    
    # Get current crispyx prior_scale
    valid = np.isfinite(mle_lfc) & np.isfinite(mle_se) & (mle_se > 0)
    crispyx_prior_scale = _estimate_apeglm_prior_scale(mle_lfc[valid], mle_se[valid])
    print(f"crispyx estimated prior_scale: {crispyx_prior_scale:.6f}")
    
    # CRITICAL: PyDESeq2 estimates prior_scale from ln-scale LFCs!
    # If crispyx has log2-scale LFCs, we need to convert for proper comparison
    if lfc_base == "log2":
        # Convert to ln-scale for fair comparison
        mle_lfc_ln = mle_lfc * np.log(2)  # log2 -> ln
        mle_se_ln = mle_se * np.log(2)
        pydeseq2_style_prior = _estimate_apeglm_prior_scale(mle_lfc_ln[valid], mle_se_ln[valid])
        print(f"Prior scale if using ln-scale LFCs: {pydeseq2_style_prior:.6f}")
        print(f"Ratio log2/ln prior: {crispyx_prior_scale / pydeseq2_style_prior:.4f}")
    
    # Test different prior_scale values
    test_scales = [
        crispyx_prior_scale,
        crispyx_prior_scale / np.log(2),  # Adjusted for log2 -> ln
        0.05,  # Smaller scale = more shrinkage
        0.1,
        0.2,
    ]
    
    from scipy.stats import pearsonr
    
    print("\nTesting different prior_scale values:")
    for scale in test_scales:
        # Convert to ln-scale for shrinkage (as PyDESeq2 does)
        if lfc_base == "log2":
            shrunk_ln, shrunk_se_ln, conv, needs_refit = shrink_lfc_apeglm_from_stats(
                mle_lfc=mle_lfc * np.log(2),  # Convert to ln
                mle_se=mle_se * np.log(2),
                prior_scale=scale,
            )
            # Convert back to log2
            shrunk_log2 = shrunk_ln / np.log(2)
        else:
            shrunk_log2, _, conv, needs_refit = shrink_lfc_apeglm_from_stats(
                mle_lfc=mle_lfc,
                mle_se=mle_se,
                prior_scale=scale,
            )
        
        # Align with pertpy
        common_genes = [g for g in crispyx_genes if g in pertpy_lfc_map]
        crispyx_aligned = np.array([shrunk_log2[np.where(crispyx_genes == g)[0][0]] for g in common_genes])
        pertpy_aligned = np.array([pertpy_lfc_map[g] for g in common_genes])
        
        valid_comp = np.isfinite(crispyx_aligned) & np.isfinite(pertpy_aligned)
        if not valid_comp.any():
            continue
        
        r, _ = pearsonr(crispyx_aligned[valid_comp], pertpy_aligned[valid_comp])
        
        # Compute ratio of magnitudes
        ratio = np.abs(crispyx_aligned[valid_comp]) / np.maximum(np.abs(pertpy_aligned[valid_comp]), 1e-10)
        median_ratio = np.median(ratio[np.isfinite(ratio)])
        
        print(f"  prior_scale={scale:.4f}: Pearson r={r:.4f}, median |crispyx|/|pertpy|={median_ratio:.4f}")
    
    print("\nConclusion:")
    print("  If a smaller prior_scale gives median_ratio closer to 1.0, the issue is prior estimation")
    print("  If no prior_scale gets median_ratio~1.0, the issue is the shrinkage formula itself")


def main():
    """Main debug routine."""
    print("=" * 60)
    print("lfcShrink Debug Script")
    print("=" * 60)
    
    results = load_results()
    
    # Load crispyx base results (with MLE LFCs and SEs)
    if results["crispyx_base"]:
        print(f"\nLoading crispyx base: {results['crispyx_base']}")
        crispyx_base = ad.read_h5ad(results["crispyx_base"])
        print(f"  Shape: {crispyx_base.shape}")
        print(f"  Layers: {list(crispyx_base.layers.keys())}")
        print(f"  Uns keys: {list(crispyx_base.uns.keys())[:10]}")
        
        # Check lfc_base
        lfc_base = crispyx_base.uns.get("lfc_base", "unknown")
        print(f"  lfc_base: {lfc_base}")
        
        # Analyze prior scales
        prior_scales = compare_prior_scales(crispyx_base, None)
    else:
        print("No crispyx base results found!")
        crispyx_base = None
        prior_scales = None
    
    # Load shrunk results for comparison
    if results["crispyx_shrunk"] and results["pertpy_shrunk"]:
        print(f"\nLoading crispyx shrunk: {results['crispyx_shrunk']}")
        crispyx_shrunk = ad.read_h5ad(results["crispyx_shrunk"])
        print(f"  Shape: {crispyx_shrunk.shape}")
        
        print(f"\nUsing pertpy shrunk CSV: {results['pertpy_shrunk']}")
        
        # Compare shrunk LFCs
        compare_shrunk_lfcs(crispyx_shrunk, results["pertpy_shrunk"], perturbation_idx=0)
        
        # Test shrinkage formula with different prior_scale values
        if crispyx_base is not None:
            test_shrinkage_formula(crispyx_shrunk, results["pertpy_shrunk"], crispyx_base, perturbation_idx=0)
    else:
        print("Missing shrunk results for comparison!")
    
    print("\n" + "=" * 60)
    print("Debug complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
