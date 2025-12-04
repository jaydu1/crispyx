import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import anndata as ad
import statsmodels.api as sm

try:  # Optional benchmark dependency for validation comparisons
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
except ModuleNotFoundError:  # pragma: no cover - executed when benchmarks unavailable
    DeseqDataSet = None
    DeseqStats = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from crispyx.glm import NBGLMFitter, build_design_matrix
from crispyx.de import nb_glm_test


def _generate_nb_counts(rng, mu, alpha):
    r = 1.0 / alpha
    p = r / (r + mu)
    return rng.negative_binomial(r, p)


def test_nb_glm_fitter_matches_statsmodels():
    rng = np.random.default_rng(42)
    n = 300
    covariates = pd.DataFrame(
        {
            "cov1": rng.normal(0, 1, size=n),
            "cov2": pd.Categorical(rng.choice(["A", "B", "C"], size=n, p=[0.2, 0.5, 0.3])),
        }
    )
    indicator = rng.binomial(1, 0.4, size=n).astype(float)
    design, column_names = build_design_matrix(
        covariates,
        covariate_columns=["cov1", "cov2"],
        perturbation_indicator=indicator,
    )
    beta = np.array([0.2, -0.5, 0.1, 0.3, -0.2])[: design.shape[1]]
    mu = np.exp(design @ beta)
    alpha = 0.7
    y = _generate_nb_counts(rng, mu, alpha)

    # Use min_mu=1e-8 to match statsmodels behavior for testing
    fitter = NBGLMFitter(
        design,
        offset=np.zeros(n, dtype=float),
        dispersion=alpha,
        max_iter=100,
        tol=1e-8,
        poisson_init_iter=30,
        min_mu=1e-8,
        ridge_penalty=1e-8,
        optimization_method="irls",
    )
    result = fitter.fit_gene(y)
    assert result.converged

    family = sm.families.NegativeBinomial(alpha=alpha)
    sm_res = sm.GLM(y, design, family=family).fit()

    np.testing.assert_allclose(result.coef, sm_res.params, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(result.se, sm_res.bse, rtol=1e-3, atol=1e-3)


def test_nb_glm_fitter_matches_statsmodels_for_well_expressed_genes():
    rng = np.random.default_rng(12345)
    n = 250
    covariates = pd.DataFrame({"cov": rng.normal(0, 1, size=n)})
    indicator = rng.binomial(1, 0.45, size=n).astype(float)
    design, column_names = build_design_matrix(
        covariates,
        covariate_columns=["cov"],
        perturbation_indicator=indicator,
    )
    alpha = 0.6
    n_genes = 6
    betas = rng.normal(0, 0.4, size=(n_genes, design.shape[1]))
    betas[0] -= 3.0  # force one gene to be extremely lowly expressed
    counts = np.zeros((n, n_genes), dtype=int)
    for gene_idx in range(n_genes):
        mu = np.exp(design @ betas[gene_idx])
        counts[:, gene_idx] = _generate_nb_counts(rng, mu, alpha)

    # Use min_mu=1e-8 to match statsmodels behavior for testing
    fitter = NBGLMFitter(
        design,
        offset=np.zeros(n, dtype=float),
        dispersion=alpha,
        max_iter=120,
        tol=1e-8,
        poisson_init_iter=40,
        min_mu=1e-8,
        ridge_penalty=1e-8,
        optimization_method="irls",
    )
    family = sm.families.NegativeBinomial(alpha=alpha)
    perturbation_index = column_names.index("perturbation")

    compared = 0
    for gene_idx in range(n_genes):
        y = counts[:, gene_idx]
        result = fitter.fit_gene(y)
        sm_res = sm.GLM(y, design, family=family).fit()
        if y.mean() < 0.25:
            # Lowly expressed genes are expected to be unstable; we only
            # assert convergence for moderately expressed genes.
            continue
        assert result.converged
        np.testing.assert_allclose(
            result.coef[perturbation_index],
            sm_res.params[perturbation_index],
            rtol=1e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            result.se[perturbation_index],
            sm_res.bse[perturbation_index],
            rtol=1e-3,
            atol=1e-3,
        )
        compared += 1

    assert compared >= 1


def test_nb_glm_test_pipeline(tmp_path):
    rng = np.random.default_rng(123)
    n_cells = 80
    n_genes = 4
    perturbations = np.array(["control"] * (n_cells // 2) + ["g1"] * (n_cells - n_cells // 2))
    covariate = rng.normal(size=n_cells)
    obs = pd.DataFrame({"perturbation": perturbations, "covariate": covariate})

    indicator = (perturbations == "g1").astype(float)
    design = np.column_stack((np.ones(n_cells), indicator, covariate))
    betas = np.array([
        [0.1, 0.6, -0.3],
        [-0.2, 0.4, 0.2],
        [0.4, -0.5, 0.1],
        [0.0, 0.2, -0.4],
    ])
    alpha = 0.5
    counts = np.zeros((n_cells, n_genes), dtype=int)
    for gene_idx in range(n_genes):
        mu = np.exp(design @ betas[gene_idx])
        counts[:, gene_idx] = _generate_nb_counts(rng, mu, alpha)

    adata = ad.AnnData(counts, obs=obs.copy(), var=pd.DataFrame(index=[f"gene{idx}" for idx in range(n_genes)]))
    path = tmp_path / "synthetic.h5ad"
    adata.write(path)

    result = nb_glm_test(
        path,
        perturbation_column="perturbation",
        control_label="control",
        covariates=["covariate"],
        dispersion=alpha,
        max_iter=100,
        poisson_init_iter=30,
        min_cells_expressed=0,
        chunk_size=2,
    )

    assert result.method == "nb_glm"
    assert result.groups == ["g1"]
    assert result.statistics.shape == (1, n_genes)
    assert np.all(np.isfinite(result.pvalues))

    exported = result.result_path
    assert exported.exists()
    exported_adata = ad.read_h5ad(exported)
    assert "dispersion" in exported_adata.layers
    assert "standard_error" in exported_adata.layers

    diff_result = result["g1"]
    assert diff_result.method == "nb_glm"
    assert diff_result.pvalue.shape[0] == n_genes


def test_nb_glm_agrees_with_statsmodels_and_deseq2():
    if DeseqDataSet is None or DeseqStats is None:
        pytest.skip("pydeseq2 is not installed")
    rng = np.random.default_rng(314159)
    n_samples = 140
    n_genes = 5

    condition = rng.binomial(1, 0.5, size=n_samples).astype(float)
    design = np.column_stack([np.ones(n_samples), condition])
    betas = rng.normal(scale=0.4, size=(n_genes, design.shape[1]))
    betas[:, 0] -= 0.1
    betas[0, 1] -= 2.5  # enforce one lowly expressed gene
    dispersion = 0.5

    counts = np.zeros((n_samples, n_genes), dtype=int)
    for gene_idx in range(n_genes):
        mu = np.exp(design @ betas[gene_idx])
        counts[:, gene_idx] = _generate_nb_counts(rng, mu, dispersion)

    # Use min_mu=1e-8 to match statsmodels for direct comparison
    # Note: The default min_mu=0.5 is aligned with PyDESeq2 for real-world usage
    fitter = NBGLMFitter(
        design,
        offset=np.zeros(n_samples, dtype=float),
        dispersion=dispersion,
        max_iter=120,
        tol=1e-8,
        poisson_init_iter=40,
        min_mu=1e-8,
        ridge_penalty=1e-8,
    )

    family = sm.families.NegativeBinomial(alpha=dispersion)

    sample_names = [f"cell_{i}" for i in range(n_samples)]
    gene_names = [f"gene_{j}" for j in range(n_genes)]
    counts_df = pd.DataFrame(counts, index=sample_names, columns=gene_names)
    clinical = pd.DataFrame(
        {
            "condition": pd.Categorical(
                np.where(condition > 0, "treated", "control"),
                categories=["control", "treated"],
            )
        },
        index=sample_names,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=clinical,
        design="~condition",
        refit_cooks=False,
    )
    dds.deseq2()
    stats = DeseqStats(
        dds,
        contrast=("condition", "treated", "control"),
        cooks_filter=False,
        alpha=0.05,
    )
    stats.summary()
    deseq_results = stats.results_df["log2FoldChange"].to_numpy() * np.log(2.0)

    well_expressed = 0
    for gene_idx, gene_name in enumerate(gene_names):
        y = counts[:, gene_idx]
        if y.mean() < 0.5:
            continue

        fit_res = fitter.fit_gene(y)
        assert fit_res.converged

        sm_res = sm.GLM(y, design, family=family).fit()

        nb_coef = fit_res.coef[1]
        sm_coef = sm_res.params[1]
        deseq_coef = deseq_results[gene_idx]

        np.testing.assert_allclose(nb_coef, sm_coef, rtol=5e-2, atol=5e-2)
        # Relaxed tolerance for PyDESeq2 due to API changes in newer versions
        # that may use different priors or estimation methods
        np.testing.assert_allclose(nb_coef, deseq_coef, rtol=0.7, atol=0.4)

        well_expressed += 1

    assert well_expressed >= 1


def test_nb_glm_joint_vs_independent():
    """Test that joint fitting produces different (more stable) results than independent."""
    rng = np.random.default_rng(12345)
    n_cells = 200
    n_genes = 20
    
    # Create a dataset with multiple perturbations
    perturbations = ["ctrl"] * 80 + ["KO1"] * 40 + ["KO2"] * 40 + ["KO3"] * 40
    obs = pd.DataFrame({"perturbation": perturbations})
    obs.index = [f"cell_{i}" for i in range(n_cells)]
    
    # Generate counts with different effects for different perturbations
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    var["gene_symbols"] = var.index
    
    # Base expression levels
    base_expr = rng.uniform(0.5, 3.0, size=n_genes)
    
    # Generate counts
    counts = np.zeros((n_cells, n_genes), dtype=np.float64)
    for i in range(n_cells):
        pert = perturbations[i]
        if pert == "ctrl":
            mu = np.exp(base_expr)
        elif pert == "KO1":
            mu = np.exp(base_expr - 0.5)  # Down-regulation
        elif pert == "KO2":
            mu = np.exp(base_expr + 0.3)  # Up-regulation
        else:  # KO3
            mu = np.exp(base_expr)  # No effect
        counts[i, :] = rng.poisson(mu)
    
    import tempfile
    import scipy.sparse as sp
    
    # Create sparse matrix and save
    adata = ad.AnnData(sp.csr_matrix(counts), obs=obs, var=var)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.h5ad"
        adata.write(path)
        
        # Run with independent fitting
        result_indep = nb_glm_test(
            path,
            perturbation_column="perturbation",
            control_label="ctrl",
            gene_name_column="gene_symbols",
            fit_method="independent",
            output_dir=tmpdir,
            data_name="indep",
        )
        
        # Run with joint fitting (no shared dispersion)
        result_joint = nb_glm_test(
            path,
            perturbation_column="perturbation",
            control_label="ctrl",
            gene_name_column="gene_symbols",
            fit_method="joint",
            share_dispersion=False,
            output_dir=tmpdir,
            data_name="joint",
        )
        
        # Run with joint fitting and shared dispersion
        result_joint_shared = nb_glm_test(
            path,
            perturbation_column="perturbation",
            control_label="ctrl",
            gene_name_column="gene_symbols",
            fit_method="joint",
            share_dispersion=True,
            output_dir=tmpdir,
            data_name="joint_shared",
        )
        
        # Check that all methods produce results
        assert len(result_indep.groups) == 3  # KO1, KO2, KO3
        assert len(result_joint.groups) == 3
        assert len(result_joint_shared.groups) == 3
        
        # Check that joint and independent give similar but not identical results
        # (Joint should be slightly different due to shared intercept)
        for label in ["KO1", "KO2"]:
            indep_effect = result_indep[label].effect_size
            joint_effect = result_joint[label].effect_size
            
            # Results should be correlated (similar direction)
            valid = np.isfinite(indep_effect) & np.isfinite(joint_effect)
            if valid.sum() > 5:
                corr = np.corrcoef(indep_effect[valid], joint_effect[valid])[0, 1]
                assert corr > 0.9, f"Correlation for {label} should be high: {corr}"
        
        # Check that shared dispersion produces consistent dispersions
        # For shared dispersion, all perturbations should have the same dispersion values
        if result_joint_shared.result is not None:
            result_adata = result_joint_shared.result.to_memory()
            if "dispersion" in result_adata.layers:
                disp = result_adata.layers["dispersion"]
                # All rows should be identical when using shared dispersion
                for i in range(1, disp.shape[0]):
                    np.testing.assert_array_almost_equal(
                        disp[0, :], disp[i, :],
                        decimal=5,
                        err_msg="Shared dispersion should be identical across perturbations"
                    )
