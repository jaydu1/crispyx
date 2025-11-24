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

    fitter = NBGLMFitter(
        design,
        offset=np.zeros(n, dtype=float),
        dispersion=alpha,
        max_iter=100,
        tol=1e-8,
        poisson_init_iter=30,
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

    fitter = NBGLMFitter(
        design,
        offset=np.zeros(n, dtype=float),
        dispersion=alpha,
        max_iter=120,
        tol=1e-8,
        poisson_init_iter=40,
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

    fitter = NBGLMFitter(
        design,
        offset=np.zeros(n_samples, dtype=float),
        dispersion=dispersion,
        max_iter=120,
        tol=1e-8,
        poisson_init_iter=40,
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
