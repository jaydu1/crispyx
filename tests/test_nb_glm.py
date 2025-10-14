import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import anndata as ad

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from streamlined_crispr.glm import NBGLMFitter, build_design_matrix
from streamlined_crispr.de import nb_glm_test


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
