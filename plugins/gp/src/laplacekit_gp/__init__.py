"""GP methods for function-space Laplace: ELLA, VaLLA and FSP-Laplace.

The methods are ELLA (Deng et al. 2022), VaLLA (Ortega et al. 2024) and FSP-Laplace
(Cinquin et al. 2024). ELLA approximates the linearized network's kernel with Nyström
features. VaLLA fits variational inducing points. FSP-Laplace places a GP prior on the
network's outputs.

Code reaches this package as `lk.gp`, which imports it on first use.
"""

import laplacekit as lk
from laplacekit_gp.fsp import (
    MAX_CONTEXT_JACOBIAN_ENTRIES,
    FSPPosterior,
    FunctionPrior,
    RangePrecision,
    fsp_laplace,
    rkhs_norm,
)
from laplacekit_gp.kernels import RBF, Kernel, Matern, Periodic, Product, Sum
from laplacekit_gp.nystrom import ELLA, MAX_BASIS_ENTRIES, NystromGP
from laplacekit_gp.points import (
    MAX_KMEANS_ENTRIES,
    Drawn,
    Picked,
    Points,
    fixed,
    from_data,
    grid,
    kmeans,
    uniform,
)
from laplacekit_gp.variational import (
    VaLLA,
    VariationalGP,
    fit_variational,
    variational_loss,
)

__all__: list[str] = [
    "MAX_BASIS_ENTRIES",
    "MAX_CONTEXT_JACOBIAN_ENTRIES",
    "MAX_KMEANS_ENTRIES",
    "RBF",
    "Drawn",
    "FSPPosterior",
    "FunctionPrior",
    "Kernel",
    "Matern",
    "ELLA",
    "NystromGP",
    "Periodic",
    "Picked",
    "Points",
    "Product",
    "RangePrecision",
    "Sum",
    "VaLLA",
    "VariationalGP",
    "fit_variational",
    "fixed",
    "from_data",
    "fsp_laplace",
    "grid",
    "kmeans",
    "rkhs_norm",
    "uniform",
    "variational_loss",
]

plugin = lk.Plugin(supports=">=0.1a0,<0.2")
