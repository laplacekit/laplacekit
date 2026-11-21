"""Laplace approximations for PyTorch.

Docs are at https://laplacekit.org. Coding agents can read
https://laplacekit.org/llms.txt, an index of Markdown copies of every page,
the API reference included.
"""

import importlib
from types import ModuleType
from typing import TYPE_CHECKING

from laplacekit import (
    core,
    curvature,
    data,
    distributed,
    function_space,
    likelihoods,
    objectives,
    predictive,
    priors,
    subsets,
    testing,
    tuning,
)
from laplacekit._health import (
    HealthCheck,
    HealthReport,
    Plugin,
    PluginInfo,
    Plugins,
    health,
    plugins,
)
from laplacekit.core import (
    Batch,
    CapabilityError,
    LaplacePosterior,
    StructureError,
    load,
    per_block,
    predict_outputs,
    sample_outputs,
    save,
)
from laplacekit.curvature import (
    DenseGGN,
    DenseHessian,
    DiagGGN,
    LowRankGGN,
)
from laplacekit.curvlinops import KFAC
from laplacekit.front import laplace, predict
from laplacekit.likelihoods import Bernoulli, Categorical, Gaussian
from laplacekit.predictive import MC, Bridge, Linearized, Probit, WeightSamples
from laplacekit.tuning import GridSearch, MarginalLikelihood, Minimize, tune

if TYPE_CHECKING:
    import laplacekit_diffusers as diffusers  # noqa: F401  # loaded by __getattr__
    import laplacekit_gp as gp  # noqa: F401  # loaded by __getattr__
    import laplacekit_heteroscedastic as heteroscedastic  # noqa: F401  # loaded by __getattr__
    import laplacekit_marglik as marglik  # noqa: F401  # loaded by __getattr__
    import laplacekit_reward as reward  # noqa: F401  # loaded by __getattr__
    import laplacekit_subnet as subnet  # noqa: F401  # loaded by __getattr__
    import laplacekit_transformers as transformers  # noqa: F401  # loaded by __getattr__

__all__ = [
    "Batch",
    "Bernoulli",
    "Bridge",
    "CapabilityError",
    "Categorical",
    "DenseGGN",
    "DenseHessian",
    "DiagGGN",
    "Gaussian",
    "GridSearch",
    "HealthCheck",
    "HealthReport",
    "KFAC",
    "LaplacePosterior",
    "Linearized",
    "LowRankGGN",
    "MC",
    "MarginalLikelihood",
    "Minimize",
    "Plugin",
    "PluginInfo",
    "Plugins",
    "Probit",
    "StructureError",
    "WeightSamples",
    "core",
    "curvature",
    "data",
    "distributed",
    "function_space",
    "health",
    "laplace",
    "likelihoods",
    "load",
    "objectives",
    "per_block",
    "plugins",
    "predict",
    "predict_outputs",
    "predictive",
    "priors",
    "sample_outputs",
    "save",
    "subsets",
    "testing",
    "tune",
    "tuning",
]

if TYPE_CHECKING:
    # Type checkers see the lazy plugin aliases as public, so pyright accepts `lk.gp`.
    # At runtime they stay out of `__all__`: `from laplacekit import *` imports no plugin.
    __all__ += [
        "diffusers",
        "gp",
        "heteroscedastic",
        "marglik",
        "reward",
        "subnet",
        "transformers",
    ]


# Plugin packages reached as `lk.<alias>`, imported on first use. The aliases are not in
# `__all__`: `from laplacekit import *` would then import transformers, or fail without the
# plugin.
_ALIASES = {
    "transformers": "laplacekit_transformers",
    "diffusers": "laplacekit_diffusers",
    "subnet": "laplacekit_subnet",
    "heteroscedastic": "laplacekit_heteroscedastic",
    "reward": "laplacekit_reward",
    "gp": "laplacekit_gp",
    "marglik": "laplacekit_marglik",
}


def __getattr__(name: str) -> ModuleType:
    """`lk.<alias>` imports the plugin package behind it on first use.

    Args:
        name: The attribute name, such as `"gp"` for `laplacekit-gp`.

    Returns:
        The plugin package.

    Raises:
        AttributeError: `name` is not a plugin alias.
        ImportError: The plugin package is not installed. The message gives the
            `pip install "laplacekit[<alias>]"` command.
        ModuleNotFoundError: The plugin is installed but a module it imports is missing.
    """

    module = _ALIASES.get(name)

    if module is None:
        raise AttributeError(f"module 'laplacekit' has no attribute {name!r}")

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:
        if e.name != module:
            raise  # the plugin is there but broken: report what is missing

        raise ImportError(
            f'lk.{name} needs the laplacekit-{name} package: pip install "laplacekit[{name}]"'
        ) from None
