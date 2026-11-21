"""A check, shared by the GP posteriors, that the model's fixed parameters are unchanged.

Each posterior's `check_versions` calls it. Having that method makes the posterior a
`HasVersions`, so `lk.core.mean_outputs`, `lk.predict_outputs` and `lk.sample_outputs` also call it
before they run a weight-space posterior (`FSPPosterior`). Function-space posteriors
(`NystromGP`, `VariationalGP`) call it in `at`.
"""

from collections.abc import Mapping

from laplacekit.core import ModelFn


def check_versions(model: ModelFn, versions: Mapping[str, int], used: str) -> None:
    """Raise `RuntimeError` if a fixed parameter of `model`'s module changed in place.

    `versions` holds the version counters taken at fit time. `used` names what the posterior
    computed from the old value, for the error message.
    """

    module = model.module
    live = dict(module.named_parameters()) if module is not None else {}

    for name, version in versions.items():
        now = live[name]._version if name in live else None

        if now != version:
            raise RuntimeError(
                f"fixed parameter {name!r} changed in place since the fit (version "
                f"{version}, now {now}); {used}; refit the posterior"
            )
