r"""Laplace on denoisers from Hugging Face `diffusers`.

Give `laplace` a denoiser, its DDPM-family scheduler and the clean training samples:

```python
post = lk.diffusers.laplace(unet, scheduler, loader, n_data=60_000)
eps = lk.predict_outputs(post, {"sample": x_t, "timestep": torch.tensor(t)})
eps.mean, eps.variance  # the predicted noise and its variance per pixel
```

A denoiser takes a noisy sample $x_t$ and a timestep $t$. It predicts the noise, the
velocity or the clean sample, whichever its scheduler's `prediction_type` names. For
each batch, `laplace` draws $t$ and the noise from the batch's generator, builds $x_t$
with the scheduler's `add_noise`, and fits a Gaussian likelihood with unit noise to the
target.
The model's outputs must have the target's shape, so a model that also predicts a
variance is refused. Flow-matching schedulers are not supported yet.

Code reaches this package as `lk.diffusers`, which imports it on first use.
"""

from importlib import metadata

import laplacekit as lk
from laplacekit_diffusers.ddpm import laplace
from laplacekit_diffusers.denoiser import denoiser

__all__ = ["denoiser", "health", "laplace"]


def health() -> tuple[lk.HealthCheck, ...]:
    """HealthCheck for `lk.health()` that `diffusers` is installed.

    Returns:
        One check, `ok` with the version when `diffusers` is installed and `error`
        otherwise.
    """

    try:
        return (lk.HealthCheck("diffusers", "ok", metadata.version("diffusers")),)
    except metadata.PackageNotFoundError:
        return (lk.HealthCheck("diffusers", "error", "not installed (required)"),)


plugin = lk.Plugin(supports=">=0.1a0,<0.2", health=health)
"""The record behind the `laplacekit.plugins` entry point."""
