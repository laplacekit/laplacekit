"""Marginal-likelihood training (Immer et al. 2021).

`train` trains a network as usual. Every few epochs, it also fits the prior precision and
the likelihood's noise by maximizing the Laplace approximation of the log marginal
likelihood:

```python
trained = lk.marglik.train(
    model, loader, lk.Gaussian(), n_data=len(dataset),
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-2),  # no weight decay
    epochs=300, marglik_frequency=5,
)
post = trained.posterior  # the Laplace fit of the kept epoch, at the learned prior and noise
```

`prior=` sets how the weights share the prior precision: `lk.priors.Isotropic` gives one
precision for all weights, `lk.priors.Layerwise` (the default) one per parameter block,
and `lk.priors.Diagonal` one per weight.

Code reaches this package as `lk.marglik`, which imports it on first use.
"""

import laplacekit as lk
from laplacekit_marglik.training import Trained, train

__all__ = ["Trained", "train"]


plugin = lk.Plugin(supports=">=0.1a0,<0.2")
