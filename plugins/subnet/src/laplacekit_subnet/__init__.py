"""Subnetwork Laplace (Daxberger et al. 2021): the approximation over `n` chosen weight entries.

Each strategy returns a `Subset` for `lk.laplace(..., subset=)`, which picks the entries:

```python
diag = lk.laplace(model, data, likelihood=lik, subset=lk.subsets.all, hessian_approx=lk.DiagGGN())
post = lk.laplace(
    model, data, likelihood=lik,
    subset=lk.subnet.largest_variance(diag, 51),  # or largest_magnitude(51), random(51)
    hessian_approx=lk.DenseGGN(joint=True),
    prior=lk.subnet.rescaled_prior(lam, 51, model),
)
```

`random` draws the entries uniformly, `largest_magnitude` takes the largest weights, and
`largest_variance` the largest marginal variances. Those can come from a diagonal fit, as
above, or from `swag_variance`. `rescaled_prior` scales the prior precision to the
subnetwork as Daxberger et al. do. To choose entries by hand, use `lk.subsets.from_mask`
or `lk.subsets.named` from the core.

Code reaches this package as `lk.subnet`, which imports it on first use.
"""

import laplacekit as lk
from laplacekit_subnet.prior import rescaled_prior
from laplacekit_subnet.selection import largest_magnitude, largest_variance, random
from laplacekit_subnet.swag import swag_variance

__all__ = ["largest_magnitude", "largest_variance", "random", "rescaled_prior", "swag_variance"]


plugin = lk.Plugin(supports=">=0.1a0,<0.2")
"""The record behind the `laplacekit.plugins` entry point."""
