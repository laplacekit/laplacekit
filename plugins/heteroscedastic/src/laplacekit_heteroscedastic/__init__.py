r"""Heteroscedastic regression, with Gaussian likelihoods whose variance depends on the input.

End the model with `NaturalHead`, fit with `NaturalGaussian`, and split the predictive
variance with `moments`:

```python
lik = lk.heteroscedastic.NaturalGaussian()  # the model ends with NaturalHead()
post = lk.laplace(model, data, likelihood=lik, subset=lk.subsets.all, hessian_approx=lk.DenseGGN())
parts = lk.heteroscedastic.moments(lik, lk.predict_outputs(post, x_test))
parts.mean, parts.epistemic, parts.aleatoric
```

There are two likelihoods. In both, the model's last axis holds two halves of $K$ entries
each, for $K$ targets.

- `NaturalGaussian` follows Immer et al. 2023. The halves are the natural parameters
  $[\eta_1, \eta_2]$ of $K$ Gaussians, with $\eta_2 < 0$. This is laplace-torch's
  `join_mean_variance` layout, with natural parameters in place of the mean and variance.
- `MeanVarianceGaussian` takes the halves as `[mean, var]`, and the Laplace approximation covers
  the mean only, as in laplace-torch's `HeteroscedasticGaussianLikelihood`.

`moments` gives the predictive mean with its epistemic and aleatoric variances.
`NoisyMean`, the default predictive of both likelihoods, is a Normal whose variance is
the sum of the two.

`lk.marglik.train` does marginal-likelihood training, the other half of Immer et al.,
with either likelihood. The output Hessian of `NaturalGaussian` stays differentiable in
$f$, and none of its checks run under vmap, so that training can use it.

Code reaches this package as `lk.heteroscedastic`, which imports it on first use.
"""

import laplacekit as lk
from laplacekit_heteroscedastic.mean_variance import MeanVarianceGaussian
from laplacekit_heteroscedastic.natural import NaturalGaussian, NaturalHead
from laplacekit_heteroscedastic.predictive import Moments, NoisyMean, moments

__all__ = [
    "MeanVarianceGaussian",
    "Moments",
    "NaturalGaussian",
    "NaturalHead",
    "NoisyMean",
    "moments",
]


plugin = lk.Plugin(supports=">=0.1a0,<0.2")
