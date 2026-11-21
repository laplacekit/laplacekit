"""The training loop of `train`, with marginal-likelihood steps (Immer et al. 2021, Alg. 1).

Each epoch takes one pass over the data, with one optimizer step per batch on the
negative log joint per example. Every few epochs, a Laplace fit gives the log marginal
likelihood, and the hyperparameters take Adam steps on it. `train` keeps the epoch whose
estimation had the best marginal likelihood and returns its fit in `Trained`.
"""

import copy
import dataclasses
import math
from collections.abc import Iterable
from typing import Any, Generic, TypeVar, cast

import torch
from torch import Tensor, nn

import laplacekit as lk
import laplacekit.core as core
from laplacekit.core import (
    Batch,
    HessianApprox,
    LaplacePosterior,
    Likelihood,
    ModelFn,
    Params,
)
from laplacekit_marglik.hypers import (
    PriorStructure,
    _Hypers,
    _init,
    _noise_update,
    _penalty,
    _prior,
    _prior_leaves,
    _with_sigma,
)

__all__ = ["Trained", "train"]


# Explicit variance, as for `LaplacePosterior`: the likelihood's type, covariant, so a
# result with a Gaussian is assignable to a bare `Trained`.
_L_co = TypeVar("_L_co", bound=Likelihood, covariant=True, default=Likelihood)


@dataclasses.dataclass(frozen=True)
class Trained(Generic[_L_co]):  # noqa: UP046
    """The result of `train`: the kept fit and the training history.

    Args:
        posterior: The Laplace fit of the kept epoch. Its type carries the likelihood that
            `train` was given.
        margliks: The negative log marginal likelihood at every hyperparameter step.
        losses: The mean training loss per example of every epoch.
        best_epoch: The kept epoch, or `None` when no estimation ran.
    """

    posterior: LaplacePosterior[Any, Any, _L_co]
    margliks: tuple[float, ...]
    losses: tuple[float, ...]
    best_epoch: int | None


@dataclasses.dataclass(frozen=True)
class _Best:
    """The estimation that `train` keeps so far.

    It holds the estimation's last value, its epoch, the model's state dict at that epoch,
    and the prior leaves and σ at which that value was computed.
    """

    marglik: float
    epoch: int
    state: dict[str, Any]
    prior_precision: tuple[Tensor, ...]
    sigma: Tensor | None


type _BuiltinPrior = lk.priors.Isotropic | lk.priors.Layerwise | lk.priors.Diagonal

_STRUCTURES: dict[type, PriorStructure] = {
    lk.priors.Isotropic: "scalar",
    lk.priors.Layerwise: "layerwise",
    lk.priors.Diagonal: "diagonal",
}


def _structure(prior: object) -> tuple[PriorStructure, float]:
    """The structure of a built-in prior (class or instance), and its starting precision."""

    cls = prior if isinstance(prior, type) else type(prior)
    structure = _STRUCTURES.get(cls)

    if structure is None:
        raise TypeError(
            "train takes lk.priors.Isotropic, Layerwise or Diagonal (a class, or an instance "
            f"with one shared precision), got {prior!r}"
        )

    if isinstance(prior, type):
        return structure, 1.0

    lam = cast(_BuiltinPrior, prior).prior_precision

    if not isinstance(lam, Tensor) or lam.numel() != 1:
        raise ValueError(
            f"train starts every precision at the prior's one shared value, but {prior!r} has "
            f"several; pass {cls.__name__}(1.0), say, or the class {cls.__name__}"
        )

    if cast(_BuiltinPrior, prior).mean is not None:
        raise ValueError("train needs a zero-mean prior, but the prior has a mean")

    init = float(lam)

    if not math.isfinite(init):
        raise ValueError(f"train needs a finite prior precision, got {init}")

    return structure, init


def _check_options(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    n_hypersteps: int,
    marglik_frequency: int,
) -> None:
    for name, value in (
        ("epochs", epochs),
        ("n_hypersteps", n_hypersteps),
        ("marglik_frequency", marglik_frequency),
    ):
        if value < 1:
            raise ValueError(f"train needs {name} >= 1, got {value}")

    for i, group in enumerate(optimizer.param_groups):
        decay = group.get("weight_decay", 0.0)

        if decay != 0:
            raise ValueError(
                f"train sets the weight decay from the prior precision, but the optimizer's "
                f"param group {i} has weight_decay={decay}; pass weight_decay=0"
            )


def _data(data: lk.data.Data | Iterable[Any], size: int | None) -> tuple[lk.data.Data, int]:
    """The data source, and N from `data.n_data` or `n_data=`."""

    if isinstance(data, lk.data.Data):
        if size is not None and size != data.n_data:
            raise ValueError(f"train got n_data={size}, but the data has size {data.n_data}")

        src = data
    else:
        src = lk.data.from_loader(data, n_data=size)

    if src.n_data is None:
        raise ValueError(
            "train needs the number of training examples: pass n_data= with a plain loader, "
            "or data with a size"
        )

    return src, src.n_data


def _check_sigma_init(likelihood: object, sigma_init: float | Tensor | None) -> None:
    if sigma_init is None:
        return

    if not (isinstance(likelihood, lk.Gaussian) and likelihood.learned):
        raise ValueError(
            "train(sigma_init=...) needs a learned sigma, but the likelihood has none; pass "
            "lk.Gaussian() to learn it"
        )

    value = torch.as_tensor(sigma_init)

    if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
        raise ValueError(f"train needs a positive finite sigma_init, got {sigma_init}")


def train[L: Likelihood](
    model: nn.Module,
    data: lk.data.Data | Iterable[Any],
    likelihood: L,
    *,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    n_data: int | None = None,
    hessian_approx: HessianApprox[Any] | None = None,
    prior: type[lk.priors.Isotropic | lk.priors.Layerwise | lk.priors.Diagonal]
    | lk.priors.Isotropic
    | lk.priors.Layerwise
    | lk.priors.Diagonal = lk.priors.Layerwise,
    lr_hyper: float = 0.1,
    n_hypersteps: int = 10,
    marglik_frequency: int = 1,
    n_epochs_burnin: int = 0,
    sigma_init: float | Tensor | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    generator: torch.Generator | None = None,
) -> Trained[L]:
    r"""Train `model` and its hyperparameters together (Immer et al. 2021, Alg. 1).

    Each epoch takes one pass over the data. Each batch takes one `optimizer` step on the
    negative log joint per example,

    $$-\frac{\sum_i w_i \log p(y_i \mid f_i)}{\sum_i w_i}
    + \frac{1}{2N} \sum_b \lambda_b \lVert \theta_b \rVert^2,$$

    at the current prior precision $\lambda$ and noise $\sigma$, followed by one
    `scheduler` step if there is a scheduler. Here $w_i$ are the batch's weights (1 when
    there are none), $N$ is the number of training examples, and $b$ runs over the groups
    of weights that share a $\lambda_b$.

    After epoch $e$, when `e % marglik_frequency == 0` and `e >= n_epochs_burnin`,
    `lk.laplace` fits the trainable parameters with `hessian_approx` and without tuning.
    `n_hypersteps` Adam steps at `lr_hyper` then minimize the fit's negative log marginal
    likelihood $-\log p(\mathcal{D} \mid \lambda, \sigma)$ in $\log \lambda$, and
    also in $\log \sigma$ for a learned `lk.Gaussian()`. One Adam optimizer, in float64,
    runs through the whole training. These are laplace-torch's `marglik_training`
    arguments.

    A per-output $\sigma$, `lk.Gaussian()` with `sigma_init=torch.ones(k)`, is fixed
    during a fit, so Adam cannot move it. MacKay's closed-form update sets it after the
    Adam steps of each estimation:

    $$\sigma_k^2 = \frac{\sum_n w_n \big[(y_{nk} - f_{nk})^2
    + \operatorname{Var} f_{nk}\big]}{\sum_n w_n}.$$

    This is the weighted mean, over the data, of the squared residual plus the variance of
    $f_k$ from `lk.predict_outputs` without noise, at the $\lambda$ of the estimation's last
    marginal likelihood. It takes one more pass over the data per estimation. Later
    epochs and estimations use the new $\sigma$.

    `train` keeps the estimation with the lowest final negative log marginal likelihood,
    as laplace-torch's `marglik_training` does. At the end, the model gets the state dict
    of that epoch back, and `posterior` is a new fit at the $\lambda$ and $\sigma$ of that
    value. A per-output $\sigma$ is the one that estimation was fitted with, before its
    update. When no estimation ran, `posterior` is fitted at the initial $\lambda$ and
    $\sigma$. A 0-dim $\lambda$ stays in float64, as in the Adam steps, and a vector
    $\lambda$ takes the model's dtype.

    Args:
        model: The network. Its trainable parameters are trained in place, and its train
            or eval mode is left unchanged.
        data: Training data with targets, as an `lk.data.Data` or a re-iterable loader
            that `lk.data.from_loader` wraps.
        likelihood: The likelihood. `lk.Gaussian()` learns $\sigma$ from 1, and
            `lk.Gaussian(sigma=0.1)` keeps it fixed.
        optimizer: The optimizer over the model's parameters. Its param groups need
            `weight_decay=0`, since $\lambda$ sets the weight decay.
        epochs: The number of epochs, at least 1.
        n_data: The number of training examples $N$ for a plain loader. An `lk.data.Data`
            gives it as `data.n_data`.
        hessian_approx: The curvature method of each fit. `None` means `lk.laplace`'s
            default.
        prior: How $\lambda$ is shared, as a prior class or an instance with one shared
            precision: `lk.priors.Isotropic` (laplace-torch's `"scalar"`),
            `lk.priors.Layerwise` (one $\lambda$ per `Layout.by_module` block) or
            `lk.priors.Diagonal` (one per weight). An instance's precision is the starting
            value of every $\lambda$; a class starts at 1.
        lr_hyper: The Adam learning rate of the hypersteps.
        n_hypersteps: The number of Adam steps per estimation, at least 1.
        marglik_frequency: The number of epochs between estimations, at least 1.
        n_epochs_burnin: The first epoch that may run an estimation.
        sigma_init: The start of a learned $\sigma$ (`lk.Gaussian()`), a float or one
            value per output, as laplace-torch's `sigma_noise_init`. `None` starts at 1.
            Only for a learned $\sigma$: with a fixed one it raises.
        scheduler: A learning-rate scheduler for `optimizer`, stepped after every
            optimizer step.
        generator: Seeds the data passes and the fits. One seed is drawn from it, or from
            torch's global RNG when `None`, so `torch.manual_seed` makes training
            repeatable. The passes and fits draw, in turn, from one CPU generator seeded
            from it.

    Returns:
        The `Trained` result.

    Raises:
        TypeError: `prior` is not one of the three built-in priors, or `data` is its own
            iterator (a generator, for example), or an item is neither a `Batch` nor an
            `(inputs, targets)` pair.
        ValueError: Before the first optimizer step, when `epochs`, `n_hypersteps` or
            `marglik_frequency` is below 1, `sigma_init` is given without a learned
            $\sigma$ or is not positive and finite, the prior has several precisions or a
            mean, an
            optimizer param group has a nonzero `weight_decay`, the module has no trainable
            parameter, `n_data` differs from
            `data.n_data`, neither gives the number of examples, the data has no batches,
            or the first batch has no targets. Only the first batch is checked for
            targets. A later batch without targets raises when training reaches it, after
            the earlier batches have changed the parameters. A fit can also raise it, as
            in `lk.laplace`.
        StructureError: Before the first optimizer step, a dry run of `lk.laplace` on
            the first batch raises it when the curvature or the prior refuses the
            structure (for example KFAC with `"diagonal"`), or when the log marginal
            likelihood refuses the fit (as in `LaplacePosterior.log_marglik`). Later fits
            and estimations can raise it the same way.
        CapabilityError: The dry run raises it when the curvature or the prior refuses
            the model's capabilities. Otherwise as in `lk.laplace`.
    """

    _check_options(optimizer, epochs, n_hypersteps, marglik_frequency)
    _check_sigma_init(likelihood, sigma_init)
    structure, prior_init = _structure(prior)
    params = {k: p for k, p in model.named_parameters() if p.requires_grad}

    if not params:
        raise ValueError("train needs a trainable parameter; the module has none")

    src, n_data = _data(data, n_data)
    gen = lk.core.spawn_generator(generator)

    fn = ModelFn.from_module(model)
    dtype = next(iter(params.values())).dtype

    gauss = likelihood if isinstance(likelihood, lk.Gaussian) else None

    # A learned sigma starts at `sigma_init`, a scalar or one per output.
    if gauss is not None and gauss.learned:
        init = 1.0 if sigma_init is None else sigma_init
        start = torch.as_tensor(init, dtype=dtype, device=gauss.sigma.device)
        gauss = copy.replace(gauss, sigma=start)
        likelihood = cast(L, gauss)

    sigma0 = None if gauss is None else gauss.sigma
    learn_sigma = gauss is not None and gauss.learned and not gauss.per_output

    # A per-output σ is a fit-time leaf: MacKay's update sets it after each estimation.
    vec = sigma0 if gauss is not None and gauss.per_output else None
    learn_vec = gauss is not None and gauss.learned and vec is not None

    hypers = _init(structure, params, prior_init, sigma0 if learn_sigma else None)
    hyper_opt = torch.optim.Adam(hypers.logs, lr=lr_hyper)

    def fit_values(logs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """The prior leaves for a fit, detached.

        0-dim in float64, larger in the model's dtype, as in the hypersteps.
        """

        return tuple(x.detach().exp() if x.ndim == 0 else x.detach().exp().to(dtype) for x in logs)

    def step_values(logs: tuple[Tensor, ...], like: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """Leaves for a hyperstep: 0-dim in float64, larger ones in their own dtype."""

        return tuple(
            x.exp().reshape(t.shape) if t.ndim == 0 else x.exp().reshape(t.shape).to(t.dtype)
            for x, t in zip(logs, like, strict=True)
        )

    # `prior_init` itself, not exp(log(prior_init)), which can be one ulp off.
    init_prior = tuple(
        torch.full(x.shape, prior_init, dtype=torch.float64 if x.ndim == 0 else dtype)
        for x in hypers.prior
    )

    def fit(
        source: lk.data.Data, precision: tuple[Tensor, ...], sigma: Tensor | None
    ) -> LaplacePosterior[Any, Any, L]:
        return lk.laplace(
            model,
            source,
            _with_sigma(likelihood, sigma),
            hessian_approx=hessian_approx,
            subset=lk.subsets.trainable,
            prior=_prior(hypers, precision),
            hyper_tuner=None,
            generator=gen,
        )

    def sigma_fit() -> Tensor | None:
        """σ for a fit.

        The learned one-element σ, the per-output vector, or the likelihood's own (`None`).
        """

        if hypers.sigma is None or sigma0 is None:
            return vec

        return hypers.sigma.detach().exp().reshape(sigma0.shape).to(sigma0.dtype)

    def neg_log_marglik(post: LaplacePosterior[Any, Any]) -> tuple[Tensor, tuple[Tensor, ...]]:
        """-log Z at the current logs, and the leaves it was evaluated at."""

        old = _prior_leaves(post, hypers)
        new = step_values(hypers.prior, old)

        if hypers.sigma is not None:
            old_sigma = cast(Tensor, cast(Any, post.likelihood).sigma)
            old = (*old, old_sigma)
            new = (*new, *step_values((hypers.sigma,), (old_sigma,)))

        return -core.replace_leaves(post, old, new).log_marglik(), new

    # The dry run: every refusal of the curvature and the prior fires on the first batch,
    # before the first optimizer step. It draws one batch from `src`, so a shuffling loader
    # consumes the global RNG once.
    first = next(iter(src(gen)), None)

    if first is None:
        raise ValueError("train: the data yielded no batches")

    if first.targets is None:
        raise ValueError("train needs data with targets, but the first batch has none")

    dry = fit(lk.data.from_loader([first], n_data=n_data), init_prior, None)
    dry = copy.replace(dry, curvature=core.with_eigen(dry.curvature))

    with torch.no_grad():
        neg_log_marglik(dry)

    margliks: list[float] = []
    losses: list[float] = []
    best: _Best | None = None

    for e in range(1, epochs + 1):
        lik = _with_sigma(likelihood, sigma_fit())
        prec = fit_values(hypers.prior)
        total = 0.0

        for batch in src(gen):
            total += _train_step(fn, params, lik, hypers, prec, batch, n_data, optimizer)

            if scheduler is not None:
                scheduler.step()

        losses.append(total / n_data)

        if e % marglik_frequency != 0 or e < n_epochs_burnin:
            continue

        post = fit(src, fit_values(hypers.prior), sigma_fit())
        post = copy.replace(post, curvature=core.with_eigen(post.curvature))

        with torch.enable_grad():
            for _ in range(n_hypersteps):
                hyper_opt.zero_grad()
                value, at = neg_log_marglik(post)
                margliks.append(value.item())
                value.backward()
                hyper_opt.step()

        # Early stopping on the evidence: the state and the (λ, σ) of the last value. A
        # per-output σ is the one this estimation was fit with, from before its update.
        n_prior = len(hypers.prior)

        if best is None or margliks[-1] < best.marglik:
            kept = tuple(t.detach() for t in at[:n_prior])
            sigma = vec

            if sigma0 is not None and hypers.sigma is not None:
                sigma = at[n_prior].detach().to(sigma0.dtype)

            best = _Best(margliks[-1], e, copy.deepcopy(model.state_dict()), kept, sigma)

        # MacKay's update, at the λ of the estimation's last marglik value, as laplace-torch's
        # `noise_per_output_update`.
        if learn_vec and vec is not None:
            lam = tuple(t.detach() for t in at[:n_prior])
            at_lam = core.replace_leaves(post, _prior_leaves(post, hypers), lam)
            vec = _noise_update(at_lam, src, gen, vec)

    if best is None:
        post = fit(src, init_prior, None)
        note = f"marglik, no estimation in {epochs} epochs"
    else:
        model.load_state_dict(best.state)
        post = fit(src, best.prior_precision, best.sigma)
        note = f"marglik, best epoch {best.epoch} of {epochs}"

    post = copy.replace(post, provenance=(*post.provenance, ("training", note)))

    return Trained(
        posterior=post,
        margliks=tuple(margliks),
        losses=tuple(losses),
        best_epoch=None if best is None else best.epoch,
    )


def _train_step(
    fn: ModelFn,
    params: Params,
    likelihood: Likelihood,
    hypers: _Hypers,
    prec: tuple[Tensor, ...],
    batch: Batch,
    n_data: int,
    optimizer: torch.optim.Optimizer,
) -> float:
    """One optimizer step on the batch's negative log joint per example.

    Returns the loss times the batch's total weight, for the epoch mean.
    """

    if batch.targets is None:
        raise ValueError("train needs data with targets, but a batch has none")

    optimizer.zero_grad()

    with torch.enable_grad():
        f = fn(params, batch.inputs)
        log_p = core.log_prob(likelihood, f, batch.targets, inputs=batch.inputs)
        weights = batch.weights

        if weights is None:
            count = float(log_p.shape[0])
            data_term = -log_p.sum() / count
        else:
            count = weights.sum().item()
            data_term = -(weights * log_p).sum() / weights.sum()

        loss = data_term + _penalty(hypers, prec, params) / n_data
        loss.backward()

    optimizer.step()

    return loss.item() * count
