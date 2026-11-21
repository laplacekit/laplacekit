"""The front door: `lk.laplace` and `lk.predict`, thin compositions of public steps."""

import copy
import dataclasses
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import torch
from torch import nn
from torch.distributions import Distribution

import laplacekit.core as core
from laplacekit import curvature, priors, subsets
from laplacekit.core import (
    Capabilities,
    FitContext,
    HasTemplate,
    HessianApprox,
    HyperTuner,
    Inputs,
    LaplacePosterior,
    Likelihood,
    LinkApprox,
    ModelFn,
    Outputs,
    Params,
    Posterior,
    Predictive,
    Prior,
    Selection,
    Subset,
    as_pytree,
    fit_curvature,
    spawn_generator,
    tree_leaves,
    tree_map,
)
from laplacekit.data import Data, as_data
from laplacekit.predictive import Linearized
from laplacekit.tuning import MarginalLikelihood

__all__ = ["laplace", "predict"]


def _name(obj: Any) -> str:
    """The function's own name, or the factory's name for a closure it returned."""

    qual = getattr(obj, "__qualname__", None)

    return qual.split(".<locals>")[0] if qual else type(obj).__name__


def _detach(t: Any) -> Any:
    return t.detach() if isinstance(t, torch.Tensor) else t


# laplace-torch's strings, and what to pass here instead.
_HESSIAN_HINTS = {
    "kron": "lk.KFAC()",
    "kfac": "lk.KFAC()",
    "diag": "lk.DiagGGN()",
    "full": "lk.DenseGGN()",
    "lowrank": "lk.LowRankGGN(rank)",
    "gp": "lk.function_space.laplace(...)",
}
_LIKELIHOOD_HINTS = {
    "classification": "lk.Categorical()",
    "regression": "lk.Gaussian()",
    "binary": "lk.Bernoulli()",
}
_SUBSET_HINTS = {
    "all": "lk.subsets.all",
    "last_layer": "lk.subsets.last_layer",
    "subnetwork": "lk.subnet.largest_magnitude(n)",
}


def _instance_hint(arg: str, got: type, example: str) -> str:
    return (
        f"{arg} takes an instance, got the class {got.__name__}; pass {got.__name__}(...), "
        f"such as {arg}={example}"
    )


def _string_hint(arg: str, got: str, hints: Mapping[str, str], example: str) -> str:
    hint = hints.get(got.lower().replace("-", "_"), example)

    return f"{arg} takes an object, not a string: {arg}={got!r} is {arg}={hint}"


def _check_arguments(
    likelihood: object,
    hessian_approx: object,
    subset: object,
    prior: object,
    hyper_tuner: object,
) -> None:
    """One `TypeError` per argument of the wrong kind, naming what to pass instead."""

    if isinstance(likelihood, str):
        raise TypeError(
            _string_hint("likelihood", likelihood, _LIKELIHOOD_HINTS, "lk.Categorical()")
        )

    if isinstance(likelihood, type):
        raise TypeError(_instance_hint("likelihood", likelihood, "lk.Categorical()"))

    if not isinstance(likelihood, Likelihood):
        raise TypeError(
            f"likelihood takes a Likelihood, such as lk.Categorical() or lk.Gaussian(), got "
            f"{type(likelihood).__name__}"
        )

    if isinstance(hessian_approx, str):
        raise TypeError(_string_hint("hessian_approx", hessian_approx, _HESSIAN_HINTS, "lk.KFAC()"))

    if isinstance(hessian_approx, type):
        raise TypeError(_instance_hint("hessian_approx", hessian_approx, "lk.KFAC()"))

    if hessian_approx is not None and not isinstance(hessian_approx, core.Curvature | core.Staged):
        raise TypeError(
            "hessian_approx takes a Hessian approximation, such as lk.KFAC() or lk.DiagGGN(), got "
            f"{type(hessian_approx).__name__}"
        )

    if isinstance(subset, str):
        raise TypeError(_string_hint("subset", subset, _SUBSET_HINTS, "lk.subsets.last_layer"))

    if not (callable(subset) or isinstance(subset, Mapping | Selection)):
        raise TypeError(
            "subset takes a subset function such as lk.subsets.all, a parameter mapping or a "
            f"Selection, got {type(subset).__name__}"
        )

    if isinstance(prior, float | int):
        raise TypeError(
            f"prior takes a prior object, not a number: prior={prior!r} is "
            f"prior=lk.priors.Isotropic({float(prior)!r})"
        )

    if isinstance(prior, type):
        raise TypeError(_instance_hint("prior", prior, "lk.priors.Isotropic(1.0)"))

    if not isinstance(prior, Prior):
        raise TypeError(
            "prior takes a prior, such as lk.priors.Isotropic(1.0) or an earlier posterior, "
            f"got {type(prior).__name__}"
        )

    if isinstance(hyper_tuner, str):
        raise TypeError(
            f"hyper_tuner takes a HyperTuner, not a string: hyper_tuner={hyper_tuner!r} is "
            "hyper_tuner=lk.MarginalLikelihood() (or lk.GridSearch(objective, values))"
        )

    if isinstance(hyper_tuner, type):
        raise TypeError(_instance_hint("hyper_tuner", hyper_tuner, "lk.MarginalLikelihood()"))

    if hyper_tuner is not None and not isinstance(hyper_tuner, HyperTuner):
        raise TypeError(
            "hyper_tuner takes a HyperTuner, such as lk.MarginalLikelihood(), or None, got "
            f"{type(hyper_tuner).__name__}"
        )


def laplace[L: Likelihood](
    model: nn.Module | ModelFn,
    data: Data | Iterable[Any],
    likelihood: L,
    *,
    hessian_approx: HessianApprox[Any] | None = None,
    subset: Subset | Params | Selection = subsets.last_layer,
    prior: Prior = priors.Isotropic(),  # noqa: B008  # immutable
    hyper_tuner: HyperTuner | None = MarginalLikelihood(),  # noqa: B008  # immutable
    generator: torch.Generator | None = None,
) -> LaplacePosterior[Any, Any, L]:
    """Fit a Laplace approximation in weight space, then tune its hyperparameters.

    ```python
    post = lk.laplace(model, train_loader, lk.Categorical())
    probs = lk.predict(post, x).probs
    ```

    The probe runs first and checks what the model supports, so a method the model can't
    run raises before any data past the first batch is read. The default picks the last
    layer and a curvature per block (`lk.curvature.default`), and tunes the prior
    precision (and a learned `sigma`) by the marginal likelihood.

    `subset` may give a `Selection`. Its entries are fitted as parameters of their own
    (`"fc.weight[idx]"`), and `post.index` holds their indices. Hook-based methods (`KFAC`)
    then raise `CapabilityError`, and the default never picks them for those blocks.

    The result holds a copy of the fitted parameters as its mean. Training the module
    afterwards leaves it in place, so it can be the `prior` of the next fit (continual
    learning). The prior and the new curvature must add: a KFAC posterior can't be the
    prior of a KFAC fit, which raises `StructureError`, and `lk.priors.KronSum.of(post)` is
    the approximate opt-in that sums the Kronecker factors. A `LowRankGGN` posterior as the
    prior of a `LowRankGGN` fit is exact, but the width grows with every fit;
    `lk.priors.LowRankSum.of(post)` truncates it to a fixed rank. When the method has a
    template (`core.HasTemplate`; the built-ins do), a prior that doesn't fit raises before
    any data past the probe's batch, with the message prefixed "before the data pass:".

    On several ranks (a `torch.distributed` process group), each rank passes its own shard
    and the statistics are combined with `all_reduce`. Every rank needs at least one batch.

    Args:
        model: The model. A module is wrapped with `ModelFn.from_module`.
        data: A re-iterable (a list, a `DataLoader`) of `(inputs, targets)` pairs or
            `Batch`es, or a `Data` source such as `lk.data.from_loader(..., n_data=N)`.
        likelihood: The likelihood, such as `lk.Categorical()` or `lk.Gaussian()`.
        hessian_approx: The curvature method, such as `lk.KFAC()` or `lk.DiagGGN()`.
            `None` picks one per block from the probe (`lk.curvature.default`). Its needs
            are checked against the likelihood before any data is read.
        subset: The parameters to fit: a subset function such as `lk.subsets.all`, called
            with the model and the probe's capabilities, a parameter mapping, or a
            `Selection`.
        prior: The prior. Its location must match the fitted parameters, which is checked
            before the fit. A `Layerwise` or `Diagonal` prior with one shared precision is
            expanded to one precision per block or per weight first.
        hyper_tuner: Tunes the prior precision and the likelihood's learned
            hyperparameters after the fit. `None` skips the tune. A posterior with nothing
            to tune (a prior that is not a `TunablePrior`, say, and a likelihood that
            learns nothing) is returned untuned. Each `Dense` curvature gets its
            eigendecomposition first (`core.with_eigen`), so the tune needs no Cholesky per
            step.
        generator: Seeds the data passes and the method's draws. One seed is drawn from
            it, or from torch's global RNG when `None`, so `torch.manual_seed` makes the
            fit repeatable.

    Returns:
        The posterior, a `LaplacePosterior`. `print(post)` shows each resolved choice.
        With targets, it keeps the data-fit term, so `post.log_marglik()` works.

    Raises:
        TypeError: An argument has the wrong kind: a string (`"kron"`), a class instead of
            an instance (`lk.KFAC`), a number as the prior, or data that is one
            `(inputs, targets)` pair or a `Dataset`. The message names what to pass.
        CapabilityError: The likelihood or the model lacks what the method needs, the probe's
            batch fails the method's batch requirements, a hook-based method meets a
            `Selection`, or `last_layer` finds no parameterized module.
        StructureError: The prior's `tunable()` breaks its rules, the prior's location
            doesn't match the fitted parameters, the prior can't be added to the curvature,
            the subset refuses the model, or the probe or the fit finds badly shaped
            outputs or targets.
        ValueError: The data yields no batches, the likelihood finds a bad target value
            (float targets for a `Categorical`, say), a method refuses its settings, or
            the hyper tuner can't work on this data (`MarginalLikelihood` without targets).
    """

    _check_arguments(likelihood, hessian_approx, subset, prior, hyper_tuner)

    gen = spawn_generator(generator)
    fn = ModelFn.from_module(model) if isinstance(model, nn.Module) else model
    src = as_data(data)

    # An explicit method's likelihood needs don't depend on the probe: check them before data.
    if hessian_approx is not None:
        core.check_likelihood(hessian_approx.requires, likelihood, repr(hessian_approx))

    first = next(iter(src(spawn_generator(gen))), None)

    if first is None:
        raise ValueError("laplace: the data yielded no batches")

    # The order pass alone picks the subset; the transforms are then probed on the subset
    # only, so the rest of the model is never cloned or differentiated.
    order: Capabilities = core.probe(fn, likelihood, first, params={})
    picked: Params | Selection

    if isinstance(subset, Selection | Mapping):
        picked, subset_name = cast(Params | Selection, subset), "given"
    else:
        picked, subset_name = subset(fn, order), _name(subset)

    # A `Selection` fits its entries as parameters of their own, through a model that
    # scatters them back.
    fn, chosen = core.select(fn, picked)
    index = picked.selected_index if isinstance(picked, Selection) else {}

    # Detached views: the fit reads the live parameters (hook-based KFAC needs them) without
    # attaching to the user's autograd graph. The posterior gets its own copy below.
    chosen = {k: v.detach() for k, v in chosen.items()}

    caps: Capabilities = core.probe(fn, likelihood, first, params=chosen)

    chosen_approx: HessianApprox[Any] = (
        curvature.default(chosen, caps, likelihood=likelihood, weighted=first.weights is not None)
        if hessian_approx is None
        else hessian_approx
    )
    caps.check(chosen_approx.requires, repr(chosen_approx))
    core.check_likelihood(chosen_approx.requires, likelihood, repr(chosen_approx))
    chosen_approx.requires.check_batch(first, repr(chosen_approx))

    # A shared precision becomes one per block or weight, so each is tuned on its own.
    chosen_prior = prior.expand(chosen) if isinstance(prior, core.HasExpand) else prior

    # Learned hyperparameters tune in the dtype of the statistic: a float64 model under a
    # float32 default dtype tunes its prior precision and sigma in float64.
    stat = core.stat_dtype(next(iter(chosen.values())).dtype)
    chosen_prior = _in_dtype(chosen_prior, core.prior_hyperparameters(chosen_prior), stat)
    likelihood = _as_dtype(likelihood, stat)

    # A prior whose location does not match the parameters raises before any fitting.
    chosen_prior.location(chosen)

    # The tune runs after the fit: check what it will tune before the other batches.
    if hyper_tuner is not None:
        hyper_tuner.check(chosen_prior, likelihood, first)

    # The check sees the fit's context, with the unit likelihood the fit uses.
    fit = fit_curvature(fn, likelihood, chosen, chosen_approx, generator=gen, capabilities=caps)
    _check_prior_fits(chosen_prior, chosen_approx, chosen, fit.ctx)

    # Under no_grad: statistics are values here; by-hand fits stay differentiable.
    done = core.run(fit, src, generator=gen)
    result = tree_map(_detach, done.result)
    stats = tree_map(_detach, done.stats)
    last = stats[-1]

    size = src.n_data
    dtype = core.stat_dtype(next(iter(chosen.values())).dtype)
    n_data = torch.tensor(float(size), dtype=dtype) if size is not None else done.count.to(dtype)

    provenance = (
        ("subset", f"{subset_name} {_shapes(chosen)}"),
        ("hessian_approx", repr(chosen_approx)),
        ("hyper_tuner", repr(hyper_tuner)),
    )
    # The posterior owns a copy of its mean, so training the module later moves neither it
    # nor a later fit's prior (continual learning). The guard then fires for fixed
    # parameters only. The fit holds `likelihood` itself (typed `Likelihood`), so the
    # posterior's is L.
    fitted = cast(
        LaplacePosterior[Any, Any, L],
        fit.posterior(result, prior=chosen_prior, n_data=n_data).snapshot(),
    )
    data_fit = () if last.data_fit is None else (last.data_fit,)
    post = dataclasses.replace(fitted, provenance=provenance, data_fit=data_fit, index=index)

    if hyper_tuner is None:
        return post

    if not core.hyperparameters(post):
        rows = tuple(
            (k, "None (nothing to tune)" if k == "hyper_tuner" else v) for k, v in provenance
        )
        return dataclasses.replace(post, provenance=rows)

    # One eigendecomposition per dense curvature, so the tune does no Cholesky per step.
    post = copy.replace(post, curvature=core.with_eigen(post.curvature))

    return hyper_tuner(post)


def _in_dtype[T](obj: T, leaves: Mapping[str, torch.Tensor], dtype: torch.dtype) -> T:
    """`obj` with each floating leaf in `leaves` of lower precision cast up to `dtype`."""

    def narrower(t: torch.Tensor) -> bool:
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            return False

        return torch.finfo(t.dtype).bits < torch.finfo(dtype).bits

    old = tuple(t for t in leaves.values() if narrower(t))

    if not old:
        return obj

    return core.replace_leaves(obj, old, tuple(t.to(dtype) for t in old))


def _as_dtype[T](likelihood: T, dtype: torch.dtype) -> T:
    """`likelihood` with every floating tensor (a sigma, say) in the fit's `dtype`.

    A float sigma is stored in double precision, so it is exact in a float64 fit and
    rounded once in a float32 one.
    """

    old = tuple(
        t
        for t in tree_leaves(likelihood)
        if isinstance(t, torch.Tensor) and t.is_floating_point() and t.dtype != dtype
    )

    if not old:
        return likelihood

    return core.replace_leaves(likelihood, old, tuple(t.to(dtype) for t in old))


_SHOWN_PARAMS = 6
"""Up to this many fitted tensors, `print` lists them all; above, the first three."""


def _shapes(params: Params) -> str:
    """The fitted tensors' names and shapes, or a summary when there are many.

    `post.mean` holds them all.
    """

    items = [f'"{n}": {tuple(p.shape)}' for n, p in params.items()]

    if len(items) <= _SHOWN_PARAMS:
        return "{" + ", ".join(items) + "}"

    size = sum(p.numel() for p in params.values())

    return "{" + ", ".join(items[:3]) + f", ...}} ({len(items)} tensors, {size} parameters)"


def _check_prior_fits(
    prior: Prior, approx: HessianApprox[Any], params: Params, ctx: FitContext
) -> None:
    """Add `prior` to `approx`'s template, so a misfit prior raises before the data pass.

    Without a template, it raises when the posterior is assembled.
    """

    if not isinstance(approx, HasTemplate):
        return

    try:
        prior.add_precision_to(approx.template(params, ctx))
    except core.StructureError as e:
        raise core.StructureError(f"before the data pass: {e}") from e


def predict[P: Posterior](
    posterior: P,
    inputs: Inputs,
    method: Predictive[P] | None = None,
    *,
    select: Callable[[Outputs, Inputs], Outputs] | None = None,
    generator: torch.Generator | None = None,
) -> Distribution:
    """Predictive distribution over targets, observation noise included.

    ```python
    probs = lk.predict(post, x).probs
    probs = lk.predict(post, x, method=lk.Linearized(link=lk.Probit())).probs
    ```

    `lk.predict_outputs` gives the Gaussian over the model outputs alone.

    Args:
        posterior: The posterior, weight-space or function-space.
        inputs: The inputs. A mapping that is not a pytree node (a tokenizer's
            `BatchEncoding`) becomes a `dict`, so `select` and the method see the same
            inputs.
        method: The predictive method. `None` means `lk.Linearized()`, resolved against the
            posterior (and the event size of the selected outputs) when it runs. A method
            accepts some posteriors only: `WeightSamples` needs a weight-space one.
        select: `select(f, inputs)` picks the outputs to predict, e.g.
            `lambda f, x: f[:, -1]` for the last position of a sequence model. The
            posterior's model is replaced by the selected one before the method runs, so
            `Linearized` linearizes the selected outputs only. `None` keeps all outputs.
        generator: The generator of a sampled method (`WeightSamples`, sampled outputs or
            an `MC` link). `None` means torch's global RNG, so `torch.manual_seed` makes
            the prediction repeatable.

    Returns:
        The distribution over targets, from the method.

    Raises:
        TypeError: `method` is not a predictive method (a link such as `lk.Probit()`,
            which goes inside `lk.Linearized(link=...)`), or it takes another kind of
            posterior.
        CapabilityError: The method, or its link, doesn't support the likelihood (one with
            no distribution, say) or the model lacks what the method needs.
        ValueError: The method refuses its settings or its numbers (`Probit` with negative
            logit variances, say).
        RuntimeError: A parameter changed in place since the fit (`check_versions`).
        StructureError: The method refuses the outputs, or the posterior's covariance
            products fail.
        NotImplementedError: The model does not return a single tensor.
    """

    if isinstance(method, LinkApprox):
        name = type(method).__name__
        raise TypeError(
            f"method takes a predictive method; {name} is a link, so wrap it: "
            f"method=lk.Linearized(link=lk.{name}())"
        )

    if isinstance(method, type):
        raise TypeError(_instance_hint("method", method, "lk.Linearized()"))

    if method is not None and not isinstance(method, Predictive):
        raise TypeError(
            "method takes a predictive method, such as lk.Linearized() or "
            f"lk.WeightSamples(100), got {type(method).__name__}"
        )

    # A mapping that is not a pytree node (a tokenizer's `BatchEncoding`) becomes a dict, so
    # `select` and the predictive see the same inputs, and `vmap` sees their tensors.
    inputs = as_pytree(inputs)

    if select is not None:
        model = posterior.model
        pick = select

        def selected(p: Params, x: Inputs) -> Outputs:
            return pick(model(p, x), x)

        posterior = copy.replace(cast(Any, posterior), model=ModelFn(selected, model.module))

    chosen: Predictive[P] = Linearized() if method is None else method

    return chosen(posterior, inputs, generator=generator)
