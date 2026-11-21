"""Orchestration: the steps a user calls from their own loop.

Single-pass curvature:

    stat = fit.init()
    for batch in loader:
        stat = fit.update(stat, batch)
    op = fit.finalize(all_reduce(stat))

Any curvature, including multi-pass ones (the same loop, once per pass):

    step = fit.start()
    while isinstance(step, Running):
        stat = step.init()
        for batch in loader:
            stat = step.update(stat, batch)
        step = step.finish(all_reduce(stat))
    op = step.result

`run(fit, data)` is that loop under `no_grad`, as the front doors use it.

`fit_curvature` draws one seed, from its `generator` or from torch's global RNG, so
`torch.manual_seed` makes a fit repeatable. Each batch's `update` gets a fresh generator
derived from that seed, the pass number and the examples seen before the batch. A batch
draws the same numbers however the statistics are merged later, and updates stay pure.
"""

import copy
from collections.abc import Callable, Iterable
from typing import Any, Self, cast, overload

import torch
import torch.distributed as dist
from torch import Tensor

from laplacekit.core._functorch import is_transformed
from laplacekit.core._functorch import transformed as _transformed
from laplacekit.core._pytree import (
    pytree,
    static,
    tree_leaves,
    tree_leaves_with_path,
    tree_map,
    tree_map_with_path,
)
from laplacekit.core._seed import _batch_seeds, _Seed
from laplacekit.core.curvature import (
    Curvature,
    Done,
    FitContext,
    HessianApprox,
    Pass,
    PerBlockCurvature,
    Staged,
    StagedPerBlock,
    Stat,
    stat_dtype,
)
from laplacekit.core.layout import Layout
from laplacekit.core.likelihoods import (
    HasTargetCheck,
    LateLikelihood,
    Likelihood,
    _mask,
    check_likelihood,
    log_prob,
    unit,
)
from laplacekit.core.model import ModelFn
from laplacekit.core.ops import Operator
from laplacekit.core.posteriors import LaplacePosterior, Prior, param_versions
from laplacekit.core.types import (
    Batch,
    Capabilities,
    CapabilityError,
    Params,
    StructureError,
)

_FIT_SEED = 0x5EED
"""Folded into the fit's seed for `FitContext.seed`, apart from the per-batch draws."""


def _device(params: Params) -> torch.device:
    p = next(iter(params.values()), None)

    return p.device if p is not None else torch.device("cpu")


def _update_generator(seed: _Seed, pass_index: int, count: int, params: Params) -> torch.Generator:
    """The generator of one batch's `update`, on the params' device."""

    _, update = _batch_seeds(seed.fold_in(pass_index), count)

    return update.generator(_device(params))


@pytree
class FitStat[S: Stat]:
    """The curvature statistic plus what objectives need from the same pass.

    It merges under `+` like its parts: the data-fit terms add, and a `None` term is the
    identity.

    Args:
        curvature: The curvature's statistic.
        data_fit: The data-fit term at the mean, `None` when no targets were seen: the
            summed `fit_stat` for a `LateLikelihood`, the summed log-likelihood otherwise.
            `log_lik(lik, data_fit)` gives the log-likelihood either way.
        count: Examples seen, a 0-dim int64 tensor.
    """

    curvature: S
    data_fit: Tensor | None
    count: Tensor

    def __add__(self, other: Self, /) -> Self:
        """The statistic of both data sets together.

        Args:
            other: The statistic of other data, from the same fit.

        Returns:
            A `FitStat` whose curvature statistic, data-fit term and count cover both data
            sets.
        """

        if self.data_fit is None:
            fit = other.data_fit
        elif other.data_fit is None:
            fit = self.data_fit
        else:
            fit = self.data_fit + other.data_fit

        return type(self)(self.curvature + other.curvature, fit, self.count + other.count)


def _init_stat[S: Stat](
    curvature: Curvature[S, Any], params: Params, ctx: FitContext
) -> FitStat[S]:
    return FitStat(curvature.init(params, ctx), None, torch.zeros((), dtype=torch.int64))


def _update_stat[S: Stat](
    curvature: Curvature[S, Any],
    stat: FitStat[S],
    params: Params,
    batch: Batch,
    seed: _Seed,
    pass_index: int,
    ctx: FitContext,
    lik: Likelihood,
    checked: bool,
) -> FitStat[S]:
    """Fold one batch into the curvature statistic and the data-fit term.

    `ctx.likelihood` is the unit likelihood the curvature sees; `lik` the real one, for
    the data-fit term. A `HasTargetCheck` likelihood checks the targets first, eagerly,
    unless a target leaf is transformed (a `vmap` over batches) and can't be read. Under a
    `grad` over the parameters the targets are plain, so the check runs.
    """

    if checked:
        curvature.requires.check_batch(batch, repr(curvature))

    if (
        batch.targets is not None
        and isinstance(lik, HasTargetCheck)
        and not any(isinstance(t, Tensor) and is_transformed(t) for t in tree_leaves(batch.targets))
    ):
        lik.check_targets(batch.targets)

    gen = _update_generator(seed, pass_index, int(stat.count.item()), params)

    if batch.targets is None:
        new = curvature.update(stat.curvature, params, batch, gen, ctx)

        return FitStat(new, stat.data_fit, stat.count + int(batch.size))

    # Under a torch.func transform nothing is recorded: the core runs its own forward.
    recorder = None if _transformed() else _Recorder(ctx.model, params, batch.inputs)

    if recorder is None:
        new = curvature.update(stat.curvature, params, batch, gen, ctx)
    else:
        recording = copy.replace(ctx, model=ModelFn(recorder, ctx.model.module))
        new = curvature.update(stat.curvature, params, batch, gen, recording)

    with torch.no_grad():
        recorded = None if recorder is None else recorder.output
        out = recorded if recorded is not None else ctx.model(params, batch.inputs)
        batch_fit = _batch_data_fit(lik, tree_map(_promote, out), batch)

    fit = batch_fit if stat.data_fit is None else stat.data_fit + batch_fit

    return FitStat(new, fit, stat.count + int(batch.size))


type _Fingerprint = tuple[int, tuple[int, ...], tuple[int, ...], torch.dtype]


def _fingerprint(t: Tensor) -> _Fingerprint:
    return t.data_ptr(), tuple(t.shape), t.stride(), t.dtype


class _Recorder:
    """The model. It keeps the output of the first call on the whole batch at the fit params.

    That call gets the batch's own `inputs` object and params with the same storage, shape,
    stride, and dtype as the fit params. Calls under a `torch.func` transform never count.
    The output is kept detached, so the data-fit term stays a value, as with the core's own
    forward.
    """

    def __init__(self, model: ModelFn, params: Params, inputs: Any) -> None:
        self.model = model
        self.inputs = inputs
        self.params = {k: _fingerprint(v) for k, v in params.items()}
        self.output: Any = None

    def __call__(self, params: Params, inputs: Any) -> Any:
        out = self.model(params, inputs)

        if self.output is None and self._at_fit(params, inputs):
            self.output = tree_map(_detach, out)

        return out

    def _at_fit(self, params: Params, inputs: Any) -> bool:
        if inputs is not self.inputs or params.keys() != self.params.keys() or _transformed():
            return False

        return all(_fingerprint(v) == self.params[k] for k, v in params.items())


def _detach(t: Any) -> Any:
    return t.detach() if isinstance(t, Tensor) else t


def _promote(t: Any) -> Any:
    """A floating tensor in its statistic dtype (at least float32)."""

    if isinstance(t, Tensor) and t.is_floating_point():
        return t.to(stat_dtype(t.dtype))

    return t


def _batch_data_fit(lik: Likelihood, f: Any, batch: Batch) -> Tensor:
    """The batch's data-fit term, weighted per example when `batch.weights` is set.

    `fit_stat` for a `LateLikelihood`, the summed log p(y|f) otherwise. `fit_stat` takes no
    inputs, so a late likelihood with an active mask raises `StructureError`.
    """

    assert batch.targets is not None

    if isinstance(lik, LateLikelihood):
        if _mask(lik, f, batch.inputs) is not None:
            raise StructureError(
                f"{type(lik).__name__} is a LateLikelihood with a mask, but fit_stat takes no "
                "inputs and can't apply it; drop one of the two"
            )

        weights = None if batch.weights is None else batch.weights.to(cast(Tensor, f).dtype)

        return lik.fit_stat(f, batch.targets, weights)

    per = log_prob(lik, f, batch.targets, inputs=batch.inputs)

    if batch.weights is None:
        return per.sum()

    return (per * batch.weights.to(per.dtype)).sum()


def _check_likelihood(method: Curvature[Any, Any] | Staged[Any], lik: Likelihood) -> None:
    """Check each per-block member on its own (for the message), then the whole method."""

    if isinstance(method, PerBlockCurvature | StagedPerBlock):
        for member in method.rules.values():
            _check_likelihood(member, lik)

    check_likelihood(method.requires, lik, repr(method))


def _check(ctx: FitContext, curvature: Curvature[Any, Any] | Staged[Any], checked: bool) -> None:
    if checked:
        ctx.caps.check(curvature.requires, repr(curvature))


@pytree
class Running[R]:
    """One pass of a fit in progress.

    Pure: `finish` returns the next step. `start()` on a fit builds the first one.

    Args:
        params: The params the curvature is taken at.
        ctx: The fit's context. Its likelihood is the unit one.
        likelihood: The real likelihood, for the data-fit term.
        pass_: The current pass.
        index: The pass number, which seeds each batch's generator with the fit's seed.
        checked: Whether to check each pass's requirements against the capabilities and
            each batch.
        done: Statistics of the finished passes.
    """

    params: Params
    ctx: FitContext = static()
    likelihood: Likelihood = static()
    pass_: Pass[R] = static()
    index: int = static()
    checked: bool = static(default=False)
    done: tuple[FitStat[Any], ...] = ()
    _seed: _Seed = static(default_factory=_Seed.draw, kw_only=True)

    def init(self) -> FitStat[Any]:
        """The empty statistic of this pass.

        Returns:
            The identity of `+`: the curvature's `init`, no data-fit term and a zero
            count.

        Raises:
            CapabilityError: The likelihood lacks something the pass needs (checked
                always). With `checked`: the capabilities lack something the pass needs.
            StructureError: The pass's curvature refuses the params or layout.
            ValueError: The pass's curvature refuses its settings for these params (e.g.
                `LowRankGGN` with `rank + oversample` above the parameter count).
        """

        _check(self.ctx, self.pass_.curvature, self.checked)
        _check_likelihood(self.pass_.curvature, self.ctx.likelihood)

        return _init_stat(self.pass_.curvature, self.params, self.ctx)

    def update(self, stat: FitStat[Any], batch: Batch, /) -> FitStat[Any]:
        """Fold one batch into the statistic.

        A `HasTargetCheck` likelihood checks the batch's targets first.

        Args:
            stat: The statistic so far.
            batch: The next batch. Its generator comes from the fit's seed, `index` and
                the examples seen in this pass before it.

        Returns:
            `stat` with the batch added.

        Raises:
            CapabilityError: With `checked`, the batch fails the pass's requirements
                (`Requirements.check_batch`). With or without it, the curvature needs
                targets the batch lacks.
            ValueError: The likelihood's `check_targets` finds a bad target value.
            StructureError: The likelihood's `check_targets` finds badly shaped targets,
                or a `LateLikelihood` has an active mask, which `fit_stat` can't apply.
        """

        return _update_stat(
            self.pass_.curvature,
            stat,
            self.params,
            batch,
            self._seed,
            self.index,
            self.ctx,
            self.likelihood,
            self.checked,
        )

    def finish(self, stat: FitStat[Any], /) -> "Running[R] | Finished[R]":
        """End the pass and move to the next step.

        Args:
            stat: The pass's statistic over all the data, after `all_reduce`.

        Returns:
            The next pass as a new `Running`, or `Finished` after the last pass. Either
            keeps `stat` with the statistics of the earlier passes.

        Raises:
            ValueError: The curvature's `finalize` refuses `stat`, e.g. a zero total
                weight.
        """

        out = self.pass_.then(self.pass_.curvature.finalize(stat.curvature))
        stats = (*self.done, stat)

        if isinstance(out, Done):
            return Finished(out.result, stats)

        return Running(
            self.params,
            self.ctx,
            self.likelihood,
            out,
            self.index + 1,
            self.checked,
            stats,
            _seed=self._seed,
        )


@pytree
class Finished[R]:
    """A completed fit.

    Args:
        result: The result of the fit's last step.
        stats: The statistics, one per pass. The last one feeds the marginal likelihood.
    """

    result: R
    stats: tuple[FitStat[Any], ...]

    @property
    def count(self) -> Tensor:
        """Examples seen in the last pass: the default `n_data`."""

        return self.stats[-1].count


@pytree
class CurvatureFit[S: Stat, Op: Operator[Any]]:
    """A model, likelihood, params and single-pass method bound for fitting.

    The fit steps are the plain fold, plus `start()` for uniform code. Build it with
    `fit_curvature`, which checks the likelihood.

    Args:
        model: The model as a function of the params.
        likelihood: The likelihood. The curvature sees its unit version (see
            `LateLikelihood`).
        params: The params the curvature is taken at.
        curvature: The single-pass curvature.
        caps: The probed capabilities. `None` skips the model checks.
    """

    model: ModelFn = static()
    likelihood: Likelihood = static()
    params: Params
    curvature: Curvature[S, Op] = static()
    caps: Capabilities | None = static(default=None)
    _seed: _Seed = static(default_factory=_Seed.draw, kw_only=True)

    @property
    def ctx(self) -> FitContext:
        """The context the curvature sees, with the unit likelihood.

        With no caps given, everything is False (not probed) and requirements are not
        checked.
        """

        return _fit_context(self)

    def posterior[C: Operator[Any], Pr: Prior](
        self, curvature: C, /, prior: Pr, n_data: Tensor
    ) -> LaplacePosterior[C, Pr]:
        """Build the posterior without repeating the model, likelihood, and layout.

        Args:
            curvature: The finalized curvature, the per-example average.
            prior: The prior. Its location must match the params.
            n_data: The effective data size $N$ the curvature is multiplied by.

        Returns:
            The Laplace posterior at the fit's params.

        Raises:
            StructureError: The prior's location doesn't match the params, or the
                curvature can't absorb the prior precision (e.g. KFAC on KFAC).
        """

        return _posterior(self, curvature, prior, n_data)

    def init(self) -> FitStat[S]:
        """The empty statistic.

        Returns:
            The identity of `+`: the curvature's `init`, no data-fit term and a zero
            count.

        Raises:
            CapabilityError: With `caps`, the capabilities lack something the curvature
                needs.
            StructureError: The curvature refuses the params or layout.
            ValueError: The curvature refuses its settings for these params (e.g.
                `LowRankGGN` with `rank + oversample` above the parameter count).
        """

        _check(self.ctx, self.curvature, self.caps is not None)
        return _init_stat(self.curvature, self.params, self.ctx)

    def update(self, stat: FitStat[S], batch: Batch, /) -> FitStat[S]:
        """Fold one batch into the statistic.

        A `HasTargetCheck` likelihood checks the batch's targets first.

        Args:
            stat: The statistic so far.
            batch: The next batch. Its generator comes from the fit's seed and the
                examples seen before it.

        Returns:
            `stat` with the batch added.

        Raises:
            CapabilityError: With `caps`, the batch fails the curvature's requirements
                (`Requirements.check_batch`). With or without `caps`, the curvature needs
                targets the batch lacks.
            ValueError: The likelihood's `check_targets` finds a bad target value.
            StructureError: The likelihood's `check_targets` finds badly shaped targets,
                or a `LateLikelihood` has an active mask, which `fit_stat` can't apply.
        """

        checked = self.caps is not None

        return _update_stat(
            self.curvature,
            stat,
            self.params,
            batch,
            self._seed,
            0,
            self.ctx,
            self.likelihood,
            checked,
        )

    def finalize(self, stat: FitStat[S], /) -> Op:
        r"""The curvature from the statistic over all the data.

        Args:
            stat: The statistic over all the data, after `all_reduce`.

        Returns:
            The curvature's `finalize`: the per-example average $\bar{C}$. The posterior
            multiplies it by `n_data`.

        Raises:
            ValueError: The curvature's `finalize` refuses `stat`, e.g. a zero total
                weight.
        """

        return self.curvature.finalize(stat.curvature)

    def start(self) -> Running[Op]:
        """The first (and only) pass, for code that also runs multi-pass methods.

        Returns:
            A `Running` step whose `finish` gives the curvature as `Finished.result`.

        Raises:
            CapabilityError: With `caps`, the capabilities lack something the curvature
                needs.
        """

        ctx = self.ctx
        checked = self.caps is not None
        _check(ctx, self.curvature, checked)
        pass_ = Pass(self.curvature, lambda r: Done(r))

        return Running(self.params, ctx, self.likelihood, pass_, 0, checked, _seed=self._seed)


@pytree
class StagedCurvatureFit[R]:
    """A model, likelihood, params and multi-pass method bound for fitting.

    Only `start()`: the number of passes is the method's business. Build it with
    `fit_curvature`, which checks the likelihood.

    Args:
        model: The model as a function of the params.
        likelihood: The likelihood. The method sees its unit version (see
            `LateLikelihood`).
        params: The params the curvature is taken at.
        curvature: The multi-pass method.
        caps: The probed capabilities. `None` skips the model checks.
    """

    model: ModelFn = static()
    likelihood: Likelihood = static()
    params: Params
    curvature: Staged[R] = static()
    caps: Capabilities | None = static(default=None)
    _seed: _Seed = static(default_factory=_Seed.draw, kw_only=True)

    @property
    def ctx(self) -> FitContext:
        """The context the method sees, with the unit likelihood.

        With no caps given, everything is False and requirements are not checked.
        """

        return _fit_context(self)

    def posterior[Op: Operator[Any], Pr: Prior](
        self, curvature: Op, /, prior: Pr, n_data: Tensor
    ) -> LaplacePosterior[Op, Pr]:
        """Build the posterior without repeating the model, likelihood, and layout.

        Args:
            curvature: The finalized curvature, the per-example average.
            prior: The prior. Its location must match the params.
            n_data: The effective data size $N$ the curvature is multiplied by.

        Returns:
            The Laplace posterior at the fit's params.

        Raises:
            StructureError: The prior's location doesn't match the params, or the
                curvature can't absorb the prior precision (e.g. KFAC on KFAC).
        """

        return _posterior(self, curvature, prior, n_data)

    def start(self) -> Running[R]:
        """The first pass of the method's plan.

        Returns:
            A `Running` step. Its `finish` gives the next pass, or `Finished` after the
            last.

        Raises:
            CapabilityError: The method's `plan` refuses the fit, with or without
                `caps`. With `caps`, also when the capabilities lack something the
                method needs.
            StructureError: The method's `plan` refuses the params or layout.
        """

        ctx = self.ctx
        checked = self.caps is not None
        _check(ctx, self.curvature, checked)

        plan = self.curvature.plan(self.params, ctx)

        return Running(self.params, ctx, self.likelihood, plan, 0, checked, _seed=self._seed)


def _fit_context(fit: "CurvatureFit[Any, Any] | StagedCurvatureFit[Any]") -> FitContext:
    """The context the curvature sees, with the unit likelihood (see `LateLikelihood`)."""

    caps = fit.caps if fit.caps is not None else Capabilities.unprobed()

    seed = fit._seed.fold_in(_FIT_SEED).value % (1 << 63)

    return FitContext(fit.model, unit(fit.likelihood), Layout.by_module(fit.params), caps, seed)


def _posterior[Op: Operator[Any], Pr: Prior](
    fit: "CurvatureFit[Any, Any] | StagedCurvatureFit[Any]",
    curvature: Op,
    prior: Pr,
    n_data: Tensor,
) -> LaplacePosterior[Op, Pr]:
    ctx = _fit_context(fit)

    # A prior whose location does not match the parameters raises here.
    prior.location(fit.params)

    posterior = LaplacePosterior(
        fit.params,
        curvature,
        prior,
        n_data,
        ctx.model,
        fit.likelihood,
        ctx.layout,
        ctx.caps,
        param_versions(ctx.model, fit.params),
    )

    # A prior precision the curvature cannot absorb (e.g. KFAC on KFAC) raises here, not at
    # the first prediction. The result is discarded, so no graph is needed (D8 is kept: the
    # posterior rebuilds its precision on use).
    with torch.no_grad():
        posterior.precision()

    return posterior


@overload
def fit_curvature[S: Stat, Op: Operator[Any]](
    model: ModelFn,
    likelihood: Likelihood,
    params: Params,
    hessian_approx: Curvature[S, Op],
    /,
    *,
    generator: torch.Generator | None = None,
    capabilities: Capabilities | None = None,
) -> CurvatureFit[S, Op]: ...


@overload
def fit_curvature[R](
    model: ModelFn,
    likelihood: Likelihood,
    params: Params,
    hessian_approx: Staged[R],
    /,
    *,
    generator: torch.Generator | None = None,
    capabilities: Capabilities | None = None,
) -> StagedCurvatureFit[R]: ...


@overload
def fit_curvature(
    model: ModelFn,
    likelihood: Likelihood,
    params: Params,
    hessian_approx: HessianApprox[Any],
    /,
    *,
    generator: torch.Generator | None = None,
    capabilities: Capabilities | None = None,
) -> CurvatureFit[Any, Any] | StagedCurvatureFit[Any]: ...


def fit_curvature(
    model: ModelFn,
    likelihood: Likelihood,
    params: Params,
    hessian_approx: Any,
    /,
    *,
    generator: torch.Generator | None = None,
    capabilities: Capabilities | None = None,
) -> Any:
    """Bind a model, likelihood, params and curvature method for fitting.

    The likelihood needs of the method (each per-block member) are always checked, and
    those of each pass when it starts.

    Args:
        model: The model as a function of the params.
        likelihood: The likelihood.
        params: The params the curvature is taken at.
        hessian_approx: A single-pass `Curvature` or a multi-pass `Staged` method.
        generator: Seeds the fit. One seed is drawn from it, or from torch's global RNG
            when `None`, so `torch.manual_seed` makes the fit repeatable. Each batch's
            `update` gets a fresh generator derived from that seed, the pass number and
            the examples seen before the batch.
        capabilities: The probed capabilities. `None` skips the model checks.

    Returns:
        A `CurvatureFit` for a `Curvature`, a `StagedCurvatureFit` for a `Staged` method.

    Raises:
        CapabilityError: The likelihood lacks something the method needs, or `capabilities` was
            probed for parameters that don't cover `params`.
    """

    _check_likelihood(hessian_approx, likelihood)
    caps = capabilities

    if caps is not None and caps.probed is not None:
        unprobed = [n for n in params if n not in caps.probed]

        if unprobed:
            raise CapabilityError(
                "capabilities were probed for other parameters; call probe(..., params=...) "
                f"with the fitted ones. Not probed: {', '.join(unprobed)}"
            )

    seed = _Seed.draw(generator)

    if isinstance(hessian_approx, Staged):
        staged = cast(Staged[Any], hessian_approx)

        return StagedCurvatureFit(model, likelihood, params, staged, caps, _seed=seed)

    return CurvatureFit(model, likelihood, params, hessian_approx, caps, _seed=seed)


@overload
def run[Op: Operator[Any]](
    fit: CurvatureFit[Any, Op],
    data: Callable[[torch.Generator], Iterable[Batch]],
    /,
    *,
    generator: torch.Generator | None = None,
) -> Finished[Op]: ...


@overload
def run[R](
    fit: StagedCurvatureFit[R],
    data: Callable[[torch.Generator], Iterable[Batch]],
    /,
    *,
    generator: torch.Generator | None = None,
) -> Finished[R]: ...


def run(
    fit: CurvatureFit[Any, Any] | StagedCurvatureFit[Any],
    data: Callable[[torch.Generator], Iterable[Batch]],
    /,
    *,
    generator: torch.Generator | None = None,
) -> Finished[Any]:
    """Run every pass of `fit` over `data`, under `no_grad`.

    Its statistic goes through `all_reduce` before `finish`. This is the loop the front
    doors run. Under `no_grad` the result carries no graph; for gradients through the fit,
    write the loop by hand.

    Args:
        fit: The bound fit, from `fit_curvature`.
        data: Batches per pass: pass $i$ reads the batches of `data(g)`, with `g` a fresh
            CPU generator for that pass.
        generator: Seeds the data passes. One seed is drawn from it, or from torch's
            global RNG when `None`, and each pass's generator is derived from that seed
            and the pass number. The fit's own draws come from the seed `fit_curvature`
            drew.

    Returns:
        The finished fit.

    Raises:
        CapabilityError: A pass or batch fails the method's requirements, including a
            curvature that needs targets the batch lacks.
        StructureError: The method refuses the params or layout, the likelihood's
            `check_targets` finds badly shaped targets, or a `LateLikelihood` has an
            active mask, which `fit_stat` can't apply.
        ValueError: The likelihood's `check_targets` finds a bad target value, or a
            curvature refuses its settings or its statistic (e.g. a zero total weight).
    """

    seed = _Seed.draw(generator)

    with torch.no_grad():
        step: Running[Any] | Finished[Any] = fit.start()

        while isinstance(step, Running):
            stat = step.init()

            for batch in data(seed.fold_in(step.index).generator()):
                stat = step.update(stat, batch)

            step = step.finish(all_reduce(stat))

    return step


def all_reduce[T](stat: T, /, group: Any = None) -> T:
    """Combine a statistic across the ranks of `group`.

    Every rank gathers all statistics and folds them with `+` in rank order, so all ranks
    hold the same result: the statistic of one process over all their data, for any
    monoid. The rank's own statistic enters as it is and keeps its autograd graph; the
    others arrive as detached copies on the same devices as the matching leaves here. A
    leaf this rank lacks (for example the data-fit term of a rank that saw no batches) goes
    to the device of the first local leaf off the CPU, or stays on the CPU.

    Batch generators come from each rank's own example count and its rank, so a sampled
    method on several ranks draws differently from one process over the same data.

    Args:
        stat: This rank's statistic.
        group: The process group. `None` means the default group.

    Returns:
        The statistic over all ranks. Without a process group, `stat` unchanged.
    """

    if not (dist.is_available() and dist.is_initialized()):
        return stat

    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, tree_map(_to_cpu, stat), group=group)

    devices = {
        path: leaf.device for path, leaf in tree_leaves_with_path(stat) if isinstance(leaf, Tensor)
    }
    home = next((d for d in devices.values() if d.type != "cpu"), torch.device("cpu"))

    def place(path: str, leaf: Any) -> Any:
        return leaf.to(devices.get(path, home)) if isinstance(leaf, Tensor) else leaf

    me = dist.get_rank(group)
    parts = [stat if r == me else tree_map_with_path(place, s) for r, s in enumerate(gathered)]
    total: Any = parts[0]

    for part in parts[1:]:
        total = total + part

    return total


def _to_cpu(t: Any) -> Any:
    return t.detach().cpu() if isinstance(t, Tensor) else t
