"""The conformance kit: built-ins pass, broken doubles fail naming the broken law."""

import copy
import dataclasses
import math
from typing import Any, Self, cast

import pytest
import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Normal

import laplacekit as lk
import laplacekit.core as core
from laplacekit.core import Batch, FitContext, Params, Requirements, pytree
from laplacekit.priors import Diagonal, Isotropic, Layerwise
from laplacekit.testing import (
    check_curvature,
    check_likelihood,
    check_link,
    check_objective,
    check_predictive,
    check_prior,
    check_subset,
)

F64 = torch.float64
N_BATCHES = 4
BATCH = 6


def _model(width: int = 5, out: int = 3) -> nn.Module:
    torch.manual_seed(0)

    return nn.Sequential(nn.Linear(3, width), nn.Tanh(), nn.Linear(width, out)).to(F64)


def _params(model: nn.Module) -> Params:
    return {k: v.detach() for k, v in model.named_parameters()}


def _batches(kind: str = "cat", out: int = 3, weights: bool = False) -> list[Batch]:
    g = torch.Generator().manual_seed(1)
    bs = []

    for _ in range(N_BATCHES):
        x = torch.randn(BATCH, 3, generator=g, dtype=F64)
        y = (
            torch.randint(0, out, (BATCH,), generator=g)
            if kind == "cat"
            else torch.randn(BATCH, out, generator=g, dtype=F64)
        )
        w = torch.rand(BATCH, generator=g, dtype=F64) + 0.5 if weights else None
        bs.append(Batch(x, y, w))

    return bs


def _lik(kind: str) -> Any:
    return (
        lk.Categorical()
        if kind == "cat"
        else copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.7, dtype=F64)))
    )


# --- built-in curvatures pass ---------------------------------------------------------


@pytest.mark.parametrize("kind", ["cat", "gauss"])
@pytest.mark.parametrize(
    "curv",
    [
        lk.DiagGGN(),
        lk.DenseGGN(),
        lk.DiagGGN(fisher="mc", mc_samples=2),
        lk.KFAC(),
        lk.KFAC(fisher="mc"),
        lk.per_block({"0": lk.DenseGGN(), "2": lk.DiagGGN()}),
        lk.per_block({"0": lk.KFAC(), "2": lk.DenseGGN()}),
    ],
    ids=repr,
)
def test_builtin_curvatures_pass(curv: Any, kind: str) -> None:
    model = _model()

    check_curvature(curv, model, _lik(kind), _params(model), _batches(kind))


@pytest.mark.parametrize("curv", [lk.DiagGGN(), lk.DenseGGN(), lk.DiagGGN(fisher="mc")], ids=repr)
def test_weighted_batches_pass(curv: Any) -> None:
    model = _model()

    check_curvature(curv, model, lk.Categorical(), _params(model), _batches(weights=True))


def test_model_fn_and_subset_pass() -> None:
    model = _model()
    params = {k: v for k, v in _params(model).items() if k.startswith("2.")}

    check_curvature(
        lk.DenseGGN(), lk.core.ModelFn.from_module(model), lk.Categorical(), params, _batches()
    )


def test_above_512_params_skips_dense_checks() -> None:
    model = _model(width=200)  # 3*200 + 200 + 200*3 + 3 parameters

    check_curvature(_WrongDiagonal(), model, lk.Categorical(), _params(model), _batches())


def test_needs_float64() -> None:
    model = _model().float()
    params = _params(model)

    with pytest.raises(ValueError, match="float64"):
        check_curvature(lk.DiagGGN(), model, lk.Categorical(), params, _batches())


def test_single_operator_needs_params_grouped_by_module() -> None:
    model = _model()
    p = _params(model)
    interleaved = {k: p[k] for k in ["0.weight", "2.weight", "0.bias", "2.bias"]}

    with pytest.raises(ValueError, match="grouped by module"):
        check_curvature(_Toy(), model, lk.Categorical(), interleaved, _batches())


def test_needs_two_batches() -> None:
    model = _model()

    with pytest.raises(ValueError, match="two batches"):
        check_curvature(lk.DiagGGN(), model, lk.Categorical(), _params(model), _batches()[:1])


class _LogProbOnly:
    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return -((f - y) ** 2).sum(-1)


def test_likelihood_needs_are_checked_like_a_fit() -> None:
    model = _model()

    with pytest.raises(lk.CapabilityError, match=r"DiagGGN\(fisher='mc'\).*HasSample"):
        check_curvature(
            lk.DiagGGN(fisher="mc"), model, _LogProbOnly(), _params(model), _batches("gauss")
        )

    observed = lk.DiagGGN(observed=True)
    check_curvature(observed, model, _LogProbOnly(), _params(model), _batches("gauss"))


# --- broken curvatures fail -------------------------------------------------------------


@pytree
class _ToyStat:
    s: Tensor
    w: Tensor

    @property
    def total_weight(self) -> Tensor:
        return self.w

    def __add__(self, other: Self, /) -> Self:
        return type(self)(self.s + other.s, self.w + other.w)

    def scaled(self, s: Tensor, /) -> Self:
        return type(self)(s * self.s, s * self.w)


@pytree
class _DropsWeight(_ToyStat):
    def __add__(self, other: Self, /) -> Self:
        return type(self)(self.s + other.s, self.w)  # drops other.w


@pytree
class _ScalesHalf(_ToyStat):
    def scaled(self, s: Tensor, /) -> Self:
        return type(self)(s * self.s, self.w)  # forgets to scale the weight


@pytree
class _ScaledDiag:
    """A diagonal operator whose `diagonal()` is wrong by a factor 2."""

    diag: Tensor

    def mv(self, v: Tensor, /) -> Tensor:
        return self.diag * v

    def diagonal(self) -> Tensor:
        return 2 * self.diag

    def inverse_diagonal(self) -> Tensor:
        return 1 / self.diag

    def logdet(self) -> Tensor:
        return self.diag.log().sum()

    def solve(self, v: Tensor, /) -> Tensor:
        return v / self.diag

    def cov_sqrt_mv(self, v: Tensor, /) -> Tensor:
        return v / self.diag.sqrt()

    def scaled(self, s: Tensor, /) -> Self:
        return type(self)(s * self.diag)

    def add_isotropic(self, prec: Tensor, /) -> Self:
        return type(self)(self.diag + prec)

    def add_diagonal(self, d: Tensor, /) -> Self:
        return type(self)(self.diag + d)

    def add_same(self, other: Any, /) -> Self:
        return type(self)(self.diag + other.diag)


@pytree
class _BadInverseDiag(_ScaledDiag):
    """Right diagonal, wrong inverse diagonal."""

    def diagonal(self) -> Tensor:
        return self.diag

    def inverse_diagonal(self) -> Tensor:
        return 2 / self.diag


@pytree
class _Toy:
    """Per-example squared outputs times a fixed positive vector, flat over all params."""

    @property
    def requires(self) -> Requirements:
        return Requirements()

    def _stat(self, s: Tensor, w: Tensor) -> _ToyStat:
        return _ToyStat(s, w)

    def init(self, params: Params, ctx: FitContext, /) -> _ToyStat:
        n = sum(p.numel() for p in params.values())

        return self._stat(torch.zeros(n, dtype=F64), torch.zeros((), dtype=F64))

    def update(
        self,
        stat: _ToyStat,
        params: Params,
        batch: Batch,
        generator: torch.Generator,
        ctx: FitContext,
        /,
    ) -> _ToyStat:
        f = ctx.model(params, batch.inputs)
        assert isinstance(f, Tensor)
        w = torch.ones(batch.size, dtype=F64) if batch.weights is None else batch.weights
        per = (f**2).sum(-1) + 1
        n = stat.s.numel()
        v = torch.arange(1, n + 1, dtype=F64) / n

        return self._stat(stat.s + (w * per).sum() * v, stat.w + w.sum())

    def finalize(self, stat: _ToyStat, /) -> Any:
        return core.Diag(stat.s / stat.w)


@pytree
class _DropsField(_Toy):
    def _stat(self, s: Tensor, w: Tensor) -> _ToyStat:
        return _DropsWeight(s, w)


@pytree
class _BadScaled(_Toy):
    def _stat(self, s: Tensor, w: Tensor) -> _ToyStat:
        return _ScalesHalf(s, w)


@pytree
class _ReturnsSums(_Toy):
    def finalize(self, stat: _ToyStat, /) -> Any:
        return core.Diag(stat.s)


@pytree
class _WrongDiagonal(_Toy):
    def finalize(self, stat: _ToyStat, /) -> Any:
        return _ScaledDiag(stat.s / stat.w)


@pytree
class _WrongInverseDiagonal(_Toy):
    def finalize(self, stat: _ToyStat, /) -> Any:
        return _BadInverseDiag(stat.s / stat.w)


@pytree
class _WrongSize(_Toy):
    def finalize(self, stat: _ToyStat, /) -> Any:
        return core.Diag((stat.s / stat.w)[1:])


@pytree
class _NotAnOperator(_Toy):
    def finalize(self, stat: _ToyStat, /) -> Any:
        return stat.s / stat.w


@pytree
class _NoUpdate:
    @property
    def requires(self) -> Requirements:
        return Requirements()

    def init(self, params: Params, ctx: FitContext, /) -> _ToyStat:
        raise NotImplementedError


def test_toy_curvature_passes() -> None:
    model = _model()

    check_curvature(_Toy(), model, lk.Categorical(), _params(model), _batches())


@pytest.mark.parametrize(
    ("curv", "law"),
    [
        (_DropsField(), r"monoid law: fold\(a \+\+ b\) != fold\(a\) \+ fold\(b\) for _DropsField"),
        (_ReturnsSums(), "weight scaling: .* for _ReturnsSums"),
        (_BadScaled(), r"scaled: scaled\(2\) != stat \+ stat .* for _BadScaled"),
        (_WrongDiagonal(), "diagonal: .* for _WrongDiagonal"),
        (_WrongInverseDiagonal(), "inverse_diagonal: .* for _WrongInverseDiagonal"),
        (_WrongSize(), "layout: .* for _WrongSize"),
        (_NotAnOperator(), "operator protocol: .* for _NotAnOperator"),
        (_NoUpdate(), "protocol: satisfies neither Curvature nor Staged for _NoUpdate"),
    ],
    ids=lambda x: type(x).__name__ if not isinstance(x, str) else "",
)
def test_broken_curvatures_fail(curv: Any, law: str) -> None:
    model = _model()

    with pytest.raises(AssertionError, match=law):
        check_curvature(curv, model, lk.Categorical(), _params(model), _batches())


def test_monoid_message_reports_error() -> None:
    model = _model()

    with pytest.raises(AssertionError, match=r"max rel err [0-9.e+-]+\)"):
        check_curvature(_DropsField(), model, lk.Categorical(), _params(model), _batches())


@pytree
class _Templated(_Toy):
    """`_Toy` with a template of the right structure."""

    def template(self, params: Params, ctx: FitContext, /) -> Any:
        n = sum(p.numel() for p in params.values())

        return core.Diag(torch.zeros(n, dtype=F64))


@pytree
class _ShortTemplate(_Toy):
    def template(self, params: Params, ctx: FitContext, /) -> Any:
        return core.Diag(torch.zeros(3, dtype=F64))


@pytree
class _Float32Template(_Toy):
    def template(self, params: Params, ctx: FitContext, /) -> Any:
        n = sum(p.numel() for p in params.values())

        return core.Diag(torch.zeros(n))


@pytree
class _DenseTemplate(_Toy):
    def template(self, params: Params, ctx: FitContext, /) -> Any:
        n = sum(p.numel() for p in params.values())

        return core.Dense(torch.zeros(n, n, dtype=F64))


@pytree
class _BlockTemplate(_Toy):
    def template(self, params: Params, ctx: FitContext, /) -> Any:
        return core.BlockDiag(
            {b: core.Diag(torch.zeros(ctx.layout.size(b), dtype=F64)) for b in ctx.layout.blocks}
        )


def test_toy_curvature_with_template_passes() -> None:
    model = _model()

    assert isinstance(_Templated(), core.HasTemplate)
    check_curvature(_Templated(), model, lk.Categorical(), _params(model), _batches())


@pytest.mark.parametrize(
    "curv", [_ShortTemplate(), _Float32Template(), _DenseTemplate(), _BlockTemplate()], ids=repr
)
def test_check_curvature_template_mismatch(curv: Any) -> None:
    model = _model()
    name = type(curv).__name__

    with pytest.raises(AssertionError, match=f"template: .* for {name}"):
        check_curvature(curv, model, lk.Categorical(), _params(model), _batches())


@pytest.mark.parametrize(
    "curv",
    [
        lk.per_block({"0": lk.DenseGGN(), "2": lk.DiagGGN()}),
        lk.per_block({"0": lk.KFAC(), "*": lk.DenseGGN()}),
        lk.DenseGGN(joint=True),
        lk.LowRankGGN(4),
    ],
    ids=repr,
)
def test_builtin_templates_pass(curv: Any) -> None:
    model = _model()

    assert isinstance(curv, core.HasTemplate)
    check_curvature(curv, model, lk.Categorical(), _params(model), _batches())


# --- a joint operator over all blocks --------------------------------------------------

_JOINT_SIZES = (("0", 20), ("2", 18))  # _model(): 3*5 + 5 and 5*3 + 3 parameters


@pytree
class _ToyJoint(_Toy):
    """`_Toy` with one dense matrix across the given blocks."""

    sizes: tuple[tuple[str, int], ...] = core.static(default=_JOINT_SIZES)

    def finalize(self, stat: _ToyStat, /) -> Any:
        m = stat.s / stat.w
        n = m.numel()

        return core.Joint(core.Dense(torch.diag(m) + torch.outer(m, m) / n), self.sizes)


def test_check_curvature_on_joint_operator() -> None:
    model = _model()

    check_curvature(_ToyJoint(), model, lk.Categorical(), _params(model), _batches())


@pytree
class _SwappedJointTemplate(_ToyJoint):
    def template(self, params: Params, ctx: FitContext, /) -> Any:
        return core.Joint(core.Dense(torch.zeros(38, 38, dtype=F64)), (("2", 18), ("0", 20)))


def test_check_curvature_template_with_other_blocks_fails() -> None:
    model = _model()

    with pytest.raises(AssertionError, match="template: its structure differs .*Joint"):
        check_curvature(
            _SwappedJointTemplate(), model, lk.Categorical(), _params(model), _batches()
        )


def test_joint_with_wrong_blocks_fails_layout() -> None:
    model = _model()
    curv = _ToyJoint((("2", 18), ("0", 20)))

    with pytest.raises(AssertionError, match=r"layout: Joint\[Dense\] keys \['2', '0'\]"):
        check_curvature(curv, model, lk.Categorical(), _params(model), _batches())


def test_per_block_refuses_joint_member() -> None:
    model = _model()
    curv = lk.per_block({"0": _ToyJoint((("0", 20),)), "2": lk.DiagGGN()})

    with pytest.raises(core.StructureError, match="'0' returned a joint operator over all"):
        check_curvature(curv, model, lk.Categorical(), _params(model), _batches())


# --- priors ------------------------------------------------------------------------------


def _prior_params() -> Params:
    return _params(_model())


@pytest.mark.parametrize(
    "prior",
    [
        Isotropic(torch.tensor(2.0, dtype=F64)),
        Diagonal(torch.linspace(0.5, 3.0, 38, dtype=F64)),
        Layerwise({"0": torch.tensor(0.5, dtype=F64), "2": torch.tensor(3.0, dtype=F64)}),
    ],
    ids=repr,
)
def test_builtin_priors_pass(prior: Any) -> None:
    check_prior(prior, _prior_params())


@pytree
class _ChangesType:
    """Turns each block into a Dense matrix."""

    prior_precision: Tensor

    def log_prob(self, params: Params, /) -> Tensor:
        return Isotropic(self.prior_precision).log_prob(params)

    def location(self, like: Params, /) -> Params:
        return {k: torch.zeros_like(v) for k, v in like.items()}

    def add_precision_to(self, op: Any, /) -> Any:
        if not isinstance(op, core.BlockDiag):
            raise core.StructureError("needs a BlockDiag")

        blocks = {
            k: core.Dense(torch.diag(b.diagonal() + self.prior_precision))
            for k, b in op.blocks.items()
        }

        return core.BlockDiag(blocks)

    def cov_mv(self, v: Params, /) -> Params:
        return {k: x / self.prior_precision for k, x in v.items()}


@pytree
class _WrongCov(_ChangesType):
    def add_precision_to(self, op: Any, /) -> Any:
        return op.add_isotropic(self.prior_precision)

    def cov_mv(self, v: Params, /) -> Params:
        return {k: x * self.prior_precision for k, x in v.items()}


@pytree
class _NaNLogProb(_WrongCov):
    def log_prob(self, params: Params, /) -> Tensor:
        return torch.tensor(math.nan, dtype=F64)

    def cov_mv(self, v: Params, /) -> Params:
        return {k: x / self.prior_precision for k, x in v.items()}


@pytree
class _FlattensJoint(_WrongCov):
    """Hands back the inner operator of a `Joint`."""

    def add_precision_to(self, op: Any, /) -> Any:
        if isinstance(op, core.Joint):
            return op.op.add_isotropic(self.prior_precision)

        return op.add_isotropic(self.prior_precision)

    def cov_mv(self, v: Params, /) -> Params:
        return {k: x / self.prior_precision for k, x in v.items()}


@pytree
class _DropsLowRank(_WrongCov):
    """Turns a `LowRankPlusDiag` into a `Diag` of its diagonal."""

    def add_precision_to(self, op: Any, /) -> Any:
        if isinstance(op, core.LowRankPlusDiag):
            return core.Diag(op.diagonal() + self.prior_precision)

        return op.add_isotropic(self.prior_precision)

    def cov_mv(self, v: Params, /) -> Params:
        return {k: x / self.prior_precision for k, x in v.items()}


@pytree
class _DropsKron(_WrongCov):
    """Turns each `KronEigenbasis` block into a `Diag` of its diagonal."""

    def add_precision_to(self, op: Any, /) -> Any:
        if isinstance(op, core.BlockDiag) and all(
            isinstance(b, core.KronEigenbasis) for b in op.blocks.values()
        ):
            return core.BlockDiag(
                {k: core.Diag(b.diagonal() + self.prior_precision) for k, b in op.blocks.items()}
            )

        return op.add_isotropic(self.prior_precision)

    def cov_mv(self, v: Params, /) -> Params:
        return {k: x / self.prior_precision for k, x in v.items()}


@pytree
class _DiagonalOnly(_WrongCov):
    """Refuses every dense structure."""

    def add_precision_to(self, op: Any, /) -> Any:
        blocks = op.blocks.values() if isinstance(op, core.BlockDiag) else [op]

        if not all(isinstance(b, core.Diag) for b in blocks):
            raise core.StructureError("diagonal operators only")

        return op.add_isotropic(self.prior_precision)

    def cov_mv(self, v: Params, /) -> Params:
        return {k: x / self.prior_precision for k, x in v.items()}


def test_diagonal_only_prior_passes() -> None:
    check_prior(_DiagonalOnly(torch.tensor(2.0, dtype=F64)), _prior_params())


@pytree
class _DiagonalOnlyWrongCov(_DiagonalOnly):
    def cov_mv(self, v: Params, /) -> Params:
        return {k: x * self.prior_precision for k, x in v.items()}


@pytree
class _TunableDiagonalOnly(_DiagonalOnly):
    def tunable(self) -> Tensor:
        return self.prior_precision


@pytree
class _CopiesTunable(_DiagonalOnly):
    def tunable(self) -> Tensor:
        return self.prior_precision.clone()


@pytree
class _FloatTunable(_DiagonalOnly):
    def tunable(self) -> Any:
        return float(self.prior_precision)


def test_tunable_prior_passes() -> None:
    check_prior(_TunableDiagonalOnly(torch.tensor(2.0, dtype=F64)), _prior_params())


@pytest.mark.parametrize(
    ("prior", "law"),
    [
        (
            _CopiesTunable(torch.tensor(2.0, dtype=F64)),
            r"tunable: tunable\(\) returned a tensor that is not a leaf of the prior .* "
            r"for _CopiesTunable",
        ),
        (
            _FloatTunable(torch.tensor(2.0, dtype=F64)),
            r"tunable: .* got float for _FloatTunable",
        ),
        (
            _TunableDiagonalOnly(torch.tensor(-2.0, dtype=F64)),
            r"tunable: .* positive .* for _TunableDiagonalOnly",
        ),
        (
            _ChangesType(torch.tensor(2.0, dtype=F64)),
            r"add_precision_to keeps operator type: turned BlockDiag\[Diag\] into "
            r"BlockDiag\[Dense\] for _ChangesType",
        ),
        (
            _FlattensJoint(torch.tensor(2.0, dtype=F64)),
            r"add_precision_to keeps operator type: turned Joint\[Dense\] into Dense "
            r"for _FlattensJoint",
        ),
        (
            _DropsLowRank(torch.tensor(2.0, dtype=F64)),
            r"add_precision_to keeps operator type: turned LowRankPlusDiag into Diag "
            r"for _DropsLowRank",
        ),
        (
            _DropsKron(torch.tensor(2.0, dtype=F64)),
            r"add_precision_to keeps operator type: turned BlockDiag\[KronEigenbasis\] into "
            r"BlockDiag\[Diag\] for _DropsKron",
        ),
        (_WrongCov(torch.tensor(2.0, dtype=F64)), "cov_mv: .* for _WrongCov"),
        (_NaNLogProb(torch.tensor(2.0, dtype=F64)), "log_prob: .* for _NaNLogProb"),
        (
            _DiagonalOnlyWrongCov(torch.tensor(2.0, dtype=F64)),
            "cov_mv: .* for _DiagonalOnlyWrongCov",
        ),
    ],
    ids=lambda x: type(x).__name__ if not isinstance(x, str) else "",
)
def test_broken_priors_fail(prior: Any, law: str) -> None:
    with pytest.raises(AssertionError, match=law):
        check_prior(prior, _prior_params())


# --- links ---------------------------------------------------------------------------------

N, C = 5, 3


def _outputs() -> core.OutputGaussian:
    g = torch.Generator().manual_seed(2)
    mu = torch.randn(N, C, generator=g, dtype=F64)
    a = torch.randn(N, C, C, generator=g, dtype=F64)

    return core.OutputGaussian(mu, core.PerInputCov(a @ a.mT / C, (C,)))


@pytest.mark.parametrize(
    ("link", "kind"),
    [
        (lk.Probit(), "cat"),
        (lk.Probit(), "gauss"),  # supports is False: must raise
        (lk.MC(n_samples=20), "cat"),
        (lk.MC(n_samples=20), "gauss"),
        (lk.predictive.Exact(), "gauss"),
        (lk.predictive.Exact(), "cat"),  # supports is False: must raise
    ],
    ids=lambda x: x if isinstance(x, str) else type(x).__name__,
)
def test_builtin_links_pass(link: Any, kind: str) -> None:
    check_link(link, _lik(kind), _outputs())


@pytree
class _ClaimsAll:
    """Says it supports everything, but raises for non-Categorical likelihoods."""

    def supports(self, lik: Any, f_mu: Tensor, /) -> bool:
        return True

    def __call__(self, lik: Any, out: core.OutputGaussian, /, *, generator=None) -> Distribution:
        return lk.Probit()(lik, out)


@pytree
class _ClaimsNone:
    """Says it supports nothing, but works."""

    def supports(self, lik: Any, f_mu: Tensor, /) -> bool:
        return False

    def __call__(self, lik: Any, out: core.OutputGaussian, /, *, generator=None) -> Distribution:
        return lk.Probit()(lik, out)


@pytree
class _Unnormalized:
    def supports(self, lik: Any, f_mu: Tensor, /) -> bool:
        return True

    def __call__(self, lik: Any, out: core.OutputGaussian, /, *, generator=None) -> Distribution:
        mu = out.mean
        assert isinstance(mu, Tensor)

        return torch.distributions.Categorical(probs=torch.softmax(mu, -1), validate_args=False)


@pytree
class _Unnormalizes(_Unnormalized):
    def __call__(self, lik: Any, out: core.OutputGaussian, /, *, generator=None) -> Distribution:
        d = super().__call__(lik, out, generator=generator)
        assert isinstance(d, torch.distributions.Categorical)
        object.__setattr__(d, "probs", d.probs * 2)

        return d


@pytree
class _NaNScale:
    def supports(self, lik: Any, f_mu: Tensor, /) -> bool:
        return True

    def __call__(self, lik: Any, out: core.OutputGaussian, /, *, generator=None) -> Distribution:
        mu = out.mean
        assert isinstance(mu, Tensor)

        return Normal(mu, torch.full_like(mu, math.nan), validate_args=False)


class _NegVarNormal(Normal):
    @property
    def variance(self) -> Tensor:
        return -super().variance


@pytree
class _NegativeVariance(_NaNScale):
    def __call__(self, lik: Any, out: core.OutputGaussian, /, *, generator=None) -> Distribution:
        mu = out.mean
        assert isinstance(mu, Tensor)

        return _NegVarNormal(mu, torch.ones_like(mu))


@pytree
class _NoMoments(_NaNScale):
    def __call__(self, lik: Any, out: core.OutputGaussian, /, *, generator=None) -> Distribution:
        return Distribution(batch_shape=torch.Size([N]), validate_args=False)


@pytest.mark.parametrize(
    ("link", "kind", "law"),
    [
        (
            _ClaimsAll(),
            "gauss",
            "supports: supports returned True for Gaussian but applying it raised "
            ".* for _ClaimsAll",
        ),
        (
            _ClaimsNone(),
            "cat",
            "supports: supports returned False for Categorical but applying it did not "
            "raise for _ClaimsNone",
        ),
        (_Unnormalizes(), "cat", "probabilities: .* for _Unnormalizes"),
        (_NaNScale(), "gauss", "moments: .* for _NaNScale"),
        (_NegativeVariance(), "gauss", "moments: .* non-negative for _NegativeVariance"),
        (_NoMoments(), "gauss", "moments: Distribution has no mean or variance for _NoMoments"),
    ],
    ids=lambda x: type(x).__name__ if not isinstance(x, str) else x[:5],
)
def test_broken_links_fail(link: Any, kind: str, law: str) -> None:
    with pytest.raises(AssertionError, match=law):
        check_link(link, _lik(kind), _outputs())


# --- subsets -------------------------------------------------------------------------------


def _subset_setup() -> tuple[nn.Module, core.Capabilities]:
    model = _model()
    caps = core.probe(
        lk.core.ModelFn.from_module(model), lk.Categorical(), _batches()[0], params={}
    )

    return model, caps


def _detached_last(model: lk.core.ModelFn, caps: core.Capabilities, /) -> Params:
    """Detached views share the parameters' storage, so they still count as the module's own."""

    return {n: p.detach() for n, p in lk.subsets.last_layer(model, caps).items()}


@pytest.mark.parametrize(
    "subset",
    [
        lk.subsets.last_layer,
        lk.subsets.all,
        lk.subsets.trainable,
        lk.subsets.named("0.*"),
        _detached_last,
    ],
    ids=lambda s: s.__qualname__,
)
def test_builtin_subsets_pass(subset: Any) -> None:
    model, caps = _subset_setup()

    check_subset(subset, model, caps)
    check_subset(subset, lk.core.ModelFn.from_module(model), caps)


def _module(model: lk.core.ModelFn) -> nn.Module:
    assert model.module is not None

    return model.module


def _copies(model: lk.core.ModelFn, caps: core.Capabilities, /) -> Params:
    return {n: p.clone() for n, p in _module(model).named_parameters() if n.startswith("2.")}


def _unknown_name(model: lk.core.ModelFn, caps: core.Capabilities, /) -> Params:
    return {"head.weight": _module(model).get_parameter("2.weight")}


def _wrong_name(model: lk.core.ModelFn, caps: core.Capabilities, /) -> Params:
    return {"0.bias": _module(model).get_parameter("2.bias")}


def _not_a_tensor(model: lk.core.ModelFn, caps: core.Capabilities, /) -> Params:
    return cast(Params, {"2.weight": 1.0})


def _empty(model: lk.core.ModelFn, caps: core.Capabilities, /) -> Params:
    return {}


def _a_list(model: lk.core.ModelFn, caps: core.Capabilities, /) -> Params:
    return cast(Params, list(_module(model).parameters()))


class _Flaky:
    """The first layer on odd calls, the last on even ones."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model: lk.core.ModelFn, caps: core.Capabilities, /) -> Params:
        self.calls += 1
        prefix = "0." if self.calls % 2 else "2."

        return {n: p for n, p in _module(model).named_parameters() if n.startswith(prefix)}


@pytest.mark.parametrize(
    ("subset", "law"),
    [
        (_copies, "own parameters: '2.weight' is a copy, not the module's parameter for _copies"),
        (_unknown_name, "names: 'head.weight' is not a parameter name of the module"),
        (_wrong_name, "own parameters: '0.bias' is a copy, not the module's parameter"),
        (_not_a_tensor, "own parameters: '2.weight' is float, not a Tensor"),
        (_empty, "non-empty: selected no parameters"),
        (_a_list, "mapping: the result is list, not a Mapping"),
        (_Flaky(), "deterministic: a second call selected .* for _Flaky"),
    ],
    ids=lambda x: x if isinstance(x, str) else type(x).__name__,
)
def test_broken_subsets_fail(subset: Any, law: str) -> None:
    model, caps = _subset_setup()

    with pytest.raises(AssertionError, match=law):
        check_subset(subset, model, caps)


def _last_entries(model: lk.core.ModelFn, caps: core.Capabilities, /) -> core.Selection:
    """The first column of the last layer's weight, as an index set."""

    w = _module(model).get_parameter("2.weight")
    mask = torch.zeros(w.shape, dtype=torch.bool)
    mask[:, 0] = True

    return core.Selection.from_masks({"2.weight": w}, {"2.weight": mask})


def test_check_subset_accepts_selection() -> None:
    model, caps = _subset_setup()

    check_subset(_last_entries, model, caps)
    check_subset(_last_entries, lk.core.ModelFn.from_module(model), caps)


def _copied_entries(model: lk.core.ModelFn, caps: core.Capabilities, /) -> core.Selection:
    sel = _last_entries(model, caps)

    return core.Selection({"2.weight": sel.base["2.weight"].clone()}, sel.index)


def _unknown_entries(model: lk.core.ModelFn, caps: core.Capabilities, /) -> core.Selection:
    sel = _last_entries(model, caps)

    return core.Selection(
        {"head.weight": sel.base["2.weight"]}, {"head.weight": sel.index["2.weight"]}
    )


class _FlakyEntries:
    """Column 0 of the last weight on odd calls, column 1 on even ones."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, model: lk.core.ModelFn, caps: core.Capabilities, /) -> core.Selection:
        self.calls += 1
        w = _module(model).get_parameter("2.weight")
        mask = torch.zeros(w.shape, dtype=torch.bool)
        mask[:, self.calls % 2] = True

        return core.Selection.from_masks({"2.weight": w}, {"2.weight": mask})


@pytest.mark.parametrize(
    ("subset", "law"),
    [
        (_copied_entries, "own parameters: '2.weight' is a copy, not the module's parameter"),
        (_unknown_entries, "names: 'head.weight' is not a parameter name of the module"),
        (_FlakyEntries(), "deterministic: a second call selected other entries .* _FlakyEntries"),
    ],
    ids=lambda x: x if isinstance(x, str) else type(x).__name__,
)
def test_broken_selection_subsets_fail(subset: Any, law: str) -> None:
    model, caps = _subset_setup()

    with pytest.raises(AssertionError, match=law):
        check_subset(subset, model, caps)


def test_subset_refusal_propagates() -> None:
    model, caps = _subset_setup()

    with pytest.raises(core.StructureError, match="no parameter matches"):
        check_subset(lk.subsets.named("nothing.*"), model, caps)


def test_subset_needs_a_module() -> None:
    _, caps = _subset_setup()
    fn = lk.core.ModelFn(lambda p, x: x)

    with pytest.raises(ValueError, match="needs a module"):
        check_subset(lk.subsets.all, fn, caps)


# --- objectives ----------------------------------------------------------------------------


def _fit(kind: str) -> tuple[Any, list[Batch]]:
    model = _model()
    batches = _batches(kind)
    post = lk.laplace(model, batches, likelihood=_lik(kind), hyper_tuner=None)

    return post, batches


def _marglik(post: Any) -> Any:
    return lk.objectives.neg_log_marglik


def test_builtin_objectives_pass() -> None:
    post, batches = _fit("gauss")

    assert list(core.hyperparameters(post)) == ["prior_precision", "sigma"]

    check_objective(_marglik(post), post)
    check_objective(lk.objectives.predictive_nll(batches[:1]), post)


def _vector(p: Any) -> Tensor:
    return p.prior.prior_precision * p.likelihood.sigma * torch.ones(2, dtype=F64)


def _python_float(p: Any) -> Tensor:
    return cast(Tensor, 1.0)


def _nan(p: Any) -> Tensor:
    return p.prior.prior_precision * math.nan


def _detached(p: Any) -> Tensor:
    return p.prior.prior_precision.detach() * 2


def _prec_only(p: Any) -> Tensor:
    return p.prior.prior_precision**2


def _nan_grad(p: Any) -> Tensor:
    return (p.prior.prior_precision * 0).sqrt() + p.likelihood.sigma


@pytest.mark.parametrize(
    ("obj", "law"),
    [
        (_vector, r"0-dim: the result must be a 0-dim tensor, got shape \(2,\) for _vector"),
        (_python_float, "0-dim: the result is float, not a Tensor for _python_float"),
        (_nan, "finite: the result is nan for _nan"),
        (_detached, "differentiable: the result does not depend on the selected leaves"),
        (_prec_only, "differentiable: no gradient reaches sigma"),
        (_nan_grad, "differentiable: the gradient for prior_precision is not finite"),
    ],
    ids=lambda x: x if isinstance(x, str) else x.__name__,
)
def test_broken_objectives_fail(obj: Any, law: str) -> None:
    post, _ = _fit("gauss")

    with pytest.raises(AssertionError, match=law):
        check_objective(obj, post)


def test_objective_needs_hyperparameters() -> None:
    post, _ = _fit("gauss")
    plain = dataclasses.replace(post, prior=post, likelihood=lk.Gaussian(sigma=1.0))

    with pytest.raises(ValueError, match="no hyperparameters"):
        check_objective(_marglik(post), plain)


def test_objective_leaves_the_posterior_alone() -> None:
    post, _ = _fit("gauss")
    prec = post.prior.prior_precision

    check_objective(_marglik(post), post)

    assert post.prior.prior_precision is prec
    assert not prec.requires_grad


# --- predictives ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["cat", "gauss"])
@pytest.mark.parametrize(
    "pred", [lk.Linearized(), lk.WeightSamples(n_samples=3)], ids=lambda p: type(p).__name__
)
def test_builtin_predictives_pass(pred: Any, kind: str) -> None:
    post, batches = _fit(kind)

    check_predictive(pred, post, batches[0].inputs)


class _ReturnsTensor:
    @property
    def requires(self) -> Requirements:
        return Requirements()

    def __call__(self, posterior: Any, inputs: Any, /, *, generator=None) -> Distribution:
        return cast(Distribution, posterior.model(posterior.mean, inputs))


class _Pooled(_ReturnsTensor):
    """One distribution for the whole batch."""

    def __call__(self, posterior: Any, inputs: Any, /, *, generator=None) -> Distribution:
        return Normal(posterior.model(posterior.mean, inputs).mean(0), 1.0)


class _Transposed(_ReturnsTensor):
    def __call__(self, posterior: Any, inputs: Any, /, *, generator=None) -> Distribution:
        return Normal(posterior.model(posterior.mean, inputs).T, 1.0)


class _NoRequirements(_ReturnsTensor):
    @property
    def requires(self) -> Requirements:
        return cast(Requirements, None)


class _NoGenerator(_ReturnsTensor):
    """Predicts per example, but takes no `generator`."""

    def __call__(self, posterior: Any, inputs: Any, /) -> Distribution:  # ty: ignore[invalid-method-override]
        return Normal(posterior.model(posterior.mean, inputs), 1.0)


@pytree
class _LinkWithoutGenerator(_Unnormalized):
    """A valid link, except that it takes no `generator`."""

    def __call__(self, lik: Any, out: core.OutputGaussian, /) -> Distribution:  # ty: ignore[invalid-method-override]
        return super().__call__(lik, out)


def test_a_link_or_predictive_without_a_generator_fails() -> None:
    post, batches = _fit("cat")

    with pytest.raises(AssertionError, match="generator"):
        check_predictive(_NoGenerator(), post, batches[0].inputs)

    with pytest.raises(AssertionError, match="generator"):
        check_link(_LinkWithoutGenerator(), _lik("cat"), _outputs())


@pytest.mark.parametrize(
    ("pred", "law"),
    [
        (_ReturnsTensor(), "distribution: the result is Tensor, not a Distribution"),
        (_Pooled(), r"batch shape: batch_shape \(3,\) does not start with the 6 examples"),
        (_Transposed(), r"batch shape: batch_shape \(3, 6\) does not start with the 6 examples"),
        (_NoRequirements(), "requires: requires is NoneType, not Requirements for _NoRequirements"),
    ],
    ids=lambda x: x if isinstance(x, str) else type(x).__name__,
)
def test_broken_predictives_fail(pred: Any, law: str) -> None:
    post, batches = _fit("gauss")

    with pytest.raises(AssertionError, match=law):
        check_predictive(pred, post, batches[0].inputs)


def test_exported_as_sl_testing() -> None:
    assert lk.testing.check_likelihood is check_likelihood
    assert lk.testing.check_curvature is check_curvature
    assert lk.testing.check_subset is check_subset
    assert lk.testing.check_objective is check_objective
    assert lk.testing.check_predictive is check_predictive


# --- likelihoods ------------------------------------------------------------------------


def _lik_data(kind: str) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(3)
    f = torch.randn(5, 3, generator=g, dtype=F64)

    if kind == "cat":
        return f, torch.randint(0, 3, (5,), generator=g)

    return f, torch.randn(5, 3, generator=g, dtype=F64)


@pytest.mark.parametrize(
    ("lik", "kind"),
    [
        (lk.Categorical(), "cat"),
        (copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.7, dtype=F64))), "gauss"),
        (
            copy.replace(
                lk.Gaussian(), sigma=torch.as_tensor(torch.tensor([0.7, 0.2, 1.3], dtype=F64))
            ),
            "gauss",
        ),
        (lk.Gaussian(1.0), "gauss"),
        (
            copy.replace(
                lk.Gaussian(event_dims=1), sigma=torch.as_tensor(torch.tensor(0.7, dtype=F64))
            ),
            "gauss",
        ),
        (lk.likelihoods.from_log_prob(lambda f, y: -((f - y) ** 2).sum(1)), "gauss"),
    ],
    ids=["categorical", "gaussian", "per-output", "unit", "event", "from-log-prob"],
)
def test_builtin_likelihoods_pass(lik: Any, kind: str) -> None:
    f, y = _lik_data(kind)

    check_likelihood(lik, f, y)


def test_bernoulli_passes() -> None:
    g = torch.Generator().manual_seed(4)
    f = torch.randn(5, 2, generator=g, dtype=F64)
    y = torch.randint(0, 2, (5, 2), generator=g).to(F64)

    check_likelihood(lk.Bernoulli(), f, y)


@pytree
class _CopyingUnit:
    """A late likelihood without a late hyperparameter whose `unit()` copies itself."""

    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return -((f - y) ** 2).sum(1)

    def unit(self) -> "_CopyingUnit":
        return _CopyingUnit()

    def curvature_scale(self) -> Tensor:
        return torch.ones(())

    def fit_stat(self, f: Tensor, y: Tensor, weights: Tensor | None, /) -> Tensor:
        return -((f - y) ** 2).sum()

    def log_lik(self, stat: Tensor, /) -> Tensor:
        return stat


@pytree
class _UnstableUnit(_CopyingUnit):
    """A late hyperparameter, but `unit()` of the unit is yet another object."""

    scale: Tensor = dataclasses.field(default_factory=lambda: torch.tensor(2.0, dtype=F64))

    def unit(self) -> "_CopyingUnit":
        return _CopyingUnit()


@pytree
class _ScalarLogProb:
    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return -((f - y) ** 2).sum()


@pytree
class _PerEntryWrongShape:
    """`[N + 1]`: one value too many."""

    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return torch.cat([-((f - y) ** 2).sum(1), torch.zeros(1, dtype=f.dtype)])


@pytree
class _WrongDistribution:
    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return -((f - y) ** 2).sum(1)

    def distribution(self, f: Tensor, /) -> Distribution:
        return torch.distributions.Independent(Normal(f, 1.0), 1)


@pytree
class _AsymmetricHessian:
    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return -((f - y) ** 2).sum(1)

    def output_hessian(self, f: Tensor, /) -> Any:
        h = torch.eye(3, dtype=f.dtype).expand(f.shape[0], 3, 3).clone()
        h[:, 0, 1] = 0.5

        return _DenseHessian(h)


@pytree
class _IndefiniteHessian(_AsymmetricHessian):
    def output_hessian(self, f: Tensor, /) -> Any:
        return _DenseHessian(-torch.eye(3, dtype=f.dtype).expand(f.shape[0], 3, 3))


@pytree
class _WrongMv(_AsymmetricHessian):
    def output_hessian(self, f: Tensor, /) -> Any:
        return _DenseHessian(torch.eye(3, dtype=f.dtype).expand(f.shape[0], 3, 3), mv_scale=2.0)


@pytree
class _DenseHessian:
    h: Tensor
    mv_scale: float = core.static(default=1.0)

    def mv(self, v: Tensor, /) -> Tensor:
        return self.mv_scale * torch.einsum("nef,nf->ne", self.h, v)

    def sqrt_mv(self, v: Tensor, /) -> Tensor:
        return v

    def diagonal(self) -> Tensor:
        return torch.diagonal(self.h, dim1=1, dim2=2)

    def dense(self) -> Tensor:
        return self.h


@pytest.mark.parametrize(
    ("lik", "law"),
    [
        (_CopyingUnit(), "unit identity"),
        (_UnstableUnit(), "unit idempotence"),
        (_ScalarLogProb(), "log_prob shape"),
        (_PerEntryWrongShape(), "log_prob shape"),
        (_WrongDistribution(), "distribution"),
        (_AsymmetricHessian(), "output Hessian symmetry"),
        (_IndefiniteHessian(), "output Hessian PSD"),
        (_WrongMv(), "output Hessian mv"),
    ],
    ids=[
        "copying-unit",
        "unstable-unit",
        "scalar",
        "n-plus-one",
        "distribution",
        "asym",
        "indef",
        "mv",
    ],
)
def test_broken_likelihoods_fail(lik: Any, law: str) -> None:
    f, y = _lik_data("gauss")

    with pytest.raises(AssertionError, match=rf"^{law}:.*{type(lik).__name__}"):
        check_likelihood(lik, f, y)
