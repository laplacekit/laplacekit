import copy
import math
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical as TorchCategorical
from torch.distributions import Distribution, Normal
from torch.func import grad, hessian, vmap

from laplacekit.core import (
    CapabilityError,
    HasDistribution,
    HasMask,
    HasOutputHessian,
    HasSample,
    HasTargetCheck,
    Requirements,
    StructureError,
    can_sample,
    check_likelihood,
    default_log_prob,
    default_sample,
    distribution,
    log_prob,
    observed_output_hessian,
    output_hessian,
    pytree,
    sample,
    spawn_generator,
    static,
)
from laplacekit.likelihoods import Bernoulli, Categorical, Gaussian


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _f() -> Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 3)


def test_categorical_distribution_and_log_prob() -> None:
    f, y = _f(), torch.tensor([0, 2, 1, 1])
    lik = Categorical()
    d = lik.distribution(f)

    assert isinstance(d, TorchCategorical)
    torch.testing.assert_close(d.logits, F.log_softmax(f, -1))
    torch.testing.assert_close(log_prob(lik, f, y), -F.cross_entropy(f, y, reduction="none"))


def test_categorical_class_dim() -> None:
    f, y = _f(), torch.tensor([0, 2, 1, 1])
    f_ct = f.unsqueeze(-1)  # [N, C, 1]: classes on axis 1, one position
    torch.testing.assert_close(
        log_prob(Categorical(class_dim=1), f_ct, y.unsqueeze(-1)), log_prob(Categorical(), f, y)
    )


def test_categorical_rejects_the_batch_axis_as_class_axis() -> None:
    with pytest.raises(StructureError, match="batch axis"):
        Categorical(class_dim=0)


def _nll_hessian(f: Tensor, class_dim: int) -> Tensor:
    """Per-example Hessian of -sum log softmax at fixed targets, flattened in f's order."""

    def nll(fi: Tensor) -> Tensor:
        logp = F.log_softmax(fi, dim=class_dim).movedim(class_dim, -1)
        return -logp[..., 0].sum()

    e = f[0].numel()

    return torch.stack([hessian(nll)(fi).reshape(e, e) for fi in f])


def test_categorical_dense_hessian_over_positions() -> None:
    torch.manual_seed(0)
    f = torch.randn(2, 3, 4, dtype=torch.float64)  # [N, T, C]
    h = output_hessian(Categorical(), f, generator=_gen()).dense()

    assert h.shape == (2, 12, 12)
    torch.testing.assert_close(h, _nll_hessian(f, class_dim=-1))


def test_categorical_dense_hessian_with_class_dim() -> None:
    torch.manual_seed(0)
    f = torch.randn(2, 4, 3, dtype=torch.float64)  # [N, C, T]
    h = output_hessian(Categorical(class_dim=1), f, generator=_gen()).dense()

    assert h.shape == (2, 12, 12)
    torch.testing.assert_close(h, _nll_hessian(f, class_dim=0))


def test_categorical_dense_hessian_limits_event_size() -> None:
    f = torch.zeros(1, 100, 50)

    with pytest.raises(StructureError, match="5000"):
        output_hessian(Categorical(), f, generator=_gen()).dense()


def test_categorical_hessian_matches_autograd() -> None:
    f = _f()
    h = output_hessian(Categorical(), f, generator=_gen())

    assert h.dense().shape == (4, 3, 3)

    for i in range(4):
        for y in range(3):
            ref = hessian(lambda fi, y=y: -F.log_softmax(fi, -1)[y])(f[i])
            torch.testing.assert_close(h.dense()[i], ref)


def test_categorical_hessian_ops() -> None:
    f = _f()
    h = output_hessian(Categorical(), f, generator=_gen())
    dense = h.dense()
    a = torch.stack([h.sqrt_mv(torch.eye(3).expand(4, 3, 3)[:, :, j]) for j in range(3)], -1)
    torch.testing.assert_close(a @ a.mT, dense)
    v = torch.randn(4, 3)
    torch.testing.assert_close(h.mv(v), (dense @ v.unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(h.diagonal(), dense.diagonal(dim1=-2, dim2=-1))


def test_gaussian_log_prob_and_hessian() -> None:
    f, y = torch.randn(4, 3), torch.randn(4, 3)
    lik = copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3)))
    torch.testing.assert_close(log_prob(lik, f, y), Normal(f, 0.3).log_prob(y).sum(1))
    h = output_hessian(lik, f, generator=_gen())
    torch.testing.assert_close(h.dense(), torch.eye(3).expand(4, 3, 3) / 0.09)
    v = torch.randn(4, 3)
    torch.testing.assert_close(h.mv(v), v / 0.09)
    torch.testing.assert_close(h.sqrt_mv(h.sqrt_mv(v)), v / 0.09)
    torch.testing.assert_close(h.diagonal(), torch.full((4, 3), 1 / 0.09))


def test_gaussian_default_sigma_and_event_dims() -> None:
    f, y = torch.randn(4, 3), torch.randn(4, 3)
    lik = Gaussian(event_dims=1)

    assert lik.distribution(f).log_prob(y).shape == (4,)
    torch.testing.assert_close(log_prob(lik, f, y), Normal(f, 1.0).log_prob(y).sum(1))
    torch.testing.assert_close(
        output_hessian(lik, f, generator=_gen()).dense(), torch.eye(3).expand(4, 3, 3)
    )


def test_default_sigma_instances_do_not_share() -> None:
    a, b = Gaussian(), Gaussian()

    assert a.sigma is not None and b.sigma is not None
    assert a.sigma is not b.sigma
    torch.testing.assert_close(
        a.sigma, torch.tensor(1.0, dtype=torch.float64)
    )  # a float stays double until the fit
    assert not a.unit().learned and float(a.unit().sigma) == 1.0
    assert Gaussian(sigma=1.0).unit().learned is False


@pytree
class _PaddedGaussian(Gaussian):
    """A Gaussian subclass with a mask and a field of its own."""

    pad: int = static(default=-1)

    def mask(self, f: Tensor, inputs: Tensor) -> Tensor | None:
        return (inputs != self.pad).unsqueeze(-1).expand_as(f)


def test_gaussian_unit_keeps_subclass() -> None:
    """`unit()` drops sigma only: the subclass, its mask and every other field stay."""

    lik = _PaddedGaussian(torch.tensor(0.5), event_dims=0, learned=True, pad=7)
    unit = lik.unit()

    assert type(unit) is _PaddedGaussian
    assert float(unit.sigma) == 1.0 and not unit.learned and unit.pad == 7 and unit.event_dims == 0
    assert isinstance(unit, HasMask)

    inputs = torch.tensor([[1, 7], [7, 2]])
    want = torch.tensor([[[True], [False]], [[False], [True]]])

    assert torch.equal(cast(Tensor, unit.mask(torch.zeros(2, 2, 1), inputs)), want)

    # Nothing late to drop: the likelihood itself, as before.
    assert unit.unit() is unit
    per_output = _PaddedGaussian(torch.tensor([0.5, 0.2]))
    assert per_output.unit() is per_output


@pytest.mark.parametrize("sigma", [torch.tensor(1.0), torch.tensor([0.3])], ids=["0d", "1el"])
def test_gaussian_noise_is_broadcast_on_the_device_of_f(monkeypatch, sigma) -> None:
    """Sigma lives on the CPU (the default does); what is broadcast to f's shape lives on f's
    device. The meta device stands in for a GPU on a CPU-only machine."""

    monkeypatch.setattr(Distribution, "_validate_args", False)  # validation reads values
    f = torch.empty(4, 3, device="meta")
    lik = copy.replace(Gaussian(), sigma=torch.as_tensor(sigma))

    assert lik.distribution(f).scale.device == f.device
    assert lik.output_hessian(f).diagonal().device == f.device


def test_sample_reproducible() -> None:
    f = _f()
    a = sample(Categorical(), f, generator=_gen())
    b = sample(Categorical(), f, generator=_gen())

    assert isinstance(a, Tensor)
    assert a.shape == (4,)
    torch.testing.assert_close(a, b)


class _HeteroGaussian:
    """Only `log_prob` and `distribution`: f = [mu, log_sigma] per example."""

    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return self.distribution(f).log_prob(y)

    def distribution(self, f: Tensor, /) -> Distribution:
        return Normal(f[:, 0], f[:, 1].exp())


class _LogProbOnly:
    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return -((f - y) ** 2).sum(-1)


def test_log_prob_per_example() -> None:
    torch.manual_seed(0)
    f = torch.randn(4, 5, 3)  # [N, T, C]
    y = torch.randint(0, 3, (4, 5))
    per_position = TorchCategorical(logits=f).log_prob(y)

    assert Categorical().log_prob(f, y).shape == (4, 5)
    torch.testing.assert_close(log_prob(Categorical(), f, y), per_position.sum(1))

    g, z = torch.randn(4, 5, 2), torch.randn(4, 5, 2)
    lik = copy.replace(Gaussian(event_dims=1), sigma=torch.as_tensor(torch.tensor(0.7)))

    assert lik.log_prob(g, z).shape == (4, 5)
    torch.testing.assert_close(log_prob(lik, g, z), Normal(g, 0.7).log_prob(z).sum((1, 2)))
    torch.testing.assert_close(
        log_prob(Gaussian(sigma=1.0), g, z), Normal(g, 1.0).log_prob(z).sum((1, 2))
    )


def test_log_prob_rejects_a_scalar() -> None:
    class Scalar:
        def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
            return -((f - y) ** 2).sum()

    with pytest.raises(StructureError, match=r"\[N\]"):
        log_prob(Scalar(), torch.zeros(4, 2), torch.zeros(4, 2))


def test_protocols_and_helpers_on_a_log_prob_only_likelihood() -> None:
    lik, f = _LogProbOnly(), torch.zeros(4, 2)

    assert not isinstance(lik, HasDistribution | HasSample | HasOutputHessian)
    assert not can_sample(lik)
    assert can_sample(Categorical()) and can_sample(_HeteroGaussian())

    with pytest.raises(CapabilityError, match="HasSample or HasDistribution"):
        sample(lik, f, generator=_gen())

    with pytest.raises(CapabilityError, match="HasDistribution"):
        distribution(lik, f)

    with pytest.raises(CapabilityError, match="HasOutputHessian"):
        output_hessian(lik, f, generator=_gen())


def test_sample_from_distribution_follows_the_generator() -> None:
    f = torch.zeros(8, 2)
    lik = _HeteroGaussian()
    a, b, c = (cast(Tensor, sample(lik, f, generator=_gen(k))) for k in (0, 0, 1))

    assert a.shape == (8,)
    torch.testing.assert_close(a, b)

    assert not torch.equal(a, c)


def test_default_output_hessian_is_mc_fisher() -> None:
    # Fisher of N(mu, sigma^2) in (mu, log sigma): diag(1 / sigma^2, 2). The per-sample
    # estimate of the (log sigma, log sigma) entry has variance 56; averaging 64 identical
    # examples of 20000 draws each puts the 2e-2 tolerance at about 6 standard errors.
    n, mu, log_sigma = 64, 0.3, 0.5
    f = torch.tensor([mu, log_sigma], dtype=torch.float64).expand(n, 2).clone()
    lik = _HeteroGaussian()

    assert not isinstance(lik, HasOutputHessian | HasSample)

    h = output_hessian(lik, f, generator=_gen(), mc_samples=20000).dense()

    assert h.shape == (n, 2, 2)

    mean = h.mean(0)
    want = torch.tensor([math.exp(-2 * log_sigma), 2.0], dtype=torch.float64)
    torch.testing.assert_close(mean.diagonal(), want, rtol=2e-2, atol=0)
    # The off-diagonal estimate has variance 10 / sigma^2 per draw: 1e-2 is about 6
    # standard errors.
    assert abs(float(mean[0, 1])) < 1e-2

    assert bool((torch.linalg.eigvalsh(h) >= -1e-12).all())

    a = output_hessian(lik, f, generator=torch.Generator().manual_seed(1), mc_samples=3).dense()
    b = output_hessian(lik, f, generator=torch.Generator().manual_seed(1), mc_samples=3).dense()
    c = output_hessian(lik, f, generator=torch.Generator().manual_seed(2), mc_samples=3).dense()
    torch.testing.assert_close(a, b)

    assert not torch.allclose(a, c)


def test_mc_fisher_ops_agree_with_dense() -> None:
    f = torch.randn(4, 2, dtype=torch.float64)
    h = output_hessian(_HeteroGaussian(), f, generator=_gen(), mc_samples=5)
    v = torch.randn(4, 2, dtype=torch.float64)
    dense = h.dense()
    torch.testing.assert_close(h.mv(v), (dense @ v.unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(h.sqrt_mv(h.sqrt_mv(v)), h.mv(v))
    torch.testing.assert_close(h.diagonal(), dense.diagonal(dim1=-2, dim2=-1))


def test_mc_fisher_limits_event_size() -> None:
    class Wide:
        def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
            return self.distribution(f).log_prob(y).sum(-1)

        def distribution(self, f: Tensor, /) -> Distribution:
            return Normal(f, 1.0)

    with pytest.raises(StructureError, match="5000"):
        output_hessian(Wide(), torch.zeros(1, 5000), generator=_gen())


def _hetero_nll(fi: Tensor, yi: Tensor) -> Tensor:
    mu, log_sigma = fi[0], fi[1]
    return log_sigma + 0.5 * ((yi - mu) / log_sigma.exp()) ** 2


def test_observed_hessian_can_be_indefinite() -> None:
    f = torch.tensor([[0.0, 0.0], [0.5, -0.3]], dtype=torch.float64)
    y = torch.tensor([1.5, -1.0], dtype=torch.float64)
    h = observed_output_hessian(_HeteroGaussian(), f, y).dense()
    want = torch.stack([hessian(_hetero_nll)(fi, yi) for fi, yi in zip(f, y, strict=True)])
    torch.testing.assert_close(h, want)

    assert bool((torch.linalg.eigvalsh(h)[:, 0] < 0).all())


def test_check_likelihood() -> None:
    log_only, hetero = _LogProbOnly(), _HeteroGaussian()

    check_likelihood(Requirements(needs_targets=True), log_only, "X")
    check_likelihood(Requirements(needs_output_hessian=True), Categorical(), "X")
    check_likelihood(
        Requirements(needs_sample=True, needs_distribution=True, needs_output_hessian=True),
        hetero,
        "X",
    )

    with pytest.raises(CapabilityError, match="DiagMCFisher needs .*HasSample"):
        check_likelihood(Requirements(needs_sample=True), log_only, "DiagMCFisher")

    with pytest.raises(CapabilityError, match="HasDistribution"):
        check_likelihood(Requirements(needs_distribution=True), log_only, "X")

    with pytest.raises(CapabilityError, match="HasOutputHessian"):
        check_likelihood(Requirements(needs_output_hessian=True), log_only, "X")

    with pytest.raises(CapabilityError, match="'b'"):
        check_likelihood(Requirements(needs_sample=True), {"a": hetero, "b": log_only}, "X")


def test_gaussian_hessian_differentiable_in_sigma() -> None:
    f = torch.randn(4, 3, dtype=torch.float64)
    s = torch.tensor(0.3, dtype=torch.float64)
    g = grad(
        lambda s: (
            copy.replace(Gaussian(), sigma=torch.as_tensor(s)).output_hessian(f).diagonal().sum()
        )
    )(s)
    torch.testing.assert_close(g, -2 / s**3 * f.numel())


def test_categorical_hessian_differentiable_in_f() -> None:
    f = torch.randn(4, 3, dtype=torch.float64)

    def fn(f: Tensor) -> Tensor:
        return Categorical().output_hessian(f).diagonal().sum()

    g = grad(fn)(f)

    assert g.abs().max() > 1e-6
    eps = 1e-6
    num = torch.zeros_like(f)

    for i in range(4):
        for j in range(3):
            d = torch.zeros_like(f)
            d[i, j] = eps
            num[i, j] = (fn(f + d) - fn(f - d)) / (2 * eps)

    torch.testing.assert_close(g, num, atol=1e-7, rtol=1e-5)


def test_hessian_objects_pass_through_vmap() -> None:
    f = _f()
    per_example = vmap(lambda fi: output_hessian(Categorical(), fi, generator=_gen()).diagonal())(f)
    torch.testing.assert_close(
        per_example, output_hessian(Categorical(), f, generator=_gen()).diagonal()
    )
    h = vmap(lambda fi: output_hessian(Categorical(), fi, generator=_gen()))(f)

    assert isinstance(h.diagonal(), Tensor)
    gh = vmap(lambda fi: output_hessian(Gaussian(sigma=1.0), fi, generator=_gen()))(f)

    assert isinstance(gh.diagonal(), Tensor)
    f2 = torch.randn(4, 2)
    dense = output_hessian(_HeteroGaussian(), f2, generator=_gen())

    assert isinstance(dense.sqrt_mv(f2), Tensor)


def test_default_log_prob_and_sample_derive_from_the_distribution() -> None:
    torch.manual_seed(0)
    f, y = torch.randn(4, 2), torch.randn(4)
    lik = _HeteroGaussian()
    ref = Normal(f[:, 0], f[:, 1].exp())

    assert default_log_prob(lik, f, y).shape == (4,)
    torch.testing.assert_close(default_log_prob(lik, f, y), ref.log_prob(y))

    # A seed drawn from the generator seeds torch's RNG for `Distribution.sample()`.
    seed = spawn_generator(_gen(7)).initial_seed()

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        want = ref.sample()

    before = torch.get_rng_state()
    got = default_sample(lik, f, generator=_gen(7))

    torch.testing.assert_close(got, want)
    torch.testing.assert_close(sample(lik, f, generator=_gen(7)), want)
    assert torch.equal(torch.get_rng_state(), before)

    # Without a generator: torch's global RNG.
    torch.manual_seed(3)
    unseeded = default_sample(lik, f)
    torch.manual_seed(3)

    torch.testing.assert_close(default_sample(lik, f), unseeded)

    g = torch.randn(4, 5, 3)  # per position: the class axis is the event
    labels = torch.randint(0, 3, (4, 5))
    torch.testing.assert_close(
        default_log_prob(Categorical(), g, labels), TorchCategorical(logits=g).log_prob(labels)
    )


# --- Bernoulli -----------------------------------------------------------------------------


def test_bernoulli_output_hessian_matches_autograd() -> None:
    torch.manual_seed(0)
    f = torch.randn(4, 3, dtype=torch.float64)
    y = torch.randint(0, 2, (4, 3)).to(torch.float64)
    lik = Bernoulli()
    h = output_hessian(lik, f, generator=_gen())
    want = observed_output_hessian(lik, f, y).dense()

    torch.testing.assert_close(h.dense(), want)
    torch.testing.assert_close(h.diagonal(), want.diagonal(dim1=-2, dim2=-1))

    eye = torch.eye(3, dtype=torch.float64).expand(4, 3, 3)
    a = torch.stack([h.sqrt_mv(eye[:, :, j]) for j in range(3)], -1)
    v = torch.randn(4, 3, dtype=torch.float64)

    torch.testing.assert_close(a @ a.mT, want)
    torch.testing.assert_close(h.mv(v), (want @ v.unsqueeze(-1)).squeeze(-1))


def test_bernoulli_soft_labels() -> None:
    torch.manual_seed(0)
    f = torch.randn(4, 3, dtype=torch.float64)
    y = torch.rand(4, 3, dtype=torch.float64)
    p = 1 / (1 + torch.exp(-f))
    want = y * torch.log(p) + (1 - y) * torch.log(1 - p)

    torch.testing.assert_close(Bernoulli().log_prob(f, y), want)

    # Over the last axis as one event: summed there, one value per example.
    torch.testing.assert_close(Bernoulli(event_dims=1).log_prob(f, y), want.sum(-1))


def test_bernoulli_casts_bool_and_int_targets() -> None:
    torch.manual_seed(0)
    f = torch.randn(4, 3, dtype=torch.float64)
    y = torch.randint(0, 2, (4, 3))
    want = torch.where(y == 1, -torch.log1p(torch.exp(-f)), -torch.log1p(torch.exp(f)))

    torch.testing.assert_close(Bernoulli().log_prob(f, y), want)
    torch.testing.assert_close(Bernoulli().log_prob(f, y.bool()), want)


def test_bernoulli_distribution() -> None:
    torch.manual_seed(0)
    f = torch.randn(4, 3)
    d = distribution(Bernoulli(), f)
    d1 = distribution(Bernoulli(event_dims=1), f)

    assert isinstance(d, torch.distributions.Bernoulli)
    torch.testing.assert_close(d.probs, torch.sigmoid(f))
    assert isinstance(d1, torch.distributions.Independent) and d1.event_shape == (3,)
    torch.testing.assert_close(d1.mean, torch.sigmoid(f))
    assert sample(Bernoulli(), f, generator=_gen()).shape == (4, 3)


def test_bernoulli_refuses_targets_not_shaped_like_f() -> None:
    f = torch.zeros(4, 3)

    with pytest.raises(StructureError, match="shaped like f"):
        Bernoulli().log_prob(f, torch.zeros(4))


def test_bernoulli_refuses_negative_event_dims() -> None:
    with pytest.raises(StructureError, match="event_dims"):
        Bernoulli(event_dims=-1)


def test_bernoulli_check_targets_names_the_bad_range() -> None:
    lik = Bernoulli()

    assert isinstance(lik, HasTargetCheck)

    lik.check_targets(torch.tensor([[0.0, 0.3, 1.0]]))
    lik.check_targets(torch.tensor([[True, False]]))

    with pytest.raises(ValueError, match=r"\[0, 1\].*\[-1\.0, 2\.0\]"):
        lik.check_targets(torch.tensor([[0.0, 2.0], [-1.0, 1.0]]))


def _bernoulli_data(bad_batch: int):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(2, 3, bias=False))
    xs = [torch.randn(4, 2) for _ in range(2)]
    ys = [torch.randint(0, 2, (4, 3)).float() for _ in range(2)]
    ys[bad_batch][0, 0] = 2.0  # a class index where a 0/1 label belongs

    return model, list(zip(xs, ys, strict=True))


@pytest.mark.parametrize("bad_batch", [0, 1])
def test_laplace_refuses_bernoulli_targets_before_the_curvature_sees_them(
    bad_batch, monkeypatch
) -> None:
    import laplacekit as lk
    from laplacekit.curvlinops import KFAC

    model, data = _bernoulli_data(bad_batch)
    seen: list[Tensor] = []
    update = KFAC.update

    def spy(self, stat, params, batch, generator, ctx, /):
        seen.append(cast(Tensor, batch.targets))
        return update(self, stat, params, batch, generator, ctx)

    monkeypatch.setattr(KFAC, "update", spy)

    with pytest.raises(ValueError, match=r"Bernoulli targets must lie in \[0, 1\]"):
        lk.laplace(
            model, data, likelihood=lk.Bernoulli(), hessian_approx=lk.KFAC(), hyper_tuner=None
        )

    # The curvature never got the bad batch.
    assert len(seen) == bad_batch
    assert all(bool((y <= 1).all()) for y in seen)


def test_by_hand_fit_checks_targets_per_batch() -> None:
    import laplacekit as lk

    model, data = _bernoulli_data(1)
    fn = lk.core.ModelFn.from_module(model)
    lik = lk.Bernoulli()
    caps = lk.core.probe(fn, lik, lk.Batch.from_pair(data[0]))
    params = lk.subsets.all(fn, caps)
    fit = lk.core.fit_curvature(fn, lik, params, lk.DenseGGN(), capabilities=caps)
    stat = fit.update(fit.init(), lk.Batch.from_pair(data[0]))

    with pytest.raises(ValueError, match=r"\[0, 2\.0\]|\[0\.0, 2\.0\]"):
        fit.update(stat, lk.Batch.from_pair(data[1]))


def _shape_data(lik_name: str):
    torch.manual_seed(0)
    x = torch.randn(8, 2)

    if lik_name == "categorical":
        model = torch.nn.Sequential(torch.nn.Linear(2, 3))
        y = torch.randint(0, 3, (8, 1))  # class indices with a trailing axis
    else:
        model = torch.nn.Sequential(torch.nn.Linear(2, 1))
        y = torch.randn(8)  # one-output regression targets without the output axis

    return model, x, y


@pytest.mark.parametrize(
    ("lik_name", "pattern"),
    # The probe runs on the first examples of the batch, so N is not pinned.
    [
        ("categorical", r"shaped \(\d+,\).*got \(\d+, 1\)"),
        ("gaussian", r"shaped \(\d+, 1\).*got \(\d+,\)"),
    ],
)
def test_mis_shaped_targets_refused_by_the_probe(lik_name, pattern) -> None:
    import laplacekit as lk

    model, x, y = _shape_data(lik_name)
    lik = lk.Categorical() if lik_name == "categorical" else lk.Gaussian()
    fn = lk.core.ModelFn.from_module(model)

    with pytest.raises(StructureError, match=pattern):
        lk.core.probe(fn, lik, lk.Batch(x, y))

    with pytest.raises(StructureError, match=pattern):
        lk.laplace(model, [(x, y)], likelihood=lik, hyper_tuner=None)


@pytest.mark.parametrize(
    ("f", "y"),
    [
        (torch.randn(4, 3), torch.randint(0, 3, (4,))),
        (torch.randn(4, 5, 3), torch.randint(0, 3, (4, 5))),  # sequence outputs [N, T, K]
    ],
)
def test_categorical_accepts_targets_without_the_class_axis(f, y) -> None:
    lp = Categorical().log_prob(f, y)

    torch.testing.assert_close(lp, TorchCategorical(logits=f).log_prob(y))


def test_categorical_class_dim_targets_checked() -> None:
    f = torch.randn(4, 3, 5)  # class axis 1
    lik = Categorical(class_dim=1)

    assert lik.log_prob(f, torch.randint(0, 3, (4, 5))).shape == (4, 5)

    with pytest.raises(StructureError, match=r"\(4, 5\).*\(4, 3\)"):
        lik.log_prob(f, torch.randint(0, 3, (4, 3)))


@pytest.mark.parametrize("sigma", [None, torch.tensor(0.5), torch.tensor([0.5, 1.0, 2.0])])
def test_gaussian_log_prob_refuses_broadcast_targets(sigma) -> None:
    f = torch.randn(4, 3)
    lik = Gaussian() if sigma is None else copy.replace(Gaussian(), sigma=sigma)

    assert lik.log_prob(f, torch.randn(4, 3)).shape == (4, 3)

    with pytest.raises(StructureError, match=r"\(4, 3\).*\(4, 1\)"):
        lik.log_prob(f, torch.randn(4, 1))


@pytest.mark.parametrize("sigma", [None, torch.tensor([0.5])])
def test_gaussian_fit_stat_refuses_mis_shaped_targets(sigma) -> None:
    f = torch.randn(6, 1)

    with pytest.raises(StructureError, match=r"\(6, 1\).*\(6,\)"):
        (Gaussian() if sigma is None else copy.replace(Gaussian(), sigma=sigma)).fit_stat(
            f, torch.randn(6), None
        )


def test_gaussian_per_output_fit_stat_refuses_mis_shaped_targets() -> None:
    f = torch.randn(6, 3)
    lik = copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor([0.5, 1.0, 2.0])))

    with pytest.raises(StructureError, match=r"\(6, 3\).*\(6, 1\)"):
        lik.fit_stat(f, torch.randn(6, 1), None)
