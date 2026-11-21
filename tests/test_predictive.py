import copy
import dataclasses
import math
import re
from functools import partial
from typing import Literal

import pytest
import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, Normal, Poisson

import laplacekit as lk
import laplacekit.core as core
from laplacekit.core import (
    Capabilities,
    CapabilityError,
    Dense,
    LaplacePosterior,
    Layout,
    ModelFn,
    Requirements,
    pytree,
)
from laplacekit.likelihoods import Bernoulli, Categorical, Gaussian
from laplacekit.predictive import MC, Linearized, Probit, WeightSamples, _average, _per_position
from laplacekit.priors import Isotropic

C, D, N = 3, 4, 5
F64 = torch.float64


CAPS = Capabilities(vjp=True, vmap=True)
VJP = Capabilities(vjp=True, vjp_twice=True)  # no vmap: sampled outputs


def post(lik, caps=CAPS, n_data=5.0):
    g = torch.Generator().manual_seed(0)
    w = torch.randn(C, D, generator=g, dtype=F64)
    a = torch.randn(C * D, C * D, generator=g, dtype=F64)
    p = LaplacePosterior(
        mean={"fc.weight": w},
        curvature=Dense(a @ a.T / (C * D)),
        prior=Isotropic(torch.tensor(1.0, dtype=F64)),
        n_data=torch.tensor(n_data, dtype=F64),
        model=ModelFn(lambda p, x: x @ p["fc.weight"].T),
        likelihood=lik,
        layout=Layout.by_module({"fc.weight": w}),
        caps=caps,
    )
    x = torch.randn(N, D, generator=g, dtype=F64)

    return p, x


def sigma(p: LaplacePosterior) -> Tensor:
    return torch.linalg.inv(p.precision().mat)


def test_outputs_match_analytic_jacobian() -> None:
    p, x = post(Categorical())
    out = core.predict_outputs(p, x)
    s = sigma(p)

    for i in range(N):
        j = torch.kron(torch.eye(C, dtype=F64), x[i : i + 1])  # [C, C*D]
        assert torch.allclose(out.cov.per_input()[i], j @ s @ j.T, rtol=1e-8, atol=1e-12)

    assert torch.allclose(out.mean, core.mean_outputs(p, x))


def _out(mu: Tensor, cov: Tensor) -> core.OutputGaussian:
    return core.OutputGaussian(mu, core.PerInputCov(cov, (mu.shape[-1],)))


def _random_cov(n: int, k: int, seed: int = 1) -> Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, k, k, generator=g, dtype=F64)

    return a @ a.mT / k


def test_probit_zero_variance_is_softmax() -> None:
    mu = torch.randn(N, C, dtype=F64)
    zero = _out(mu, torch.zeros(N, C, C, dtype=F64))

    assert torch.allclose(Probit()(Categorical(), zero).probs, torch.softmax(mu, -1))


def test_probit_ignores_shared_logit_shift_variance() -> None:
    mu = torch.randn(N, C, dtype=F64)
    cov = _random_cov(N, C)
    shared = cov + 1e4 * torch.ones(C, C, dtype=F64)

    a = Probit()(Categorical(), _out(mu, cov)).probs
    b = Probit()(Categorical(), _out(mu, shared)).probs

    assert torch.allclose(a, b)


def test_probit_matches_centred_variance_formula() -> None:
    mu = torch.randn(N, C, dtype=F64)
    cov = _random_cov(N, C)
    cov = cov - cov.mean(-1, keepdim=True)  # remove the shared component
    cov = cov - cov.mean(-2, keepdim=True)
    out = _out(mu, cov)
    got = Probit()(Categorical(), out).probs

    for i in range(N):
        c = torch.eye(C, dtype=F64) - torch.ones(C, C, dtype=F64) / C
        v = torch.diag(c @ cov[i] @ c.T)
        g = mu[i] - mu[i].mean()
        ref = torch.softmax(g / torch.sqrt(1 + math.pi / 8 * v), -1)

        assert torch.allclose(got[i], ref)

    # Without a shared component the centred variance equals the marginal one only
    # through the formula: check it against the explicit sums too.
    v = cov.diagonal(dim1=-2, dim2=-1) - 2 / C * cov.sum(-1) + cov.sum((-1, -2))[:, None] / C**2
    ref = torch.softmax((mu - mu.mean(-1, keepdim=True)) / torch.sqrt(1 + math.pi / 8 * v), -1)

    assert torch.allclose(got, ref)


def test_probit_needs_per_input_blocks() -> None:
    class Sampled:
        """An OutputCov without dense blocks, as a sampled covariance will be."""

        def diagonal(self) -> Tensor:
            return torch.ones(N, C, dtype=F64)

        def sample(self, n, /, *, generator=None) -> Tensor:
            return torch.zeros(n, N, C, dtype=F64)

        def per_input(self) -> Tensor:
            raise core.StructureError("Sampled has no per-input blocks")

        def scaled(self, s, /):
            return self

    out = core.OutputGaussian(torch.randn(N, C, dtype=F64), Sampled())

    with pytest.raises(core.StructureError, match="per-input"):
        Probit()(Categorical(), out)


T = 4


def _sequence_case(seed: int = 2) -> tuple[Tensor, Tensor]:
    """Logits [N, T, C] with one dense [T*C, T*C] covariance per input, row-major."""

    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(N, T, C, generator=g, dtype=F64)
    a = torch.randn(N, T * C, T * C, generator=g, dtype=F64)

    return mu, a @ a.mT / (T * C)


def _per_position_reference(mu: Tensor, cov: Tensor) -> Tensor:
    """The [N, C] probit applied to each position t with its own [C, C] block."""

    full = cov.reshape(N, T, C, T, C)
    probs = [Probit()(Categorical(), _out(mu[:, t], full[:, t, :, t, :])).probs for t in range(T)]

    return torch.stack(probs, 1)  # [N, T, C]


def test_probit_classes_last_with_extra_axis() -> None:
    mu, cov = _sequence_case()
    out = core.OutputGaussian(mu, core.PerInputCov(cov, (T, C)))
    got = Probit()(Categorical(class_dim=-1), out).probs

    assert got.shape == (N, T, C)
    assert torch.allclose(got, _per_position_reference(mu, cov))


def test_probit_classes_at_dim_one() -> None:
    mu, cov = _sequence_case()  # [N, T, C] layout for the reference
    ref = _per_position_reference(mu, cov)

    # The same model with outputs [N, C, T]: permute mean and covariance to match.
    mu_ct = mu.transpose(1, 2)
    cov_ct = cov.reshape(N, T, C, T, C).permute(0, 2, 1, 4, 3).reshape(N, C * T, C * T)
    out = core.OutputGaussian(mu_ct, core.PerInputCov(cov_ct, (C, T)))
    got = Probit()(Categorical(class_dim=1), out).probs  # probs put classes last

    assert got.shape == (N, T, C)
    assert torch.allclose(got, ref)


def test_probit_rejects_class_dim_on_batch_axis() -> None:
    mu, cov = _sequence_case()
    out = core.OutputGaussian(mu, core.PerInputCov(cov, (T, C)))

    with pytest.raises(CapabilityError, match="class_dim"):
        Probit()(Categorical(class_dim=-3), out)  # [N, T, C]: axis -3 is the batch axis


def test_probit_close_to_mc_with_weak_prior() -> None:
    p, x = post(Categorical())
    train = torch.randn(40, D, generator=torch.Generator().manual_seed(2), dtype=F64)
    probs = torch.softmax(core.mean_outputs(p, train), -1)
    ggn = torch.zeros(C * D, C * D, dtype=F64)

    for i in range(len(train)):  # softmax GGN, null along the shared logit direction
        j = torch.kron(torch.eye(C, dtype=F64), train[i : i + 1])
        h = torch.diag(probs[i]) - torch.outer(probs[i], probs[i])
        ggn = ggn + j.T @ h @ j

    p = dataclasses.replace(p, curvature=Dense(ggn), prior=Isotropic(torch.tensor(1e-4, dtype=F64)))
    out = core.predict_outputs(p, x)
    a = MC(n_samples=20000)(Categorical(), out).probs
    b = Probit()(Categorical(), out).probs

    assert (a - b).abs().max() < 0.05


def test_mc_agrees_with_probit() -> None:
    mu = torch.randn(N, C, dtype=F64)
    cov = 0.5 * torch.eye(C, dtype=F64).expand(N, C, C)
    out = core.OutputGaussian(mu, core.PerInputCov(cov, (C,)))
    a = MC(n_samples=20000)(Categorical(), out).probs
    b = Probit()(Categorical(), out).probs

    assert (a - b).abs().max() < 0.03


def test_weight_samples_matches_mc_linearized() -> None:
    p, x = post(Categorical())
    a = WeightSamples(n_samples=20000)(p, x).probs
    b = Linearized(link=MC(n_samples=20000))(p, x).probs

    assert (a - b).abs().max() < 0.02


def test_sampling_class_dim_nonnegative() -> None:
    lik = Categorical(class_dim=1)
    f = torch.randn(7, N, C, dtype=F64)
    ref = torch.softmax(f, dim=2).mean(0)

    assert torch.allclose(_average(lik, f).probs, ref)
    mu = torch.randn(N, C, dtype=F64)
    out = core.OutputGaussian(mu, core.PerInputCov(torch.zeros(N, C, C, dtype=F64), (C,)))

    assert torch.allclose(MC(n_samples=3)(lik, out).probs, torch.softmax(mu, 1))


def test_gaussian_predictive_adds_noise() -> None:
    p, x = post(copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=F64))))
    out = core.predict_outputs(p, x)
    d = Linearized()(p, x)

    assert isinstance(d, Normal)
    f_var = out.cov.diagonal()

    assert torch.allclose(d.variance, f_var + 0.09)
    assert torch.allclose(out.cov.diagonal(), torch.diagonal(out.cov.per_input(), dim1=-2, dim2=-1))


def test_sampling_gaussian_moments() -> None:
    p, x = post(copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=F64))))
    ref = Linearized()(p, x)

    for pred in (WeightSamples(n_samples=20000), Linearized(link=MC(n_samples=20000))):
        d = pred(p, x)

        assert torch.allclose(d.mean, ref.mean, atol=0.05)
        assert torch.allclose(d.variance, ref.variance, rtol=0.1)


def test_sampling_rejects_other_likelihoods() -> None:
    p, x = post(object())

    with pytest.raises(CapabilityError, match="object"):
        WeightSamples(n_samples=2)(p, x)

    assert not MC().supports(object(), x)


def test_outputs_needs_caps() -> None:
    p, x = post(Categorical(), caps=Capabilities(vjp=True))

    with pytest.raises(CapabilityError, match="vmap"):
        core.predict_outputs(p, x, n_samples="exact")


def test_requires_and_resolve() -> None:
    # Exact output covariances vmap over inputs and output entries; sampled ones do not.
    exact = (Requirements(jvp=True, vmap=True), Requirements(vjp=True, vmap=True))
    sampled = (Requirements(jvp=True), Requirements(vjp=True, vjp_twice=True))

    assert Linearized().requires == Requirements(one_of=exact + sampled)
    assert Linearized(n_samples="exact").requires == Requirements(one_of=exact)
    assert Linearized(n_samples=8).requires == Requirements(one_of=sampled)
    assert Linearized(strategy="reverse-mode").requires == Requirements(
        vjp=True, one_of=(Requirements(vmap=True), Requirements(vjp_twice=True))
    )
    assert Linearized(strategy="reverse-mode", n_samples="exact").requires == Requirements(
        vjp=True, vmap=True
    )
    assert Linearized(strategy="reverse-mode", n_samples=8).requires == Requirements(
        vjp=True, vjp_twice=True
    )
    assert Linearized(strategy="forward-mode", n_samples="exact").requires == Requirements(
        jvp=True, vmap=True
    )
    assert Linearized(strategy="forward-mode", n_samples=8).requires == Requirements(jvp=True)
    p, _ = post(Categorical())
    r = Linearized().resolve(p)

    assert "strategy='reverse-mode'" in repr(r)
    assert isinstance(r.link, Probit)
    assert r.n_samples == "exact"
    assert "n_samples='exact'" in repr(r)

    # Before any forward, from the probe's event size and the layout: P = 12 < E = 20.
    full = Capabilities(vjp=True, vjp_twice=True, jvp=True, vmap=True)
    wide, _ = post(Categorical(), dataclasses.replace(full, event_size=20))
    r = Linearized().resolve(wide)

    assert r.strategy == "forward-mode" and r.n_samples == "exact" and isinstance(r.link, Probit)
    assert (
        Linearized().resolve(post(Categorical(), full)[0]).strategy == "reverse-mode"
    )  # P = 12 >= E = 3

    p2, _ = post(Categorical(), Capabilities())

    with pytest.raises(CapabilityError, match="vjp.*WeightSamples"):
        Linearized().resolve(p2)

    with pytest.raises(CapabilityError, match="vmap"):
        Linearized(n_samples="exact").resolve(post(Categorical(), Capabilities(vjp=True))[0])


def test_explicit_samples_resolve_to_the_double_vjp() -> None:
    """P = 12 >= E = 3 picks vjp for exact outputs. Sampled draws take a VJP of a VJP (one
    recorded forward) when the probe found vjp_twice, else one JVP each."""

    full, _ = post(Categorical(), FULL)

    assert Linearized().resolve(full).strategy == "reverse-mode"
    assert Linearized(n_samples=8).resolve(full).strategy == "reverse-mode"

    no_twice, _ = post(Categorical(), Capabilities(vjp=True, jvp=True, vmap=True))
    r = Linearized(n_samples=8).resolve(no_twice)

    assert r.strategy == "forward-mode" and r.n_samples == 8


def test_sampled_outputs_need_the_double_vjp() -> None:
    """vjp without vjp_twice (SDPA attention): refused before any forward pass."""

    p, x = post(Categorical(), Capabilities(vjp=True))
    calls = []
    model = p.model

    def counted(params, inputs):
        calls.append(1)
        return model(params, inputs)

    p = dataclasses.replace(p, model=ModelFn(counted, model.module))
    fix = r"(?s)vjp_twice.*attn_implementation=.eager.*WeightSamples"

    with pytest.raises(CapabilityError, match="vjp_twice"):
        core.predict_outputs(p, x, n_samples=8)

    with pytest.raises(CapabilityError, match="vjp_twice"):
        core.predict_outputs(p, x)

    with pytest.raises(CapabilityError, match=fix):
        Linearized().resolve(p)

    with pytest.raises(CapabilityError, match=fix):
        Linearized()(p, x)

    assert calls == []
    assert WeightSamples(n_samples=3)(p, x).probs.shape == (N, C)


def test_linearized_generator_seeds_the_sampled_outputs(monkeypatch) -> None:
    import laplacekit.predictive as predictive

    p, x = post(Categorical(), VJP)
    seen = []
    real = predictive.predict_outputs

    def spy(*args, **kwargs):
        seen.append(kwargs.get("generator"))
        return real(*args, **kwargs)

    monkeypatch.setattr(predictive, "predict_outputs", spy)
    gens = [torch.Generator().manual_seed(s) for s in (0, 0, 5)]
    base, same, other = (Linearized(link=Probit())(p, x, generator=g).probs for g in gens)

    assert torch.equal(base, same)
    assert not torch.equal(base, other)
    assert seen == gens


def test_default_link_on_sampled_cov_is_mc_over_the_draws() -> None:
    """Categorical: Probit on the exact covariance, MC over all draws of a SampleCov."""

    exact, _ = post(Categorical())

    assert isinstance(Linearized().resolve(exact).link, Probit)

    p, x = post(Categorical(), VJP)
    r = Linearized().resolve(p)

    assert r.link == MC(n_samples=64) and r.n_samples == 64
    assert "link=MC(n_samples=64" in repr(r)

    # By hand: the softmax averaged over mean + each draw J z_s.
    # The MC link takes all 64 draws, so their order doesn't matter.
    out = core.predict_outputs(p, x, n_samples=64, generator=torch.Generator().manual_seed(0))
    draws = out.cov.draws
    want = torch.softmax(out.mean.unsqueeze(0) + draws, dim=-1).mean(0)
    got = Linearized()(p, x, generator=torch.Generator().manual_seed(0)).probs

    torch.testing.assert_close(got, want)

    # An explicit link is kept.
    assert isinstance(Linearized(link=Probit()).resolve(p).link, Probit)


@pytest.mark.parametrize("s", [2, 5])
def test_sampled_outputs_centre_and_rescale_the_noise(s) -> None:
    """Draws are J (z_s - mean z) sqrt(S / (S - 1)), with J z = x z^T for this linear model."""

    p, x = post(Categorical(), VJP)

    # `outputs` draws one seed from its generator and pushes the noise of that seed.
    noise_gen = core.spawn_generator(torch.Generator().manual_seed(7))
    z = p.sample_centered(s, generator=noise_gen)["fc.weight"]  # [S, C, D]
    t = (z - z.mean(0)) * math.sqrt(s / (s - 1))
    want = torch.einsum("nd,scd->snc", x, t)

    got = core.predict_outputs(
        p, x, n_samples=s, generator=torch.Generator().manual_seed(7)
    ).cov.draws

    torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-12)


def test_resolve_without_vmap_uses_samples() -> None:
    p, x = post(Categorical(), Capabilities(vjp=True, vjp_twice=True, jvp=True))
    r = Linearized().resolve(p)

    assert r.n_samples == 64
    assert "n_samples=64" in repr(r)
    assert Linearized().requires.satisfies(p.caps)
    assert Linearized()(p, x).probs.shape == (N, C)


def test_large_event_resolves_to_samples() -> None:
    p, _ = post(Categorical(), Capabilities(vjp=True, vjp_twice=True, vmap=True, event_size=5000))

    assert Linearized().resolve(p).n_samples == 64
    assert Linearized(n_samples="exact").resolve(p).n_samples == "exact"


def test_mc_link_with_sampled_cov(monkeypatch) -> None:
    import laplacekit.predictive as predictive

    p, x = post(Categorical(), VJP)

    assert Linearized(link=MC(n_samples=100)).resolve(p).n_samples == 100
    assert Linearized(link=MC(n_samples=10)).resolve(p).n_samples == 64

    # The link asks for exactly as many draws as SampleCov holds, and no more.
    held = []
    real = predictive.predict_outputs

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        held.append(out.cov.draws.shape[0])
        return out

    monkeypatch.setattr(predictive, "predict_outputs", spy)
    probs = Linearized(link=MC(n_samples=100))(p, x).probs

    assert held == [100] and probs.shape == (N, C)
    monkeypatch.undo()

    with pytest.raises(ValueError, match="10"):
        Linearized(link=MC(n_samples=100), n_samples=10)(p, x)

    # With many draws, it agrees with MC on the exact output covariance.
    q, _ = post(Categorical())
    a = Linearized(link=MC(n_samples=20000))(p, x).probs
    b = Linearized(link=MC(n_samples=20000))(q, x).probs

    assert (a - b).abs().max() < 1e-2


def test_probit_on_sampled_cov_matches_its_blocks() -> None:
    p, x = post(Categorical(), VJP)
    out = core.predict_outputs(p, x, n_samples=200)

    assert isinstance(out.cov, core.SampleCov)

    dense = core.OutputGaussian(out.mean, core.PerInputCov(out.cov.per_input(), (C,)))

    torch.testing.assert_close(
        Probit()(Categorical(), out).probs, Probit()(Categorical(), dense).probs
    )


def test_probit_on_sampled_cov_reads_only_the_diagonal(monkeypatch) -> None:
    g = torch.Generator().manual_seed(4)
    mu = torch.randn(2, 3, 5000, generator=g, dtype=F64)
    draws = torch.randn(8, 2, 3, 5000, generator=g, dtype=F64)
    cov = core.SampleCov(draws)

    monkeypatch.setattr(core.SampleCov, "per_input", lambda self: pytest.fail("per_input"))
    probs = Probit()(Categorical(class_dim=-1), core.OutputGaussian(mu, cov)).probs

    assert probs.shape == (2, 3, 5000)
    assert torch.isfinite(probs).all()


@core.pytree
class _OtherGaussian:
    """A WeightPosterior that is not a LaplacePosterior, as a plugin would write."""

    inner: LaplacePosterior

    @property
    def mean(self):
        return self.inner.mean

    @property
    def model(self):
        return self.inner.model

    @property
    def likelihood(self):
        return self.inner.likelihood

    @property
    def layout(self):
        return self.inner.layout

    @property
    def caps(self):
        return self.inner.caps

    def precision(self):
        return self.inner.precision()

    def cov_mv(self, v, /):
        return self.inner.cov_mv(v)

    def cov_sqrt_mv(self, v, /):
        return self.inner.cov_sqrt_mv(v)

    def sample(self, n, /, *, generator=None):
        return self.inner.sample(n, generator=generator)

    def sample_centered(self, n, /, *, generator=None):
        return self.inner.sample_centered(n, generator=generator)

    @property
    def variance(self):
        return self.inner.variance

    def log_prob(self, params, /):
        return self.inner.log_prob(params)

    def location(self, like, /):
        return self.inner.location(like)

    def add_precision_to(self, op, /):
        return self.inner.add_precision_to(op)


def test_resolve_takes_any_weight_gaussian() -> None:
    p, x = post(Categorical())
    other = _OtherGaussian(p)

    assert isinstance(other, core.WeightPosterior)

    r = Linearized().resolve(other)

    assert isinstance(r.link, Probit)
    torch.testing.assert_close(Linearized()(other, x).probs, Linearized()(p, x).probs)


def test_unsupported_link_fails_before_the_pushforward(monkeypatch) -> None:
    import laplacekit.predictive as predictive

    p, x = post(Gaussian(sigma=1.0))

    def boom(*args, **kwargs):
        raise AssertionError("outputs ran")

    monkeypatch.setattr(predictive, "predict_outputs", boom)

    with pytest.raises(CapabilityError, match="Probit does not support Gaussian"):
        Linearized(link=Probit())(p, x)


class _InPlace(torch.nn.Module):
    """Writes its output in place into a fresh tensor: no vmap."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(D, 5)
        self.l2 = torch.nn.Linear(5, C)

    def forward(self, x: Tensor) -> Tensor:
        out = torch.zeros(x.shape[0], C, dtype=x.dtype)
        out.add_(self.l2(torch.tanh(self.l1(x))))

        return out


def test_no_vmap_model_predicts_linearized() -> None:
    import laplacekit as lk

    torch.manual_seed(0)
    module = _InPlace().double()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(40, D, generator=g, dtype=F64)
    y = torch.randint(0, C, (40,), generator=g)
    post = lk.laplace(module, [(x, y)], likelihood=Categorical(), hyper_tuner=None)

    assert not post.caps.vmap
    assert Linearized().resolve(post).n_samples == 64
    assert lk.predict(post, x[:N]).probs.shape == (N, C)

    # Reference: J by autograd without vmap; Sigma column by column from cov_mv.
    names = list(post.mean)
    shapes = [post.mean[n].shape for n in names]
    sizes = [post.mean[n].numel() for n in names]
    w = torch.cat([post.mean[n].reshape(-1) for n in names])

    def unflat(v: Tensor) -> dict[str, Tensor]:
        parts = torch.split(v, sizes)
        return {n: t.reshape(sh) for n, t, sh in zip(names, parts, shapes, strict=True)}

    def flat(p: dict[str, Tensor]) -> Tensor:
        return torch.cat([p[n].reshape(-1) for n in names])

    def f(v: Tensor, xi: Tensor) -> Tensor:
        return post.model(unflat(v), xi.unsqueeze(0))[0]

    jac = torch.stack(
        [torch.autograd.functional.jacobian(partial(f, xi=xi), w) for xi in x[:N]]
    )  # [N, C, P]
    eye = torch.eye(w.numel(), dtype=F64)
    sigma = torch.stack([flat(post.cov_mv(unflat(e))) for e in eye])
    ref = torch.einsum("ncp,pq,ncq->nc", jac, sigma, jac)
    out = lk.predict_outputs(post, x[:N], n_samples=20000)

    assert isinstance(out.cov, core.SampleCov)
    assert (out.cov.diagonal() - ref).norm() / ref.norm() < 2e-2


def test_linearized_chunk_reaches_outputs(monkeypatch) -> None:
    import laplacekit.predictive as predictive

    p, x = post(Categorical())
    chunks = []
    real = predictive.predict_outputs

    def spy(*args, **kwargs):
        chunks.append(kwargs.get("chunk_size"))
        return real(*args, **kwargs)

    monkeypatch.setattr(predictive, "predict_outputs", spy)
    one = Linearized(chunk_size=1)(p, x).probs
    default = Linearized()(p, x).probs

    assert chunks == [1, None]
    assert "chunk_size=1" in repr(Linearized(chunk_size=1))
    torch.testing.assert_close(one, default)


def _counting(p: LaplacePosterior) -> tuple[LaplacePosterior, list[int]]:
    calls: list[int] = []
    model = p.model

    def counted(params, inputs):
        calls.append(1)
        return model(params, inputs)

    return dataclasses.replace(p, model=ModelFn(counted, model.module)), calls


@pytest.mark.parametrize(
    ("n_samples", "link", "match"),
    [
        (1, None, "n_samples >= 2.*got 1"),
        (0, None, "n_samples >= 2.*got 0"),
        (10, MC(n_samples=100), "10.*100|100.*10"),
    ],
)
def test_linearized_rejects_bad_samples_before_forward(n_samples, link, match) -> None:
    q, x = post(Categorical(), VJP)
    p, calls = _counting(q)
    pred = (
        Linearized(n_samples=n_samples)
        if link is None
        else Linearized(link=link, n_samples=n_samples)
    )

    with pytest.raises(ValueError, match=match):
        pred.resolve(p)

    with pytest.raises(ValueError, match=match):
        pred(p, x)

    assert calls == []


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_linearized_rejects_chunk_zero_before_forward(chunk_size: int) -> None:
    q, x = post(Categorical(), VJP)
    p, calls = _counting(q)
    pred = Linearized(chunk_size=chunk_size)

    with pytest.raises(
        ValueError, match=f"chunk_size must be an int of at least 1, got {chunk_size}"
    ):
        pred.resolve(p)

    with pytest.raises(
        ValueError, match=f"chunk_size must be an int of at least 1, got {chunk_size}"
    ):
        pred(p, x)

    assert calls == []


def test_samples_rule_has_one_message() -> None:
    """`outputs` and `Linearized` refuse n_samples < 2 with the same words."""

    q, x = post(Categorical(), VJP)

    with pytest.raises(ValueError) as by_outputs:
        core.predict_outputs(q, x, n_samples=1)

    with pytest.raises(ValueError) as by_linearized:
        Linearized(n_samples=1).resolve(q)

    assert str(by_outputs.value) == str(by_linearized.value)


def test_exact_fits_policy() -> None:
    full = Capabilities(vjp=True, vmap=True)

    assert core.exact_fits(full, None)
    assert core.exact_fits(full, core.MAX_DENSE_EVENT)
    assert not core.exact_fits(full, core.MAX_DENSE_EVENT + 1)
    assert not core.exact_fits(Capabilities(vjp=True), 3)
    assert not core.exact_fits(Capabilities(vmap=True), 3)


FULL = Capabilities(vjp=True, vjp_twice=True, jvp=True, vmap=True)


@pytest.mark.parametrize(("shape", "class_dim"), [((C,), -1), ((T, C), -1), ((C, T), 1)])
def test_probit_with_factor_cov_matches_per_input(shape, class_dim, monkeypatch) -> None:
    g = torch.Generator().manual_seed(8)
    mu = torch.randn(N, *shape, generator=g, dtype=F64)
    factor = torch.randn(N, *shape, 6, generator=g, dtype=F64)
    e = math.prod(shape)
    flat = factor.reshape(N, e, 6)
    dense = core.OutputGaussian(mu, core.PerInputCov(flat @ flat.mT, shape))
    lik = Categorical(class_dim=class_dim)
    want = Probit()(lik, dense).probs

    monkeypatch.setattr(core.FactorCov, "per_input", lambda self: pytest.fail("per_input"))
    got = Probit()(lik, core.OutputGaussian(mu, core.FactorCov(factor))).probs

    torch.testing.assert_close(got, want, rtol=0, atol=1e-10)


def _image_post(lik) -> tuple[LaplacePosterior, Tensor]:
    """Outputs [N, 2, 2, 2] from a tanh layer with P = 8 * D weights."""

    g = torch.Generator().manual_seed(0)
    w = torch.randn(8, D, generator=g, dtype=F64)
    a = torch.randn(8 * D, 8 * D, generator=g, dtype=F64)
    p = LaplacePosterior(
        mean={"fc.weight": w},
        curvature=Dense(a @ a.T / (8 * D)),
        prior=Isotropic(torch.tensor(1.0, dtype=F64)),
        n_data=torch.tensor(5.0, dtype=F64),
        model=ModelFn(lambda p, x: torch.tanh(x @ p["fc.weight"].T).reshape(-1, 2, 2, 2)),
        likelihood=lik,
        layout=Layout.by_module({"fc.weight": w}),
        caps=FULL,
    )

    return p, torch.randn(N, D, generator=g, dtype=F64)


def test_linearized_exact_link_factor_equals_per_input() -> None:
    p, x = _image_post(
        copy.replace(Gaussian(event_dims=3), sigma=torch.as_tensor(torch.tensor(0.3, dtype=F64)))
    )

    assert isinstance(
        core.predict_outputs(p, x, strategy="forward-mode", n_samples="exact").cov, core.FactorCov
    )

    a = Linearized(strategy="forward-mode", n_samples="exact")(p, x)
    b = Linearized(strategy="reverse-mode", n_samples="exact")(p, x)

    assert a.event_shape == b.event_shape == (2, 2, 2)
    torch.testing.assert_close(a.mean, b.mean, rtol=0, atol=1e-12)
    torch.testing.assert_close(a.variance, b.variance, rtol=0, atol=1e-10)


def test_linearized_mc_factor_matches_per_input() -> None:
    p, x = _image_post(
        copy.replace(Gaussian(event_dims=3), sigma=torch.as_tensor(torch.tensor(0.3, dtype=F64)))
    )
    link = MC(n_samples=4000)
    a = Linearized(link=link, strategy="forward-mode", n_samples="exact")(p, x)
    b = Linearized(link=link, strategy="reverse-mode", n_samples="exact")(p, x)
    exact = Linearized(strategy="reverse-mode", n_samples="exact")(p, x)
    scale = exact.variance.sqrt()

    assert ((a.mean - exact.mean) / scale).abs().max() < 0.15
    assert ((b.mean - exact.mean) / scale).abs().max() < 0.15
    assert (a.variance / exact.variance - 1).abs().max() < 0.15
    assert (b.variance / exact.variance - 1).abs().max() < 0.15


def test_resolve_names_a_posterior_of_neither_kind() -> None:
    _, x = post(Categorical())
    prior = Isotropic(torch.tensor(1.0, dtype=F64))

    with pytest.raises(
        TypeError,
        match="Linearized needs a weight-space or a function-space posterior.*got Isotropic",
    ):
        Linearized().resolve(prior)

    with pytest.raises(TypeError, match="got Isotropic"):
        lk.predict(prior, x)


def test_gaussian_predictive_noise_is_on_the_device_of_f(monkeypatch) -> None:
    """The default sigma is a CPU tensor; the noise added to the output variance is built on
    the outputs' device. The meta device stands in for a GPU on a CPU-only machine."""

    monkeypatch.setattr(torch.distributions.Distribution, "_validate_args", False)
    meta = torch.device("meta")
    lik = Gaussian()
    mu = torch.empty(5, 2, device=meta)
    out = core.OutputGaussian(mu, core.PerInputCov(torch.empty(5, 2, 2, device=meta), (2,)))

    assert lk.predictive.Exact()(lik, out).scale.device == meta
    assert _average(lik, torch.empty(7, 5, 2, device=meta)).scale.device == meta


@pytree
class _Poisson:
    """A test likelihood outside the built-ins: Poisson counts with rate exp(f)."""

    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return self.distribution(f).log_prob(y)

    def distribution(self, f: Tensor, /) -> Distribution:
        return Independent(Poisson(f.exp()), 1)

    def default_link(self, n_samples: int | Literal["exact"], /) -> core.LinkApprox:
        return _LogNormalMean()

    def average(self, f: Tensor, /) -> Distribution:
        return Independent(Poisson(f.exp().mean(0)), 1)


@pytree
class _LogNormalMean:
    """Poisson with the rate E[exp f] = exp(mu + v / 2), exact for f ~ N(mu, v)."""

    def supports(self, lik: object, f_mu: Tensor, /) -> bool:
        return isinstance(lik, _Poisson)

    def __call__(self, lik: object, out: core.OutputGaussian, /, *, generator=None) -> Distribution:
        mu = out.mean
        assert isinstance(mu, Tensor)
        v = out.cov.diagonal().reshape(mu.shape)

        return Independent(Poisson(torch.exp(mu + v / 2)), 1)


@pytree
class _NoLink:
    """A likelihood with a distribution but no default link and no sample average."""

    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return self.distribution(f).log_prob(y)

    def distribution(self, f: Tensor, /) -> Distribution:
        return Independent(Poisson(f.exp()), 1)


def _dense_output_cov(p: LaplacePosterior, x: Tensor) -> Tensor:
    """J Sigma J^T per input, with J from `jacrev` and Sigma the dense inverse precision."""

    w = p.mean["fc.weight"]
    jac = torch.func.jacrev(lambda w: x @ w.T)(w).reshape(N, C, C * D)  # [N, C, C*D]

    return jac @ sigma(p) @ jac.mT


def test_likelihood_hooks_are_core_only() -> None:
    # Like the other likelihood hooks (HasDistribution, HasTargetCheck, ...).
    for name in ("HasDefaultLink", "HasSampleAverage"):
        assert name in core.__all__
        assert name not in lk.__all__


def test_protocols_are_runtime_checkable() -> None:
    assert isinstance(_Poisson(), core.HasDefaultLink)
    assert isinstance(_Poisson(), core.HasSampleAverage)
    assert not isinstance(_NoLink(), core.HasDefaultLink)
    assert not isinstance(_NoLink(), core.HasSampleAverage)
    assert not isinstance(Categorical(), core.HasDefaultLink | core.HasSampleAverage)


def test_linearized_uses_likelihood_default_link() -> None:
    """The likelihood's link gets the exact output Gaussian, compared with the closed form."""

    p, x = post(_Poisson())

    assert Linearized().resolve(p).link == _LogNormalMean()

    d = lk.predict(p, x, Linearized())

    # E[exp f] for f ~ N(mu, v), in closed form.
    mu = x @ p.mean["fc.weight"].T
    v = torch.diagonal(_dense_output_cov(p, x), dim1=-2, dim2=-1)
    want = torch.exp(mu + v / 2)

    assert isinstance(d, Independent) and isinstance(d.base_dist, Poisson)
    torch.testing.assert_close(d.mean, want, rtol=1e-10, atol=0)

    # The covariance matters here: without it the rate is off by far more than the tolerance.
    assert ((torch.exp(mu) - want) / want).abs().max() > 0.1


def test_weight_samples_uses_average() -> None:
    p, x = post(_Poisson())
    n = 30

    d = WeightSamples(n_samples=n)(p, x, generator=torch.Generator().manual_seed(5))

    # By hand: the weight draws pushed through the model, then exp averaged.
    # WeightSamples draws from a generator spawned from the one it is given.
    gen = core.spawn_generator(torch.Generator().manual_seed(5))
    draws = p.sample(n, generator=gen)["fc.weight"]  # [n, C, D]
    f = torch.stack([x @ draws[i].T for i in range(n)])
    want = f.exp().mean(0)

    assert isinstance(d, Independent) and isinstance(d.base_dist, Poisson)
    torch.testing.assert_close(d.mean, want, rtol=1e-12, atol=1e-12)


def test_mc_link_accepts_has_sample_average() -> None:
    p, x = post(_Poisson())
    mc = MC(n_samples=50)

    assert mc.supports(_Poisson(), x)

    # Exact outputs draw nothing, so the link spawns its generator from the given one.
    d = Linearized(link=mc)(p, x, generator=torch.Generator().manual_seed(6))

    # By hand: the same `PerInputCov` draws, from the jacrev covariance, then exp averaged.
    mu = x @ p.mean["fc.weight"].T
    out = core.OutputGaussian(mu, core.PerInputCov(_dense_output_cov(p, x), (C,)))
    f = mu + out.cov.sample(50, generator=core.spawn_generator(torch.Generator().manual_seed(6)))
    want = f.exp().mean(0)

    assert isinstance(d.base_dist, Poisson)
    torch.testing.assert_close(d.mean, want, rtol=1e-8, atol=1e-10)


def test_no_link_names_both_protocols() -> None:
    p, x = post(_NoLink())

    with pytest.raises(CapabilityError, match="no default link for _NoLink.*HasDefaultLink"):
        Linearized().resolve(p)

    with pytest.raises(CapabilityError, match="_NoLink.*HasSampleAverage"):
        WeightSamples(n_samples=2)(p, x)

    with pytest.raises(CapabilityError, match="_NoLink.*HasSampleAverage"):
        _average(_NoLink(), torch.zeros(2, N, C, dtype=F64))

    assert not MC().supports(_NoLink(), x)


def test_builtin_links_unchanged() -> None:
    cat, x = post(Categorical())
    gauss, _ = post(copy.replace(Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=F64))))

    assert repr(Linearized().resolve(cat).link) == "Probit()"
    assert repr(Linearized().resolve(gauss).link) == "Exact()"

    # Values: Exact is f_var + sigma^2 with the jacrev covariance; Probit the moderated
    # softmax on centred logits.
    cov = _dense_output_cov(gauss, x)
    mu = x @ gauss.mean["fc.weight"].T
    d = Linearized()(gauss, x)

    torch.testing.assert_close(d.mean, mu)
    torch.testing.assert_close(d.variance, torch.diagonal(cov, dim1=-2, dim2=-1) + 0.09)

    cov = _dense_output_cov(cat, x)
    mu = x @ cat.mean["fc.weight"].T
    centre = torch.eye(C, dtype=F64) - 1 / C
    var = torch.diagonal(centre @ cov @ centre, dim1=-2, dim2=-1)
    g = mu - mu.mean(-1, keepdim=True)
    want = torch.softmax(g / torch.sqrt(1 + math.pi / 8 * var), -1)

    torch.testing.assert_close(Linearized()(cat, x).probs, want)


@pytree
class _WantsManyDraws:
    """A likelihood whose default link needs more draws than `Linearized` resolves to."""

    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return self.distribution(f).log_prob(y)

    def distribution(self, f: Tensor, /) -> Distribution:
        return Independent(Poisson(f.exp()), 1)

    def default_link(self, n_samples: int | Literal["exact"], /) -> core.LinkApprox:
        return MC(n_samples=1000)

    def average(self, f: Tensor, /) -> Distribution:
        return Independent(Poisson(f.exp().mean(0)), 1)


@pytest.mark.parametrize("n_samples", [None, 64])
def test_default_mc_link_too_large_for_samples_fails_at_resolve(n_samples, monkeypatch) -> None:
    p, x = post(_WantsManyDraws(), VJP)

    def no_outputs(*args, **kwargs):
        raise AssertionError("the pushforward ran")

    monkeypatch.setattr("laplacekit.predictive.predict_outputs", no_outputs)

    with pytest.raises(ValueError, match=r"MC\(n_samples=1000\) needs at least 1000 draws"):
        Linearized(n_samples=n_samples).resolve(p)

    with pytest.raises(ValueError, match="needs at least 1000 draws"):
        Linearized(n_samples=n_samples)(p, x)


# --- Bernoulli -----------------------------------------------------------------------------


def test_bernoulli_default_link_follows_samples() -> None:
    exact, _ = post(Bernoulli())
    sampled, _ = post(Bernoulli(), VJP)

    assert Linearized().resolve(exact).link == Probit()
    assert Linearized().resolve(sampled).link == MC(n_samples=64)
    assert Linearized(n_samples=80).resolve(sampled).link == MC(n_samples=80)


@pytest.mark.parametrize("event_dims", [0, 1])
def test_bernoulli_probit_is_the_sigmoid_probit(event_dims) -> None:
    """sigmoid(mu / sqrt(1 + pi/8 var)) per output, with var from the jacrev covariance."""

    p, x = post(Bernoulli(event_dims=event_dims))
    d = Linearized()(p, x)

    mu = x @ p.mean["fc.weight"].T
    var = torch.diagonal(_dense_output_cov(p, x), dim1=-2, dim2=-1)
    want = 1 / (1 + torch.exp(-mu / torch.sqrt(1 + math.pi / 8 * var)))

    torch.testing.assert_close(d.mean, want)
    assert d.batch_shape == ((N, C) if event_dims == 0 else (N,))
    assert d.event_shape == (() if event_dims == 0 else (C,))


def test_bernoulli_probit_reads_the_diagonal_of_any_cov() -> None:
    torch.manual_seed(0)
    mu = torch.randn(N, C, dtype=F64)
    draws = torch.randn(N, C, 2, dtype=F64)
    out = core.OutputGaussian(mu, core.FactorCov(draws))
    var = (draws**2).sum(-1)
    want = 1 / (1 + torch.exp(-mu / torch.sqrt(1 + math.pi / 8 * var)))

    torch.testing.assert_close(Probit()(Bernoulli(), out).mean, want)
    assert Probit().supports(Bernoulli(), mu)


def test_bernoulli_mc_averages_the_sigmoid() -> None:
    p, x = post(Bernoulli(), VJP)
    d = Linearized()(p, x, generator=torch.Generator().manual_seed(2))

    # By hand: the sigmoid averaged over mean + each draw (MC takes all 64, in any order).
    out = core.predict_outputs(p, x, n_samples=64, generator=torch.Generator().manual_seed(2))
    f = out.mean.unsqueeze(0) + out.cov.draws
    want = (1 / (1 + torch.exp(-f))).mean(0)

    assert isinstance(d, torch.distributions.Bernoulli)
    torch.testing.assert_close(d.mean, want)


def test_bernoulli_weight_samples_average_the_sigmoid() -> None:
    p, x = post(Bernoulli(event_dims=1))
    n = 30
    d = WeightSamples(n_samples=n)(p, x, generator=torch.Generator().manual_seed(5))

    gen = core.spawn_generator(torch.Generator().manual_seed(5))
    draws = p.sample(n, generator=gen)["fc.weight"]
    f = torch.stack([x @ draws[i].T for i in range(n)])
    want = (1 / (1 + torch.exp(-f))).mean(0)

    assert isinstance(d, Independent) and d.event_shape == (C,)
    torch.testing.assert_close(d.mean, want)


def _log_sigmoid64(z: Tensor) -> Tensor:
    """log sigmoid(z) in float64: -log(1 + exp(-z)), split by sign so neither side overflows."""

    z = z.double()

    return torch.where(z > 0, -torch.log1p(torch.exp(-z)), z - torch.log1p(torch.exp(z)))


def test_bernoulli_probit_stays_finite_when_saturated() -> None:
    mu = torch.tensor([[20.0, -20.0, 3.0]])
    var = torch.tensor([[0.01, 0.01, 0.5]])
    out = _out(mu, torch.diag_embed(var))
    d = Probit()(Bernoulli(), out)

    z = mu.double() / torch.sqrt(1 + math.pi / 8 * var.double())
    one, zero = torch.ones_like(mu), torch.zeros_like(mu)

    assert mu.dtype == torch.float32
    torch.testing.assert_close(d.log_prob(one).double(), _log_sigmoid64(z))
    torch.testing.assert_close(d.log_prob(zero).double(), _log_sigmoid64(-z))


def test_bernoulli_mc_stays_finite_when_saturated() -> None:
    g = torch.Generator().manual_seed(0)
    f = torch.tensor([20.0, -20.0, 3.0]) + 0.1 * torch.randn(50, 1, 3, generator=g)  # [S, N, C]
    d = _average(Bernoulli(), f)

    # log of the mean probability, in float64, from the log-sigmoids.
    f64 = f.double()
    p1 = torch.logsumexp(_log_sigmoid64(f64), 0) - math.log(50)
    p0 = torch.logsumexp(_log_sigmoid64(-f64), 0) - math.log(50)
    one = torch.ones(1, 3)

    assert f.dtype == torch.float32
    torch.testing.assert_close(d.log_prob(one).double(), p1)
    torch.testing.assert_close(d.log_prob(1 - one).double(), p0)


@pytree
class _MyBernoulli(Bernoulli):
    def default_link(self, n_samples: int | Literal["exact"], /) -> core.LinkApprox:
        return MC(n_samples=7)


def test_bernoulli_subclass_default_link_is_used() -> None:
    p, _ = post(_MyBernoulli())

    assert isinstance(Bernoulli(), core.HasDefaultLink)
    assert Linearized().resolve(p).link == MC(n_samples=7)


def test_bernoulli_predictive_refuses_targets_outside_zero_one() -> None:
    p, x = post(Bernoulli())
    d = Linearized()(p, x)

    with pytest.raises(ValueError, match="support"):
        d.log_prob(torch.full((N, C), 2.0, dtype=F64))


def _binary_case(shape: tuple[int, ...], cd: int, kind: str) -> tuple[Tensor, object, Tensor]:
    """Logits with two classes on axis `cd`, a covariance of `kind`, and its dense blocks."""

    g = torch.Generator().manual_seed(11)
    mu = 3 * torch.randn(N, *shape, generator=g, dtype=F64)
    e = math.prod(shape)

    if kind == "draws":
        draws = torch.randn(40, N, *shape, generator=g, dtype=F64)
        flat = draws.reshape(40, N, e)
        dense = torch.einsum("sne,snf->nef", flat, flat) / 40

        return mu, core.SampleCov(draws), dense

    factor = torch.randn(N, *shape, 5, generator=g, dtype=F64)
    flat = factor.reshape(N, e, 5)
    dense = flat @ flat.mT
    cov = core.FactorCov(factor) if kind == "factor" else core.PerInputCov(dense, shape)

    return mu, cov, dense


@pytest.mark.parametrize("kind", ["blocks", "factor", "draws"])
@pytest.mark.parametrize(("shape", "cd"), [((2,), 1), ((T, 2), 2), ((2, T), 1)])
def test_probit_binary_categorical_is_probit_of_difference(kind, shape, cd) -> None:
    mu, cov, dense = _binary_case(shape, cd, kind)
    got = Probit()(Categorical(class_dim=cd), core.OutputGaussian(mu, cov)).probs  # classes last

    # By hand: each position's 2x2 block from the dense covariance, d = f_1 - f_2.
    e = math.prod(shape)
    index = torch.arange(e).reshape(shape).movedim(cd - 1, -1).reshape(-1, 2)  # [P, 2]
    m = mu.reshape(N, e)

    for i in range(N):
        for p, (a, b) in enumerate(index.tolist()):
            v_d = dense[i, a, a] + dense[i, b, b] - 2 * dense[i, a, b]
            z = (m[i, a] - m[i, b]) / torch.sqrt(1 + math.pi / 8 * v_d)
            want = torch.stack([torch.sigmoid(z), torch.sigmoid(-z)])

            torch.testing.assert_close(got.reshape(N, -1, 2)[i, p], want)


def test_binary_probit_closer_to_mc() -> None:
    # mu_d = 2 and v_d = 16 + 16 = 32.
    mu = torch.tensor([[1.0, -1.0]], dtype=F64)
    cov = torch.diag(torch.tensor([16.0, 16.0], dtype=F64)).unsqueeze(0)
    got = Probit()(Categorical(), _out(mu, cov)).probs[0, 0]

    g = torch.Generator().manual_seed(0)
    d = 2 + math.sqrt(32) * torch.randn(1_000_000, generator=g, dtype=F64)
    mc = torch.sigmoid(d).mean()

    # The centred softmax probit: g = [1, -1], Var(g_k) = 8 each.
    old = torch.sigmoid(torch.tensor(2 / math.sqrt(1 + math.pi / 8 * 8), dtype=F64))

    assert abs(mc.item() - 0.632) < 0.001
    assert abs(old.item() - 0.728) < 0.001
    assert (got - mc).abs() < 0.005
    assert (old - mc).abs() > 0.005


def test_binary_probit_stays_finite_when_saturated() -> None:
    mu = torch.tensor([[30.0, -30.0], [-25.0, 25.0], [2.0, 1.0]])
    cov = torch.diag_embed(torch.tensor([[0.01, 0.01], [0.02, 0.0], [0.5, 0.5]]))
    d = Probit()(Categorical(), _out(mu, cov))

    v_d = cov.double().diagonal(dim1=-2, dim2=-1).sum(-1)
    z = (mu[:, 0] - mu[:, 1]).double() / torch.sqrt(1 + math.pi / 8 * v_d)

    assert mu.dtype == torch.float32
    torch.testing.assert_close(d.log_prob(torch.zeros(3)).double(), _log_sigmoid64(z))
    torch.testing.assert_close(d.log_prob(torch.ones(3)).double(), _log_sigmoid64(-z))


def test_probit_one_class_points_to_outputs() -> None:
    mu = torch.randn(N, 1, dtype=F64)
    out = _out(mu, torch.ones(N, 1, 1, dtype=F64))
    msg = (
        "a Categorical over one class gives it probability 1; for a scalar output such as a "
        "reward, use lk.predict_outputs(posterior, inputs)"
    )

    with pytest.raises(CapabilityError, match=re.escape(msg)):
        Probit()(Categorical(), out)


@pytest.mark.parametrize("kind", ["blocks", "factor", "draws"])
@pytest.mark.parametrize(("shape", "cd"), [((C,), 1), ((T, C), 2), ((C, T), 1)])
def test_per_position_moments_match_dense_blocks(kind, shape, cd, monkeypatch) -> None:
    g = torch.Generator().manual_seed(12)
    mu = torch.randn(N, *shape, generator=g, dtype=F64)
    e = math.prod(shape)

    if kind == "draws":
        draws = torch.randn(30, N, *shape, generator=g, dtype=F64)
        flat = draws.reshape(30, N, e)
        dense = torch.einsum("sne,snf->nef", flat, flat) / 30
        cov = core.SampleCov(draws)
    else:
        factor = torch.randn(N, *shape, 4, generator=g, dtype=F64)
        flat = factor.reshape(N, e, 4)
        dense = flat @ flat.mT
        cov = core.FactorCov(factor) if kind == "factor" else core.PerInputCov(dense, shape)

    if kind != "blocks":
        monkeypatch.setattr(type(cov), "per_input", lambda self: pytest.fail("per_input"))

    m, s1, ss, diag = _per_position(core.OutputGaussian(mu, cov), mu, cd)

    # By hand: each position's [K, K] block of the dense covariance.
    index = torch.arange(e).reshape(shape).movedim(cd - 1, -1).reshape(-1, C)  # [P, K]
    blocks = dense[:, index[:, :, None], index[:, None, :]]  # [N, P, K, K]
    ones = torch.ones(C, dtype=F64)

    torch.testing.assert_close(m, mu.reshape(N, e)[:, index])
    torch.testing.assert_close(s1, blocks @ ones)
    torch.testing.assert_close(ss, blocks @ ones @ ones)
    torch.testing.assert_close(diag, blocks.diagonal(dim1=-2, dim2=-1))


def test_exact_link_adds_per_output_noise() -> None:
    """Exact and the sample average add sigma² of each output to its variance, and refuse
    a sigma that doesn't fit the outputs."""

    g = torch.Generator().manual_seed(0)
    mu = torch.randn(N, 2, generator=g, dtype=F64)
    root = torch.randn(N, 2, 2, generator=g, dtype=F64)
    cov = root @ root.mT
    out = core.OutputGaussian(mu, core.PerInputCov(cov, (2,)))
    sigma = torch.tensor([0.2, 0.9], dtype=F64)
    var = torch.diagonal(cov, dim1=-2, dim2=-1)

    d = lk.predictive.Exact()(copy.replace(Gaussian(), sigma=torch.as_tensor(sigma)), out)

    torch.testing.assert_close(d.mean, mu)
    torch.testing.assert_close(d.variance, var + torch.tensor([0.04, 0.81], dtype=F64))

    f = torch.randn(64, N, 2, generator=g, dtype=F64)
    avg = _average(copy.replace(Gaussian(), sigma=torch.as_tensor(sigma)), f)

    torch.testing.assert_close(avg.variance, f.var(0) + torch.tensor([0.04, 0.81], dtype=F64))

    wrong = copy.replace(Gaussian(), sigma=torch.as_tensor(torch.ones(3, dtype=F64)))
    message = r"Gaussian sigma of shape torch.Size\(\[3\]\) doesn't fit"

    with pytest.raises(core.StructureError, match=message):
        lk.predictive.Exact()(wrong, out)

    with pytest.raises(core.StructureError, match=message):
        _average(wrong, f)


# The Laplace bridge (Hobbhahn et al., 2022).


def _naive_bridge(mu: Tensor, cov: Tensor, normalized: bool = False) -> Tensor:
    """The bridge from dense [N, K, K] blocks, by the formula, with plain exponentials."""

    k = mu.shape[-1]
    s1 = cov.sum(-1)
    ss = s1.sum(-1, keepdim=True)
    m = mu - s1 / ss * mu.sum(-1, keepdim=True)
    v = cov.diagonal(dim1=-2, dim2=-1) - s1**2 / ss

    if normalized:
        c = v.sum(-1, keepdim=True) / k / math.sqrt(k / 2)
        m = m / c.sqrt()

    alpha = (1 - 2 / k + m.exp() * (-m).exp().sum(-1, keepdim=True) / k**2) / v

    return alpha / alpha.sum(-1, keepdim=True)


@pytest.mark.parametrize("normalized", [False, True])
def test_bridge_matches_naive_formula(normalized) -> None:
    g = torch.Generator().manual_seed(21)
    mu = torch.randn(N, 4, generator=g, dtype=F64)
    cov = _random_cov(N, 4, seed=22) + 0.1 * torch.eye(4, dtype=F64)

    got = lk.Bridge(normalized=normalized)(Categorical(), _out(mu, cov)).probs

    torch.testing.assert_close(got, _naive_bridge(mu, cov, normalized), rtol=1e-10, atol=0)


@pytest.mark.parametrize("v", [0.01, 1.0, 16.0])
def test_bridge_two_classes_is_softmax_of_corrected_mean(v) -> None:
    # laplace-torch's bridge gives 0.8808 = sigmoid(2) for every v.
    mu = torch.tensor([[1.0, -1.0]], dtype=F64)
    cov = v * torch.eye(2, dtype=F64).unsqueeze(0)

    got = lk.Bridge()(Categorical(), _out(mu, cov)).probs[0, 0]

    assert got.item() == pytest.approx(0.8807970779778823, rel=1e-10)


@pytest.mark.parametrize(
    ("v", "want"), [(0.01, 0.99999999999948), (1.0, 0.94419278), (16.0, 0.66976155)]
)
def test_bridge_norm_two_classes_depends_on_variance(v, want) -> None:
    # laplace-torch's bridge_norm on the same inputs.
    mu = torch.tensor([[1.0, -1.0]], dtype=F64)
    cov = v * torch.eye(2, dtype=F64).unsqueeze(0)

    got = lk.Bridge(normalized=True)(Categorical(), _out(mu, cov)).probs[0, 0]

    assert got.item() == pytest.approx(want, rel=1e-4)


def _degenerate_case() -> tuple[Tensor, Tensor]:
    """Logits [N, 4] and blocks; input 0's block has a zero row (and column) for class 0."""

    g = torch.Generator().manual_seed(23)
    mu = 2 * torch.randn(N, 4, generator=g, dtype=F64)
    cov = _random_cov(N, 4, seed=24) + 0.1 * torch.eye(4, dtype=F64)
    cov[0, 0, :] = 0
    cov[0, :, 0] = 0

    return mu, cov


def _laplace_torch_fallback(mu: Tensor, cov: Tensor) -> Tensor:
    """`BaseLaplace._glm_classification_link`'s fallback rows, written out from its code:
    the probit on the shift-corrected moments, before any normalization."""

    s1 = cov.sum(-1)
    ss = s1.sum(-1, keepdim=True)
    m = mu - s1 / ss * mu.sum(-1, keepdim=True)
    v = cov.diagonal(dim1=-2, dim2=-1) - s1**2 / ss

    return torch.softmax(m / torch.sqrt(1 + math.pi / 8 * v.clamp_min(0)), -1)


@pytest.mark.parametrize("normalized", [False, True])
def test_bridge_fallback_matches_laplace_torch(normalized) -> None:
    mu, cov = _degenerate_case()
    out = _out(mu, cov)

    with pytest.raises(ValueError, match=r'1 of 5 rows .*fallback="probit"'):
        lk.Bridge(normalized=normalized)(Categorical(), out)

    got = lk.Bridge(normalized=normalized, fallback="probit")(Categorical(), out).probs

    torch.testing.assert_close(got[:1], _laplace_torch_fallback(mu[:1], cov[:1]))
    torch.testing.assert_close(got[1:], _naive_bridge(mu[1:], cov[1:], normalized))


# laplace-torch's `bridge` and `bridge_norm` on `_centred_degenerate_case`, for mu and mu + 5
# alike: the fallback row of the centred branch.
_CENTRED_FALLBACK = [0.1510219018580297, 0.3872370531482036, 0.06378910366960748]
_CENTRED_FALLBACK += [0.1423456742839385, 0.25560626704022066]


def _centred_degenerate_case() -> tuple[Tensor, Tensor]:
    """K = 5: class 0 has no variance, and classes 1..4 are centred, so 1^T Sigma 1 = 0
    exactly (small integers, and 1/4 is exact)."""

    a = torch.tensor([[1, -2, 0, 1], [2, 1, -1, 0], [0, 1, 2, -1], [1, 0, 1, 2]], dtype=F64)
    centre = torch.eye(4, dtype=F64) - 1 / 4
    cov = torch.zeros(1, 5, 5, dtype=F64)
    cov[0, 1:, 1:] = centre @ a @ a.T @ centre
    mu = torch.tensor([[0.5, 2.0, -1.0, 0.3, 1.2]], dtype=F64)

    return mu, cov


@pytest.mark.parametrize("normalized", [False, True])
def test_bridge_centred_fallback_is_shift_invariant(normalized) -> None:
    mu, cov = _centred_degenerate_case()
    bridge = lk.Bridge(normalized=normalized, fallback="probit")

    assert cov.sum() == 0

    want = torch.tensor([_CENTRED_FALLBACK], dtype=F64)

    for shift in (0.0, 5.0, -40.0):
        got = bridge(Categorical(), _out(mu + shift, cov)).probs

        torch.testing.assert_close(got, want, rtol=1e-10, atol=1e-14)


def test_bridge_zero_cov_raises_and_fallback() -> None:
    # A zero output covariance, as where the subset's Jacobian vanishes.
    g = torch.Generator().manual_seed(25)
    mu = torch.randn(N, 4, generator=g, dtype=F64)
    cov = _random_cov(N, 4, seed=26) + 0.1 * torch.eye(4, dtype=F64)
    cov[1:3] = 0
    out = _out(mu, cov)

    with pytest.raises(ValueError, match=r'2 of 5 rows .*fallback="probit"'):
        lk.Bridge()(Categorical(), out)

    got = lk.Bridge(fallback="probit")(Categorical(), out).probs

    # laplace-torch: the centred branch, no variance, so the softmax of the mean.
    torch.testing.assert_close(got[1:3], torch.softmax(mu[1:3], -1))
    torch.testing.assert_close(got[[0, 3, 4]], _naive_bridge(mu[[0, 3, 4]], cov[[0, 3, 4]]))


def test_bridge_skips_the_correction_when_the_logit_sum_is_certain() -> None:
    # Centred logits: 1^T Sigma 1 = 0, so laplace-torch keeps mu and diag Sigma. Small
    # integers and K = 4 keep every entry exact, so the sum is exactly zero.
    g = torch.Generator().manual_seed(27)
    a = torch.randint(-3, 4, (N, 4, 4), generator=g).to(F64)
    centre = torch.eye(4, dtype=F64) - 1 / 4
    cov = centre @ a @ a.mT @ centre
    mu = torch.randn(N, 4, generator=g, dtype=F64)

    assert (cov.sum((-2, -1)) == 0).all()
    v = cov.diagonal(dim1=-2, dim2=-1)

    alpha = (1 - 2 / 4 + mu.exp() * (-mu).exp().sum(-1, keepdim=True) / 16) / v
    got = lk.Bridge()(Categorical(), _out(mu, cov)).probs

    torch.testing.assert_close(got, alpha / alpha.sum(-1, keepdim=True))


@pytest.mark.parametrize("normalized", [False, True])
def test_bridge_cov_types_agree(normalized, monkeypatch) -> None:
    g = torch.Generator().manual_seed(28)
    mu = torch.randn(N, 4, generator=g, dtype=F64)
    factor = torch.randn(N, 4, 6, generator=g, dtype=F64)
    dense = factor @ factor.mT
    draws = math.sqrt(6) * factor.permute(2, 0, 1)  # mean outer product: F F^T
    bridge = lk.Bridge(normalized=normalized)

    want = bridge(Categorical(), _out(mu, dense)).probs

    for cov in (core.FactorCov(factor), core.SampleCov(draws)):
        monkeypatch.setattr(type(cov), "per_input", lambda self: pytest.fail("per_input"))
        got = bridge(Categorical(), core.OutputGaussian(mu, cov)).probs

        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("normalized", [False, True])
def test_bridge_positions(normalized) -> None:
    mu, cov = _sequence_case(seed=29)
    out = core.OutputGaussian(mu, core.PerInputCov(cov, (T, C)))
    bridge = lk.Bridge(normalized=normalized)

    got = bridge(Categorical(class_dim=-1), out).probs
    full = cov.reshape(N, T, C, T, C)
    want = [bridge(Categorical(), _out(mu[:, t], full[:, t, :, t, :])).probs for t in range(T)]

    assert got.shape == (N, T, C)
    torch.testing.assert_close(got, torch.stack(want, 1))

    # Classes on axis 1: [N, C, T].
    cov_ct = full.permute(0, 2, 1, 4, 3).reshape(N, C * T, C * T)
    out_ct = core.OutputGaussian(mu.transpose(1, 2), core.PerInputCov(cov_ct, (C, T)))

    torch.testing.assert_close(bridge(Categorical(class_dim=1), out_ct).probs, got)


@pytest.mark.parametrize("normalized", [False, True])
def test_bridge_float32_large_logits_finite(normalized) -> None:
    mu = torch.tensor(
        [[1e4, -1e4, 0.0, 5e3], [-1e4, 1e4, 1e4, -1e4], [3.0, 1.0, 0.0, -2.0]],
        dtype=torch.float32,
    )
    cov = torch.diag_embed(torch.tensor([[0.5, 1.0, 2.0, 0.1], [1.0, 1.0, 1.0, 1.0], [1e-3] * 4]))
    bridge = lk.Bridge(normalized=normalized)

    got = bridge(Categorical(), _out(mu, cov)).probs
    want = bridge(Categorical(), _out(mu.double(), cov.double())).probs

    assert got.dtype == torch.float32
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got.sum(-1), torch.ones(3))
    torch.testing.assert_close(got.double(), want, rtol=1e-5, atol=1e-6)

    # laplace-torch's `bridge` / `bridge_norm` on the same float32 inputs.
    third = (
        [1.0, 1.917126098002186e-38, 0.0, 0.0]
        if normalized
        else [0.7614017724990845, 0.13153131306171417, 0.06921317428350449, 0.03785376995801926]
    )
    reference = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], third], dtype=F64)

    torch.testing.assert_close(got.double(), reference, rtol=1e-5, atol=1e-6)


def test_bridge_check_link() -> None:
    mu = torch.randn(N, 4, dtype=F64)
    out = _out(mu, _random_cov(N, 4) + 0.1 * torch.eye(4, dtype=F64))

    lk.testing.check_link(lk.Bridge(), Categorical(), out)
    lk.testing.check_link(lk.Bridge(normalized=True), Categorical(), out)
    lk.testing.check_link(lk.Bridge(), Gaussian(), out)

    assert not lk.Bridge().supports(Bernoulli(), mu)


_ONE_CLASS = re.escape(
    "a Categorical over one class gives it probability 1; for a scalar output such as a "
    "reward, use lk.predict_outputs(posterior, inputs)"
)


@pytest.mark.parametrize("link", [lk.Bridge(), MC(n_samples=8)], ids=["bridge", "mc"])
def test_one_class_refused_by_every_link(link) -> None:
    mu = torch.randn(N, 1, dtype=F64)
    out = _out(mu, torch.ones(N, 1, 1, dtype=F64))

    with pytest.raises(CapabilityError, match=_ONE_CLASS):
        link(Categorical(), out)

    # [N, T, 1]: one class at every position.
    seq = core.OutputGaussian(
        torch.randn(N, T, 1, dtype=F64), core.FactorCov(torch.ones(N, T, 1, 2, dtype=F64))
    )

    with pytest.raises(CapabilityError, match=_ONE_CLASS):
        link(Categorical(), seq)

    # Scalar outputs [N]: no class axis at all.
    scalar = core.OutputGaussian(torch.randn(N, dtype=F64), core.SampleCov(torch.ones(8, N)))

    with pytest.raises(CapabilityError, match=_NO_CLASS_AXIS):
        link(Categorical(), scalar)


def _one_output_post(caps: Capabilities) -> tuple[LaplacePosterior, Tensor]:
    g = torch.Generator().manual_seed(30)
    w = torch.randn(1, D, generator=g, dtype=F64)
    p = LaplacePosterior(
        mean={"fc.weight": w},
        curvature=Dense(torch.eye(D, dtype=F64)),
        prior=Isotropic(torch.tensor(1.0, dtype=F64)),
        n_data=torch.tensor(5.0, dtype=F64),
        model=ModelFn(lambda p, x: x @ p["fc.weight"].T),
        likelihood=Categorical(),
        layout=Layout.by_module({"fc.weight": w}),
        caps=caps,
    )

    return p, torch.randn(N, D, generator=g, dtype=F64)


@pytest.mark.parametrize(
    "link",
    [None, Probit(), MC(n_samples=8), lk.Bridge()],
    ids=["default", "probit", "mc", "bridge"],
)
def test_one_class_refused_before_the_pushforward(link, monkeypatch) -> None:
    def fail(*args, **kwargs):
        pytest.fail("predict_outputs")

    monkeypatch.setattr("laplacekit.predictive.predict_outputs", fail)

    # The probe's event size is 1: refused at resolve, before any forward pass.
    p, x = _one_output_post(Capabilities(vjp=True, vmap=True, event_size=1))

    with pytest.raises(CapabilityError, match=_ONE_CLASS):
        Linearized(link=link).resolve(p)

    # No probed event size: refused after the mean's forward pass.
    p, x = _one_output_post(CAPS)

    with pytest.raises(CapabilityError, match=_ONE_CLASS):
        lk.predict(p, x, Linearized(link=link))


def test_one_class_refused_by_weight_samples() -> None:
    p, x = _one_output_post(CAPS)

    with pytest.raises(CapabilityError, match=_ONE_CLASS):
        lk.predict(p, x, WeightSamples(n_samples=2))


_NO_CLASS_AXIS = re.escape("with class_dim off the batch axis")


class _Counting(torch.nn.Module):
    """A linear map with `out` outputs (`None`: scalar outputs [N]); counts forwards."""

    def __init__(self, out: int | None) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(3, out or 1).double()
        self.scalar = out is None
        self.calls = 0

    def forward(self, x: Tensor) -> Tensor:
        self.calls += 1
        f = self.fc(x)

        return f.squeeze(-1) if self.scalar else f


@pytest.mark.parametrize(
    "predictive",
    [WeightSamples(n_samples=3), Linearized(link=MC(n_samples=3)), Linearized()],
    ids=["weight_samples", "mc", "default"],
)
def test_one_class_refused_on_entry(predictive) -> None:
    # The probe knows the event size is 1, so predict refuses before any forward pass.
    g = torch.Generator().manual_seed(31)
    x = torch.randn(16, 3, generator=g, dtype=F64)
    y = torch.zeros(16, dtype=torch.long)
    model = _Counting(1)
    post = lk.laplace(
        model, [(x, y)], likelihood=Categorical(), hessian_approx=lk.DiagGGN(), hyper_tuner=None
    )
    model.calls = 0

    with pytest.raises(CapabilityError, match=_ONE_CLASS):
        lk.predict(post, x[:4], predictive)

    assert model.calls == 0


def test_scalar_outputs_refused_by_the_fit() -> None:
    # Scalar outputs [N] make the batch axis the class axis: one Categorical over N classes,
    # which cannot take [N] class indices. The probe refuses before the fit sees data.
    g = torch.Generator().manual_seed(31)
    x = torch.randn(16, 3, generator=g, dtype=F64)
    y = torch.zeros(16, dtype=torch.long)

    with pytest.raises(core.StructureError, match=r"shaped \(\).*got \(\d+,\)"):
        lk.laplace(
            _Counting(None),
            [(x, y)],
            likelihood=Categorical(),
            hessian_approx=lk.DiagGGN(),
            hyper_tuner=None,
        )


def test_weight_samples_refuses_scalar_outputs_at_the_first_forward() -> None:
    # No probed event size: refused after the first of the three forward passes.
    g = torch.Generator().manual_seed(32)
    model = _Counting(None)
    x = torch.randn(N, 3, generator=g, dtype=F64)
    params = {k: v.detach() for k, v in model.named_parameters()}
    p = LaplacePosterior(
        mean=params,
        curvature=Dense(torch.eye(4, dtype=F64)),  # 3 weights and a bias
        prior=Isotropic(torch.tensor(1.0, dtype=F64)),
        n_data=torch.tensor(5.0, dtype=F64),
        model=ModelFn.from_module(model),
        likelihood=Categorical(),
        layout=Layout.by_module(params),
        caps=CAPS,
    )

    with pytest.raises(CapabilityError, match=_NO_CLASS_AXIS):
        lk.predict(p, x, WeightSamples(n_samples=3))

    assert model.calls == 1


@pytest.mark.parametrize("k", [2, 3])
def test_probit_refuses_a_negative_variance_from_float32_cancellation(k) -> None:
    # Rows sharing a large component: a low-precision block J Jᵀ loses the small
    # differences, and the difference (K = 2) or centred (K = 3) variance comes out
    # negative. The block is rounded once to a 2^16 grid rather than formed by a float32
    # matmul, whose rounding depends on the CPU's kernels. Every grid value and the sums
    # Probit forms are exact in float32, so the sign is the same on every machine.
    g = torch.Generator().manual_seed(12)
    c = torch.randn(64, generator=g, dtype=F64) * 1e4
    j = c + torch.randn(k, 64, generator=g, dtype=F64)
    grid = 2.0**16
    cov = (torch.round(j @ j.T / grid) * grid).float().unsqueeze(0)
    mu = torch.zeros(1, k)

    with pytest.raises(ValueError, match=r"negative.*float32.*float64.*strategy=\"forward-mode\""):
        Probit()(Categorical(), _out(mu, cov))

    # In float64 the same rows give a positive variance.
    out64 = _out(mu.double(), (j @ j.T).unsqueeze(0))
    assert torch.isfinite(Probit()(Categorical(), out64).probs).all()


@pytest.mark.parametrize("k", [2, C], ids=["two-classes", "three-classes"])
def test_probit_under_vmap(k: int) -> None:
    """The negative-variance check can't read a vmapped variance, so it is skipped there."""

    mu = torch.randn(N, k, dtype=F64)
    cov = _random_cov(N, k)

    def probs(m: Tensor, c: Tensor) -> Tensor:
        return Probit()(Categorical(), _out(m[None], c[None])).probs[0]

    got = torch.func.vmap(probs)(mu, cov)
    ref = Probit()(Categorical(), _out(mu, cov)).probs

    torch.testing.assert_close(got, ref)
