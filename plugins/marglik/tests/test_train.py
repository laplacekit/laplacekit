"""`lk.marglik.train`: the hyperparameter steps against a dense marginal likelihood built
here, the refusals before training, and the early stopping on the evidence."""

import copy
import math

import pytest
import torch
from torch import Tensor, nn
from torch.func import functional_call, jacrev
from torch.utils.data import DataLoader, TensorDataset

import laplacekit as lk

DT = torch.float64


def _data(n: int = 20) -> tuple[Tensor, Tensor]:
    g = torch.Generator().manual_seed(3)
    x = torch.randn(n, 2, generator=g, dtype=DT)
    y = torch.sin(x[:, :1]) + 0.1 * torch.randn(n, 1, generator=g, dtype=DT)

    return x, y


def _mlp(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1)).to(DT)


def _dense(model: nn.Module, x: Tensor, lam: Tensor, sigma: Tensor):
    """The outputs, the flat parameters, their `jacrev` Jacobian [N, K, P] and the full GGN
    posterior precision with an isotropic prior, for a scalar or per-output σ."""

    params = {k: v.detach() for k, v in model.named_parameters()}
    shapes = {k: v.shape for k, v in params.items()}
    sizes = [v.numel() for v in params.values()]
    theta = torch.cat([v.reshape(-1) for v in params.values()])

    def f(flat: Tensor) -> Tensor:
        parts = torch.split(flat, sizes)
        ps = {k: p.reshape(shapes[k]) for k, p in zip(params, parts, strict=True)}

        return functional_call(model, ps, (x,))

    out = f(theta)
    jac = jacrev(f)(theta).reshape(*out.shape, theta.numel())
    noise = (1 / sigma**2).expand(out.shape)
    ggn = torch.einsum("nkp,nk,nkq->pq", jac, noise, jac)
    prec = lam * torch.eye(theta.numel(), dtype=DT) + ggn

    return out, theta, jac, prec


def _dense_neg_log_marglik(model: nn.Module, x: Tensor, y: Tensor, lam: Tensor, sigma: Tensor):
    """-log Z of the full GGN Laplace with an isotropic prior, from `jacrev` Jacobians."""

    out, theta, _, prec = _dense(model, x, lam, sigma)
    log_lik = torch.distributions.Normal(out, sigma).log_prob(y).sum()
    occam = torch.logdet(prec) - theta.numel() * torch.log(lam) + lam * theta @ theta

    return -(log_lik - 0.5 * occam)


def _sgd0(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.0)


def _params(model: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in model.named_parameters()}


def test_first_step_matches_dense():
    x, y = _data()
    model = _mlp()

    # Two estimations, one hyperstep each: the second marglik is evaluated where Adam's
    # first step went. lr 0 keeps the network (and its GGN) fixed.
    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=lk.Gaussian(),
        sigma_init=torch.tensor(1.0, dtype=DT),
        optimizer=_sgd0(model),
        epochs=2,
        hessian_approx=lk.DenseGGN(joint=True),
        prior=lk.priors.Isotropic,
        n_hypersteps=1,
    )

    log_lam = torch.zeros((), dtype=DT, requires_grad=True)
    log_sigma = torch.zeros((), dtype=DT, requires_grad=True)
    first = _dense_neg_log_marglik(model, x, y, log_lam.exp(), log_sigma.exp())
    g_lam, g_sigma = torch.autograd.grad(first, (log_lam, log_sigma))

    # Adam's first step is lr * sign(g), up to its eps.
    lam_1 = torch.exp(-0.1 * g_lam.sign())
    sigma_1 = torch.exp(-0.1 * g_sigma.sign())
    second = _dense_neg_log_marglik(model, x, y, lam_1, sigma_1)

    assert len(trained.margliks) == 2
    assert trained.margliks[0] == pytest.approx(first.item(), rel=1e-6)
    assert trained.margliks[1] == pytest.approx(second.item(), rel=1e-6)

    # The step improved the evidence, so the second estimation is the one kept.
    assert trained.best_epoch == 2
    torch.testing.assert_close(trained.posterior.prior.prior_precision, lam_1, rtol=1e-6, atol=0)
    torch.testing.assert_close(trained.posterior.likelihood.sigma, sigma_1, rtol=1e-6, atol=0)


class _Counting:
    """Batches in a re-iterable, recording which ones were yielded."""

    def __init__(self, batches: list) -> None:
        self.batches = batches
        self.seen: set[int] = set()

    def __iter__(self):
        for i, b in enumerate(self.batches):
            self.seen.add(i)

            yield b


def _two_batches(targets: bool = True) -> list:
    x, y = _data()

    if not targets:
        return [lk.Batch(x[:10]), lk.Batch(x[10:])]

    return [(x[:10], y[:10]), (x[10:], y[10:])]


_REFUSALS = {
    "weight decay": (
        ValueError,
        "weight_decay",
        0,
        dict(optimizer=lambda m: torch.optim.Adam(m.parameters(), weight_decay=1e-3)),
    ),
    "weight decay in a second group": (
        ValueError,
        "weight_decay",
        0,
        dict(
            optimizer=lambda m: torch.optim.SGD(
                [
                    {"params": list(m[0].parameters())},
                    {"params": list(m[2].parameters()), "weight_decay": 1e-2},
                ],
                lr=0.1,
            )
        ),
    ),
    "no targets": (ValueError, "targets", 1, dict(targets=False)),
    "no n_data": (ValueError, "n_data=", 0, dict(n_data=None)),
    "epochs": (ValueError, "epochs", 0, dict(epochs=0)),
    "hypersteps": (ValueError, "hypersteps", 0, dict(n_hypersteps=0)),
    "marglik_frequency": (ValueError, "marglik_frequency", 0, dict(marglik_frequency=0)),
    "sigma_init with a fixed sigma": (
        ValueError,
        "sigma_init.*learned sigma",
        0,
        dict(likelihood=lk.Gaussian(sigma=0.3), sigma_init=0.5),
    ),
    "sigma_init not positive": (ValueError, "sigma_init", 0, dict(sigma_init=-1.0)),
    "kfac diagonal": (
        lk.StructureError,
        "diagonal prior",
        1,
        dict(hessian_approx=lk.KFAC(), prior=lk.priors.Diagonal),
    ),
}


@pytest.mark.parametrize("case", list(_REFUSALS))
def test_refusals_before_training(case):
    error, match, batches, opts = _REFUSALS[case]
    opts = dict(opts)
    model = _mlp()
    data = _Counting(_two_batches(opts.pop("targets", True)))
    make_opt = opts.pop("optimizer", lambda m: torch.optim.SGD(m.parameters(), lr=0.1))
    kwargs = dict(n_data=20, epochs=2, likelihood=lk.Gaussian()) | opts
    before = _params(model)
    versions = {k: v._version for k, v in model.named_parameters()}

    with pytest.raises(error, match=match):
        lk.marglik.train(model, data, optimizer=make_opt(model), **kwargs)

    assert len(data.seen) == batches

    for k, v in model.named_parameters():
        assert torch.equal(v, before[k])
        assert v._version == versions[k]


def test_no_trainable_parameter():
    model = _mlp().requires_grad_(False)
    data = _Counting(_two_batches())
    other = torch.zeros(1, requires_grad=True)

    with pytest.raises(ValueError, match="trainable parameter"):
        lk.marglik.train(
            model,
            data,
            n_data=20,
            likelihood=lk.Gaussian(),
            optimizer=torch.optim.SGD([other], lr=0.1),
            epochs=1,
        )

    assert len(data.seen) == 0


def test_size_disagrees_with_data():
    x, y = _data()
    model = _mlp()
    data = lk.data.from_loader([(x, y)], n_data=20)

    with pytest.raises(ValueError, match="n_data="):
        lk.marglik.train(
            model, data, n_data=19, likelihood=lk.Gaussian(), optimizer=_sgd0(model), epochs=1
        )


def test_initial_prior_precision_must_be_finite():
    x, y = _data()
    model = _mlp()

    with pytest.raises(ValueError, match="finite"):
        lk.marglik.train(
            model,
            [(x, y)],
            n_data=20,
            likelihood=lk.Gaussian(),
            optimizer=_sgd0(model),
            epochs=1,
            prior=lk.priors.Layerwise(float("inf")),
        )


class _Wander(torch.optim.Optimizer):
    """Gradient descent for the first `turn` steps, ascent after, so the evidence gets
    worse late."""

    def __init__(self, params, lr: float, turn: int) -> None:
        super().__init__(params, {"lr": lr, "weight_decay": 0.0})
        self.turn = turn
        self.count = 0

    @torch.no_grad()
    def step(self, closure=None):
        sign = -1.0 if self.count < self.turn else 1.0

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.add_(p.grad, alpha=sign * group["lr"])

        self.count += 1


def test_best_epoch_restored():
    x, y = _data()
    model = _mlp()
    hypersteps = 3

    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=lk.Gaussian(),
        sigma_init=torch.tensor(1.0, dtype=DT),
        optimizer=_Wander(model.parameters(), lr=0.2, turn=3),
        epochs=6,
        hessian_approx=lk.DenseGGN(joint=True),
        prior=lk.priors.Isotropic,
        n_hypersteps=hypersteps,
        marglik_frequency=1,
    )

    last = torch.tensor(trained.margliks[hypersteps - 1 :: hypersteps])
    best = int(last.argmin())

    assert len(last) == 6
    assert trained.best_epoch == 1 + best
    assert trained.best_epoch < 6

    post = trained.posterior

    for k, v in model.named_parameters():
        assert torch.equal(post.mean[k], v)

    restored = _dense_neg_log_marglik(
        model, x, y, post.prior.prior_precision, post.likelihood.sigma
    )

    assert restored.item() == pytest.approx(last[best].item(), rel=1e-6)
    assert ("training", f"marglik, best epoch {trained.best_epoch} of 6") in post.provenance


def test_never_estimated():
    x, y = _data()
    model = _mlp()

    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=lk.Gaussian(),
        sigma_init=torch.tensor(0.7, dtype=DT),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        epochs=2,
        marglik_frequency=3,
        prior=lk.priors.Layerwise(2.5),
    )

    assert trained.best_epoch is None
    assert trained.margliks == ()
    assert len(trained.losses) == 2

    prior = trained.posterior.prior

    assert isinstance(prior, lk.priors.Layerwise)
    assert list(prior.prior_precision) == ["0", "2"]

    for v in prior.prior_precision.values():
        assert v.item() == 2.5

    assert trained.posterior.likelihood.sigma.item() == 0.7


def test_noise_fixed():
    x, y = _data()
    model = _mlp()
    sigma = torch.tensor(0.3, dtype=DT)

    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=lk.Gaussian(sigma),
        optimizer=_sgd0(model),
        epochs=2,
        prior=lk.priors.Diagonal,
    )

    assert torch.equal(trained.posterior.likelihood.sigma, sigma)

    prec = trained.posterior.prior.prior_precision

    assert prec.shape == (17,) and prec.dtype == DT
    assert not torch.allclose(prec, torch.ones_like(prec))


def test_losses_are_the_negative_log_joint_per_example():
    x, y = _data()
    model = _mlp()
    lam = 2.0

    trained = lk.marglik.train(
        model,
        [(x[:10], y[:10]), (x[10:], y[10:])],
        n_data=len(x),
        likelihood=lk.Gaussian(),
        sigma_init=torch.tensor(0.5, dtype=DT),
        optimizer=_sgd0(model),
        epochs=1,
        marglik_frequency=2,
        prior=lk.priors.Isotropic(lam),
    )

    theta = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    nll = -torch.distributions.Normal(model(x), 0.5).log_prob(y).sum() / len(x)
    expected = nll + 0.5 * lam * theta @ theta / len(x)

    assert trained.losses[0] == pytest.approx(expected.item(), rel=1e-12)


def test_size_for_plain_loader():
    x, y = _data()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=10)
    model = _mlp()

    trained = lk.marglik.train(
        model,
        loader,
        n_data=len(dataset),
        likelihood=lk.Gaussian(),
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        epochs=1,
        n_hypersteps=2,
    )

    assert trained.best_epoch == 1
    assert len(trained.margliks) == 2
    assert math.isfinite(trained.posterior.log_marglik().item())

    with pytest.raises(ValueError, match="n_data="):
        lk.marglik.train(model, loader, likelihood=lk.Gaussian(), optimizer=_sgd0(model), epochs=1)


def _two_outputs(n: int = 16) -> tuple[Tensor, Tensor, nn.Module]:
    g = torch.Generator().manual_seed(5)
    x = torch.randn(n, 1, generator=g, dtype=DT)
    y = torch.cat([torch.sin(x), torch.cos(x)], 1) + 0.2 * torch.randn(n, 2, generator=g, dtype=DT)
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(1, 3), nn.Tanh(), nn.Linear(3, 2)).to(DT)

    return x, y, model


def _per_output(
    model: nn.Module, x: Tensor, y: Tensor, sigma: Tensor, learned: bool = True, **kwargs
) -> lk.marglik.Trained:
    likelihood = lk.Gaussian() if learned else lk.Gaussian(sigma)
    kwargs = {"sigma_init": sigma, **kwargs} if learned else kwargs

    return lk.marglik.train(
        model,
        [(x[:10], y[:10]), (x[10:], y[10:])],
        n_data=len(x),
        likelihood=likelihood,
        optimizer=_sgd0(model),
        hessian_approx=lk.DenseGGN(joint=True),
        prior=lk.priors.Isotropic,
        n_hypersteps=1,
        **kwargs,
    )


def test_noise_update_matches_dense():
    x, y, model = _two_outputs()
    sigma_0 = torch.tensor([3.0, 0.5], dtype=DT)
    lam_0 = torch.tensor(1.0, dtype=DT)

    # MacKay's update at the λ of the first marglik (the initial one, with one hyperstep):
    # σ_k² = mean_n [(y_nk - f_nk)² + (J_n Σ J_nᵀ)_kk], Σ the dense inverse of the precision.
    out, _, jac, prec = _dense(model, x, lam_0, sigma_0)
    cov = torch.linalg.inv(prec)
    var = torch.einsum("nkp,pq,nkq->nk", jac, cov, jac)
    sigma_1 = ((y - out).square() + var).mean(0).sqrt()

    log_lam = torch.zeros((), dtype=DT, requires_grad=True)
    first = _dense_neg_log_marglik(model, x, y, log_lam.exp(), sigma_0)
    (g_lam,) = torch.autograd.grad(first, (log_lam,))
    lam_1 = torch.exp(-0.1 * g_lam.sign())
    second = _dense_neg_log_marglik(model, x, y, lam_1, sigma_1)

    trained = _per_output(model, x, y, sigma_0, epochs=2)

    assert trained.margliks[0] == pytest.approx(first.item(), rel=1e-6)
    assert trained.margliks[1] == pytest.approx(second.item(), rel=1e-6)

    # The second estimation, fit with the updated σ, is better and kept with that σ.
    assert trained.best_epoch == 2
    torch.testing.assert_close(trained.posterior.likelihood.sigma, sigma_1, rtol=1e-6, atol=0)

    # With one estimation, the kept σ is the one it was fit with, not the update after it.
    once = _per_output(model, x, y, sigma_0, epochs=1)

    assert torch.equal(once.posterior.likelihood.sigma, sigma_0)


def test_noise_vector_fixed():
    x, y, model = _two_outputs()
    sigma = torch.tensor([3.0, 0.5], dtype=DT)

    trained = _per_output(model, x, y, sigma, learned=False, epochs=2)

    log_lam = torch.zeros((), dtype=DT, requires_grad=True)
    first = _dense_neg_log_marglik(model, x, y, log_lam.exp(), sigma)
    (g_lam,) = torch.autograd.grad(first, (log_lam,))
    second = _dense_neg_log_marglik(model, x, y, torch.exp(-0.1 * g_lam.sign()), sigma)

    assert trained.margliks[1] == pytest.approx(second.item(), rel=1e-6)
    assert torch.equal(trained.posterior.likelihood.sigma, sigma)


def test_scheduler_steps_once_per_batch():
    model = _mlp()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    lk.marglik.train(
        model,
        _two_batches(),
        n_data=20,
        likelihood=lk.Gaussian(),
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=3,
        n_hypersteps=1,
    )

    assert scheduler.last_epoch == 6
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01 * 0.5**6)


@pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
def test_mode_left_alone(training):
    model = _mlp().train(training)

    lk.marglik.train(
        model,
        _two_batches(),
        n_data=20,
        likelihood=lk.Gaussian(),
        optimizer=_sgd0(model),
        epochs=1,
        n_hypersteps=1,
    )

    assert all(m.training == training for m in model.modules())


def test_later_batch_without_targets():
    x, y = _data()
    model = _mlp()
    before = _params(model)
    data = [(x[:10], y[:10]), lk.Batch(x[10:])]

    # The dry run sees only the first batch, so the second fails mid-epoch, after the
    # first batch's step.
    with pytest.raises(ValueError, match="a batch has none"):
        lk.marglik.train(
            model,
            data,
            n_data=20,
            likelihood=lk.Gaussian(),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            epochs=1,
        )

    assert any(not torch.equal(v, before[k]) for k, v in model.named_parameters())


def test_scalar_precision_stays_float64():
    x, y = _data()
    model = _mlp().float()

    scalar = lk.marglik.train(
        model,
        [(x.float(), y.float())],
        n_data=20,
        likelihood=lk.Gaussian(),
        optimizer=_sgd0(model),
        epochs=1,
        prior=lk.priors.Isotropic,
        n_hypersteps=2,
    )
    diagonal = lk.marglik.train(
        model,
        [(x.float(), y.float())],
        n_data=20,
        likelihood=lk.Gaussian(),
        optimizer=_sgd0(model),
        epochs=1,
        prior=lk.priors.Diagonal,
        n_hypersteps=2,
    )
    never = lk.marglik.train(
        model,
        [(x.float(), y.float())],
        n_data=20,
        likelihood=lk.Gaussian(),
        optimizer=_sgd0(model),
        epochs=1,
        marglik_frequency=2,
        prior=lk.priors.Layerwise,
    )

    assert scalar.posterior.prior.prior_precision.dtype == DT
    assert diagonal.posterior.prior.prior_precision.dtype == torch.float32
    assert all(v.dtype == DT for v in never.posterior.prior.prior_precision.values())


@pytest.mark.parametrize("prior", [lk.priors.Isotropic, lk.priors.Layerwise, lk.priors.Diagonal])
def test_initial_prior_is_the_given_precision_exactly(prior):
    # exp(log(3.0)) is one ulp off 3.0 in float64.
    assert math.exp(math.log(3.0)) != 3.0

    x, y = _data()
    model = _mlp()

    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=lk.Gaussian(),
        optimizer=_sgd0(model),
        epochs=1,
        marglik_frequency=2,
        prior=prior(3.0),
    )

    prec = trained.posterior.prior.prior_precision
    leaves = list(prec.values()) if isinstance(prec, dict) else [prec]

    for v in leaves:
        assert v.dtype == DT
        assert bool((v == 3.0).all())


@pytest.mark.parametrize(
    "sigma", [[1.0, 1.0, 1.0], [[1.0, 1.0]] * 2, [[1.0], [1.0]]], ids=["3", "2x2", "2x1"]
)
def test_noise_vector_shape_refused_before_training(sigma):
    x, y, model = _two_outputs()
    data = _Counting([(x[:10], y[:10]), (x[10:], y[10:])])
    before = _params(model)

    with pytest.raises(lk.StructureError, match=r"sigma of shape .* event shape"):
        lk.marglik.train(
            model,
            data,
            n_data=len(x),
            likelihood=lk.Gaussian(),
            sigma_init=torch.tensor(sigma, dtype=DT),
            optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
            epochs=2,
        )

    assert len(data.seen) == 1

    for k, v in model.named_parameters():
        assert torch.equal(v, before[k])


def test_noise_update_weighted_matches_dense():
    from laplacekit_marglik.hypers import _noise_update

    x, y, model = _two_outputs()
    sigma = torch.tensor([3.0, 0.5], dtype=DT)
    lam = torch.tensor(1.0, dtype=DT)
    w = torch.linspace(0.5, 2.0, len(x), dtype=DT)

    # The posterior from unweighted data; the update reads weighted batches.
    post = lk.laplace(
        model,
        [(x, y)],
        likelihood=copy.replace(lk.Gaussian(), sigma=sigma),
        hessian_approx=lk.DenseGGN(joint=True),
        subset=lk.subsets.all,
        prior=lk.priors.Isotropic(lam),
        hyper_tuner=None,
    )
    weighted = lk.data.from_loader(
        [lk.Batch(x[:10], y[:10], weights=w[:10]), lk.Batch(x[10:], y[10:], weights=w[10:])],
        n_data=len(x),
    )

    # σ_k² = Σ_n w_n [(y_nk - f_nk)² + (J_n Σ J_nᵀ)_kk] / Σ_n w_n.
    out, _, jac, prec = _dense(model, x, lam, sigma)
    var = torch.einsum("nkp,pq,nkq->nk", jac, torch.linalg.inv(prec), jac)
    terms = (y - out).square() + var
    expected = ((w[:, None] * terms).sum(0) / w.sum()).sqrt()

    got = _noise_update(post, weighted, torch.Generator().manual_seed(0), sigma)

    torch.testing.assert_close(got, expected, rtol=1e-8, atol=0)


def test_noise_vector_restored_with_best():
    x, y, model = _two_outputs()
    sigma_0 = torch.tensor([3.0, 0.5], dtype=DT)

    # Descent, then ascent: the evidence gets worse late while MacKay's update keeps
    # moving σ after every estimation.
    trained = lk.marglik.train(
        model,
        [(x, y)],
        n_data=len(x),
        likelihood=lk.Gaussian(),
        sigma_init=sigma_0,
        optimizer=_Wander(model.parameters(), lr=0.2, turn=3),
        epochs=6,
        hessian_approx=lk.DenseGGN(joint=True),
        prior=lk.priors.Isotropic,
        n_hypersteps=1,
    )

    last = torch.tensor(trained.margliks)
    best = int(last.argmin())

    assert trained.best_epoch == 1 + best
    assert 1 < trained.best_epoch < 6

    post = trained.posterior
    sigma = post.likelihood.sigma

    assert not torch.equal(sigma, sigma_0)

    # The restored (θ, λ, σ) give back the best value: σ is the vector that estimation
    # was fit with, not a later update.
    restored = _dense_neg_log_marglik(model, x, y, post.prior.prior_precision, sigma)

    assert restored.item() == pytest.approx(last[best].item(), rel=1e-6)
