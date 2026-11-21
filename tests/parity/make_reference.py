"""Write reference.json from laplace-torch (float64, CPU).

- Regression: laplace-torch's regression example (diag, full, KFAC), with the GGN and the
  Hessian.
- Subnetwork: largest-magnitude, largest-variance (diagonal Laplace) and parameter-name
  masks on the regression example, full (and diagonal) curvature over the subnetwork.
- Continual KFAC: the bias-free regression fit on two (or three) parts of the data in turn,
  with `fit(..., override=False)`.
- Joint tuning: on the regression example, `prior_prec` and `sigma` maximize laplace-torch's
  `log_marginal_likelihood(prior_precision, sigma_noise)`, run to convergence with L-BFGS.
- Classification: the setup of the docs classification example (make_blobs with the same
  arguments, the same network and training), last layer, full and KFAC; with the
  empirical Fisher, also full over all weights.
- Per-output sigma: a two-output regression with `GaussianLikelihood(sigma_noise=[0.1, 0.5])`
  and the Laplace `sigma_noise` at 1, which laplace-torch requires with a vector sigma.
- Bernoulli: the blobs as three one-vs-rest labels (the fourth blob is all zeros), with
  the probit predictive.
- Bridge: the Laplace bridge, plain and normalized, on the last-layer full classification
  posterior, with a per-row flag that the corrected variances are positive.
- Metrics: Brier scores computed in numpy and ECE values from torchmetrics (15 bins, l1),
  for multiclass and binary probabilities, plain and with saturated entries.
- Heteroscedastic: a natural-parameter network and a mean-variance network. laplace-torch
  has no natural-parameter likelihood, so the reference for it is laplace-torch's generic
  GGN path (its Jacobians, `J^T H J` and its marginal likelihood) with a custom
  `Likelihood` subclass here that supplies the closed-form output Hessian. Its mean and
  variances come from the delta method on the GLM predictive over the natural
  parameters. The mean-variance network uses `HeteroscedasticGaussianLikelihood`.
- Reward: a Bradley-Terry reward model fit on pairs, predicting one reward per item.
- Functional: `FunctionalLaplace` with the subset of data set to all rows, so its GP prior
  isn't rescaled by M / N. On the regression example, on a two-output regression (sigma
  0.2) and on a three-class blobs toy, with joint and independent outputs. The script
  checks that every output-Hessian diagonal entry of the toy is positive, so laplace-torch's
  `nan_to_num(1 / L, posinf=10)` changes nothing.
- ELLA: on the regression example (100 rows, 20 directions) and the classification
  example (200 rows, 50 directions), seed 1234. It stores the subset `indices`, the output
  coordinate drawn for each subset row (`coords`) and the eigenvalues of `feature_gram`,
  which don't depend on the signs of the basis. The script replays the coordinate draws,
  builds the Jacobian rows with `torch.func.jacrev` and checks that they reproduce
  `dual_directions @ dual_directions.T` to 1e-10. It also checks that the classification
  kernel has at least 50 directions above ELLA's tolerance.
- VaLLA: on the regression example, 20 inducing inputs on [0, 8] (output 0), sigma 0.3 and
  prior precision 1. For alpha 1 and 0 it stores `_objective` on all data at L = I and its
  gradients (L unpacked to a full lower triangle, the log prior precision, the log noise
  variance and the inducing inputs). The script sets what `fit` sets before its first step,
  and checks that `fit` records the same first objective. At a seeded factor it stores
  the KL (`_objective` with `n_data = 0`) and the latent moments. After 50 Adam steps at
  lr 1e-2 it stores the learned prior precision, noise, inducing inputs, L, the objective
  and the variances. On the classification example, the inducing inputs are 20 random
  training rows with their labels, probit and alpha 1, fit for 30 steps.
- Marglik training: `marglik_training` with `CurvlinopsGGN`, hyperparameter Adam at lr 0.1,
  10 hypersteps per estimate and prior precision 1 at the start. `margliks` are
  laplace-torch's negative log marginal likelihoods, one per hyperstep. On the three-class
  blobs of the functional section, a bias-free tanh network (its seeded init stored)
  trains with Adam at lr 1e-2 for 20 epochs, marglik every 5 epochs after a 5-epoch
  burn-in, as (kron, layerwise), (full, scalar) and (diag, diag); each run stores the
  margliks, the losses, the final prior precision, the final state dict and the log
  marginal likelihood. The regression example (full; kron on the bias-free model), the
  two-output toy of the functional section with a per-output sigma starting at [1, 1]
  (full) and the mean-variance network (full, `HeteroscedasticGaussianLikelihood(full=True)`
  as above) run 3, 3 and 2 epochs of SGD at lr 0 with a scalar prior, so only the
  hyperparameters move.
  The trajectories are comparable because the loaders are full-batch and unshuffled, the
  KFAC network has no biases (laplace-torch keeps a bias as its own Kronecker block), the
  regression runs use lr 0 and no network has a mode-dependent layer (`Laplace.fit`
  leaves the model in eval mode).
- SWAG: `fit_diagonal_swag_var` with 5 snapshots, one per epoch, at lr 0.01 and the other
  defaults (momentum 0.9, weight decay 3e-4, `min_var` 1e-30), on the classification
  example (`CrossEntropyLoss`) and the regression example (`MSELoss`), with full-batch,
  unshuffled loaders. It trains a deep copy in train mode, so the classifier's BatchNorm
  layers use batch statistics and update the copy's running stats. Each stores the
  variances, the state dict SWAG started from and the indices of
  `LargestVarianceSWAGSubnetMask` (30 parameters, the same SWAG settings). The script
  checks that these indices are the top 30 of the stored variances, with a positive gap
  at the cut.

Every stored `f_mu, f_var` of these function-space sections is a latent moment without
observation noise. For regression it comes from `la(X)`, which is latent for a Gaussian
likelihood. For classification it comes from `la.predictive_moments(X)`, because `la(X)`
returns probit probabilities there.

Every section after the first literal seeds itself, so the earlier numbers don't change.
torchmetrics must be installed next to laplace-torch.

Run with a Python that has laplace-torch installed:
    python tests/parity/make_reference.py
"""

import itertools
import json
import warnings
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from laplace import (
    ELLA,
    BernoulliLikelihood,
    BradleyTerryLikelihood,
    CategoricalLikelihood,
    DiagLaplace,
    FunctionalLaplace,
    GaussianLikelihood,
    HeteroscedasticGaussianLikelihood,
    KronLaplace,
    Laplace,
    Likelihood,
    VaLLA,
    marglik_training,
)
from laplace.curvature import CurvlinopsEF, CurvlinopsGGN, CurvlinopsHessian
from laplace.utils import (
    LargestMagnitudeSubnetMask,
    LargestVarianceDiagLaplaceSubnetMask,
    LargestVarianceSWAGSubnetMask,
    ParamNameSubnetMask,
    fit_diagonal_swag_var,
)
from laplace.utils.functional_utils import subset_loader
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional.classification import (
    binary_calibration_error,
    multiclass_calibration_error,
)

torch.set_default_dtype(torch.float64)
torch.manual_seed(711)

X = torch.rand(150, 1) * 8.0
Y = torch.sin(X) + 0.3 * torch.randn(150, 1)
X_TEST = torch.linspace(-2.0, 10.0, 50).reshape(-1, 1)

model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1)).to("cpu")
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(1000):
    opt.zero_grad()
    nn.functional.mse_loss(model(X), Y).backward()
    opt.step()

model.eval()

loader = DataLoader(TensorDataset(X, Y), batch_size=150)


def summarize(la) -> dict[str, list]:
    f_mu, f_var = la(X_TEST, pred_type="glm")

    return {
        "log_marglik": float(la.log_marginal_likelihood()),
        "f_mu": f_mu.detach().tolist(),
        "f_var": f_var.detach().tolist(),
    }


def fit(
    subset: str,
    structure: str,
    net: nn.Module = model,
    **kwargs,
) -> dict[str, list]:
    la = Laplace(
        net,
        "regression",
        subset_of_weights=subset,
        hessian_structure=structure,
        sigma_noise=0.3,
        prior_precision=1.0,
        backend=CurvlinopsGGN,
        **kwargs,
    )
    la.fit(loader)
    out = summarize(la)

    if structure == "kron":
        out["precision"] = la.posterior_precision.to_matrix().tolist()

    return out


def fit_hessian(subset: str) -> dict[str, list | float]:
    """Full Hessian at prior precision 1, or at the first power of 2 that makes the
    posterior precision positive definite (then stored as `prior_prec`)."""

    prior_prec = 1.0

    while True:
        la = Laplace(
            model,
            "regression",
            subset_of_weights=subset,
            hessian_structure="full",
            sigma_noise=0.3,
            prior_precision=prior_prec,
            backend=CurvlinopsHessian,
        )
        la.fit(loader)
        precision = la.H * la._H_factor + prior_prec * torch.eye(la.n_params)
        min_eig = float(torch.linalg.eigvalsh(precision).min())
        print(f"Hessian {subset}: prior {prior_prec}, smallest eigenvalue {min_eig:.3e}")

        if min_eig > 0:
            break

        prior_prec *= 2.0

    out = summarize(la)

    if prior_prec != 1.0:
        out["prior_prec"] = prior_prec

    return out


def score_gap(scores: torch.Tensor, k: int) -> float:
    """The k-th largest score minus the (k+1)-th."""

    top = scores.sort(descending=True).values

    return float(top[k - 1] - top[k])


def fit_subnet(indices: torch.Tensor, structures: tuple[str, ...] = ("full",)) -> dict:
    out: dict = {"indices": indices.tolist()}

    for structure in structures:
        out[structure] = fit("subnetwork", structure, subnetwork_indices=indices)

    return out


def subnet_magnitude() -> dict:
    mask = LargestMagnitudeSubnetMask(model, 30)
    out = fit_subnet(mask.select(), ("full", "diag"))
    out["gap"] = score_gap(mask._param_scores, 30)

    return out


def subnet_variance() -> dict:
    diag = DiagLaplace(
        model, "regression", sigma_noise=0.3, prior_precision=1.0, backend=CurvlinopsGGN
    )
    mask = LargestVarianceDiagLaplaceSubnetMask(model, 30, diag)
    out = fit_subnet(mask.select(loader))
    out["gap"] = score_gap(mask._param_scores, 30)

    return out


def subnet_names() -> dict:
    return fit_subnet(ParamNameSubnetMask(model, ["0.bias", "2.weight"]).select())


def tune_jointly(subset: str, structure: str, net: nn.Module = model) -> dict[str, float]:
    """Fit at sigma 0.3 and prior precision 1, then tune both on laplace-torch's marglik."""

    la = Laplace(
        net,
        "regression",
        subset_of_weights=subset,
        hessian_structure=structure,
        sigma_noise=0.3,
        prior_precision=1.0,
        backend=CurvlinopsGGN,
    )
    la.fit(loader)
    logs = [torch.zeros(1, requires_grad=True), torch.tensor([0.3]).log().requires_grad_()]
    opt = torch.optim.LBFGS(
        logs,
        max_iter=1000,
        tolerance_grad=1e-12,
        tolerance_change=1e-15,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = -la.log_marginal_likelihood(logs[0].exp(), logs[1].exp())
        loss.backward()

        return loss

    opt.step(closure)
    marglik = la.log_marginal_likelihood(logs[0].exp(), logs[1].exp())
    grads = torch.autograd.grad(marglik, logs)
    assert all(float(g.abs()) < 1e-6 for g in grads), grads

    return {
        "prior_prec": logs[0].exp().item(),
        "sigma": logs[1].exp().item(),
        "log_marglik": marglik.item(),
    }


# The same regression, without biases: laplace-torch's KFAC keeps a bias as its own block,
# laplacekit folds it into the layer's Kronecker factor, so only bias-free layers agree.
torch.manual_seed(711)
nobias = nn.Sequential(nn.Linear(1, 50, bias=False), nn.Tanh(), nn.Linear(50, 1, bias=False))
opt = torch.optim.Adam(nobias.parameters(), lr=1e-2)

for _ in range(1000):
    opt.zero_grad()
    nn.functional.mse_loss(nobias(X), Y).backward()
    opt.step()

nobias.eval()


def fit_continual(splits: list[int]) -> dict[str, list]:
    """KFAC on the bias-free model, fit on the parts of the data in turn."""

    la = KronLaplace(
        nobias, "regression", sigma_noise=0.3, prior_precision=1.0, backend=CurvlinopsGGN
    )
    bounds = [0, *splits, len(X)]

    for i, (lo, hi) in enumerate(itertools.pairwise(bounds)):
        part = DataLoader(TensorDataset(X[lo:hi], Y[lo:hi]), batch_size=hi - lo)
        la.fit(part, override=i == 0)

    f_mu, f_var = la(X_TEST, pred_type="glm")

    return {
        "precision": la.posterior_precision.to_matrix().tolist(),
        "f_mu": f_mu.detach().tolist(),
        "f_var": f_var.detach().tolist(),
    }


def make_blobs(
    n_samples: int, centers: int, cluster_std: float, center_box: tuple[float, float], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """sklearn's `make_blobs` (1.x), without sklearn: the same draws from the same RandomState."""

    rng = np.random.RandomState(seed)
    mu = rng.uniform(center_box[0], center_box[1], size=(centers, 2))
    sizes = [n_samples // centers + (i < n_samples % centers) for i in range(centers)]
    xs = np.concatenate(
        [rng.normal(loc=mu[i], scale=cluster_std, size=(n, 2)) for i, n in enumerate(sizes)]
    )
    ys = np.concatenate([np.full(n, i) for i, n in enumerate(sizes)])
    idx = np.arange(n_samples)
    rng.shuffle(idx)

    return xs[idx], ys[idx]


x_np, y_np = make_blobs(500, 4, 1.2, (-10.0, 10.0), 37)
X_CLS = torch.tensor(x_np)
Y_CLS = torch.tensor(y_np)
X_CLS_TEST = torch.tensor(
    [[x, y] for x in (-15.0, -5.0, 0.0, 5.0, 15.0) for y in (-15.0, 0.0, 15.0)]
)

torch.manual_seed(99999)
classifier = nn.Sequential(
    nn.Linear(2, 20),
    nn.BatchNorm1d(20),
    nn.ReLU(),
    nn.Linear(20, 20),
    nn.BatchNorm1d(20),
    nn.ReLU(),
    nn.Linear(20, 4, bias=False),
)
opt = torch.optim.SGD(classifier.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

for _ in range(5000):
    opt.zero_grad()
    nn.functional.cross_entropy(classifier(X_CLS), Y_CLS).backward()
    opt.step()

classifier.eval()
cls_loader = DataLoader(TensorDataset(X_CLS, Y_CLS), batch_size=500)


def fit_classifier(
    structure: str, subset: str = "last_layer", backend: type = CurvlinopsGGN
) -> dict[str, list]:
    la = Laplace(
        classifier,
        "classification",
        subset_of_weights=subset,
        hessian_structure=structure,
        prior_precision=5e-4,
        backend=backend,
    )
    la.fit(cls_loader)
    f_mu, f_var = la._glm_predictive_distribution(X_CLS_TEST)

    return {
        "log_marglik": float(la.log_marginal_likelihood()),
        "f_mu": f_mu.detach().tolist(),
        "f_var": f_var.detach().tolist(),
    }


ref = {
    "state_dict": {k: v.tolist() for k, v in model.state_dict().items()},
    "X": X.tolist(),
    "Y": Y.tolist(),
    "X_test": X_TEST.tolist(),
    "last_layer_full": fit("last_layer", "full"),
    "all_diag": fit("all", "diag"),
    "all_kron": fit("all", "kron"),
    "nobias_state_dict": {k: v.tolist() for k, v in nobias.state_dict().items()},
    "nobias_all_kron": fit("all", "kron", nobias),
    "nobias_last_layer_kron": fit("last_layer", "kron", nobias),
    "classification": {
        "state_dict": {k: v.tolist() for k, v in classifier.state_dict().items()},
        "X": X_CLS.tolist(),
        "Y": Y_CLS.tolist(),
        "X_test": X_CLS_TEST.tolist(),
        "last_layer_full": fit_classifier("full"),
        "last_layer_kron": fit_classifier("kron"),
        "ef_last_layer_full": fit_classifier("full", backend=CurvlinopsEF),
        "ef_last_layer_kron": fit_classifier("kron", backend=CurvlinopsEF),
        "ef_all_full": fit_classifier("full", "all", CurvlinopsEF),
    },
    "joint_tune": {
        "last_layer_full": tune_jointly("last_layer", "full"),
        "all_diag": tune_jointly("all", "diag"),
        "nobias_all_kron": tune_jointly("all", "kron", nobias),
    },
    "all_full": fit("all", "full"),
    "hessian_last_layer_full": fit_hessian("last_layer"),
    "hessian_all_full": fit_hessian("all"),
    "subnet_magnitude": subnet_magnitude(),
    "subnet_variance": subnet_variance(),
    "subnet_names": subnet_names(),
    "nobias_online_kron": fit_continual([60]),
    "nobias_online_kron_3": fit_continual([50, 100]),
}
# Everything below seeds itself and runs after the fits above, so their numbers don't change.


def train(net: nn.Module, loss: Callable[[], torch.Tensor], steps: int = 1000) -> None:
    """Full-batch Adam at lr 1e-2, then eval mode."""

    opt = torch.optim.Adam(net.parameters(), lr=1e-2)

    for _ in range(steps):
        opt.zero_grad()
        loss().backward()
        opt.step()

    net.eval()


def state(net: nn.Module) -> dict[str, list]:
    return {k: v.tolist() for k, v in net.state_dict().items()}


# Regression with a known noise std per output.
torch.manual_seed(2024)
X_VEC = torch.rand(150, 1) * 8.0
eps = torch.randn(150, 2)
Y_VEC = torch.cat([torch.sin(X_VEC) + 0.1 * eps[:, :1], torch.cos(X_VEC) + 0.5 * eps[:, 1:]], 1)
X_VEC_TEST = torch.linspace(-2.0, 10.0, 50).reshape(-1, 1)
SIGMA = torch.tensor([0.1, 0.5])
vec_loader = DataLoader(TensorDataset(X_VEC, Y_VEC), batch_size=150)

vec_model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 2))
train(vec_model, lambda: nn.functional.mse_loss(vec_model(X_VEC), Y_VEC))

vec_nobias = nn.Sequential(nn.Linear(1, 50, bias=False), nn.Tanh(), nn.Linear(50, 2, bias=False))
train(vec_nobias, lambda: nn.functional.mse_loss(vec_nobias(X_VEC), Y_VEC))


def fit_vector_sigma(subset: str, structure: str, net: nn.Module) -> dict[str, list]:
    """The per-output sigma lives on the likelihood; laplace-torch then needs sigma_noise 1."""

    la = Laplace(
        net,
        GaussianLikelihood(sigma_noise=SIGMA),
        subset_of_weights=subset,
        hessian_structure=structure,
        sigma_noise=1.0,
        prior_precision=1.0,
        backend=CurvlinopsGGN,
    )
    la.fit(vec_loader)
    f_mu, f_var = la(X_VEC_TEST, pred_type="glm")

    return {
        "log_marglik": float(la.log_marginal_likelihood()),
        "f_mu": f_mu.detach().tolist(),
        "f_var": f_var.detach().tolist(),
    }


ref["vector_sigma"] = {
    "state_dict": state(vec_model),
    "nobias_state_dict": state(vec_nobias),
    "X": X_VEC.tolist(),
    "Y": Y_VEC.tolist(),
    "X_test": X_VEC_TEST.tolist(),
    "sigma": SIGMA.tolist(),
    "last_layer_full": fit_vector_sigma("last_layer", "full", vec_model),
    "all_diag": fit_vector_sigma("all", "diag", vec_model),
    "all_full": fit_vector_sigma("all", "full", vec_model),
    "nobias_all_kron": fit_vector_sigma("all", "kron", vec_nobias),
    "nobias_last_layer_kron": fit_vector_sigma("last_layer", "kron", vec_nobias),
}


# Multi-label classification: one sigmoid per class of the first three blobs.
torch.manual_seed(2025)
Y_BER = nn.functional.one_hot(Y_CLS, 4)[:, :3].to(torch.get_default_dtype())
ber_loader = DataLoader(TensorDataset(X_CLS, Y_BER), batch_size=500)

ber_model = nn.Sequential(
    nn.Linear(2, 20), nn.ReLU(), nn.Linear(20, 20), nn.ReLU(), nn.Linear(20, 3, bias=False)
)
train(ber_model, lambda: nn.functional.binary_cross_entropy_with_logits(ber_model(X_CLS), Y_BER))


def fit_bernoulli(structure: str, subset: str = "last_layer", backend: type = CurvlinopsGGN):
    la = Laplace(
        ber_model,
        BernoulliLikelihood(),
        subset_of_weights=subset,
        hessian_structure=structure,
        prior_precision=1e-2,
        backend=backend,
    )
    la.fit(ber_loader)
    f_mu, f_var = la._glm_predictive_distribution(X_CLS_TEST)

    return {
        "log_marglik": float(la.log_marginal_likelihood()),
        "f_mu": f_mu.detach().tolist(),
        "f_var": f_var.detach().tolist(),
        "probit": la(X_CLS_TEST, link_approx="probit").detach().tolist(),
    }


ref["bernoulli"] = {
    "state_dict": state(ber_model),
    "Y": Y_BER.tolist(),
    "last_layer_full": fit_bernoulli("full"),
    "last_layer_kron": fit_bernoulli("kron"),
    "all_diag": fit_bernoulli("diag", "all"),
    "ef_last_layer_full": fit_bernoulli("full", backend=CurvlinopsEF),
}


def bridge() -> dict[str, list]:
    """The Laplace bridge on `fit_classifier("full")`'s posterior, refit here."""

    la = Laplace(
        classifier,
        "classification",
        subset_of_weights="last_layer",
        hessian_structure="full",
        prior_precision=5e-4,
        backend=CurvlinopsGGN,
    )
    la.fit(cls_loader)
    _, f_var = la._glm_predictive_distribution(X_CLS_TEST)

    # laplace-torch's correction: Σ' = Σ − (Σ1)(Σ1)ᵀ / 1ᵀΣ1, valid when its diagonal is > 0
    row_sums = f_var.sum(-1)
    corrected = (
        f_var - row_sums[:, :, None] * row_sums[:, None, :] / row_sums.sum(-1)[:, None, None]
    )

    return {
        "probs": la(X_CLS_TEST, link_approx="bridge").detach().tolist(),
        "probs_norm": la(X_CLS_TEST, link_approx="bridge_norm").detach().tolist(),
        "valid": (corrected.diagonal(dim1=-2, dim2=-1) > 0).all(-1).tolist(),
    }


ref["bridge"] = bridge()


def metrics() -> dict[str, list | float]:
    """Brier scores in numpy, ECE from torchmetrics (15 bins, l1)."""

    torch.manual_seed(2026)
    probs = torch.distributions.Dirichlet(torch.ones(4)).sample((200,))
    labels = torch.randint(0, 4, (200,))
    probs_saturated = probs.clone()
    probs_saturated[::4] = nn.functional.one_hot(torch.randint(0, 4, (50,)), 4).to(probs)

    binary_probs = torch.rand(200, 3)
    binary_labels = torch.bernoulli(binary_probs).long()
    binary_probs_saturated = binary_probs.clone().flatten()
    picked = torch.randperm(600)[:150]
    binary_probs_saturated[picked] = torch.randint(0, 2, (150,)).to(binary_probs)
    binary_probs_saturated = binary_probs_saturated.reshape(200, 3)

    def brier(p: torch.Tensor, y: np.ndarray) -> float:
        return float(((p.numpy() - y) ** 2).sum(-1).mean())

    def ece(p: torch.Tensor) -> float:
        return float(multiclass_calibration_error(p, labels, 4, n_bins=15, norm="l1"))

    def binary_ece(p: torch.Tensor) -> float:
        return float(
            binary_calibration_error(p.flatten(), binary_labels.flatten(), n_bins=15, norm="l1")
        )

    onehot = np.eye(4)[labels.numpy()]
    binary_y = binary_labels.numpy().astype(np.float64)

    return {
        "probs": probs.tolist(),
        "labels": labels.tolist(),
        "probs_saturated": probs_saturated.tolist(),
        "binary_probs": binary_probs.tolist(),
        "binary_labels": binary_labels.tolist(),
        "binary_probs_saturated": binary_probs_saturated.tolist(),
        "ece": ece(probs),
        "ece_saturated": ece(probs_saturated),
        "brier": brier(probs, onehot),
        "brier_saturated": brier(probs_saturated, onehot),
        "binary_ece": binary_ece(binary_probs),
        "binary_ece_saturated": binary_ece(binary_probs_saturated),
        "binary_brier": brier(binary_probs, binary_y),
        "binary_brier_saturated": brier(binary_probs_saturated, binary_y),
    }


ref["metrics"] = metrics()


# Heteroscedastic regression: noise grows with |x|, and the data have a gap at 0.
torch.manual_seed(2023)
X_HET = torch.cat([torch.empty(75, 1).uniform_(-3, -0.7), torch.empty(75, 1).uniform_(0.7, 3)])
Y_HET = torch.sin(2 * X_HET) + (0.05 + 0.15 * X_HET.abs()) * torch.randn_like(X_HET)
X_HET_TEST = torch.linspace(-4.0, 4.0, 50).reshape(-1, 1)
het_loader = DataLoader(TensorDataset(X_HET, Y_HET), batch_size=150)


class NaturalHead(nn.Module):
    """(h₁, h₂) ↦ (η₁, η₂) = (h₁, −½ exp h₂), so η₂ < 0."""

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h1, h2 = h.chunk(2, dim=-1)

        return torch.cat([h1, -0.5 * h2.exp()], -1)


def natural_moments(eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """μ = −η₁ / (2η₂) and σ² = −1 / (2η₂)."""

    eta1, eta2 = eta.chunk(2, dim=-1)

    return -eta1 / (2 * eta2), -1 / (2 * eta2)


class NaturalGaussianLikelihood(Likelihood):
    """A Gaussian in its natural parameters, with the closed-form output Hessian ∇²A(η)."""

    name = "natural_gaussian"

    def distribution(self, f: torch.Tensor) -> torch.distributions.Distribution:
        mu, var = natural_moments(f)

        return torch.distributions.Normal(mu, var.sqrt())

    def check_targets(self, f: torch.Tensor, y: torch.Tensor) -> None:
        if y.shape != (*f.shape[:-1], f.shape[-1] // 2):
            raise ValueError(f"targets {tuple(y.shape)} don't fit outputs {tuple(f.shape)}")

    def output_hessian(self, f: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        mu, var = natural_moments(f)
        K = mu.shape[-1]
        H = f.new_zeros(*f.shape, f.shape[-1])

        for k in range(K):
            m, v = mu[..., k], var[..., k]
            H[..., k, k] = v
            H[..., k, K + k] = H[..., K + k, k] = 2 * m * v
            H[..., K + k, K + k] = 4 * m**2 * v + 2 * v**2

        return H


def gaussian_nll(mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    return nn.functional.gaussian_nll_loss(mean, Y_HET, var)


natural = nn.Sequential(
    nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2), NaturalHead()
)
train(natural, lambda: gaussian_nll(*natural_moments(natural(X_HET))))


class MeanVarianceNet(nn.Module):
    """As in laplace-torch's heteroscedastic example, with a 32-wide body."""

    def __init__(self) -> None:
        super().__init__()

        self.body = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.mean = nn.Linear(32, 1)
        self.var = nn.Sequential(nn.Linear(32, 1), nn.Softplus())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)

        return self.mean(h), self.var(h) + 1e-4


mean_variance = MeanVarianceNet()
train(mean_variance, lambda: gaussian_nll(*mean_variance(X_HET)))


def fit_natural(structure: str) -> dict[str, list | float]:
    """The delta method gives the moments: g = ∂μ/∂η = (σ², 2μσ²), epistemic gᵀΣg."""

    la = Laplace(
        natural,
        NaturalGaussianLikelihood(),
        subset_of_weights="all",
        hessian_structure=structure,
        prior_precision=1.0,
        backend=CurvlinopsGGN,
    )
    la.fit(het_loader)
    f_mu, f_var = la._glm_predictive_distribution(X_HET_TEST)
    mu, var = natural_moments(f_mu)
    g = torch.cat([var, 2 * mu * var], -1)

    return {
        "log_marglik": float(la.log_marginal_likelihood()),
        "f_mu": f_mu.detach().tolist(),
        "f_var": f_var.detach().tolist(),
        "mean": mu.detach().tolist(),
        "epistemic": torch.einsum("ni,nij,nj->n", g, f_var, g)[:, None].detach().tolist(),
        "aleatoric": var.detach().tolist(),
    }


def fit_mean_variance(structure: str) -> dict[str, list | float]:
    la = Laplace(
        mean_variance,
        HeteroscedasticGaussianLikelihood(full=True),
        subset_of_weights="all",
        hessian_structure=structure,
        prior_precision=1.0,
        backend=CurvlinopsGGN,
    )
    la.fit(het_loader)
    f_mean, f_alea, f_epis = la(X_HET_TEST)

    return {
        "log_marglik": float(la.log_marginal_likelihood()),
        "f_mean": f_mean.detach().tolist(),
        "f_alea": f_alea.detach().tolist(),
        "f_epis": f_epis.detach().tolist(),
    }


ref["heteroscedastic"] = {
    "X": X_HET.tolist(),
    "Y": Y_HET.tolist(),
    "X_test": X_HET_TEST.tolist(),
    "natural": {
        "state_dict": state(natural),
        "all_full": fit_natural("full"),
        "all_diag": fit_natural("diag"),
    },
    "mean_variance": {
        "state_dict": state(mean_variance),
        "all_full": fit_mean_variance("full"),
        "all_diag": fit_mean_variance("diag"),
    },
}


# Reward modeling: pairs of items, the label is the index of the preferred one.
torch.manual_seed(2027)
X_FIRST = torch.rand(300, 3) * 4.0 - 2.0
X_SECOND = torch.rand(300, 3) * 4.0 - 2.0
X_ITEMS_TEST = torch.rand(40, 3) * 4.0 - 2.0


def utility(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * x[:, 0]) + x[:, 1]


first_wins = torch.bernoulli(torch.sigmoid(3 * (utility(X_FIRST) - utility(X_SECOND))))
LABELS = (1 - first_wins).long()
X_PAIRS = torch.stack([X_FIRST, X_SECOND], 1)
pair_loader = DataLoader(TensorDataset(X_PAIRS, LABELS), batch_size=300)


class RewardNet(nn.Module):
    """laplace-torch's reward model: pairs [N, 2, D] give [N, 2], items [N, D] give [N, 1]."""

    def __init__(self, bias: bool = True) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(3, 32, bias=bias), nn.ReLU(), nn.Linear(32, 1, bias=bias)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            return self.net(x.reshape(-1, x.shape[-1])).reshape(x.shape[0], 2)

        return self.net(x)


reward = RewardNet()
train(reward, lambda: nn.functional.cross_entropy(reward(X_PAIRS), LABELS))

reward_nobias = RewardNet(bias=False)
train(reward_nobias, lambda: nn.functional.cross_entropy(reward_nobias(X_PAIRS), LABELS))


def fit_reward(subset: str, structure: str, net: nn.Module) -> dict[str, list | float]:
    la = Laplace(
        net,
        BradleyTerryLikelihood(),
        subset_of_weights=subset,
        hessian_structure=structure,
        prior_precision=1.0,
        backend=CurvlinopsGGN,
    )
    la.fit(pair_loader)
    f_mu, f_var = la(X_ITEMS_TEST)

    return {
        "log_marglik": float(la.log_marginal_likelihood()),
        "f_mu": f_mu.detach().tolist(),
        "f_var": f_var.detach().tolist(),
    }


ref["reward"] = {
    "state_dict": state(reward),
    "nobias_state_dict": state(reward_nobias),
    "X_first": X_FIRST.tolist(),
    "X_second": X_SECOND.tolist(),
    "labels": LABELS.tolist(),
    "X_test": X_ITEMS_TEST.tolist(),
    "last_layer_full": fit_reward("last_layer", "full", reward),
    "all_diag": fit_reward("all", "diag", reward),
    "nobias_all_kron": fit_reward("all", "kron", reward_nobias),
}


# FunctionalLaplace with the subset of data set to all of it, so the GP prior is not
# rescaled by M / N.
torch.manual_seed(2028)


def functional(
    net: nn.Module,
    likelihood: str,
    data: DataLoader,
    x_test: torch.Tensor,
    independent_outputs: bool = False,
    **kwargs,
) -> tuple[FunctionalLaplace, dict[str, list | float]]:
    """Fit on every row; latent moments from `la(X)` (regression) or `predictive_moments`."""

    la = FunctionalLaplace(
        net,
        likelihood,
        n_subset=len(data.dataset),
        prior_precision=1.0,
        independent_outputs=independent_outputs,
        **kwargs,
    )

    with warnings.catch_warnings():
        # laplace-torch warns against independent outputs for multi-output regression
        warnings.simplefilter("ignore", UserWarning)
        la.fit(data)

    if likelihood == "regression":
        f_mu, f_var = la(x_test)
    else:
        f_mu, f_var = la.predictive_moments(x_test)

    return la, {
        "log_marglik": float(la.log_marginal_likelihood()),
        "f_mu": f_mu.detach().tolist(),
        "f_var": f_var.detach().tolist(),
    }


X_FN2 = torch.rand(80, 1) * 6.0
Y_FN2 = torch.cat([torch.sin(X_FN2), torch.cos(X_FN2)], 1) + 0.2 * torch.randn(80, 2)
fn2_loader = DataLoader(TensorDataset(X_FN2, Y_FN2), batch_size=80)

fn2_model = nn.Sequential(nn.Linear(1, 30), nn.Tanh(), nn.Linear(30, 2))
train(fn2_model, lambda: nn.functional.mse_loss(fn2_model(X_FN2), Y_FN2))

x_np, y_np = make_blobs(120, 3, 1.5, (-3.0, 3.0), 2028)
X_FNC = torch.tensor(x_np)
Y_FNC = torch.tensor(y_np)
X_FNC_TEST = torch.rand(60, 2) * 10.0 - 5.0
fnc_loader = DataLoader(TensorDataset(X_FNC, Y_FNC), batch_size=120)

fnc_model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 3))
train(fnc_model, lambda: nn.functional.cross_entropy(fnc_model(X_FNC), Y_FNC), steps=200)


def functional_classification(independent_outputs: bool) -> dict[str, list | float]:
    """Every output-Hessian diagonal entry must be positive, so that laplace-torch's
    `nan_to_num(1 / L, posinf=10)`, which replaces only infinities, changes nothing."""

    la, out = functional(fnc_model, "classification", fnc_loader, X_FNC_TEST, independent_outputs)
    lam = torch.cat(la.L) if independent_outputs else la.L
    assert bool((lam > 0).all()), float(lam.min())
    assert torch.equal(torch.nan_to_num(1 / lam, posinf=10.0), 1 / lam)

    return out


ref["functional"] = {
    "regression": functional(model, "regression", loader, X_TEST, sigma_noise=0.3)[1],
    "regression2": {
        "state_dict": state(fn2_model),
        "X": X_FN2.tolist(),
        "Y": Y_FN2.tolist(),
        "X_test": X_TEST.tolist(),
        "joint": functional(fn2_model, "regression", fn2_loader, X_TEST, sigma_noise=0.2)[1],
        "independent": functional(
            fn2_model, "regression", fn2_loader, X_TEST, True, sigma_noise=0.2
        )[1],
    },
    "classification": {
        "state_dict": state(fnc_model),
        "X": X_FNC.tolist(),
        "Y": Y_FNC.tolist(),
        "X_test": X_FNC_TEST.tolist(),
        "joint": functional_classification(False),
        "independent": functional_classification(True),
    },
}


# ELLA: a Nyström basis from one random output coordinate per subset row.
torch.manual_seed(2029)


def jacobian_rows(net: nn.Module, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Rows J[n, coords[n]] of the parameter Jacobian, from `torch.func.jacrev`."""

    params = {k: v.detach() for k, v in net.named_parameters()}
    buffers = dict(net.named_buffers())
    jac = torch.func.jacrev(lambda p: torch.func.functional_call(net, (p, buffers), (x,)))(params)
    J = torch.cat([j.reshape(*j.shape[:2], -1) for j in jac.values()], -1)

    return J[torch.arange(len(coords)), coords]


def ella(
    net: nn.Module,
    likelihood: str,
    data: DataLoader,
    x_test: torch.Tensor,
    subsample_size: int,
    n_eigenvalues: int,
    **kwargs,
) -> dict[str, list]:
    """Fit ELLA, then replay its subset and coordinate draws to check the basis."""

    la = ELLA(
        net,
        likelihood,
        subsample_size=subsample_size,
        n_eigenvalues=n_eigenvalues,
        prior_precision=1.0,
        seed=1234,
        **kwargs,
    )
    la.fit(data)

    # `_build_basis` draws one output per row from a generator seeded with seed + 1
    indices = la._indices(data, False)
    generator = torch.Generator().manual_seed(la.seed + 1)
    xs, coords = [], []

    for x, _ in subset_loader(data, indices):
        xs.append(x)
        coords.append(torch.randint(la.n_outputs, (x.shape[0],), generator=generator))

    coords = torch.cat(coords)
    rows = jacobian_rows(net, torch.cat(xs), coords)

    # The kernel needs n_eigenvalues directions above ELLA's tolerance
    eigvals, eigvecs = torch.linalg.eigh(rows @ rows.T)
    tolerance = torch.finfo(eigvals.dtype).eps * max(rows.shape) * eigvals.abs().max()
    assert int((eigvals > tolerance).sum()) >= n_eigenvalues

    top = torch.argsort(eigvals, descending=True)[:n_eigenvalues]
    directions = rows.T @ (eigvecs[:, top] / eigvals[top].sqrt())
    projector = la.dual_directions @ la.dual_directions.T
    assert float((directions @ directions.T - projector).abs().max()) < 1e-10

    if likelihood == "regression":
        f_mu, f_var = la(x_test)
    else:
        f_mu, f_var = la.predictive_moments(x_test)

    return {
        "indices": indices.tolist(),
        "coords": coords.tolist(),
        "gram_eigvals": torch.linalg.eigvalsh(la.feature_gram).tolist(),
        "f_mu": f_mu.detach().tolist(),
        "f_var": f_var.detach().tolist(),
    }


assert len(X_CLS) >= 200
ref["ella"] = {
    "regression": ella(model, "regression", loader, X_TEST, 100, 20, sigma_noise=0.3),
    "classification": ella(classifier, "classification", cls_loader, X_CLS_TEST, 200, 50),
}


# VaLLA with explicit inducing inputs.
torch.manual_seed(2030)


def unpack(packed: torch.Tensor, m: int) -> torch.Tensor:
    """The packed lower triangle as a full [m, m] matrix, in `tril_indices` order."""

    rows, cols = torch.tril_indices(m, m)
    full = packed.new_zeros(m, m)
    full[rows, cols] = packed

    return full


def prepare(la: VaLLA, data: DataLoader) -> None:
    """What `fit` sets before its first step, so that `_objective` runs outside it."""

    x, _ = next(iter(data))
    la.n_data = len(data.dataset)
    la.n_outputs = la.model(x).shape[-1]
    la.model.output_size = la.n_outputs


def objective_and_grads(la: VaLLA, x: torch.Tensor, y: torch.Tensor) -> dict[str, list | float]:
    """`_objective` and its gradients for the trained leaves, with L unpacked."""

    objective = la._objective(x, y)
    leaves = {"L": la.L, "log_prior_precision": la.log_prior_precision}

    if la.log_noise_variance is not None:
        leaves["log_noise_variance"] = la.log_noise_variance

    leaves["inducing_locations"] = la.inducing_locations
    grads = dict(zip(leaves, torch.autograd.grad(objective, list(leaves.values())), strict=True))
    grads["L"] = unpack(grads["L"], la.num_inducing)

    return {
        "objective_init": float(objective),
        "grads_init": {k: g.tolist() for k, g in grads.items()},
    }


Z_VALLA = torch.linspace(0.0, 8.0, 20)[:, None]


def valla_regression(alpha: float) -> VaLLA:
    return VaLLA(
        model,
        "regression",
        inducing_locations=Z_VALLA,
        inducing_classes=torch.zeros(20, dtype=torch.long),
        sigma_noise=0.3,
        prior_precision=1.0,
        alpha=alpha,
    )


def valla_at_init() -> dict[str, dict]:
    out = {}

    for alpha in (1.0, 0.0):
        la = valla_regression(alpha)
        prepare(la, loader)
        out[str(alpha)] = objective_and_grads(la, X, Y)

    return out


def valla_random_factor() -> dict[str, list | float]:
    """KL and latent moments at a seeded factor. The KL is `_objective` with no data."""

    factor = torch.randn(20, 20).tril(-1) + torch.diag(1.0 + torch.rand(20))
    la = valla_regression(1.0)
    prepare(la, loader)
    rows, cols = torch.tril_indices(20, 20)

    with torch.no_grad():
        la.L.copy_(factor[rows, cols])

    f_mu, f_var = la._glm_predictive_distribution(X_TEST)
    la.n_data = 0

    return {
        "factor": factor.tolist(),
        "kl": float(la._objective(X, Y)),
        "f_mu": f_mu.tolist(),
        "f_var": f_var.tolist(),
    }


def valla_fit(init: dict) -> dict[str, list | float]:
    la = valla_regression(1.0)
    la.fit(loader, iterations=50, lr=1e-2)

    # `fit` records the objective before each step, so its first one is the initial one
    assert la.fit_history_["objective"][0] == init["objective_init"]
    _, f_var = la(X_TEST)

    return {
        "prior_precision": float(la.prior_precision),
        "sigma_noise": float(la.sigma_noise),
        "inducing_locations": la.inducing_locations.tolist(),
        "L": unpack(la.L, 20).tolist(),
        "objective_final": float(la._objective(X, Y)),
        "f_var": f_var.tolist(),
    }


def valla_classification() -> dict:
    """Twenty training rows as inducing inputs, their labels as classes, probit."""

    torch.manual_seed(2030)
    picks = torch.randperm(len(X_CLS))[:20]

    la = VaLLA(
        classifier,
        "classification",
        inducing_locations=X_CLS[picks],
        inducing_classes=Y_CLS[picks],
        prior_precision=1.0,
        alpha=1.0,
        mc_softmax_samples=0,
    )
    prepare(la, cls_loader)
    out: dict = {"indices": picks.tolist(), **objective_and_grads(la, X_CLS, Y_CLS)}

    la.fit(cls_loader, iterations=30, lr=1e-2)
    assert la.fit_history_["objective"][0] == out["objective_init"]
    f_mu, f_var = la.predictive_moments(X_CLS_TEST)
    out["fit"] = {
        "prior_precision": float(la.prior_precision),
        "inducing_locations": la.inducing_locations.tolist(),
        "f_mu": f_mu.tolist(),
        "f_var": f_var.tolist(),
    }

    return out


valla_init = valla_at_init()
ref["valla"] = {
    "regression": {
        "inducing_locations": Z_VALLA.tolist(),
        "alpha": valla_init,
        "random_factor": valla_random_factor(),
        "fit": valla_fit(valla_init["1.0"]),
    },
    "classification": valla_classification(),
}


# Marglik training: laplace-torch's `marglik_training` on full-batch, unshuffled loaders.
torch.manual_seed(2031)


def tensor_or_list(value: torch.Tensor | float) -> list | float:
    """A tensor as a (nested) list or a float; a float as is."""

    return value.detach().tolist() if isinstance(value, torch.Tensor) else float(value)


def run_marglik(
    net: nn.Module,
    data: DataLoader,
    likelihood: Likelihood | str,
    structure: str,
    prior: str,
    **kwargs,
) -> tuple[Laplace, nn.Module, dict]:
    """`marglik_training` on a copy of `net`, with the hyperparameter settings shared by all
    runs. `margliks` holds laplace-torch's negative log marginal likelihoods, one per
    hyperstep."""

    lap, trained, margliks, losses = marglik_training(
        deepcopy(net),
        data,
        likelihood,
        hessian_structure=structure,
        backend=CurvlinopsGGN,
        prior_structure=prior,
        lr_hyp=0.1,
        n_hypersteps=10,
        prior_prec_init=1.0,
        **kwargs,
    )

    return (
        lap,
        trained,
        {
            "margliks": margliks,
            "losses": losses,
            "prior_precision": tensor_or_list(lap.prior_precision),
        },
    )


marglik_net = nn.Sequential(nn.Linear(2, 16, bias=False), nn.Tanh(), nn.Linear(16, 3, bias=False))


def marglik_classification(structure: str, prior: str) -> dict:
    """Adam at lr 1e-2 for 20 epochs, marglik every 5 epochs after a 5-epoch burn-in."""

    lap, trained, out = run_marglik(
        marglik_net,
        fnc_loader,
        CategoricalLikelihood(),
        structure,
        prior,
        optimizer_kwargs={"lr": 1e-2},
        n_epochs=20,
        marglik_frequency=5,
        n_epochs_burnin=5,
    )
    out["state_dict"] = state(trained)
    out["log_marglik"] = float(lap.log_marginal_likelihood())

    return out


def marglik_regression(
    net: nn.Module,
    data: DataLoader,
    likelihood: Likelihood | str,
    structure: str = "full",
    **kwargs,
) -> tuple[Laplace, nn.Module, dict]:
    """SGD at lr 0, so the network stays put and only the hyperparameters move."""

    return run_marglik(
        net,
        data,
        likelihood,
        structure,
        "scalar",
        optimizer_cls=torch.optim.SGD,
        optimizer_kwargs={"lr": 0.0},
        **kwargs,
    )


def marglik_gaussian(net: nn.Module, structure: str) -> dict:
    lap, _, out = marglik_regression(
        net,
        loader,
        GaussianLikelihood(),
        structure=structure,
        n_epochs=3,
        marglik_frequency=1,
        sigma_noise_init=1.0,
    )
    del out["losses"]
    out["sigma_noise"] = tensor_or_list(lap.sigma_noise)

    return out


def marglik_per_output() -> dict:
    lap, _, out = marglik_regression(
        fn2_model,
        fn2_loader,
        GaussianLikelihood(sigma_noise=torch.tensor([1.0, 1.0])),
        n_epochs=3,
    )
    del out["losses"]
    out["sigma_noise"] = tensor_or_list(lap.likelihood.sigma_noise)

    return out


def marglik_heteroscedastic() -> dict:
    _, _, out = marglik_regression(
        mean_variance, het_loader, HeteroscedasticGaussianLikelihood(full=True), n_epochs=2
    )
    del out["losses"]

    return out


ref["marglik"] = {
    "classification": {
        "state_dict": state(marglik_net),
        "kron_layerwise": marglik_classification("kron", "layerwise"),
        "full_scalar": marglik_classification("full", "scalar"),
        "diag_diag": marglik_classification("diag", "diag"),
    },
    "regression": {
        "full": marglik_gaussian(model, "full"),
        "nobias_kron": marglik_gaussian(nobias, "kron"),
    },
    "per_output": marglik_per_output(),
    "heteroscedastic": marglik_heteroscedastic(),
}


# Diagonal SWAG on full-batch, unshuffled loaders.
torch.manual_seed(2032)


def swag(net: nn.Module, data: DataLoader, criterion: nn.Module, likelihood: Likelihood) -> dict:
    """SWAG variances, and the largest-variance SWAG mask on the same data, which must pick
    the top 30 of these variances with no tie at the cut."""

    start = state(net)
    variances = fit_diagonal_swag_var(
        net, data, criterion, n_snapshots_total=5, snapshot_freq=1, lr=0.01
    )

    mask = LargestVarianceSWAGSubnetMask(net, 30, likelihood, swag_n_snapshots=5, swag_lr=0.01)
    indices = mask.select(data)

    assert torch.equal(indices, variances.argsort(descending=True)[:30].sort().values)
    assert score_gap(variances, 30) > 0

    return {"state_dict": start, "variances": variances.tolist(), "indices": indices.tolist()}


ref["swag"] = {
    "classification": swag(classifier, cls_loader, nn.CrossEntropyLoss(), CategoricalLikelihood()),
    "regression": swag(model, loader, nn.MSELoss(), GaussianLikelihood()),
}

Path(__file__).with_name("reference.json").write_text(json.dumps(ref))
print({k: v["log_marglik"] for k, v in ref.items() if isinstance(v, dict) and "log_marglik" in v})
print(ref["joint_tune"])
print(
    {
        k: v["log_marglik"]
        for k, v in ref["classification"].items()
        if isinstance(v, dict) and "log_marglik" in v
    }
)
