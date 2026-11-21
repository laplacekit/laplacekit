# %% [markdown]
# ---
# title: Subnetworks
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# diag = lk.laplace(model, data, likelihood=lik, subset=lk.subsets.all, hessian_approx=lk.DiagGGN())
# post = lk.laplace(
#     model, data, likelihood=lik,
#     subset=lk.subnet.largest_magnitude(51),  # or largest_variance(diag, 51), random(51), named(...)
#     # or largest_variance(lk.subnet.swag_variance(model, data, likelihood=lik), 51)
#     hessian_approx=lk.DenseGGN(joint=True),
#     prior=lk.subnet.rescaled_prior(lam, 51, model),  # lam * 51 / d, the paper's scaling
# )
# ```
# :::
#
# The Laplace approximation over all weights $\theta \in \mathbb{R}^d$ of a
# network is
#
# $$
#   p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\theta \mid \theta_*, \Sigma) , \qquad \Sigma = (\lambda I + N G)^{-1} ,
# $$
#
# with $G$ the generalized Gauss-Newton matrix averaged over the $N$ data points.
# For most networks, the $d \times d$ matrix $\Sigma$ is too big to store, so we
# usually keep only its diagonal or a Kronecker factorization. Subnetwork
# Laplace (Daxberger et al., 2021)[^daxberger2021] goes the other way. It keeps
# every correlation, but only among a few weights. We split the weights into
# a chosen set $\theta_S$, with $|S| = n \ll d$, and the rest $\theta_R$. The
# rest stays fixed at the MAP, and the chosen ones get a full covariance:
#
# $$
#   q(\theta) = \mathcal{N}\big(\theta_S \mid \theta_{S*}, (\lambda_S I + N G_{SS})^{-1}\big) \, \delta(\theta_R - \theta_{R*}) , \qquad \lambda_S = \lambda \, \frac{n}{d} ,
# $$
#
# where $G_{SS}$ is the $n \times n$ block of $G$ for the chosen entries. So
# the covariance stays $n \times n$ however large $d$ is, and the predictive
# linearizes the network in $\theta_S$ only. The prior precision is scaled
# down by $n / d$, as in Sec. 4 of the paper. With only $n$ weights left to
# carry the uncertainty about $f$, a weaker prior on each keeps the
# predictive variance at about the size it has under the full posterior.
#
# Why could this beat a diagonal over all $d$ weights? A diagonal treats each
# weight as if it moved on its own. In the full posterior, the weights move
# together, in the directions that keep the function close to the data. The
# predictive variance $J(x) \Sigma J(x)^\top$ depends on exactly these
# correlations, so a diagonal can be far off even when each marginal looks
# fine. The subnetwork keeps the correlations it can afford and drops the
# rest of the weights altogether.
#
# Which $n$ weights? Daxberger et al. pick the subnetwork whose posterior is
# closest to the full one in the 2-Wasserstein distance. That problem is
# intractable, but if we pretend $\Sigma$ is diagonal, the squared distance is
# the sum of the variances we drop, $\sum_{j \in R} \Sigma_{jj}$. So the best
# $S$ keeps the $n$ largest marginal variances, which we read off a diagonal
# Laplace over all weights. Its variances are $1 / (\lambda + N G_{jj})$, so
# for any $\lambda$ this keeps the $n$ weights with the smallest $G_{jj}$. We
# compare this choice with three others: the $n$
# weights largest in magnitude, $n$ weights at random, and the last layer.

# %%
import matplotlib.pyplot as plt
import torch
from torch import nn

import laplacekit as lk

# %% [markdown]
# ## Data and model
#
# We draw 150 points of $y = \sin(x) + \varepsilon$, $\varepsilon \sim
# \mathcal{N}(0, 0.1^2)$, half with $x$ uniform in $[-4, -2]$ and half in $[2,
# 4]$, so there is a gap in the middle. The test set has 400 points on a grid
# from $-5$ to $5$, which covers the gap and both ends.
#
# The network has two hidden layers of 50 tanh units, $d = 2701$ weights in
# all. We train it to the MAP of the same model that the Laplace
# approximation assumes: noise $\sigma = 0.1$, prior precision $\lambda = 1$.

# %%
torch.set_default_dtype(torch.float64)
torch.manual_seed(711)

x = torch.cat([torch.rand(75, 1) * 2 - 4, torch.rand(75, 1) * 2 + 2])
y = torch.sin(x) + 0.1 * torch.randn_like(x)
x_test = torch.linspace(-5.0, 5.0, 400).reshape(-1, 1)
y_test = torch.sin(x_test) + 0.1 * torch.randn_like(x_test)
data = [(x, y)]

model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(), nn.Linear(50, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
sigma, lam = 0.1, 1.0

for _ in range(3000):
    opt.zero_grad()
    sq = ((model(x) - y) ** 2).sum() / (2 * sigma**2)
    loss = sq + lam / 2 * sum((p**2).sum() for p in model.parameters())
    (loss / len(x)).backward()
    opt.step()

print("weights:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# ## The fits
#
# Every fit uses the $\sigma$ of the training and `hyper_tuner=None`. The fits over
# all weights use the $\lambda = 1$ of the training, and the subnetworks the
# scaled $\lambda_S = \lambda \, n / d$. We could tune the prior per fit, but
# then the fits would differ in two things at once.
#
# First, the diagonal Laplace over all weights. It is our baseline, and it
# gives the marginal variances that `lk.subnet.largest_variance` sorts. Since
# the network is small, we can also afford the GGN over all 2701 weights as
# one dense matrix, with `DenseGGN(joint=True)`. That is the posterior the
# other fits approximate, so we fit it too, as a reference. Without
# `joint=True`, `DenseGGN` keeps one block per layer and drops the
# correlations between layers.

# %%
lik = lk.Gaussian(sigma=torch.tensor(sigma))
prior = lk.priors.Isotropic(lam)
fit = dict(likelihood=lik, prior=prior, hyper_tuner=None)

diag = lk.laplace(
    model,
    data,
    subset=lk.subsets.all,
    hessian_approx=lk.DiagGGN(),
    **fit,
    generator=torch.Generator().manual_seed(0),
)
full = lk.laplace(
    model,
    data,
    subset=lk.subsets.all,
    hessian_approx=lk.DenseGGN(joint=True),
    **fit,
    generator=torch.Generator().manual_seed(0),
)

# %% [markdown]
# Then four subnetworks of the same size. The last layer, `4.weight` and
# `4.bias`, has 51 weights, so we take $n = 51$ for the other three as well.
# `lk.subsets.named` takes whole parameters by name. The three strategies of
# [`lk.subnet`](../reference/laplacekit_subnet.qmd) (from the
# `laplacekit-subnet` plugin) pick single entries anywhere in the network.
# `DenseGGN(joint=True)` gives the chosen entries one $51 \times 51$
# covariance, across layers, and `lk.subnet.rescaled_prior` gives them the
# scaled $\lambda_S$. To see what the scaling does, we also fit the
# last layer once more with the unscaled $\lambda = 1$.

# %%
n = 51
subnets = {
    "magnitude": lk.subnet.largest_magnitude(n),
    "variance": lk.subnet.largest_variance(diag, n),
    "random": lk.subnet.random(n, generator=torch.Generator().manual_seed(0)),
    "last layer": lk.subsets.named("4.*"),
}
sub_fit = fit | dict(prior=lk.subnet.rescaled_prior(lam, n, model))
posts = {
    name: lk.laplace(
        model,
        data,
        subset=s,
        hessian_approx=lk.DenseGGN(joint=True),
        **sub_fit,
        generator=torch.Generator().manual_seed(0),
    )
    for name, s in subnets.items()
}
fixed = lk.laplace(
    model,
    data,
    subset=subnets["last layer"],
    hessian_approx=lk.DenseGGN(joint=True),
    **fit,
    generator=torch.Generator().manual_seed(0),
)

for name, post in posts.items():
    print(f"{name:10}", {k: v.numel() for k, v in post.mean.items()})

# %% [markdown]
# ## The predictive
#
# We predict once per fit. The full posterior is the slow one, since its
# Jacobians have 2701 columns. Below are the predictive mean and two standard deviations of each fit,
# against the full posterior's band in dashed lines.

# %%
posteriors = {"full, all": full, "diagonal, all": diag} | posts
preds = {name: lk.predict(posterior, x_test) for name, posterior in posteriors.items()}

t = x_test.squeeze()
ref = preds["full, all"]
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)

for ax, (name, p) in zip(axes.flat, preds.items(), strict=True):
    m, s = p.mean.squeeze(), p.stddev.squeeze()
    ax.scatter(x.squeeze(), y.squeeze(), s=6, color="0.5")
    ax.plot(t, m)
    ax.fill_between(t, m - 2 * s, m + 2 * s, alpha=0.3)
    ax.set_title(name)

    if p is not ref:
        for sign in (-1, 1):
            ax.plot(t, ref.mean.squeeze() + sign * 2 * ref.stddev.squeeze(), "k--", lw=0.8)

axes[0, 0].set_ylim(-3, 3)
plt.show()

# %% [markdown]
# ## The numbers
#
# For each fit, we print the log marginal likelihood $\log Z$, the predictive
# negative log-likelihood (NLL) per test point, on the whole grid and in the
# gap, $|x| < 2$, and the mean predictive standard deviation in the gap. Lower
# NLL is better. A standard deviation of $0.1$ is the noise alone, with no
# uncertainty about $f$.

# %%
gap = t.abs() < 2

for name, p in preds.items():
    nll = -p.log_prob(y_test).squeeze()
    std = p.stddev.squeeze()[gap].mean()
    print(
        f"{name:14} log Z {posteriors[name].log_marglik():8.1f}   NLL {nll.mean():6.3f}"
        f"   in gap: NLL {nll[gap].mean():6.3f}, std {std:.3f}"
    )

p = lk.predict(fixed, x_test)
nll = -p.log_prob(y_test).squeeze()
print(
    f"{'last, lam = 1':14} log Z {fixed.log_marglik():8.1f}   NLL {nll.mean():6.3f}"
    f"   in gap: NLL {nll[gap].mean():6.3f}, std {p.stddev.squeeze()[gap].mean():.3f}"
)

# %% [markdown]
# ## Reading the numbers
#
# On the whole grid, the random draw has the lowest NLL (-0.392), and the last
# layer comes next (-0.321). Both are below the full posterior (-0.290), with
# 51 weights instead of 2701. The largest magnitudes (-0.259) follow. All
# three beat the diagonal over all weights (0.073). The choice by variance
# does worst (0.322).
#
# The plot shows why the diagonal loses. Its band is wide at the data too,
# where the full posterior is sure. This is the cost of dropping the
# correlations: each weight moves alone, also in directions that the data
# rule out.
#
# In the gap, two of the subnetworks follow the full posterior closely. Its
# mean standard deviation there is 0.526, against 0.557 for the largest
# magnitudes and 0.467 for the last layer. The random draw is narrower, at
# 0.261. The scaled prior matters here. With $\lambda = 1$, the last layer reaches only
# 0.142 in the gap. That fit looks better on the NLL (-0.469 in the gap,
# against 0.076 with $\lambda_S$), because the MAP network happens to follow
# the sine there, so being sure pays off on this test set. If the true
# function did something else in the gap, the wide bands would be the safer
# bet.
#
# The choice by variance gives exactly the noise in the gap, 0.100, even
# with the scaled prior. All 51 entries it picks are in the middle layer,
# `2.weight`. Below, we check their diagonal variances against the prior's,
# $1 / \lambda = 1$, and the standard deviation of $f$ under this
# subnetwork.

# %%
var = torch.cat([v.flatten() for v in diag.variance.values()])
f_std = lk.predict_outputs(posts["variance"], x_test).cov.diagonal().sqrt()

print("smallest diagonal variance among those picked:", f"{var.topk(n).values.min():.4f}")
print("largest std of f under the variance choice:   ", f"{f_std.max():.4f}")

# %% [markdown]
# The picked variances are all at the prior's, to four digits, and $f$ barely
# moves (at most 0.0008). These are weights the data don't touch: units that
# the weight decay switched off, so their entries of $G$ are close to zero,
# and so are the Jacobians of the output with respect to them, at the data
# and away from it. Their posterior is wide, but moving them doesn't move
# $f$. The Wasserstein argument measures the distance between weight
# posteriors, which counts these variances in full. The predictive only sees
# them through $J(x)$, so they add nothing.
#
# The log marginal likelihood points the wrong way too. The choice by
# variance has the highest $\log Z$ (139.5), and the full posterior (95.1)
# and the diagonal (-2698.4) are the lowest. With $\theta_R$ fixed, only the
# $n$ chosen weights pay the Occam factor, and weights the data don't touch
# pay almost nothing, so $\log Z$ is close to the data fit at the MAP. As on
# [Choosing a curvature](curvature.py), the log marginal likelihood is only
# comparable between fits on the same parameters and the same prior. The
# four subnetworks sit on four different sets of weights. The full posterior
# and the diagonal share theirs, and there the full one is far ahead, as
# Hadamard's inequality says it must be.
#
# ::: {.callout-note}
# Daxberger et al. run nearly this toy themselves: two hidden layers of 50
# units, 2600 weights, and subnetworks of 1% to 50% of them (our $n = 51$ is
# 1.9%), with $\lambda$ fixed by hand. Tuning $\lambda$ would not change our
# selection anyway, since the ranking by $1 / (\lambda + N G_{jj})$ is the
# same for every $\lambda$. Their Sec. 6.3 reports the failure we see:
# variances from a diagonal Laplace pick dead units, whose variance is the
# prior's.[^daxberger2021-63] They switch to variances from diagonal SWAG
# there, which leaves out weights with zero training gradients. We try that
# next.
# :::
#
# ## Choosing by SWAG variances
#
# SWAG (Maddox et al., 2019)[^maddox2019] keeps training from the MAP with
# SGD and takes snapshots $\theta^{(1)}, \dots, \theta^{(T)}$ of the weights
# along the way. Its diagonal variance is that of the snapshots,
#
# $$
#   \sigma_j^2 = \frac{1}{T} \sum_{t=1}^T \big(\theta_j^{(t)}\big)^2 - \Big(\frac{1}{T} \sum_{t=1}^T \theta_j^{(t)}\Big)^2 .
# $$
#
# A weight whose gradient is zero on the data moves only through the weight
# decay, so its variance stays close to zero, and the largest variances go to
# weights that the data push around. `lk.subnet.swag_variance` trains a copy
# of the model, so our network stays at its MAP. Daxberger et al. take 40
# snapshots, and we take 10, one per epoch, which is enough on this toy. We
# pick the 51 largest variances and fit them with the same scaled prior as
# the other subnetworks.

# %%
swag = lk.subnet.swag_variance(
    model, data, likelihood=lik, n_snapshots=10, generator=torch.Generator().manual_seed(0)
)
by_swag = lk.laplace(
    model,
    data,
    subset=lk.subnet.largest_variance(swag, n),
    hessian_approx=lk.DenseGGN(joint=True),
    **sub_fit,
    generator=torch.Generator().manual_seed(0),
)
print({k: v.numel() for k, v in by_swag.mean.items()})

p = lk.predict(by_swag, x_test)
nll = -p.log_prob(y_test).squeeze()
print(
    f"{'SWAG':14} log Z {by_swag.log_marglik():8.1f}   NLL {nll.mean():6.3f}"
    f"   in gap: NLL {nll[gap].mean():6.3f}, std {p.stddev.squeeze()[gap].mean():.3f}"
)

# %% [markdown]
# SWAG picks 49 entries of the last layer's weight, `4.weight`, and 2 of the
# middle layer's. This is the fix Daxberger et al. describe for the choice by
# diagonal variances, which took 51 dead entries of `2.weight`. The NLL on the
# whole grid drops from 0.322 to -0.335, and the mean standard deviation in the
# gap grows from 0.100, the noise alone, to 0.443. That NLL is below the last
# layer (-0.321) and the full posterior (-0.290), and only the random draw
# (-0.392) does better. The gap's standard deviation is close to the last
# layer's 0.467 and below the full posterior's 0.526. In the gap itself, the NLL is worse than with the
# variance pick, 0.040 against -0.112.

# %% [markdown]
# [^daxberger2021]: Daxberger, E., Nalisnick, E., Allingham, J. U., Antorán, J., & Hernández-Lobato, J. M. (2021). Bayesian deep learning via subnetwork inference. *ICML*. <https://arxiv.org/abs/2010.14689>
#
# [^daxberger2021-63]: Daxberger et al. (2021), Sec. 6.3. <https://arxiv.org/abs/2010.14689>
#
# [^maddox2019]: Maddox, W. J., Izmailov, P., Garipov, T., Vetrov, D. P., & Wilson, A. G. (2019). A simple baseline for Bayesian uncertainty in deep learning. *NeurIPS*. <https://arxiv.org/abs/1902.02476>
