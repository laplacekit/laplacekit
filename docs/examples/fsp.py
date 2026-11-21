# %% [markdown]
# ---
# title: Function-space priors
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# prior = lk.gp.FunctionPrior(lk.gp.RBF(torch.tensor(0.3)))
# context = lk.gp.uniform(-2.0, 2.0, 100)
#
# # MAP: the likelihood plus the prior's RKHS norm at fresh context points
# c = context.draw()
# loss = -lik.log_prob(model(x), y).sum() + 0.5 * lk.gp.rkhs_norm(prior, c, model(c))
#
# post = lk.gp.fsp_laplace(
#     model, loader, likelihood=lik, prior=prior,
#     context=lk.gp.grid(-2.0, 2.0, 100), rank=100, subset=lk.subsets.all,
# )
# pred = lk.predict(post, x_test)
# ```
# :::
#
# The prior of a Laplace approximation usually lives in weight space: an
# isotropic Gaussian $\mathcal{N}(0, \lambda^{-1} I)$ over the network's
# weights. It is easy to write down and hard to interpret. What does
# $\lambda = 1$ say about the functions the network computes? That depends
# on the architecture, the activations and the input scale, and we can't
# read it off the number. If we know something about the function, say that
# it is smooth with a lengthscale of about $0.3$, or that it should go back to
# zero away from the data, a weight prior gives us no direct way to say so.
#
# A Gaussian process prior can. On this page, we look at FSP-Laplace
# (Cinquin et al., 2024)[^cinquin2024], which puts a GP prior on the
# network's outputs and then does a Laplace approximation under it.
#
# ## A GP prior on the function
#
# Let $f(x; \theta)$ be the network with $P$ parameters $\theta$ and $E$
# outputs. FSP-Laplace places the prior $f \sim \mathcal{GP}(\mu, k)$ on the
# function directly, with mean $\mu$ and kernel $k$, shared by every output.
# A GP over functions has no density, so "the MAP estimate" needs a
# definition. The paper uses the weak mode of the prior, which leads to the
# objective
#
# $$
#   \mathcal{L}(\theta) = -\sum_{i=1}^N \log p(y_i \mid f(x_i; \theta))
#     + \tfrac{1}{2} \lVert f(\cdot; \theta) - \mu \rVert_{\mathcal{H}}^2 ,
# $$
#
# where $\lVert \cdot \rVert_{\mathcal{H}}$ is the norm of the kernel's
# reproducing kernel Hilbert space (RKHS). Functions that are rough, or far
# from $\mu$, relative to what $k$ expects have a large norm.
#
# We can't compute the RKHS norm of a network exactly. Take $n_C$ context
# points $C = (c_1, \dots, c_{n_C})$, and let $K_{CC}$ be the kernel matrix at
# them. The paper estimates the norm by
#
# $$
#   \lVert f - \mu \rVert_{\mathcal{H}}^2
#   \approx \big(f(C) - \mu(C)\big)^\top K_{CC}^{-1} \big(f(C) - \mu(C)\big) ,
# $$
#
# summed over the $E$ outputs. This is the RKHS norm of the smallest function
# in the RKHS that goes through the values of $f - \mu$ at $C$. The network
# minus the mean is one such function, so the estimate is never above the
# true norm: it underestimates it. Training draws new context points at every
# step, so the penalty sees the whole input region over time.
#
# ## The Laplace approximation
#
# Let $\theta_*$ be the MAP estimate and $J(x)$ the $E \times P$ Jacobian of
# the outputs at $\theta_*$. Linearize $f$ around $\theta_*$. The RKHS term
# becomes a quadratic in $\theta$ with the precision
#
# $$
#   J(C)^\top K_{CC}^{-1} J(C) ,
# $$
#
# where $J(C)$ stacks the Jacobians at the context points (with $K_{CC}$
# repeated over the $E$ outputs). This is the pulled-back prior. It has rank
# at most $n_C E$, far below $P$ for any network we care about, so it isn't
# a proper precision over the weights. Along the directions it leaves out,
# the outputs at $C$ don't change, and the function prior says nothing.
#
# FSP-Laplace restricts the posterior to the directions the prior does
# constrain. Write the pulled-back precision as $M M^\top$ with
# $M = J(C)^\top L$ and $L L^\top \approx K_{CC}^{-1}$, and take the thin SVD
# $M = U_M D_M V_M^\top$. The columns of $U_M$ span the range of $M$. In that
# basis, the prior precision is $D_M^2$, and with the GGN
# $G = \sum_i J(x_i)^\top H_i J(x_i)$ the posterior precision is
#
# $$
#   A = D_M^2 + U_M^\top G \, U_M .
# $$
#
# With the eigendecomposition $A = U_A D_A U_A^\top$, the posterior over the
# weights is $\mathcal{N}(\theta_*, S S^\top)$ with
# $S = U_M U_A D_A^{-1/2}$. The output covariance at $x$ is
# $J(x) S S^\top J(x)^\top$, and the predictive is the linearized one.
#
# Three details make this work in practice. First, $L$ comes from a few
# steps of the Lanczos method on $K_{CC}$, started from $J(C) \mathbf{1}$,
# the row sums of the Jacobian (summed over the outputs when $E > 1$). With
# $r$ steps, $L$ has $r$ columns and $M$ has $r E$, so the SVD is cheap.
# Second, the entries of $D_M^2$ can span many decades, and then the
# smallest eigenvalues of $A$ come out at the rounding level, or even
# negative. $D_A^{-1/2}$ is then the pseudo-inverse root: those eigenvalues
# and their directions are dropped. Third, the posterior variance at a
# context point should never be above the prior variance there. With the
# exact $K_{CC}^{-1}$ it can't be, since data only shrinks the variance, but
# a truncated $L$ can break that. So FSP-Laplace drops the columns of $S$
# with the largest variance, one by one, until
#
# $$
#   \operatorname{diag}\big( J(C) S S^\top J(C)^\top \big)
#   \le \operatorname{diag}(K_{CC})
# $$
#
# holds at every context point.
#
# ## Data and models
#
# We use the 1-D toy of the paper's Appendix B.1: $y = \sin(2 \pi x)$ plus
# Gaussian noise with $\sigma = 0.1$, on two intervals,
# $[-1, -0.5] \cup [0.5, 1]$, with a gap between them. The network has two
# hidden layers of 50 tanh units, 2701 weights in all. We test on a grid of
# 201 points on $[-2, 2]$, with fresh noise.

# %%
import matplotlib.pyplot as plt
import torch
from torch import nn

import laplacekit as lk

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

x = torch.cat([torch.rand(50, 1) * 0.5 - 1.0, torch.rand(50, 1) * 0.5 + 0.5])
y = torch.sin(2 * torch.pi * x) + 0.1 * torch.randn(100, 1)
data = [(x, y)]

x_test = torch.linspace(-2.0, 2.0, 201).reshape(-1, 1)
y_test = torch.sin(2 * torch.pi * x_test) + 0.1 * torch.randn(201, 1)
lik = lk.Gaussian(sigma=0.1)

# %% [markdown]
# We train three networks from the same initialization, for 5000 Adam steps
# each. The first is a plain MAP estimate under the weight prior
# $\mathcal{N}(0, I)$. The other two use the RKHS objective with a GP prior of
# zero mean and unit variance: an RBF kernel with lengthscale $0.3$, and a
# Matérn-½ kernel with lengthscale $0.5$. The Matérn-½ kernel gives rough,
# Brownian-motion-like functions. At each step, we draw 100 context points
# uniformly from $[-2, 2]$.


# %%
def train(prior=None):
    torch.manual_seed(1)
    model = nn.Sequential(
        nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(), nn.Linear(50, 1)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    context = lk.gp.uniform(-2.0, 2.0, 100, generator=torch.Generator().manual_seed(0))

    for _ in range(5000):
        opt.zero_grad()
        loss = -lik.log_prob(model(x), y).sum()

        if prior is None:
            loss = loss + 0.5 * sum(p.square().sum() for p in model.parameters())
        else:
            c = context.draw()
            loss = loss + 0.5 * lk.gp.rkhs_norm(prior, c, model(c))

        loss.backward()
        opt.step()

    return model


priors = {
    "RBF": lk.gp.FunctionPrior(lk.gp.RBF(torch.tensor(0.3))),
    "Matérn-½": lk.gp.FunctionPrior(lk.gp.Matern("1/2", torch.tensor(0.5))),
}
nets = {"weights": train()} | {name: train(prior) for name, prior in priors.items()}

# %% [markdown]
# Before the posteriors, a look at the norm estimate. We evaluate it for the
# two trained networks on nested grids over $[-2, 2]$, from 9 to 257 points.
# Each grid contains the one before it.

# %%
with torch.no_grad():
    for name, prior in priors.items():
        norms = []

        for n in (9, 17, 33, 65, 129, 257):
            c = lk.gp.grid(-2.0, 2.0, n).draw()
            norms.append(round(float(lk.gp.rkhs_norm(prior, c, nets[name](c))), 2))

        print(f"{name:<9} RKHS norm estimate:", norms)

# %% [markdown]
# The estimate grows as we add points, as it should. With more points, the
# smallest interpolant has to match the network in more places, so its norm
# can only grow. The 9-point grid has spacing $0.5$, and
# $\sin(2 \pi x)$ is zero at every multiple of $0.5$, so both networks are
# close to zero there and the estimate is close to zero too. From 17 points
# on, it grows slowly: from 11.12 to 12.45 for the RBF prior, and from 4.27 to
# 7.01 for the Matérn-½ one. The true norm is above all of these numbers.
#
# ## Fitting the posteriors
#
# The baseline is the usual Laplace approximation: `lk.laplace` with the dense
# GGN over all weights and the isotropic prior $\lambda = 1$ that we trained
# with. We keep $\lambda$ fixed instead of tuning it by the marginal
# likelihood, which `lk.laplace` does by default, so all three priors are set
# by hand. Tuning would change the width of the weight-space band, not its
# mean. For FSP-Laplace, `lk.gp.fsp_laplace` takes the prior and the context
# points of the posterior, here a grid of 100 points on $[-2, 2]$, as in the
# paper. `rank` is the number of Lanczos steps. With 100 points, 100 steps
# are all there are.

# %%
posteriors = {
    "MAP + Laplace": lk.laplace(
        nets["weights"],
        data,
        likelihood=lik,
        hessian_approx=lk.DenseGGN(joint=True),
        prior=lk.priors.Isotropic(1.0),
        subset=lk.subsets.all,
        hyper_tuner=None,
        generator=torch.Generator().manual_seed(0),
    )
}

for name, prior in priors.items():
    post = lk.gp.fsp_laplace(
        nets[name],
        data,
        likelihood=lik,
        prior=prior,
        context=lk.gp.grid(-2.0, 2.0, 100),
        rank=100,
        subset=lk.subsets.all,
        generator=torch.Generator().manual_seed(0),
    )
    posteriors[f"FSP, {name}"] = post
    evals = post.prior_evals
    print(post)
    print(f"  prior precision in the basis: {evals.min():.1e} to {evals.max():.1e}")

    # A = D_M^2 + U_M^T G U_M; the Gaussian likelihood scales the GGN by 1/sigma^2.
    a = torch.diag(evals) + post.n_data * post.gram / lik.sigma.double() ** 2
    d = torch.linalg.eigvalsh((a + a.mT) / 2)
    small = int((d <= torch.finfo(d.dtype).eps * len(d) * d.max()).sum())
    print(f"  eigenvalues of A at the rounding level: {small} of {len(d)}")

# %% [markdown]
# `rank=38 of 38` says that the RBF posterior has 38 directions, out of 2701
# weights. The RBF kernel matrix on 100 points of $[-2, 2]$ is close to
# singular, so the basis has only 38 columns above the rounding level. The
# Matérn-½ kernel is rougher, and its basis has 81 columns. Its $D_M^2$
# spans more than 20 decades, though, and $A$ has 21 eigenvalues at the
# rounding level: directions that neither the prior nor the data pin down.
# The pseudo-inverse drops them, and $81 - 21 = 60$ are left, the 60 of
# `rank=60 of 81`. So the variance truncation drops nothing here, and
# nothing for the RBF prior either, whose $A$ has no eigenvalue at the
# rounding level and keeps all 38.
#
# ## Mean and variance
#
# Here are the three predictives, $\pm 2$ standard deviations, with the true
# function dashed.

# %%
t = x_test.squeeze()
preds = {name: lk.predict(posterior, x_test) for name, posterior in posteriors.items()}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)

for ax, (name, pred) in zip(axes, preds.items(), strict=True):
    mean, std = pred.mean.squeeze(), pred.stddev.squeeze()
    ax.plot(t, torch.sin(2 * torch.pi * t), "k--", linewidth=1, label="sin(2πx)")
    ax.plot(t, mean, label="mean")
    ax.fill_between(t, mean - 2 * std, mean + 2 * std, alpha=0.3, label="mean ± 2 std")
    ax.scatter(x.squeeze(), y.squeeze(), s=6, color="0.4")
    ax.set_title(name)
    ax.set_ylim(-5.0, 5.0)

axes[0].legend(loc="lower left")
plt.show()

for name, pred in preds.items():
    mean, std = pred.mean.squeeze(), pred.stddev.squeeze()
    at = [0, 100, 200]
    print(
        f"{name:<15} mean at x = -2, 0, 2: {[round(float(mean[i]), 2) for i in at]}",
        f" std: {[round(float(std[i]), 2) for i in at]}",
    )

# %% [markdown]
# Away from the data, both FSP posteriors go back to the prior. At
# $x = \pm 2$, their means are between $-0.27$ and $0.37$, close to the prior
# mean of zero, and their standard deviations are 0.98 to 1, the prior's. The
# weight-space posterior follows the network, which keeps going: its mean is
# $-1.82$ at $x = -2$ and $1.92$ at $x = 2$, with standard deviations of 1.49
# and 1.51. That width comes from $\lambda$ and the architecture, and nothing
# in the weight prior said how wide it should be.
#
# The two kernels differ in the gap. With the RBF prior, the mean keeps the
# sine's shape through the gap. With the Matérn-½ prior, the mean drops back
# to zero right next to the data. At $x = 0$, the standard deviations are 0.8
# (weights), 0.78 (RBF) and 0.84 (Matérn-½), all below the prior's 1, since
# the data on both sides still inform the function there.
#
# ## Test NLL
#
# Finally, the held-out NLL on the test grid, split into the data
# ($0.5 \le |x| \le 1$), the gap ($|x| < 0.5$) and outside ($|x| > 1$).

# %%
region = {
    "data": (t.abs() >= 0.5) & (t.abs() <= 1.0),
    "gap": t.abs() < 0.5,
    "outside": t.abs() > 1.0,
}
print(f"{'':<15}" + "".join(f"{r:>9}" for r in region))

for name, pred in preds.items():
    nll = -pred.log_prob(y_test).squeeze()
    print(f"{name:<15}" + "".join(f"{nll[m].mean():>9.3f}" for m in region.values()))

# %% [markdown]
# On the data, the three are close, and the Matérn-½ posterior is a bit
# worse ($-0.580$, against $-0.703$ and $-0.719$). In the gap, the RBF prior
# is the best by far, at 0.092 against 0.500 for the weight-space posterior:
# its mean follows the sine. The Matérn-½ prior is the worst there, at 0.899,
# since its mean is zero where the sine isn't. Outside, both FSP posteriors
# beat the weight-space one, with 0.921 and 1.081 against 1.793. A mean of
# zero with a standard deviation of one is about right for a sine there,
# while the weight-space mean runs off.
#
# The paper reports this toy only as figures, so there are no numbers to
# check ours against.
#
# ::: {.callout-note}
# FSP-Laplace's predictive is the linearized one, `lk.predict(post, x)`.
# Sampling weights from the posterior and pushing them through the network
# (`lk.WeightSamples`) runs, but it can be far off. The covariance is large
# along the directions that the context points barely constrain, and the
# linearization doesn't hold that far from $\theta_*$.
# :::
#
# [^cinquin2024]: Cinquin, T., Pförtner, M., Fortuin, V., Hennig, P., & Bamler, R. (2024). FSP-Laplace: Function-space priors for the Laplace approximation in Bayesian deep learning. *NeurIPS*. <https://arxiv.org/abs/2407.13711>
