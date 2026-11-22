# %% [markdown]
# ---
# title: Nyström and variational GPs
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# from laplacekit import function_space as fs
#
# ella = lk.gp.ELLA(lk.gp.from_data(100), rank=15)
# gp = fs.laplace(model, loader, likelihood=lik, method=ella, subset=lk.subsets.all)
# pred = lk.predict(gp, x_test)  # the same calls as for a weight-space posterior
#
# valla = lk.gp.VaLLA(lk.gp.kmeans(20))
# gp = fs.laplace(model, loader, likelihood=lik, method=valla, subset=lk.subsets.all)
# gp = lk.gp.fit_variational(
#     gp, loader, n_steps=1500, lr=1e-2, learn_prior=False, learn_noise=False
# )
# ```
# :::
#
# The [function-space page](function_space.py) showed that the linearized Laplace approximation is a Gaussian process over the network's outputs, and that we can condition it on a random subset of the data.
# But a subset throws the other points away.
# On this page, we look at ELLA (Deng et al., 2022)[^deng2022], which keeps all of them and approximates the kernel instead.
# In the second half, we look at VaLLA (Ortega et al., 2024)[^ortega2024], which conditions the GP on a few inducing points and learns how much to trust them.
#
# ## The linearized Laplace GP
#
# Let $\theta_*$ be the MAP estimate and $J(x)$ the $E \times P$ Jacobian of the network's $E$ outputs with respect to its $P$ parameters at $\theta_*$.
# With the isotropic prior $\Sigma_0 = \lambda^{-1} I$, the linearized network is a GP with mean $f(x; \theta_*)$ and kernel
#
# $$
#   k(x, x') = J(x) \, \Sigma_0 \, J(x')^\top = \lambda^{-1} J(x) J(x')^\top .
# $$
#
# Let $H_i$ be the $E \times E$ Hessian of the negative log-likelihood with respect to the outputs at $x_i$.
# In weight space, the Laplace posterior has covariance
#
# $$
#   \Sigma = \Big( \lambda I + \sum_{i=1}^N J(x_i)^\top H_i J(x_i) \Big)^{-1} ,
# $$
#
# and the output covariance at $x_*$ is $J(x_*) \Sigma J(x_*)^\top$.
# This needs a $P \times P$ matrix.
# In function space, the same covariance needs an $NE \times NE$ one.
# Both are cubic in their size, so neither works when the network and the data are both large.
# A subset of $m$ points brings the second down to $mE \times mE$, at the price of the $N - m$ points it drops.
#
# ## The Nyström approximation
#
# ELLA keeps every data point and approximates the kernel instead.
# We pick $m$ inputs and, for each, one row of its Jacobian (all $E$ rows if we want).
# Stacked, they form $R$, an $m \times P$ matrix.
# Let $(U, \Lambda)$ be the top $r$ eigenpairs of the $m \times m$ matrix $R R^\top$ (here $\Lambda$ holds eigenvalues, not the output Hessians of the function-space page).
# The columns of
#
# $$
#   V = R^\top U \Lambda^{-1/2} \in \mathbb{R}^{P \times r}
# $$
#
# are orthonormal, and they span the top $r$ directions of the rows of $R$.
# The Nyström approximation projects every Jacobian onto them,
#
# $$
#   J(x) J(x')^\top \approx J(x) V V^\top J(x')^\top = \varphi(x)^\top \varphi(x') ,
#   \qquad \varphi(x) = V^\top J(x)^\top \in \mathbb{R}^{r \times E} .
# $$
#
# This is the kernel of a linear model with $r$ weights $w \sim \mathcal{N}(0, \lambda^{-1} I)$ and features $\varphi$.
# Its posterior covariance is the $r \times r$ matrix
#
# $$
#   S = \Big( \lambda I + \sum_{i=1}^N \varphi(x_i) H_i \varphi(x_i)^\top \Big)^{-1} ,
# $$
#
# and the output covariance at $x_*$ is $\varphi(x_*)^\top S \varphi(x_*)$.
# The fit is a pass over the data to pick the $m$ points, an $m \times m$ eigendecomposition for $V$, and one more pass for the sum, with $r$ Jacobian-vector products per input.
#
# Note that the prior has to be isotropic for $\lambda$ to be applied late, i.e., after the fit.
# The basis comes from the Nyström matrix of the prior kernel, $\lambda^{-1} R R^\top$, and its eigenvectors $U$ are those of $R R^\top$ for every $\lambda$.
# So $V$ and the sum are the same for every $\lambda$.
# Since $V$ has orthonormal columns, the prior on $w$ is $V^\top \Sigma_0 V = \lambda^{-1} I$, and $\lambda$ enters only through the $\lambda I$ in $S$.
# A new $\lambda$ then costs one $r \times r$ solve.
# With another $\Sigma_0$, the eigenvectors of $R \Sigma_0 R^\top$ change with $\Sigma_0$, so a new prior would need a new basis and a new pass over the data.

# %%
import time

import matplotlib.pyplot as plt
import torch
from torch import nn

import laplacekit as lk
from laplacekit import function_space as fs

# %% [markdown]
# ## Data and model
#
# We use the data and the network of the [regression page](regression.py): 150 noisy samples of a sine on $[0, 8]$ and a tanh network with 151 weights.
# For testing, we draw 300 held-out points on $[-2, 10]$.
# About a third of them lie past the ends of the data, where we want wide error bars.

# %%
torch.set_default_dtype(torch.float64)
torch.manual_seed(711)

x = torch.rand(150, 1) * 8.0
y = torch.sin(x) + 0.3 * torch.randn(150, 1)
x_test = torch.linspace(-2.0, 10.0, 200).reshape(-1, 1)

model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(1000):
    opt.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    opt.step()

x_out = torch.rand(300, 1) * 12.0 - 2.0
y_out = torch.sin(x_out) + 0.3 * torch.randn(300, 1)
inside = ((x_out >= 0.0) & (x_out <= 8.0)).squeeze()
print("held-out points on [0, 8]:", int(inside.sum()), "of", len(inside))

# %% [markdown]
# ## Exact, subset, and Nyström
#
# The reference is the exact linearized Laplace: `lk.laplace` with the dense GGN over all weights.
# Its default hyper tuner fits $\lambda$ and the noise $\sigma$ on the marginal likelihood.
# We give both to the GPs, so that the five posteriors differ only in the method.
# The GPs are a subset of 50 points, and ELLA with 100 points from the data at ranks 5, 10 and 15.


# %%
data = [(x, y)]


def exact_fit():
    return lk.laplace(
        model,
        data,
        likelihood=lk.Gaussian(),
        hessian_approx=lk.DenseGGN(joint=True),
        subset=lk.subsets.all,
        generator=torch.Generator().manual_seed(0),
    )


def gp_fit(approx):
    return fs.laplace(
        model,
        data,
        likelihood=lik,
        method=approx,
        prior=prior,
        subset=lk.subsets.all,
        generator=torch.Generator().manual_seed(0),
    )


def timed(fit, *args):
    start = time.perf_counter()
    posterior = fit(*args)

    return posterior, time.perf_counter() - start


posteriors = {"Exact": timed(exact_fit)}
lik, prior = posteriors["Exact"][0].likelihood, posteriors["Exact"][0].prior
print(
    "prior precision:",
    f"{float(prior.prior_precision):.3g}",
    " sigma:",
    f"{float(lik.sigma):.3g}",
)

posteriors["Subset, m = 50"] = timed(gp_fit, fs.SubsetOfData(50))

for r in (5, 10, 15):
    posteriors[f"ELLA, r = {r}"] = timed(
        gp_fit, lk.gp.ELLA(lk.gp.from_data(100), rank=r)
    )

print(posteriors["ELLA, r = 15"][0])

# %% [markdown]
# Why not rank 50?
# With 100 points, we could ask for it, but ELLA refuses.

# %%
try:
    gp_fit(lk.gp.ELLA(lk.gp.from_data(100), rank=50))
except ValueError as err:
    print(err)

# %% [markdown]
# This network has one input and smooth tanh units, so its Jacobians on $[0, 8]$ span few directions.
# The eigenvalues of $R R^\top$ decay fast, and only 20 of them are above the tolerance (machine precision, scaled by the size of $R$ and the largest eigenvalue).
# Directions below it would only add rounding noise, so ELLA stops at the usable rank instead of returning them.
# The last few of the 20 sit close to the tolerance, so a rank near 20 could be refused on another machine.
# We use at most 15.
#
# ## Error bars
#
# Here are the predictive standard deviations on the test grid, on a log scale.
# The dotted lines mark the ends of the data.

# %%
t = x_test.squeeze()
fig, ax = plt.subplots(figsize=(8, 4))

for name, (posterior, _) in posteriors.items():
    std = lk.predict(posterior, x_test).stddev.squeeze()
    ax.plot(t, std, label=name, linestyle="--" if name == "Exact" else "-")

for end in (0.0, 8.0):
    ax.axvline(end, color="0.6", linestyle=":")

ax.set_yscale("log")
ax.set_xlabel("x")
ax.set_ylabel("predictive std")
ax.legend()
plt.show()

# %% [markdown]
# Next is the held-out NLL on the data range, left of it ($x < 0$) and right of it ($x > 8$), with the fit times.
# The exact posterior's time includes its tuning of $\lambda$ and $\sigma$.

# %%
left = (x_out < 0.0).squeeze()
right = (x_out > 8.0).squeeze()


def nll_table(posteriors):
    print(
        f"{'':<16} {'NLL on [0, 8]':>14} {'left':>7} {'right':>7} "
        f"{'fit (s)':>8}"
    )

    for name, (posterior, seconds) in posteriors.items():
        nll = -lk.predict(posterior, x_out).log_prob(y_out).squeeze()
        print(
            f"{name:<16} {nll[inside].mean():>14.3f} {nll[left].mean():>7.3f} "
            f"{nll[right].mean():>7.3f} {seconds:>8.3f}"
        )


nll_table(posteriors)

with torch.no_grad():
    error = (model(x_out) - torch.sin(x_out)).abs().squeeze()

print(
    "mean |f(x) - sin(x)| left:",
    f"{error[left].mean():.2f}",
    " right:",
    f"{error[right].mean():.2f}",
)

# %% [markdown]
# On the data, all five agree: the standard deviation is about $\sigma$, and the NLL is between 0.209 and 0.213.
# They differ past the ends.
# ELLA at rank 15 is the closest of the three ranks to the exact posterior: it matches it on the right and is narrower on the left.
# The subset of 50 points is wider than the exact posterior everywhere, since it conditions on a third of the data.
#
# Past the ends, the two sides behave differently.
# On the left, the network stays close to the sine (a mean error of 0.29), so the wide exact band costs more than it saves.
# Narrower bands score better there: rank 5 has 0.780, rank 10 has 1.088 and rank 15 has 1.577, against the exact 1.692.
# On the right, the network is off by 0.97 on average, and the wide band pays off.
# Rank 5, whose band hardly grows, scores 5.261 there against the exact 1.447.
# Rank 10 is worse than the exact posterior too, at 1.574, and rank 15 is close to it, at 1.452.
#
# So a low held-out NLL doesn't make an approximation good.
# ELLA approximates the exact posterior, so its quality is how close it gets to it.
# In the plot, that closeness improves with the rank.
#
# All fits here take a fraction of a second, so the times say little at this size.
# They grow differently with the size, though: the solve is cubic in $P$ for the exact posterior, and in $mE$ for the subset.
# ELLA's eigendecomposition is cubic in $m$, its solve in $r$, and its pass over the data linear in $N$.
#
# ## Variational inducing points
#
# ELLA approximates the kernel and keeps the exact posterior's form.
# VaLLA keeps the kernel and approximates the posterior.
# It picks $M$ inducing inputs $z_1, \dots, z_M$ and one output coordinate for each (with $E = 1$ there is only one).
# Let $f_z$ be the GP's values there.
# Under the prior, they have covariance $K_{zz} = \lambda^{-1} J_z J_z^\top$, where $J_z$ stacks the $M$ Jacobian rows, and $K_{zx} = \lambda^{-1} J_z J(x)^\top$ is their covariance with the outputs at $x$.
#
# VaLLA keeps the posterior mean at the network's output $f(x; \theta_*)$, so only the covariance is learned.
# Its posterior over $f_z$ is the prior conditioned on pseudo-observations of $f_z$ with noise covariance $(L L^\top)^{-1}$, for a learned lower-triangular $M \times M$ matrix $L$.
# That is, $q(f_z) = \mathcal{N}\big(f_z(\theta_*), (K_{zz}^{-1} + L L^\top)^{-1}\big)$.
# Passed on to any $x$ through the prior, it gives the output covariance
#
# $$
#   \Sigma(x) = K_{xx} - K_{xz} L \big( I + L^\top K_{zz} L \big)^{-1} L^\top K_{zx} .
# $$
#
# This needs $M$ Jacobian rows and an $M \times M$ Cholesky factor, but never a $P \times P$ or $NE \times NE$ matrix.
# Both $q$ and the prior have the same mean, so with $A = L^\top K_{zz} L$, the KL divergence between them is
#
# $$
#   \mathrm{KL}\big( q(f_z) \,\|\, p(f_z) \big)
#   = \tfrac{1}{2} \Big( \log \det (I + A) - \operatorname{tr}\big( (I + A)^{-1} A \big) \Big) .
# $$
#
# VaLLA fits $L$ by minimizing an $\alpha$-divergence objective,
#
# $$
#   \mathcal{L}_\alpha = -\sum_{i=1}^N \frac{1}{\alpha}
#     \log \mathbb{E}_{q} \big[ p(y_i \mid f_i)^\alpha \big]
#   + \mathrm{KL}\big( q(f_z) \,\|\, p(f_z) \big) ,
# $$
#
# where $f_i \sim \mathcal{N}(\mu_i, v_i)$ under $q$, with $\mu_i = f(x_i; \theta_*)$ and $v_i = \Sigma(x_i)$.
# On a minibatch of $B$ points, the sum is $N / B$ times the batch's.
# As $\alpha \to 0$, the data term becomes $\mathbb{E}_q[\log p(y_i \mid f_i)]$, and $\mathcal{L}_0$ is the negative ELBO.
# At $\alpha = 1$, the paper's default, the data term is the log of the predictive density.
# For a Gaussian likelihood, both are in closed form:
#
# $$
#   \alpha = 0 : \;\; \log \mathcal{N}(y_i \mid \mu_i, \sigma^2) - \frac{v_i}{2 \sigma^2} ,
#   \qquad
#   \alpha = 1 : \;\; \log \mathcal{N}(y_i \mid \mu_i, \sigma^2 + v_i) .
# $$
#
# `lk.gp.fit_variational` runs Adam on $L$ and the inducing inputs.
# It can also learn the prior precision $\lambda$ and the noise $\sigma$ on the same objective.
# But the paper advises against that for the ELBO (its §3.4), and we will see why below.
# On its 1-D toy problem, the paper holds $\lambda$ and $\sigma$ at the values that the exact linearized Laplace gets from its marginal likelihood (§5.1).
# We do the same: we keep the exact posterior's $\lambda$ and $\sigma$, put 20 inducing points on the data by k-means, and train for 1500 steps at $\alpha = 1$ and at $\alpha = 0$.

# %%
valla = fs.laplace(
    model,
    data,
    likelihood=lik,
    method=lk.gp.VaLLA(lk.gp.kmeans(20)),
    prior=prior,
    subset=lk.subsets.all,
    generator=torch.Generator().manual_seed(0),
)
print(valla)


def hold(alpha):
    return lk.gp.fit_variational(
        valla,
        data,
        n_steps=1500,
        lr=1e-2,
        alpha=alpha,
        learn_prior=False,
        learn_noise=False,
        generator=torch.Generator().manual_seed(0),
    )


for alpha in (1.0, 0.0):
    posteriors[f"VaLLA, α = {alpha:g}"] = timed(hold, alpha)

# %% [markdown]
# ### Learning $\lambda$ and $\sigma$ with the ELBO
#
# Now we let the ELBO ($\alpha = 0$) learn $\lambda$ and $\sigma$ too, starting from the same values.
# We train in three rounds of 500 steps and print both after each round.
# Note that each call to `fit_variational` starts a new Adam optimizer.


# %%
def learn():
    gp = valla

    for k in range(1, 4):
        gp = lk.gp.fit_variational(
            gp,
            data,
            n_steps=500,
            lr=1e-2,
            alpha=0.0,
            generator=torch.Generator().manual_seed(0),
        )
        print(
            f"after {500 * k:>4} steps: prior precision "
            f"{float(gp.prior.prior_precision):.3g},",
            f"sigma {float(gp.likelihood.sigma):.3g}",
        )

    return gp


posteriors["VaLLA, learned λ"] = timed(learn)

# %% [markdown]
# $\sigma$ stays near 0.3, but $\lambda$ keeps growing, to $7.43 \times 10^3$ after 1500 steps.
# Why?
# The mean is fixed at the network's output, so at $\alpha = 0$, the variance $v_i$ only enters the data term as the penalty $v_i / 2 \sigma^2$.
# The KL is smallest when $q$ equals the prior, and $A$ goes to zero as the prior's variance does.
# With $L$ and the inducing inputs held fixed, a larger $\lambda$ makes both terms smaller.
# So $\mathcal{L}_0$ keeps falling as $\lambda$ grows, and has no minimum in it.
# This is the paper's point in §3.4: with the mean fixed, the ELBO prefers a predictive variance of zero, so it can't tune the prior.
#
# The paper's $\alpha = 1$ objective, with $\lambda$ and $\sigma$ learned, has a related problem.
# It can overfit and shrink the predictive variances too much, so the paper stops it early on a validation set (§3.4, §3.5 and §5.1).
# Either way, $\lambda$ and $\sigma$ need something besides the training objective: fixed values from the marginal likelihood, as here, or early stopping on validation data.
#
# Here are the error bars of the exact posterior, ELLA at rank 15 and the three VaLLA fits.
# Below them are the inducing inputs: where k-means put them, and where the two fits with fixed $\lambda$ and $\sigma$ moved them.

# %%
fig, (ax, rug) = plt.subplots(
    2, 1, figsize=(8, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
shown = [
    "Exact",
    "ELLA, r = 15",
    "VaLLA, α = 1",
    "VaLLA, α = 0",
    "VaLLA, learned λ",
]

for name in shown:
    std = lk.predict(posteriors[name][0], x_test).stddev.squeeze()
    ax.plot(t, std, label=name, linestyle="--" if name == "Exact" else "-")
    print(f"{name:<16} std at x = -2: {std[0]:.3f}, at x = 10: {std[-1]:.3f}")

for end in (0.0, 8.0):
    ax.axvline(end, color="0.6", linestyle=":")

ax.set_yscale("log")
ax.set_ylabel("predictive std")
ax.legend()

rows = {
    "k-means": valla.inducing,
    "α = 1": posteriors["VaLLA, α = 1"][0].inducing,
    "α = 0": posteriors["VaLLA, α = 0"][0].inducing,
}

for i, (name, z) in enumerate(rows.items()):
    print(f"inducing inputs, {name:<7}: from {z.min():.2f} to {z.max():.2f}")
    rug.plot(
        z.squeeze(),
        torch.full((z.shape[0],), float(i)),
        "|",
        color="k",
        markersize=10,
    )

rug.set_yticks(range(len(rows)), list(rows))
rug.set_ylim(-0.7, len(rows) - 0.3)
rug.set_xlabel("x")
plt.show()

# %% [markdown]
# Next is the held-out NLL of all eight posteriors.
# VaLLA's times are those of its 1500 training steps.

# %%
nll_table(posteriors)

# %% [markdown]
# With $\lambda$ and $\sigma$ fixed, both VaLLA fits follow the exact posterior.
# On the data, their bands are close to it, with the $\alpha = 1$ one a bit above.
# Past the ends, both are a bit wider than the exact band: at $x = -2$, 4.733 ($\alpha = 1$) and 4.565 ($\alpha = 0$) against 4.265, and at $x = 10$, 1.578 and 1.327 against 1.100.
# ELLA at rank 15 is narrower than the exact band on the left (3.277).
# The inducing inputs moved a little and stayed close to the data, between 0 and 8.07.
# With the ELBO-learned $\lambda$, the band is flat at about $\sigma$ everywhere (0.295 at $x = -2$, 0.296 at $x = 10$).
# That is, the posterior has collapsed to the network's prediction plus noise.
#
# In the table, $\alpha = 0$ scores 0.210 on the data, like the exact posterior, and $\alpha = 1$ is a bit worse, at 0.224.
# On the right, where the network is off by 0.97, the wider bands pay off: 1.230 and 1.288, against the exact 1.447.
# On the left, where the network is close to the sine, they cost a little: 1.814 and 1.763 against 1.692.
# The collapsed fit is the opposite: it scores 0.884 on the left, like the narrow bands of rank 5, and 6.597 on the right.
# The VaLLA fits take a few seconds each, against a fraction of a second for ELLA, since every step computes the Jacobians of the 20 inducing inputs and of the 150 data points.
#
# To sum up, ELLA approximates the exact linearized Laplace posterior and gets closer with the rank.
# It has no training loop, and $\lambda$ is a knob we set after the fit, with the marginal likelihood or a validation set, as for the exact posterior.
# VaLLA, on the other hand, trains its posterior by gradient steps, so it costs more here, and it needs $\lambda$ and $\sigma$ from somewhere else.
# With the marginal-likelihood values, it came out close to the exact posterior on this data, a little wider past the ends.
# If we let it learn them, we should hold out a validation set and stop early, as the paper does.

# %% [markdown]
# [^deng2022]: Deng, Z., Zhou, F., & Zhu, J. (2022). Accelerated linearized Laplace approximation for Bayesian deep learning. *NeurIPS*. <https://arxiv.org/abs/2210.12642>
#
# [^ortega2024]: Ortega, L. A., Santana, S. R., & Hernández-Lobato, D. (2024). Variational linearized Laplace approximation for Bayesian deep learning. *ICML*. <https://arxiv.org/abs/2302.12565>
