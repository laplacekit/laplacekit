# %% [markdown]
# ---
# title: Function-space approximations
# jupyter: python3
# ---
#
# ::: {.callout-tip}
# ## New to protocols?
# The contract below is a protocol. Subclass it, as in
# `class MyClass(lk.function_space.FunctionSpaceMethod):`, and your editor marks
# a missing method in red.
# [Protocols in one page](protocols.py) shows how.
# :::
#
# A function-space approximation turns the data into a Gaussian process over
# the network's outputs. `fs.laplace` runs it, with the same probe, parameter
# subset, passes and generators as `lk.laplace`:
#
# ```python
# class FunctionSpaceMethod[B](Protocol):
#     @property
#     def requires(self) -> Requirements: ...
#
#     def plan(self, params: Params, ctx: FitContext, prior: PriorWithCovariance, /) -> Pass[B]: ...
# ```
#
# `B` is the posterior it returns. Unlike a weight-space `Staged`, `plan` also gets
# the prior, because the GP's kernel contains the prior covariance. We write
# the deterministic training conditional (DTC)[^dtc]: a GP that sees the data
# only through its outputs at $m$ fixed inputs $Z$, in one pass.

# %%
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

import laplacekit as lk
from laplacekit import function_space as fs
from laplacekit.core import (
    Capabilities,
    Likelihood,
    ModelFn,
    PriorWithCovariance,
    Requirements,
    curvature_scale,
    output_hessian,
    stat_dtype,
)


@lk.core.pytree
class Sum:
    """A sum of [k, k] terms over the examples seen, and their count."""

    gram: Tensor
    count: Tensor

    @property
    def total_weight(self) -> Tensor:
        return self.count

    def __add__(self, other: "Sum", /) -> "Sum":
        return Sum(self.gram + other.gram, self.count + other.count)

    def scaled(self, s: Tensor, /) -> "Sum":
        return Sum(s * self.gram, s * self.count)


@lk.core.pytree
class Cross:
    """One pass: the average of K_Zn H_n K_nZ over the data, with K_nZ = J_n Σ₀ J_Zᵀ."""

    rows: Tensor  # Σ₀ J_Zᵀ, as [m E, P] rows

    @property
    def requires(self) -> Requirements:
        return Requirements(vjp=True, vmap=True, needs_output_hessian=True, weights=False)

    def init(self, params, ctx, /) -> Sum:
        k = self.rows.shape[0]

        return Sum(self.rows.new_zeros(k, k), self.rows.new_zeros(()))

    def update(self, stat: Sum, params, batch, generator, ctx, /) -> Sum:
        f = ctx.model(params, batch.inputs)
        n = f.shape[0]

        k_nz = fs.jacobian(ctx.model, params, batch.inputs, self.rows.dtype) @ self.rows.T
        hess = output_hessian(ctx.likelihood, f, generator=generator, inputs=batch.inputs)

        # H_n applied to each column of K_nZ, one `mv` per column under vmap
        cols = k_nz.reshape(*f.shape, -1).to(f.dtype)
        h_k = torch.vmap(hess.mv, in_dims=-1, out_dims=-1)(cols).reshape(k_nz.shape)

        return Sum(stat.gram + k_nz.T @ h_k.to(k_nz.dtype), stat.count + n)

    def finalize(self, stat: Sum, /) -> Sum:
        return Sum(stat.gram / stat.count, stat.count)


@lk.core.pytree
class InducingGP:
    """The DTC posterior over the outputs."""

    model: ModelFn = lk.core.static()
    likelihood: Likelihood
    mean: dict[str, Tensor]
    prior: PriorWithCovariance
    caps: Capabilities = lk.core.static()
    rows: Tensor  # Σ₀ J_Zᵀ
    k_zz: Tensor
    gram: Tensor  # the per-example average from `Cross`
    n_data: Tensor  # the number of examples

    @property
    def requires(self) -> Requirements:
        return Requirements(vjp=True, vmap=True)

    def at(self, inputs, /) -> lk.core.OutputGaussian:
        self.caps.check(self.requires, "InducingGP.at")

        f = self.model(self.mean, inputs)
        n, e = f.shape[0], f[0].numel()

        j = fs.jacobian(self.model, self.mean, inputs, self.rows.dtype)
        k_xz = j @ self.rows.T
        k_xx = torch.einsum(
            "nep,nfp->nef",
            j.reshape(n, e, -1),
            fs.cov_rows(self.prior, self.mean, j).reshape(n, e, -1),
        )

        # Whitened by L, the Cholesky factor of K_ZZ: V = L⁻¹ K_Zx and W = L⁻¹ (s c S) L⁻ᵀ.
        # Then K_xZ K_ZZ⁻¹ K_Zx - K_xZ (K_ZZ + s c S)⁻¹ K_Zx = Vᵀ (I + W)⁻¹ W V.
        # 1 / σ² for a lk.Gaussian, None if the likelihood has none.
        c = curvature_scale(self.likelihood)
        weight = self.n_data * (1.0 if c is None else c)
        eye = torch.eye(len(self.k_zz), dtype=self.k_zz.dtype)
        chol = torch.linalg.cholesky(self.k_zz + 1e-10 * eye)  # jitter: K_ZZ can be near singular

        def whiten(m: Tensor) -> Tensor:
            return torch.linalg.solve_triangular(chol, m, upper=False)

        v = whiten(k_xz.T)
        w = whiten(whiten(weight * self.gram).T)
        u = torch.linalg.solve(eye + w, w @ v)

        k = len(eye)
        cov = k_xx - torch.einsum("kne,knf->nef", v.reshape(k, n, e), u.reshape(k, n, e))

        return lk.core.OutputGaussian(
            f, lk.core.PerInputCov(0.5 * (cov + cov.mT), tuple(f.shape[1:]))
        )


@lk.core.pytree
class Inducing:
    """DTC with the inducing inputs `z`, in one pass."""

    z: Tensor

    @property
    def requires(self) -> Requirements:
        return Requirements(vjp=True, vmap=True, needs_output_hessian=True, weights=False)

    def plan(self, params, ctx, prior, /):
        dtype = stat_dtype(next(iter(params.values())).dtype)
        j_z = fs.jacobian(ctx.model, params, self.z, dtype)
        rows = fs.cov_rows(prior, params, j_z)

        def done(stat: Sum) -> lk.core.Done[InducingGP]:
            gp = InducingGP(
                ctx.model,
                ctx.likelihood,
                params,
                prior,
                ctx.caps,
                rows,
                j_z @ rows.T,
                stat.gram,
                stat.count,
            )

            return lk.core.Done(gp)

        return lk.core.chain(Cross(rows), then=done)


# %% [markdown]
# We try it on the sine of the [regression page](../quickstart/regression.py),
# with 10 inducing inputs spread over the data, $[0, 8]$. As a reference, we fit
# the full dense GGN over the same 151 weights, which is the exact GP of the
# linearized network.

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

lik = lk.Gaussian(sigma=0.3)
z = torch.linspace(0.0, 8.0, 10).reshape(-1, 1)

gp = fs.laplace(model, [(x, y)], likelihood=lik, method=Inducing(z), subset=lk.subsets.all)
full = lk.laplace(
    model,
    [(x, y)],
    likelihood=lik,
    subset=lk.subsets.all,
    hessian_approx=lk.DenseGGN(joint=True),
    hyper_tuner=None,
)

grid = torch.tensor([[-2.0], [4.0], [10.0]])

for name, posterior in [("DTC", gp), ("full GGN", full)]:
    std = lk.predict(posterior, grid).stddev.squeeze()
    print(f"{name:<8} std at x = -2, 4, 10:", [round(float(s), 3) for s in std])

# %%
t = x_test.squeeze()
dtc, ref = lk.predict(gp, x_test), lk.predict(full, x_test)
mean, std = dtc.mean.squeeze(), dtc.stddev.squeeze()

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(x.squeeze(), y.squeeze(), s=8, color="0.5", label="data")
ax.plot(t, mean, label="mean")
ax.fill_between(t, mean - 2 * std, mean + 2 * std, alpha=0.3, label="DTC, mean ± 2 std")

for sign in (-1, 1):
    band = ref.mean.squeeze() + sign * 2 * ref.stddev.squeeze()
    ax.plot(t, band, "k--", lw=0.8, label="full GGN" if sign > 0 else None)

ax.plot(z.squeeze(), torch.full((10,), -3.5), "|", color="k", markersize=12, label="Z")
ax.set_ylim(-4.5, 4.0)
ax.legend(loc="upper center", ncols=3, fontsize="small")
plt.show()

# %% [markdown]
# With 10 inducing inputs, DTC is close to the full posterior: its standard
# deviations at $x = -2$, 4 and 10 are 1.671, 0.308 and 0.569, against 1.684,
# 0.308 and 0.57. In the plot, its band lies on the dashed lines of the full
# GGN.
#
# ## The contract
#
# `fs.laplace` probes the model, picks the parameters, and calls
# `plan(params, ctx, prior)` with the prior bound. `plan` returns a `Pass`,
# built with `lk.core.chain(pass, then=...)` as for a weight-space `Staged` (see
# [Curvatures](curvature.py)). The core runs the pass over the data and hands
# what `finalize` returns to `then`, which returns the next `Pass` or
# `lk.core.Done(posterior)`. Our method has one pass. `plan` computes what depends on
# $Z$ alone before any data, and `Cross` sums over the data.
#
# The pass follows the curvature conventions. Its statistic is a monoid
# (`init` is zero and `+` adds), and `finalize` returns the per-example
# average with the total weight, which the posterior multiplies back. `requires`
# names what the pass needs, and `fs.laplace` checks it against the probe
# before it reads the data. With `weights=False`, weighted batches are
# refused, since our sum has no place for the weights.
#
# The posterior needs `model`, `likelihood`, `requires` and `at(inputs)`, which
# returns an `lk.core.OutputGaussian` with one $[E, E]$ block per input
# (`lk.core.PerInputCov`). `lk.predict` and `lk.predict_outputs` then take it like a
# weight-space posterior. It must also be a dataclass with a `likelihood`
# field. For a `lk.Gaussian` with one $\sigma$, the passes see unit noise, and
# `fs.laplace` swaps the real likelihood in with `copy.replace` at the end.
# `at` reads $\sigma$ through `curvature_scale` on every call, so a new
# $\sigma$ needs no refit.
#
# `laplacekit.function_space` has four helpers for such methods.
# `fs.jacobian(model, params, inputs, dtype)` gives the $[N E, P]$ Jacobian of
# the outputs, and `fs.cov_rows(prior, params, rows)` applies the prior
# covariance $\Sigma_0$ to each row of a $[k, P]$ matrix; we use both.
# `fs.rows_of(inputs, index)` picks examples out of a batch of inputs, which
# may be a dict or another pytree, and `fs.is_shared(leaf)` says whether an
# inputs leaf is shared by all examples instead of having one row each.
#
# A real method also needs a guard against a model that changed after the
# fit. `fs.SubsetGP` and the posteriors of `laplacekit_gp` record the version
# counters of the module's fixed parameters and raise in `at` when one
# changed. We left that out.
#
# ## The math
#
# The linearized network is a GP with the kernel $k(x, x') = J(x) \Sigma_0
# J(x')^\top$. DTC conditions on the outputs $u = f(Z)$ and lets the outputs
# at the data depend on them through $f(x) \approx K_{xZ} K_{ZZ}^{-1} u$. With
# $H_n$ the output Hessian of the negative log-likelihood at $x_n$, the
# covariance of the outputs at $x$ is
#
# $$
#   \Sigma(x) = K_{xx} - K_{xZ} K_{ZZ}^{-1} K_{Zx} + K_{xZ} \Big( K_{ZZ} + \sum_{n=1}^N K_{Zn} H_n K_{nZ} \Big)^{-1} K_{Zx} .
# $$
#
# The sum is the only term over the data, so one pass computes it. For a
# Gaussian, $H_n = I / \sigma^2$: the pass sums with $\sigma = 1$, and `at`
# divides by $\sigma^2$. `at` whitens with the Cholesky factor of $K_{ZZ}$
# before it subtracts the two terms, which keeps the difference accurate when
# $K_{ZZ}$ is close to singular. With every training input in $Z$, DTC is the
# exact GP, which is the linearized Laplace posterior with the full GGN.
#
# ## The test
#
# `lk.testing` has no check for function-space approximations yet, so we
# test the laws by hand. The statistic must be the same for one batch or two
# (the monoid law). With $Z$ set to the training inputs, `at` must give the
# predictive of the dense GGN, an independent reference. We keep only 5
# training points there: with more of them close together, $K_{ZZ}$ is so
# ill-conditioned that the two sides agree to a few digits only.
# `lk.testing.check_predictive` then checks that `lk.Linearized` accepts the
# posterior.


# %%
def test_inducing():
    x_s, y_s = x[:5], y[:5]
    fit = dict(likelihood=lik, subset=lk.subsets.all)

    # The monoid law, through the front door: one batch or two give the same statistic.
    one = fs.laplace(model, [(x_s, y_s)], method=Inducing(z), **fit)
    two = fs.laplace(model, [(x_s[:2], y_s[:2]), (x_s[2:], y_s[2:])], method=Inducing(z), **fit)
    torch.testing.assert_close(two.gram, one.gram)
    torch.testing.assert_close(two.n_data, one.n_data)

    # With Z the training inputs, DTC is the exact GP: the dense GGN's predictive.
    dtc = fs.laplace(model, [(x_s, y_s)], method=Inducing(x_s), **fit)
    ref = lk.laplace(
        model, [(x_s, y_s)], hessian_approx=lk.DenseGGN(joint=True), hyper_tuner=None, **fit
    )
    torch.testing.assert_close(
        dtc.at(x_test).cov.diagonal(), lk.predict_outputs(ref, x_test).cov.diagonal()
    )

    lk.testing.check_predictive(lk.Linearized(), dtc, x_test[:7])


test_inducing()

# %% [markdown]
# [`laplacekit_gp`](https://github.com/laplacekit/laplacekit/tree/main/plugins/gp)
# is the full example, with three methods ([GP methods](../quickstart/gp.py)).
# `lk.gp.ELLA` takes two passes: one picks the points and one sums the GGN
# projected onto their basis. It checks its memory budgets before the data where
# it can.
#
# [^dtc]: Quiñonero-Candela, J., & Rasmussen, C. E. (2005). A unifying view of sparse approximate Gaussian process regression. *JMLR*, 6, 1939–1959. <https://jmlr.org/papers/v6/quinonero-candela05a.html>
