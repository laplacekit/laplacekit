# %% [markdown]
# ---
# title: Curvatures
# jupyter: python3
# ---
#
# ::: {.callout-tip}
# ## New to protocols?
# The contract below is a protocol. Subclass it, as in
# `class MyClass(lk.core.Curvature):`, and your editor marks a missing method in red.
# [Protocols in one page](protocols.py) shows how.
# :::
#
# A curvature estimates the per-example average $G$ in the Laplace precision
# $\Lambda = \Lambda_0 + N G$, where $\Lambda_0$ is the prior precision and $N$
# the number of training examples, for example the GGN $G = \frac{1}{N} \sum_i
# J_i^\top H_i J_i$. It folds batches into a statistic and turns the statistic
# into an operator:
#
# ```python
# class Curvature[S: Stat, R](Protocol):
#     @property
#     def requires(self) -> Requirements: ...
#
#     def init(self, params: Params, ctx: FitContext, /) -> S: ...
#
#     def update(
#         self, stat: S, params: Params, batch: Batch, generator: torch.Generator, ctx: FitContext, /
#     ) -> S: ...
#
#     def finalize(self, stat: S, /) -> R: ...
# ```
#
# `requires` is a read-only property, so a method reads it as `approx.requires`,
# without a call. A property or a field of a frozen dataclass both satisfy it.
#
# Here, `params` maps parameter names to tensors, and `ctx` is a `FitContext`
# holding the model function, the likelihood, the `Layout` (which parameters
# form which block, by default one block per module), and the probe's
# `Capabilities`. `batch` is a `Batch` with `inputs`, `targets` and optional
# per-example `weights`. `generator` is a fresh `torch.Generator` on the params'
# device, for all randomness. The fit seeds it from its own seed and the batch's
# position in the data. We write a Hutchinson estimate of the diagonal
# GGN, fit with it, and then test it against the contract.
#
# ## The Hutchinson diagonal
#
# We set up a small problem first: four Gaussian blobs and an MLP with 388
# parameters. We use float64, since `lk.testing` checks the laws to a tight
# tolerance.

# %%
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_blobs
from torch import nn
from torch.func import vjp, vmap

import laplacekit as lk
import laplacekit.core as core

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

x_np, y_np = make_blobs(
    n_samples=400, centers=4, cluster_std=1.5, random_state=3
)
x = torch.tensor(x_np)
y = torch.tensor(y_np)
loader = list(zip(x.split(100), y.split(100), strict=True))

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 16),
    nn.Tanh(),
    nn.Linear(16, 4),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(300):
    opt.zero_grad()
    nn.functional.cross_entropy(model(x), y).backward()
    opt.step()

lik = lk.Categorical()
print("parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
# Now the curvature itself. It needs per-example VJPs, so it requires `vjp` and
# `vmap`. The statistic is a `SumStat`: one running sum per block and the total
# weight. In `update`, we draw $S$ sign vectors from the batch's `generator`, so
# the same batch always gets the same signs and the [monoid law](statistics.qmd)
# holds exactly. `core.output_hessian` gives $H_i$ as an operator, and its
# `sqrt_mv(z)` is $R_i z$. Without a closed form it is the MC Fisher, whose
# targets are drawn from the same generator, so the curvature also declares `needs_output_hessian`. Each VJP returns one
# gradient per parameter, which we flatten into the block order given by
# `ctx.layout.blocks`. Per-example weights $w_i$ multiply
# the squares. Finally, `finalize` divides by the total weight to return the
# average, as one `Diag` per block.


# %%
@lk.core.pytree
class HutchinsonDiag:
    """Diagonal GGN from random sign vectors: the average of (J^T R z)^2."""

    n_probes: int = lk.core.static(default=16)

    @property
    def requires(self) -> lk.core.Requirements:
        return lk.core.Requirements(
            vjp=True, vmap=True, needs_output_hessian=True
        )

    def init(self, params, ctx):
        like = next(iter(params.values()))
        sums = {
            b: torch.zeros(ctx.layout.size(b), dtype=like.dtype)
            for b in ctx.layout.blocks
        }

        return core.SumStat(sums, torch.zeros((), dtype=like.dtype))

    def update(self, stat, params, batch, generator, ctx):
        p = dict(params)
        f = ctx.model(p, batch.inputs)
        w = torch.ones(batch.size) if batch.weights is None else batch.weights
        signs = torch.randint(
            0, 2, (self.n_probes, *f.shape), generator=generator
        )
        hess = core.output_hessian(ctx.likelihood, f, generator=generator)

        def pullback(x, ct):
            _, pull = vjp(lambda q: ctx.model(q, x[None]), p)

            return pull(ct[None])[0]

        sums = dict(stat.sum)

        for z in 2.0 * signs - 1.0:
            g = vmap(pullback)(batch.inputs, hess.sqrt_mv(z))

            for b, names in ctx.layout.blocks.items():
                flat = torch.cat(
                    [g[n].reshape(batch.size, -1) for n in names], dim=1
                )
                sums[b] = (
                    sums[b] + (w[:, None] * flat**2).sum(0) / self.n_probes
                )

        return core.SumStat(sums, stat.weight + w.sum())

    def finalize(self, stat):
        return core.BlockDiag(
            {b: core.Diag(s / stat.weight) for b, s in stat.sum.items()}
        )


# %% [markdown]
# The new curvature goes into `lk.laplace` like a built-in. To see how good the
# estimate is, we compare its diagonal with the exact one from `lk.DiagGGN()`,
# entry by entry.

# %%
hutch = lk.laplace(
    model,
    loader,
    likelihood=lik,
    subset=lk.subsets.all,
    hessian_approx=HutchinsonDiag(),
    generator=torch.Generator().manual_seed(0),
)
exact = lk.laplace(
    model,
    loader,
    likelihood=lik,
    subset=lk.subsets.all,
    hessian_approx=lk.DiagGGN(),
    generator=torch.Generator().manual_seed(0),
)
print(hutch)

est = torch.cat([op.diagonal() for op in hutch.curvature.blocks.values()])
ref = torch.cat([op.diagonal() for op in exact.curvature.blocks.values()])
nonzero = ref > 0

fig, ax = plt.subplots(figsize=(4.5, 4.5), layout="constrained")
ax.loglog(ref[nonzero], est[nonzero], ".", alpha=0.6)
lims = [ref[nonzero].min(), ref.max()]
ax.plot(lims, lims, color="0.5", linestyle=":")
ax.set_xlabel("exact GGN diagonal (lk.DiagGGN)")
ax.set_ylabel("HutchinsonDiag, $S = 16$")
plt.show()

rel_err = (est[nonzero] - ref[nonzero]).abs() / ref[nonzero]
print(f"median relative error: {rel_err.median():.2f}")

# %% [markdown]
# The points lie along the diagonal line, with a median relative error of about
# 6% for 16 samples. The error shrinks as $1 / \sqrt{S}$. Note that with four
# classes and $S = 16$, this costs more than `lk.DiagGGN()` on this model. We use it
# here because it is short.
#
# ## The math
#
# `lk.DiagGGN()` computes the diagonal of the GGN exactly, with one VJP per output
# class. With many classes, say a language model's vocabulary, this is too
# expensive. Instead, we can use Hutchinson's trick. Let $H_i = R_i R_i^\top$
# and let $z$ be a random vector of independent $\pm 1$ entries, so that
# $\mathbb{E}[z z^\top] = I$. Then
#
# $$
#   \mathbb{E}_z \big[ (J_i^\top R_i z)(J_i^\top R_i z)^\top \big]
#     = J_i^\top R_i \, \mathbb{E}[z z^\top] \, R_i^\top J_i = J_i^\top H_i J_i ,
# $$
#
# and taking the diagonal on both sides gives
#
# $$
#   \mathrm{diag}(J_i^\top H_i J_i) = \mathbb{E}_z \big[ (J_i^\top R_i z)^{\odot 2} \big] ,
# $$
#
# where $\odot 2$ is the elementwise square. Each sample of $z$ costs one VJP,
# whatever the number of classes. We average over $S$ samples and over the
# examples.
#
# A curvature's job is to estimate $G$ from batches of data. The fit goes as
# follows, and the boxes in orange are the curvature's code:
#
# ```{mermaid}
# flowchart TD
#     A["probe(model, likelihood, first batch)"] --> B["Capabilities<br/>vjp, vjp_twice, jvp, vmap, hooked layers"]
#     B --> C{"curvature.requires satisfied<br/>by the model and the likelihood?"}
#     C -- no --> E["CapabilityError,<br/>before any data is read"]
#     C -- yes --> D["stat = curvature.init(params, ctx)"]
#     D --> F["stat = curvature.update(stat, params, batch, generator, ctx)<br/>for every batch"]
#     F --> G["G = curvature.finalize(stat)<br/>an Operator"]
#     G --> H["LaplacePosterior<br/>precision = prior + N * G"]
#     classDef plugin fill:#fde2c8,stroke:#d9822b
#     class D,F,G plugin
# ```
#
# The statistic `S` must satisfy the `Stat` protocol: it has `total_weight`,
# `+`, and `scaled(s)`. It has to be a monoid. That is, `init` is the identity,
# $\mathrm{init} + a = a$, and folding the data in two halves gives the same
# result as folding all of it:
#
# $$
#   \mathrm{fold}(a \mathbin{+\!\!+} b) = \mathrm{fold}(a) + \mathrm{fold}(b) ,
# $$
#
# where $a \mathbin{+\!\!+} b$ is the concatenation of two lists of batches.
# This is what lets **laplacekit** split the data across processes and add the
# results with `all_reduce`. The other rule concerns scale: `finalize` returns
# the per-example average, e.g. $G$ and not $N G$. The posterior multiplies by
# $N$ later, so that the prior and $N$ can be changed without refitting. In
# practice, this means dividing by `stat.total_weight` in `finalize`.
#
# The result `R` is an `Operator`: an object with `mv`, `diagonal`,
# `inverse_diagonal`, `logdet`, `solve`, `cov_sqrt_mv`, `scaled`,
# `add_isotropic`, `add_diagonal` and `add_same`. **laplacekit** ships these,
# all in `laplacekit.core`:
#
# - `Diag` and `Dense`, a diagonal and a full matrix.
# - `KronEigenbasis`, one layer of KFAC, kept as the eigenbases of the
#   two Kronecker factors and the eigenvalues.
# - `LowRankPlusDiag`, a diagonal plus $U \mathrm{diag}(s) U^\top$, which
#   solves and takes log-determinants with the Woodbury identity. `lk.LowRankGGN`
#   returns it.
# - `BlockDiag`, one operator per block, and `Joint`, one operator over all
#   blocks, which concatenates them in layout order. The dense curvatures
#   return a `Joint` when they get `joint=True`, and `lk.LowRankGGN` always does.
#
# A new curvature should return one of these if it can, since the priors and
# predictives already know them.
#
# Plugins meet only through these core types. A prior adds itself to an operator
# through the operator's `add_isotropic` and `add_diagonal`, so a new operator
# with those methods works with the built-in priors. A per-block prior, such as
# `lk.priors.Layerwise`, also needs the blocks. It checks with `isinstance` for
# the `HasBlocks` protocol, which asks for `block_sizes()` and
# `add_per_block(prior_precision)`. `BlockDiag` and `Joint` satisfy it, so a new operator
# over several blocks should too. Two new types from different plugins, say our
# operator and someone else's prior, combine only if one of the two packages
# defines that pair. Otherwise the code that meets the unknown type raises
# `StructureError` naming both, as the built-ins do.
#
# An operator may also refuse a prior it can't represent. A `KronEigenbasis`
# keeps the prior in its eigenbasis, which works for one scalar per block and
# for nothing finer: a per-weight diagonal has no Kronecker form. So KFAC with
# `lk.priors.Diagonal` raises `StructureError`, and says what to do instead:

# %%
try:
    lk.laplace(
        model,
        loader,
        likelihood=lik,
        subset=lk.subsets.all,
        hessian_approx=lk.KFAC(),
        prior=lk.priors.Diagonal(torch.ones(388)),
        hyper_tuner=None,
        generator=torch.Generator().manual_seed(0),
    )
except lk.StructureError as e:
    print(e)

# %% [markdown]
# ## The test
#
# `lk.testing.check_curvature` runs the laws above on real batches. It checks
# that `init` is the identity, that the monoid law holds, that multiplying every
# example weight by 3 leaves the result unchanged (which is what returning an
# average means), and, for at most 512 parameters, that `diagonal`, `mv` and
# `logdet` agree with the dense matrix built from `mv`. It needs float64 and at
# least two batches.

# %%
batches = [lk.Batch.from_pair(pair) for pair in loader]
params = {k: v.detach() for k, v in model.named_parameters()}

lk.testing.check_curvature(HutchinsonDiag(), model, lik, params, batches)
print("HutchinsonDiag passes")

# %% [markdown]
# To see what a failure looks like, here is a version that forgets to divide by
# the total weight in `finalize`. It returns $N G$ instead of $G$:


# %%
@lk.core.pytree
class Unnormalized(HutchinsonDiag):
    def finalize(self, stat):
        return core.BlockDiag({b: core.Diag(s) for b, s in stat.sum.items()})


try:
    lk.testing.check_curvature(Unnormalized(), model, lik, params, batches)
except AssertionError as e:
    print(e)

# %% [markdown]
# In a plugin package, these calls go into the test suite. They check the
# contract only, so we also compare with an independent reference, as the
# scatter plot above does with `lk.DiagGGN()`.
#
# ## Requirements and capabilities
#
# Before any data is read, `lk.core.probe` runs the model on two examples and records
# what works: `vjp`, `jvp` and `vmap` through `torch.func`, the VJP of a VJP
# (`vjp_twice`, which fused attention kernels lack), and which `Linear` and
# `Conv2d` modules fire their forward hooks. The result is a `Capabilities`
# object. A curvature states what it needs as a `Requirements` object in
# `requires`, e.g. `Requirements(vjp=True, vmap=True)`. If the probe didn't find
# something the curvature requires, the fit raises `CapabilityError` naming the
# missing capability, before the first batch. A method that can only fail later,
# for example because it supports one likelihood only, should raise
# `CapabilityError` from `init`, which also runs before any data. `Requirements`
# also says whether the method honours per-example weights (`weights`, default
# `True`) and whether the statistic is a plain sum (`additive_stat`, default
# `True`).
#
# The same object states what the method needs from the data and the
# likelihood. `needs_targets` says the batches must carry targets, as for the
# empirical Fisher. `Requirements.check_batch` checks it on the first batch, and
# then on every batch. The likelihood needs are checked by `check_likelihood`,
# before any data:
#
# - `needs_sample`: the likelihood must draw targets (`HasSample` or
#   `HasDistribution`), as for the sampled Fisher.
# - `needs_distribution`: it must have `HasDistribution`.
# - `needs_output_hessian`: it must give an output Hessian, either in closed form
#   (`HasOutputHessian`) or as the MC Fisher, which needs sampling.
# - `needs_closed_form_hessian`: it must give the output Hessian in closed form;
#   the error names `fisher="mc"` as the way out.
# - `late_scale` (default `True`): the statistic is linear in the noise
#   precision of a `LateLikelihood`, so its hyperparameters can be applied after
#   the fit. A method whose statistic isn't, such as the empirical Fisher, which
#   is quadratic in the score, sets it to `False` and refuses such likelihoods
#   when they carry a late hyperparameter. A `LateLikelihood` has none when its
#   `unit()` returns itself, as a `Gaussian` with a fixed sigma does.
#
# ## Mixing with the built-ins
#
# A plugin mixes with the built-ins through `lk.per_block`, which maps glob
# patterns over block names to methods. Each block goes to the first pattern
# that matches it. Here the last layer (block `"4"`) gets the dense GGN and the
# rest gets our diagonal:

# %%
both = lk.laplace(
    model,
    loader,
    likelihood=lik,
    subset=lk.subsets.all,
    hessian_approx=lk.per_block({"4": lk.DenseGGN(), "*": HutchinsonDiag()}),
    generator=torch.Generator().manual_seed(0),
)
print(both)

# %% [markdown]
# :::{.callout-note}
# Every block must match a pattern, and every pattern must match a block. A
# pattern that no block reaches, for example one placed after `"*"`, raises
# `StructureError` when the fit starts.
# :::
#
# ## Several passes
#
# A method that needs more than one pass over the data satisfies `Staged`
# instead. For example, a method may estimate an eigenbasis in one pass and
# refit the eigenvalues in that basis in the next. It has `requires` and
# `plan(params, ctx)`, which returns a chain of passes. Each pass is a
# `Curvature`, and `chain(curvature, then=...)` hands what one pass finalizes
# to a function that returns the next pass, or `Done(result)` at the end:
#
# ```python
# def plan(self, params, ctx):
#     def refit(basis):
#         return lk.core.chain(Eigenvalues(basis), then=lk.core.Done)
#
#     return lk.core.chain(Eigenbasis(), then=refit)
# ```
#
# A `Staged` method mixes with single-pass ones in `per_block`. The map then
# runs as many passes as its longest member, and a single-pass member collects
# its statistic in the first pass and sits out the rest.
