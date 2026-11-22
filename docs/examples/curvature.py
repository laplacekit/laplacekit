# %% [markdown]
# ---
# title: Choosing a curvature
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# post = lk.laplace(model, loader, likelihood=lik)  # last layer, lk.curvature.default
# post = lk.laplace(model, loader, likelihood=lik, subset=lk.subsets.all, hessian_approx=lk.KFAC())
# print(post)  # what was fitted, with which method
# post.log_marglik()  # compare fits on the same parameters
# ```
# :::
#
# The goal of this page is to compare the curvature approximations that **laplacekit** ships with on one small CNN.
# For each of them, we look at what it approximates, how long the fit takes, the log marginal likelihood it gives, and how confident its predictions are on corrupted inputs.
# At the end, we see what `lk.laplace` picks when we don't ask for anything.
#
# First, a reminder of what the curvature is.
# Let $f_\theta$ be a network with $d$ parameters, trained on $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ to the MAP estimate $\theta_\text{MAP}$, with the Gaussian prior $\mathcal{N}(\theta \mid 0, \lambda^{-1} I)$.
# The Laplace approximation is
#
# $$
#   p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\theta \mid \theta_\text{MAP}, \Lambda^{-1}) ,
#   \qquad \Lambda = \lambda I + N G ,
# $$
#
# where $G$ is the generalized Gauss-Newton (GGN) matrix, averaged over the data:
#
# $$
#   G = \frac{1}{N} \sum_{i=1}^N J_i^\top H_i J_i .
# $$
#
# Here, $J_i = \partial f_\theta(x_i) / \partial \theta$ is the $C \times d$ Jacobian of the network's output at $\theta_\text{MAP}$, and $H_i = -\nabla^2_f \log p(y_i \mid f)$ is the $C \times C$ Hessian of the negative log-likelihood with respect to the output.
# For a softmax over $C$ classes, $H_i = \mathrm{diag}(p_i) - p_i p_i^\top$ with $p_i$ the predicted probabilities.
# The matrix $G$ is $d \times d$, so we can't store it for any real network.
# A curvature in **laplacekit** is a way to approximate it.

# %%
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits
from torch import nn

import laplacekit as lk

# %% [markdown]
# ## Data and model
#
# We use the digits dataset from `sklearn`: 1797 grayscale images of $8 \times 8$ pixels in ten classes.
# The first $N = 1200$ images are for training and the rest for testing.
# The CNN has two convolutions and two linear layers, with 9802 parameters in total, and we train it with a small weight decay.

# %%
torch.manual_seed(0)

digits = load_digits()
images = torch.tensor(digits.images, dtype=torch.float32).unsqueeze(1) / 16.0
labels = torch.tensor(digits.target)
x_train, y_train = images[:1200], labels[:1200]
x_test, y_test = images[1200:], labels[1200:]
loader = list(zip(x_train.split(200), y_train.split(200), strict=True))

model = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(8, 16, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16 * 4 * 4, 32),
    nn.ReLU(),
    nn.Linear(32, 10),
)
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

for _ in range(50):
    for xb, yb in loader:
        opt.zero_grad()
        nn.functional.cross_entropy(model(xb), yb).backward()
        opt.step()

with torch.no_grad():
    acc = (model(x_test).argmax(-1) == y_test).float().mean()

print("parameters:", sum(p.numel() for p in model.parameters()))
print(f"test accuracy: {acc:.3f}")

# %% [markdown]
# ## The log marginal likelihood
#
# We compare the fits by their log marginal likelihood $\log Z$, where $Z = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta$.
# The Laplace approximation gives
#
# $$
#   \log Z \approx \log p(\mathcal{D} \mid \theta_\text{MAP}) + \log p(\theta_\text{MAP})
#     + \frac{d}{2} \log 2\pi - \frac{1}{2} \log \det \Lambda .
# $$
#
# With the Gaussian prior above, the last three terms simplify, and we get the form that `lk.objectives.neg_log_marglik` computes (with the opposite sign):
#
# $$
#   \log Z \approx \log p(\mathcal{D} \mid \theta_\text{MAP})
#     - \frac{1}{2} \Big( \log \det \Lambda - \log \det \lambda I
#     + \lambda \lVert \theta_\text{MAP} \rVert^2 \Big) .
# $$
#
# The first term is the data fit.
# The second is the Occam factor: it grows when the posterior is much narrower than the prior.
# Since both depend on the curvature only through $\Lambda$, a better approximation of $G$ gives a more faithful $\log Z$.
# Higher is better, as long as the approximation doesn't drop curvature, which we come back to below.
# By default, `lk.laplace` also picks $\lambda$ by maximizing $\log Z$.
#
# A posterior computes $\log Z$ with `post.log_marglik()`.
# The data-fit term comes from the fit itself, since `lk.laplace` sums the log-likelihood of the training data while it collects the curvature.

# %%
lik = lk.Categorical()
posts = {}


def fit(name, **kwargs):
    start = time.perf_counter()
    posts[name] = lk.laplace(model, loader, likelihood=lik, **kwargs)
    seconds = time.perf_counter() - start

    print(
        f"{name}: {seconds:.2f} s, log marginal likelihood "
        f"{posts[name].log_marglik():.1f}"
    )


# %% [markdown]
# ## Diagonal GGN
#
# The simplest choice keeps only the diagonal of $G$, so $\Lambda$ is diagonal too.
# `lk.DiagGGN()` computes it exactly.
# Write $H_i = R_i R_i^\top$ with a square root $R_i$.
# Then
#
# $$
#   [G]_{jj} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C \big[ J_i^\top R_i e_c \big]_j^2 ,
# $$
#
# where $e_c$ is the $c$-th unit vector.
# Each $J_i^\top R_i e_c$ is one vector-Jacobian product (VJP), so the cost is $C$ backward passes per example.
# We use it on all layers.

# %%
fit("Diag", hessian_approx=lk.DiagGGN(), subset=lk.subsets.all)

# %% [markdown]
# ## Dense GGN on the last layer
#
# Alternatively, we can keep all of $G$, but only for the last layer's weights $W^L$.
# That is, the first $L - 1$ layers are a fixed feature map and only $W^L$ gets a posterior, as in last-layer Laplace (Kristiadi et al., 2020)[^kristiadi2020].
# Here, $W^L$ has $32 \times 10$ weights and 10 biases, so $G$ is a $330 \times 330$ matrix, which is cheap to store and to factorize.

# %%
fit(
    "DenseGGN (last layer)",
    hessian_approx=lk.DenseGGN(),
    subset=lk.subsets.last_layer,
)

# %% [markdown]
# ## KFAC
#
# KFAC (Martens & Grosse, 2015[^martens2015]; Ritter et al., 2018[^ritter2018]) keeps one block of $G$ per layer and approximates each block by a Kronecker product.
# Consider a linear layer $s = W a$, where $a$ is the layer's input (with a 1 appended for the bias) and $s$ its output.
# Let $g_{ic} = \partial (R_i e_c)^\top f / \partial s$ be the gradient that example $i$ sends back to $s$ along the $c$-th column of $R_i$.
# The layer's block of $G$ is then $\frac{1}{N} \sum_i \sum_c (g_{ic} g_{ic}^\top) \otimes (a_i a_i^\top)$, and KFAC averages the two factors separately:
#
# $$
#   G_l \approx B \otimes A , \qquad
#   A = \frac{1}{N} \sum_{i=1}^N a_i a_i^\top , \qquad
#   B = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C g_{ic} g_{ic}^\top .
# $$
#
# For a convolution, $a$ and $g$ also run over the spatial positions.
# Since $A = Q_A \, \mathrm{diag}(\alpha) \, Q_A^\top$ and $B = Q_B \, \mathrm{diag}(\beta) \, Q_B^\top$, we have
#
# $$
#   B \otimes A = (Q_B \otimes Q_A) \, \mathrm{diag}(\beta \otimes \alpha) \, (Q_B \otimes Q_A)^\top .
# $$
#
# The isotropic prior lives in the same eigenbasis, so the layer's precision is $(Q_B \otimes Q_A) \big( N \mathrm{diag}(\beta \otimes \alpha) + \lambda I \big) (Q_B \otimes Q_A)^\top$.
# Its log-determinant, inverse and square root then only touch the eigenvalues.
# **laplacekit** stores each layer this way, as a `KronEigenbasis`, and never forms the full block.
# The factors come from `curvlinops`.

# %%
fit("KFAC", hessian_approx=lk.KFAC(), subset=lk.subsets.all)

# %% [markdown]
# ## The empirical Fisher
#
# The GGN needs the output Hessian $H_i$.
# The empirical Fisher only needs the gradient of the log-likelihood at the label we observed:
#
# $$
#   F = \frac{1}{N} \sum_{i=1}^N g_i g_i^\top , \qquad
#   g_i = \nabla_\theta \log p(y_i \mid f_\theta(x_i)) = J_i^\top d_i , \qquad
#   d_i = \nabla_f \log p(y_i \mid f) .
# $$
#
# For a softmax, $H_i = \mathbb{E}_{y \sim p_i} [d \, d^\top]$, so the GGN is the Fisher, which averages over labels drawn from the model.
# The empirical Fisher plugs in the training label instead, and Kunstner et al. (2019)[^kunstner2019] show that the two can be far apart.
# Here, $d_i = e_{y_i} - p_i$, which is small for every training digit that the network already classifies with confidence.
# So near the MAP estimate, we expect $F$ to be smaller than $G$.
#
# `lk.DenseGGN(fisher="empirical")` computes $F$ densely, from one gradient per example.
# `lk.KFAC(fisher="empirical")` is KFAC with $d_i$ in place of the columns of $R_i$ in the factor $B$.
# Both need the labels in the batches, and both need only `log_prob` from the likelihood.
# We fit KFAC on all layers and the dense version on the last layer.
# On the last layer, we compare the size of $F$ and $G$ by their traces.

# %%
fit(
    "KFAC (empirical)",
    hessian_approx=lk.KFAC(fisher="empirical"),
    subset=lk.subsets.all,
)
fit(
    "Dense empirical (last layer)",
    hessian_approx=lk.DenseGGN(fisher="empirical"),
    subset=lk.subsets.last_layer,
)

ggn_8 = posts["DenseGGN (last layer)"].curvature.blocks["8"].mat
ef_8 = posts["Dense empirical (last layer)"].curvature.blocks["8"].mat
print(
    f"trace ratio, empirical Fisher / GGN: {ef_8.trace() / ggn_8.trace():.2f}"
)

# %% [markdown]
# The empirical Fisher has 0.38 times the trace of the GGN on the last layer.
# Less curvature makes the posterior wider and $\log \det \Lambda$ smaller, so $\log Z$ goes up.
# It is $-163.6$ against $-214.2$ for the GGN on the last layer, and $-607.2$ against $-806.9$ for KFAC on all layers.
# These higher numbers come from the curvature that $F$ misses, so they say nothing about how close $F$ is to $G$.
#
# ## The full GGN across layers
#
# Every fit so far keeps correlations within a layer at most.
# `lk.DenseGGN(joint=True)` keeps all of $G$ over the chosen parameters, including the entries between layers.
# This is `hessian_structure="full"` in `laplace-torch`.
# On all 9802 weights, $G$ is a $9802 \times 9802$ matrix.
# The search for $\lambda$ factorizes $\Lambda$ at every step, which is too slow for this page.
# So we leave out the hidden linear layer, with its 8224 weights, and keep the two convolutions and the output layer.
# On those, we fit the diagonal, the dense GGN per block, and the joint one.

# %%
small = lk.subsets.named("0.*", "2.*", "8.*")
print(
    "parameters:",
    sum(p.numel() for i in (0, 2, 8) for p in model[i].parameters()),
)

fit("Diag (small)", hessian_approx=lk.DiagGGN(), subset=small)
fit("DenseGGN per block (small)", hessian_approx=lk.DenseGGN(), subset=small)
fit(
    "DenseGGN joint (small)",
    hessian_approx=lk.DenseGGN(joint=True),
    subset=small,
)

# %% [markdown]
# The more correlations a fit keeps, the higher its $\log Z$: $-939.8$ for the diagonal, $-402.1$ per block and $-346.6$ for the joint GGN.
# This is Fischer's inequality, which generalizes Hadamard's from entries to blocks.
# For a positive definite $\Lambda$ with diagonal blocks $\Lambda_{bb}$, it says that $\det \Lambda \leq \prod_b \det \Lambda_{bb}$.
# It holds at every $\lambda$, so it also holds after the search for $\lambda$.
#
# ## Low rank
#
# Another way to keep the correlations between all layers is to keep only the largest eigenvalues of $G$, as $G \approx U \mathrm{diag}(s) U^\top$ with $U$ a $d \times r$ matrix with orthonormal columns.
# `lk.LowRankGGN(rank=r)` finds them with a Nyström sketch (Tropp et al., 2017)[^tropp2017].
# Let $\Omega$ be a fixed random $d \times k$ matrix with orthonormal columns, where $k = r + 10$ by default.
# Each batch adds its share of $Y = G \Omega$, which costs $k$ Jacobian-vector products and $k$ VJPs.
# After the last batch,
#
# $$
#   \hat{G} = Y (\Omega^\top Y)^{-1} Y^\top ,
# $$
#
# truncated to its top $r$ eigenpairs.
# In floating point, $\Omega^\top Y$ can be close to singular, so we follow their Algorithm 3.
# We shift $Y$ to $Y + \nu \Omega$ with a tiny $\nu = \sqrt{d} \, \epsilon \, \|Y\|_2$ before the formula, take $\nu$ off the eigenvalues after it, and clamp the ones that round-off pushes below zero.
# The statistic has only $d k$ entries.
# The precision is $\Lambda = \lambda I + N U \mathrm{diag}(s) U^\top$, and we never form it.
# The Woodbury identity gives its inverse,
#
# $$
#   \Lambda^{-1} = \frac{1}{\lambda} \Big( I - U \, \mathrm{diag}\Big( \frac{N s_j}{\lambda + N s_j} \Big) U^\top \Big) ,
# $$
#
# and the matrix determinant lemma its log-determinant,
#
# $$
#   \log \det \Lambda = d \log \lambda + \sum_{j=1}^r \log \Big( 1 + \frac{N s_j}{\lambda} \Big) .
# $$
#
# The result is a `Joint[LowRankPlusDiag]`, which also allows a diagonal prior.
# The sketch needs a positive semi-definite $G$, so `LowRankGGN` always uses the Fisher output Hessian.
# We fit rank 100 on all weights, and on the small set of the previous section.

# %%
fit("LowRankGGN", hessian_approx=lk.LowRankGGN(rank=100), subset=lk.subsets.all)
fit("LowRankGGN (small)", hessian_approx=lk.LowRankGGN(rank=100), subset=small)

# %% [markdown]
# Rank 100 gives $-339.3$ on all weights, which is above every other all-layer fit.
# On the small set, it gives $-288.8$, above the joint GGN's $-346.6$.
# In exact arithmetic, the Nyström approximation never exceeds $G$, i.e., $G - \hat{G}$ is positive semi-definite.
# So the low-rank fit can only lower $\log \det \Lambda$ and raise $\log Z$, the opposite of dropping correlations.
# The 9702 directions it leaves out keep the prior's variance, which we see in the predictions below.
#
# ## The exact Hessian
#
# The GGN is the Hessian of the negative log-likelihood $\ell_i(\theta) = -\log p(y_i \mid f_\theta(x_i))$ with one term dropped.
# By the chain rule,
#
# $$
#   \nabla^2_\theta \, \ell_i = J_i^\top H_i J_i
#     + \sum_{c=1}^C [r_i]_c \, \nabla^2_\theta [f_\theta(x_i)]_c ,
#   \qquad r_i = \nabla_f \ell_i .
# $$
#
# The second term needs the second derivatives of the network.
# It vanishes where the network is linear in $\theta$, and it is small where the residual $r_i$ is.
# Unlike the first term, it can be negative, so the Hessian can be indefinite.
# `lk.DenseHessian()` computes the exact Hessian per block, with `jacrev(jacrev(...))`.
# That needs the double backward (`vjp_twice`) and `vmap` from the probe, and the labels, since $r_i$ depends on them.
# On the last layer, $f_\theta$ is linear in $W^L$, so the second term is zero and the Hessian is the GGN.

# %%
fit(
    "DenseHessian (last layer)",
    hessian_approx=lk.DenseHessian(),
    subset=lk.subsets.last_layer,
)

hess_8 = posts["DenseHessian (last layer)"].curvature.blocks["8"].mat
print(f"last layer, max |Hessian - GGN|: {(hess_8 - ggn_8).abs().max():.1e}")

# %% [markdown]
# The two matrices agree up to float32 rounding, and so do the log marginal likelihoods.
#
# For a ReLU network, more is true.
# Within one layer, the network is piecewise linear in that layer's weights, since ReLU and max pooling are piecewise linear.
# So each layer's block of the second term is zero almost everywhere, and the per-block Hessian is the per-block GGN on any layer of this CNN.
# The second term lives in the blocks between layers.
# To see it, we take the joint Hessian, `lk.DenseHessian(joint=True)`, over the first convolution and the last layer, 410 parameters, together with the joint GGN on the same parameters.
# We skip the tuning (`hyper_tuner=None`) and look at the matrices first.

# %%
pair = lk.subsets.named("0.*", "8.*")
joint = {
    name: lk.laplace(
        model,
        loader,
        likelihood=lik,
        subset=pair,
        hessian_approx=approx,
        hyper_tuner=None,
    )
    for name, approx in [
        ("GGN", lk.DenseGGN(joint=True)),
        ("Hessian", lk.DenseHessian(joint=True)),
    ]
}
hess, ggn = joint["Hessian"].curvature.op.mat, joint["GGN"].curvature.op.mat
evals = torch.linalg.eigvalsh(hess)
rel = torch.linalg.matrix_norm(hess - ggn) / torch.linalg.matrix_norm(ggn)

print(f"relative difference |Hessian - GGN| / |GGN|: {rel:.1e}")
print(
    f"negative eigenvalues of the Hessian: {(evals < 0).sum()} of "
    f"{evals.numel()}"
)
print(f"N times the smallest one: {joint['Hessian'].n_data * evals.min():.1f}")
print("prior:", joint["Hessian"].prior)

try:
    joint["Hessian"].log_marglik()
except lk.StructureError as e:
    print("StructureError:", e)

# %% [markdown]
# The Hessian is close to the GGN, but 136 of its 410 eigenvalues are negative, and the smallest one times $N$ is $-17.9$.
# With the default prior precision $\lambda = 1$, the precision $\lambda I$ plus $N$ times the Hessian is not positive definite.
# Its Cholesky factorization fails, and we get a `StructureError` that names the two ways out.
# These are a larger prior precision, or a positive semi-definite curvature such as the GGN.
# The default fit fails the same way, since the search for $\lambda$ starts there.
# Any $\lambda$ above $17.9$ makes $\Lambda$ positive definite.
# At $\lambda = 20$, the two give almost the same $\log Z$:

# %%
for name, post in joint.items():
    stiffer = copy.replace(post, prior=lk.priors.Isotropic(20.0))
    print(
        f"{name}, lambda = 20: log marginal likelihood "
        f"{stiffer.log_marglik():.1f}"
    )

# %% [markdown]
# On this model, the Hessian gives about what the GGN gives, at a higher cost: its memory grows with the number of parameters times the activations.
# It is still useful away from a minimum, or to check the GGN.
#
# ## Reading the numbers
#
# The printed times include the search for $\lambda$.
# Among the all-layer fits, KFAC is far ahead of the diagonal.
# Notice that the diagonal drops every correlation between weights.
# By Hadamard's inequality, $\det \Lambda \leq \prod_j \Lambda_{jj}$.
# So, compared with the full precision, dropping them can only raise $\log \det \Lambda$, and with it the Occam factor.
#
# The other way round, a method that misses curvature lowers $\log \det \Lambda$ and raises $\log Z$.
# The empirical Fisher and the low-rank fit both do this.
# So their $\log Z$ is high for the wrong reason, and we can't rank them against the others by it.
#
# The last-layer fit scores the highest.
# However, it treats the other 9472 weights as fixed, and only 330 weights pay the Occam factor.
# So the log marginal likelihood is only comparable between fits on the same parameters.
#
# ## Confidence on corrupted digits
#
# To use the posterior for prediction, the default predictive linearizes the network around $\theta_\text{MAP}$, $f_\theta(x) \approx f_{\theta_\text{MAP}}(x) + J(x)(\theta - \theta_\text{MAP})$.
# Under the Laplace posterior, the logits are then Gaussian:
#
# $$
#   f(x) \sim \mathcal{N}\big( f_{\theta_\text{MAP}}(x), \; J(x) \Lambda^{-1} J(x)^\top \big) .
# $$
#
# The probit approximation turns this Gaussian into class probabilities, $p(y = k \mid x) \approx \mathrm{softmax}_k \big( m / \sqrt{1 + \tfrac{\pi}{8} v} \big)$, where $m$ and $v$ are the mean and variance of the logits after subtracting their mean over the classes.
# A large $v$ pulls the probabilities toward uniform.
#
# We blend the first 300 test images with uniform noise $u$, as $(1 - t) x + t u$.
# At $t = 1$, the input is pure noise, and the ideal confidence (the highest class probability) is chance, $0.1$.

# %%
ts = np.linspace(0.0, 1.0, 5)
x_plot = x_test[:300]
noise = torch.rand(x_plot.shape, generator=torch.Generator().manual_seed(1))
corrupted = [(1 - t) * x_plot + t * noise for t in ts]

with torch.no_grad():
    curves = {
        "MAP": [model(xc).softmax(-1).max(-1).values.mean() for xc in corrupted]
    }

plotted = [
    "Diag",
    "DenseGGN (last layer)",
    "KFAC",
    "KFAC (empirical)",
    "LowRankGGN",
]

for name in plotted:
    curves[name] = [
        lk.predict(posts[name], xc).probs.max(-1).values.mean()
        for xc in corrupted
    ]

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
for name, conf in curves.items():
    ax.plot(ts, conf, marker="o", label=name)
ax.axhline(0.1, color="0.6", linestyle=":", label="chance")
ax.set_xlabel("noise level $t$")
ax.set_ylabel("mean confidence")
ax.legend(loc="upper right")
plt.show()

# %% [markdown]
# Every Laplace curve stays below the MAP network, and none of them reaches chance on pure noise.
# The diagonal is underconfident even on clean digits.
# Its mean confidence there is far below the test accuracy printed at the top, so we wouldn't use it on this model.
# The last-layer posterior stays close to the MAP network along the whole range.
# The empirical-Fisher KFAC sits below KFAC everywhere, as expected from a smaller curvature.
# The low-rank fit is the least confident of all, below the diagonal, since most of its directions have the prior's variance.
#
# ## The default
#
# When we don't pass `hessian_approx`, `lk.laplace` picks a method per block (here, per layer) through `lk.curvature.default`, based on what the probe found about the model.
# Let $E$ be the number of output entries per example, e.g., $C = 10$ here, or the number of tokens times the vocabulary size for a language model.
# We say that KFAC applies to a block if all of the following hold:
#
# - module hooks reach its layer;
# - the block has the layer's weight (not only its bias);
# - KFAC supports the likelihood;
# - the batches carry no per-example weights;
# - no tensor in the block is shared with another layer (weight tying), since the hooks would see only one of its uses.
#
# Then the rules are tried in order:
#
# 1. If $E > 4096$, the exact methods are too expensive, since they cost one pass per output entry or a dense $E \times E$ output Hessian per example.
#    Every block gets a sampled method: KFAC with a Monte Carlo Fisher, `KFAC(fisher="mc")`, where KFAC applies, and the sampled Fisher diagonal otherwise.
# 2. If the model supports `vjp` and `vmap`, a block of at most 4096 parameters gets the dense GGN.
# 3. Otherwise, the block gets KFAC where KFAC applies.
# 4. Otherwise, it gets the exact diagonal, again if the model supports `vjp` and `vmap`.
# 5. Otherwise, it gets the sampled Fisher diagonal, which needs only plain backward passes.
#
# Installed plugins never change this choice.
# `print(post)` shows the resulting map.

# %%
start = time.perf_counter()
auto = lk.laplace(model, loader, likelihood=lik, subset=lk.subsets.all)
seconds = time.perf_counter() - start

print(auto)
print(f"{seconds:.2f} s, log marginal likelihood {auto.log_marglik():.1f}")

# %% [markdown]
# The two convolutions and the output layer are small, so they get the dense GGN.
# The hidden linear layer has 8224 parameters and gets KFAC.
# Since three of the four blocks keep their full GGN, the log marginal likelihood is higher than the all-layer KFAC one.
#
# The same map can be built by hand, e.g., to edit it before fitting:

# %%
fn = lk.core.ModelFn.from_module(model)
caps = lk.core.probe(fn, lik, lk.Batch.from_pair(loader[0]))
params = lk.subsets.all(fn, caps)
print(lk.curvature.default(params, caps, likelihood=lik))

# %% [markdown]
# ## Which subset and curvature
#
# We start with the defaults, `lk.laplace(model, data, likelihood=...)`: the last layer, with `lk.curvature.default`.
# Then we change one thing at a time, and compare the log marginal likelihood only between fits on the same parameters.
# The last column is what we saw on this page.
#
# | Change | When | On this page |
# |---|---|---|
# | `subset=lk.subsets.all` | The predictive stays close to the MAP network where it shouldn't, e.g., on corrupted inputs. | The last-layer curve stays close to the MAP one at every noise level. The all-layer fits come down further. |
# | `hessian_approx=lk.KFAC()` | The default's dense blocks are too slow or too large. KFAC keeps two small factors per layer. | All-layer KFAC gives $\log Z = -806.9$, against $-707.6$ for the default curvature on all layers, a mix of dense blocks and KFAC. |
# | `hessian_approx=lk.DenseGGN()` | The subset is small, up to a few thousand parameters. It is the exact GGN. | The last layer's $330 \times 330$ block. The default already picks it for blocks of at most 4096 parameters. |
# | `hessian_approx=lk.DenseGGN(joint=True)` | The subset spans several layers, is small enough for one dense matrix, and the correlations between layers matter. | On the convolutions and the output layer, $\log Z = -346.6$, against $-402.1$ per block and $-939.8$ for the diagonal. |
# | `hessian_approx=lk.LowRankGGN(rank=r)` | We want correlations between all layers, and $d (r + 10)$ numbers fit in memory. Its $\log Z$ is biased upward. | Rank 100 on all weights has the highest all-layer $\log Z$, $-339.3$, and the least confident predictions. |
# | `hessian_approx=lk.DenseHessian()` | We need the exact Hessian, away from a minimum or to check the GGN. Small blocks only, and the model needs a double backward. | It equals the GGN on the last layer. Across two layers it has negative eigenvalues, and $\lambda = 1$ fails. |
# | `hessian_approx=lk.DenseGGN(fisher="empirical")` or `lk.KFAC(fisher="empirical")` | The likelihood gives only `log_prob`, no output Hessian. | It has 0.38 times the GGN's trace on the last layer, and a higher $\log Z$ for that reason. |
# | `hessian_approx=lk.DiagGGN()` | The methods above don't apply or don't fit in memory. The default falls back to it on its own (rule 4). | It drops every correlation. It has the lowest $\log Z$, $-2307.9$, and is underconfident even on clean digits. |
# | `lk.KFAC(fisher="mc")` or `lk.DiagGGN(fisher="mc")` | Each example has many outputs, e.g., a language model's tokens times its vocabulary. The default switches to them when $E > 4096$ (rule 1). | Not needed here, with $E = 10$. |

# %% [markdown]
# [^kristiadi2020]: Kristiadi, A., Hein, M., & Hennig, P. (2020). Being Bayesian, even just a bit, fixes overconfidence in ReLU networks. *ICML*. <https://arxiv.org/abs/2002.10118>
#
# [^martens2015]: Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML*. <https://arxiv.org/abs/1503.05671>
#
# [^ritter2018]: Ritter, H., Botev, A., & Barber, D. (2018). A scalable Laplace approximation for neural networks. *ICLR*. <https://openreview.net/forum?id=Skdvd2xAZ>
#
# [^kunstner2019]: Kunstner, F., Balles, L., & Hennig, P. (2019). Limitations of the empirical Fisher approximation for natural gradient descent. *NeurIPS*. <https://arxiv.org/abs/1905.12558>
#
# [^tropp2017]: Tropp, J. A., Yurtsever, A., Udell, M., & Cevher, V. (2017). Fixed-rank approximation of a positive-semidefinite matrix from streaming data. *NeurIPS*. <https://arxiv.org/abs/1706.05736>
#
