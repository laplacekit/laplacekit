# %% [markdown]
# ---
# title: Reward models
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# fn = lk.reward.pairwise(lk.transformers.sequence_classification(model))  # one reward per text
# data = lk.data.from_loader(batches, prepare=lk.reward.prepare_preferences, n_data=n_pairs)
# post = lk.laplace(fn, data, likelihood=lk.reward.bradley_terry(), subset=..., hessian_approx=lk.KFAC())
# out = lk.predict_outputs(post, tok(prompts, responses, padding=True, return_tensors="pt"))  # reward, variance
# ```
# :::
#
# A reward model scores a response to a prompt with one number, and in RLHF a
# policy is then trained to make that number large. The reward model only saw a
# finite set of comparisons, though, and the policy is free to wander off to
# responses that look like none of them. There the reward is an extrapolation,
# and a policy that maximizes it can find responses that the reward model scores
# highly while people wouldn't. This is reward-model overoptimization (Gao et
# al., 2023)[^gao2023]. Yang et al. (2024)[^yang2024rm] put a Laplace
# approximation on the reward model, so that every reward comes with a variance,
# and penalize the rewards the model is unsure about.
#
# On this page, we fit a Laplace approximation to a published reward model and
# ask whether its reward variance is larger on prompts and responses unlike the
# ones it was trained on.
#
# ## Bradley–Terry and the Laplace approximation
#
# Let $r_\theta(x)$ be the reward of a text $x$, a prompt together with a
# response. A preference pair is two responses $a$ and $b$ to the same prompt,
# with $a$ the one a person preferred. The Bradley–Terry model
# (1952)[^bradley1952] says this happens with probability
#
# $$
#   p(a \succ b \mid \theta) = \sigma\big(r_\theta(a) - r_\theta(b)\big) ,
# $$
#
# with $\sigma$ the logistic sigmoid. That is the softmax over the two rewards,
# so `lk.reward.bradley_terry()` is a categorical likelihood over the pair's two
# rewards, with the label $0$ for "the first one won". Only the difference of
# the rewards enters, so adding a constant to every reward changes nothing.
#
# With $N$ pairs and $\theta_*$ the trained weights, the Laplace approximation is
#
# $$
#   p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\theta \mid \theta_*, \Sigma) , \qquad \Sigma^{-1} = \lambda I + \sum_{i=1}^N J_i^\top H_i J_i ,
# $$
#
# where $J_i$ is the $2 \times P$ Jacobian of the two rewards of pair $i$ and
# $H_i$ the Hessian of $-\log p$ with respect to them. With $p_i = \sigma(r(a_i)
# - r(b_i))$ and $u = (1, -1)^\top$, it is $H_i = p_i (1 - p_i) \, u u^\top$, so
#
# $$
#   J_i^\top H_i J_i = p_i (1 - p_i) \, \big(\nabla_\theta r(a_i) - \nabla_\theta r(b_i)\big) \big(\nabla_\theta r(a_i) - \nabla_\theta r(b_i)\big)^\top .
# $$
#
# With this exact GGN, the data add curvature only along the differences of the
# reward gradients. A direction that moves all rewards together, such as the
# bias of the last layer, gets none, and its variance stays at the prior's.
# This is the idealized picture; the KFAC fit below departs from it.
#
# Linearizing $r_\theta$ around $\theta_*$ gives each text a Gaussian reward,
# with mean $r_{\theta_*}(x)$ and variance
#
# $$
#   v(x) = \nabla_\theta r(x)^\top \, \Sigma \, \nabla_\theta r(x) .
# $$
#
# $v(x)$ is small when the gradient at $x$ points along directions that the
# preference data pinned down, and grows towards the prior's
# $\|\nabla_\theta r(x)\|^2 / \lambda$ when it doesn't. Yang et al. use it
# through a penalized reward, for example
#
# $$
#   \tilde r(x) = r_{\theta_*}(x) - k \sqrt{v(x)} ,
# $$
#
# with some $k > 0$. Where the reward model extrapolates, the hope is that
# $v(x)$ is large, so the penalty takes back the reward that the model can't
# vouch for, and the policy gains less from going there.
#
# The [Quickstart](../quickstart/reward.py) does all this on a toy with three
# input features, and there the reward's standard deviation grows past the
# range of the data. Here we use a real reward model.

# %%
import warnings

import datasets
import huggingface_hub
import matplotlib.pyplot as plt
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import laplacekit as lk

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
datasets.disable_progress_bars()
huggingface_hub.logging.set_verbosity_error()
warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # deprecation notices from transformers

# %% [markdown]
# ## Model and data
#
# The model is `nicholasKluge/RewardModel`[^aira] on the Hugging Face Hub,
# pinned to one revision. It is a BERT-base sequence classifier with 108
# million parameters and one output, the reward. It reads the prompt and the
# response as a BERT sentence pair. We load it with eager attention, as on the
# [Language models](llm.py) page.

# %%
MODEL = "nicholasKluge/RewardModel"
MODEL_REV = "914413962bfc33b1b703942b2920bd1b7b6e3b27"
DATA_REV = (
    "ecfeb1936069f37bee7daffee60cd7765c326a34"  # the data as of August 2023
)

tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL, revision=MODEL_REV, attn_implementation="eager"
)
_ = model.eval()

aira = load_dataset(
    "nicholasKluge/reward-aira-dataset", revision=DATA_REV, split="english"
)
print(aira)

# %% [markdown]
# The model was trained on the English part of `reward-aira-dataset`. Its
# English data was replaced in January 2024, so we pin the version that was on
# the Hub when the model was uploaded, in August 2023: 32,675 prompts, each with
# a chosen and a rejected response. We fit the Laplace
# approximation on the first 32 pairs, in four batches of eight.
#
# `prepare_preferences` takes each batch as a dictionary with the tokenizer
# output of the chosen responses under `"chosen"` and that of the rejected ones
# under `"rejected"`. The two sides must have the same length, so we tokenize
# them in one padded call and split the result. Many responses are long, and we
# cut each pair at 128 tokens to keep the fit fast on a CPU.


# %%
def preferences(rows):
    n = len(rows)
    both = tok(
        list(rows["instruction"]) * 2,
        list(rows["chosen_response"]) + list(rows["rejected_response"]),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    return {
        "chosen": {k: v[:n] for k, v in both.items()},
        "rejected": {k: v[n:] for k, v in both.items()},
    }


batches = [preferences(aira.select(range(i, i + 8))) for i in range(0, 32, 8)]

# %% [markdown]
# ## Fitting the last layer
#
# `lk.transformers.sequence_classification` makes the model return its output, and
# `lk.reward.pairwise` turns the result into a model that scores both sides of a
# pair in one forward pass. We put the Laplace approximation on the weight of
# the last layer, `classifier.weight`, the 768 numbers that map BERT's pooled
# features to the reward. We leave out its bias, a shift of all rewards that the
# likelihood can't see. The curvature is KFAC, as in Laplace-LoRA (Yang et al.,
# 2024)[^yang2024lora]. `DenseGGN` would keep the exact GGN from above, but it
# needs `vmap`, which the probe doesn't find for this BERT.
#
# `pairwise` runs both sides of a pair through the model as one batch, and KFAC
# treats the two texts as separate examples. Its
# input factor is the second moment of each text's own features, without the
# cross terms between the two sides, so it adds curvature along each text's
# features and not only along the differences. A shift of all rewards then gets
# curvature too: under KFAC, even a bias would end up with a variance below the
# prior's. So $v(x)$ here measures how far the features of $x$ stray from those
# of the 64 texts in the fit, more than how far its gradient strays from the
# directions that the preferences pin down.
#
# The 32 pairs give the average curvature, and
# `n_data=len(aira)` scales it to $N = 32{,}675$. The marginal likelihood picks
# $\lambda$.

# %%
fn = lk.reward.pairwise(lk.transformers.sequence_classification(model))
data = lk.data.from_loader(
    batches, prepare=lk.reward.prepare_preferences, n_data=len(aira)
)

post = lk.laplace(
    fn,
    data,
    likelihood=lk.reward.bradley_terry(),
    subset=lk.subsets.named("classifier.weight"),
    hessian_approx=lk.KFAC(),
    generator=torch.Generator().manual_seed(0),
)
print(post)

# %% [markdown]
# The marginal likelihood chose $\lambda \approx 85$.
#
# ::: {.callout-note}
# For Laplace-LoRA on a reward model, the model is a `peft` model with LoRA
# adapters and a trainable score head, and the call is the same with
# `subset=lk.subsets.trainable`.
# :::
#
# ## Rewards and their variance
#
# We compare two groups of four texts. The first are prompts with their chosen
# responses from the same version of the dataset, but not among the 32 pairs:
# the first four of rows 32 to 1,999 whose prompt and response together fit in
# 64 tokens. The
# second are prompts and responses unlike the training data: German, keyboard
# noise, SQL and Japanese. Our OOD texts are short, so we keep the first group
# short too, so that length doesn't differ between the groups. Each group is
# padded to its longest text, and `lk.predict_outputs` gives the reward of each text with
# its variance. Without `vmap`, the variance comes from 64 sampled draws, so we
# seed them with a generator.

# %%
held = aira.select(range(32, 2000))
lengths = [
    len(ids)
    for ids in tok(list(held["instruction"]), list(held["chosen_response"]))[
        "input_ids"
    ]
]
short = held.select([i for i, n in enumerate(lengths) if n <= 64][:4])

groups = {
    "in": list(
        zip(short["instruction"], short["chosen_response"], strict=True)
    ),
    "OOD": [
        ("Wie spät ist es?", "Es ist drei Uhr."),
        ("asdf qwer zxcv", "1 2 3 4 5 6 7 8 9"),
        ("Write a SQL query that lists all users.", "SELECT * FROM users;"),
        ("Translate 'cat' into Japanese.", "猫 (neko)"),
    ],
}

rewards = {}

for name, texts in groups.items():
    x = tok(
        [p for p, _ in texts],
        [r for _, r in texts],
        padding=True,
        return_tensors="pt",
    )
    out = lk.predict_outputs(
        post, x, generator=torch.Generator().manual_seed(0)
    )
    rewards[name] = (
        out.mean.squeeze(-1),
        out.cov.per_input().reshape(-1).sqrt(),
    )

print(f"{'':<4} {'reward':>7} {'std':>6}  prompt")

for name, texts in groups.items():
    for (prompt, _), r, s in zip(texts, *rewards[name], strict=True):
        print(
            f"{name:<4} {float(r):7.2f} {float(s):6.3f}  "
            f"{prompt.splitlines()[0][:48]}"
        )

for name, (_, s) in rewards.items():
    print(f"mean std, {name}: {float(s.mean()):.3f}")

# %% [markdown]
# The four chosen responses get rewards from 6.67 to 11.91, and the OOD texts
# from -8.21 to -1.45. The OOD texts also get a larger standard deviation on
# average, 0.517 against 0.321, but the ranges overlap: 0.409 to 0.570 for the
# OOD texts and 0.127 to 0.445 for the others. The SQL query, at 0.409, is less
# uncertain than the question about pronouncing "architecture", at 0.445. The
# largest standard deviation of all, 0.570, goes to the Japanese translation,
# which is also the OOD text with the highest reward.
#
# Next we plot each text's standard deviation against its reward, for both
# groups together.

# %%
r_all = torch.cat([r for r, _ in rewards.values()])
s_all = torch.cat([s for _, s in rewards.values()])

fig, ax = plt.subplots(figsize=(6, 4))

for name, (r, s) in rewards.items():
    ax.scatter(r, s, label=name)

ax.set_xlabel("reward")
ax.set_ylabel("std of the reward")
ax.legend()
plt.show()

print(
    "correlation of reward and std: "
    f"{float(torch.corrcoef(torch.stack([r_all, s_all]))[0, 1]):.2f}"
)

# %% [markdown]
# Over the eight texts, the correlation of reward and standard deviation is
# -0.74. It comes mostly from the split between the groups: the chosen
# responses have high rewards and smaller standard deviations, the OOD texts low
# rewards and larger ones. Within the OOD group the order goes the other way,
# since the Japanese text has both the highest reward and the largest standard
# deviation. With four texts per group, neither pattern is pinned down.
#
# For the penalized reward $\tilde r$, the Japanese text is the interesting
# case: an unusual text that scores higher than the other OOD texts, and its
# penalty is the largest of the eight. Its reward is still far below those of
# the chosen responses, though, so the page doesn't show what the variance does
# for an unusual text that the model scores as highly as a good response, which
# is the case that matters most for overoptimization.
#
# ## Wrap-up
#
# A published reward model gets a Laplace approximation in one `lk.laplace`
# call: `lk.reward.pairwise` scores both sides of a pair, `prepare_preferences`
# turns tokenized chosen and rejected responses into a batch, and
# `bradley_terry()` is the likelihood. `lk.predict_outputs` then gives every text's
# reward with its variance. On this BERT reward model, with Laplace on the 768
# weights of its last layer, four OOD texts get a larger reward standard
# deviation on average than four chosen responses from the training data. The
# groups overlap, and the OOD text with the highest reward has the largest
# standard deviation. Yang et
# al. put the Laplace approximation on LoRA adapters throughout the network,
# which `subset=lk.subsets.trainable` gives on a `peft` model.
#
# [^gao2023]: Gao, L., Schulman, J., & Hilton, J. (2023). Scaling laws for reward model overoptimization. *ICML*. <https://arxiv.org/abs/2210.10760>
#
# [^yang2024rm]: Yang, A. X., Robeyns, M., Coste, T., Shi, Z., Wang, J., Bou-Ammar, H., & Aitchison, L. (2024). Bayesian reward models for LLM alignment. arXiv:2402.13210. <https://arxiv.org/abs/2402.13210>
#
# [^bradley1952]: Bradley, R. A., & Terry, M. E. (1952). Rank analysis of incomplete block designs: I. The method of paired comparisons. *Biometrika*, 39(3/4), 324–345. <https://doi.org/10.2307/2334029>
#
# [^aira]: The model and its training data, `nicholasKluge/reward-aira-dataset`, are released under the Apache-2.0 license. <https://huggingface.co/nicholasKluge/RewardModel>
#
# [^yang2024lora]: Yang, A. X., Robeyns, M., Wang, X., & Aitchison, L. (2024). Bayesian low-rank adaptation for large language models. *ICLR*. <https://arxiv.org/abs/2308.13111>
