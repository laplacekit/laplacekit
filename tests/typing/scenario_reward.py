"""Scenario: a reward model fit on preference pairs, through `lk.reward`."""

from collections.abc import Mapping, Sequence

import torch
from _fixtures import x_test
from torch import Tensor, nn

import laplacekit as lk

reward = lk.reward
net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))
pairs: Sequence[tuple[Tensor, Tensor, Tensor]] = []

fn: lk.core.ModelFn = reward.pairwise(net)
again: lk.core.ModelFn = reward.pairwise(fn)
data = lk.data.from_loader(pairs, prepare=reward.prepare_pairs)
lik: lk.Categorical = reward.bradley_terry()
post = lk.laplace(fn, data, likelihood=lik, hessian_approx=lk.KFAC())

# The reward of single items with its variance, and the preference between two items.
out = lk.predict_outputs(post, x_test)
p: reward.Pair = reward.pair(x_test, x_test)
both = reward.Pair(first=x_test, second={"input_ids": torch.zeros(1)})
prefer = lk.predict(post, p)
batch: lk.Batch = reward.prepare_pairs((x_test, x_test, x_test), torch.Generator().manual_seed(0))

# Laplace-LoRA on a language-model reward model, from tokenized preferences.
peft_model = nn.Linear(3, 1)  # stands in for a peft-wrapped sequence classifier
preferences: Sequence[Mapping[str, Mapping[str, Tensor]]] = []
tokens = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "attention_mask": torch.ones(2, 4)}

lm: lk.core.ModelFn = reward.pairwise(lk.transformers.sequence_classification(peft_model))
lm_post = lk.laplace(
    lm,
    lk.data.from_loader(preferences, prepare=reward.prepare_preferences),
    likelihood=reward.bradley_terry(),
    subset=lk.subsets.trainable,
    hessian_approx=lk.KFAC(),
)
rewards = lk.predict_outputs(lm_post, tokens)
chosen_first: lk.Batch = reward.prepare_preferences(
    {"chosen": tokens, "rejected": tokens}, torch.Generator().manual_seed(0)
)
