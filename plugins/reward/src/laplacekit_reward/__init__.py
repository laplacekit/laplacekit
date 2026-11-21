r"""Laplace on reward models trained on preference pairs with Bradley–Terry.

A reward model maps an item to one number, its reward $r$. A preference pair holds two
items, and its label is the index of the preferred one, 0 for the first. Bradley–Terry
gives

$$p(\text{first} \succ \text{second}) = \sigma(r_1 - r_2),$$

which is the softmax over the two rewards.

A fit on pairs of tensors looks like this:

```python
fn = lk.reward.pairwise(reward_model)
data = lk.data.from_loader(loader, prepare=lk.reward.prepare_pairs)
post = lk.laplace(fn, data, likelihood=lk.reward.bradley_terry())
out = lk.predict_outputs(post, x)  # one reward per item, with its variance
probs = lk.predict(post, lk.reward.pair(a, b))  # the preference probabilities
```

`pairwise` wraps a reward model so that it scores a `Pair` as `[N, 2]` and single items
as `[N, 1]`. A `Pair` holds the two sides of a batch of pairs, and `pair` builds one.
`bradley_terry` is the likelihood, `lk.Categorical()` over the two rewards.
`prepare_pairs` turns `(first, second, label)` rows into a `Batch`.

## Language-model reward models

Laplace-LoRA (Yang et al., 2024, https://arxiv.org/abs/2308.13111) also works on a
language-model reward model. The model is a sequence classifier with `num_labels=1`,
wrapped by peft, and the Laplace approximation covers its LoRA adapters, plus the `score`
head when it is in `modules_to_save`. This needs `lk.transformers`
(`pip install "laplacekit[transformers]"`) and `peft`. `prepare_preferences` turns the
tokenizer outputs `{"chosen": ..., "rejected": ...}` into a `Batch` of pairs, with the
chosen side first.

```python
fn = lk.reward.pairwise(lk.transformers.sequence_classification(peft_model))
post = lk.laplace(
    fn,
    lk.data.from_loader(loader, prepare=lk.reward.prepare_preferences),
    likelihood=lk.reward.bradley_terry(),
    subset=lk.subsets.trainable,
    hessian_approx=lk.KFAC(),
)
out = lk.predict_outputs(post, tokenizer(texts, padding=True, return_tensors="pt"))
```

Laplace-LoRA uses KFAC, and `lk.DiagGGN()` also works. `out.mean` is the reward, `[N, 1]`,
and `out.cov.per_input()` its variance. Tokenize the chosen and rejected texts in one
padded call, so that both sides have the same length, and then split the result.

Code reaches this package as `lk.reward`, which imports it on first use.
"""

import laplacekit as lk
from laplacekit_reward.pairs import Pair, bradley_terry, pair, pairwise, prepare_pairs
from laplacekit_reward.preferences import prepare_preferences

__all__ = ["Pair", "bradley_terry", "pair", "pairwise", "prepare_pairs", "prepare_preferences"]


plugin = lk.Plugin(supports=">=0.1a0,<0.2")
