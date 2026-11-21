"""LoRA Laplace on a causal language model and a sequence classifier, through the
`lk.transformers` presets."""

from collections.abc import Mapping

import torch
from _fixtures import train_my_resnet
from torch import Tensor
from transformers import GPT2Tokenizer

import laplacekit as lk

model = train_my_resnet()  # stands in for a peft-wrapped causal LM
loader: list[Mapping[str, Tensor]] = []
prompts = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "attention_mask": torch.ones(2, 4)}

post = lk.transformers.laplace_lora(model, loader)
post = lk.transformers.laplace_lora(
    model,
    loader,
    hessian_approx=lk.DenseGGN(),
    hyper_tuner=None,
    generator=torch.Generator().manual_seed(1),
)
probs = lk.predict(post, prompts, select=lk.transformers.last_token).mean

# The same fit, spelled out.
post = lk.laplace(
    lk.transformers.causal_lm(model),
    lk.data.from_loader(loader, prepare=lk.transformers.prepare_causal_lm),
    likelihood=lk.Categorical(mask_fn=lk.transformers.padding_mask),
    subset=lk.subsets.trainable,
    hessian_approx=lk.DiagGGN(fisher="empirical"),
)

# The logits of a few answers only, not the vocabulary.
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
yes_no = lk.transformers.answer_logits(tokenizer, [" True", " False"])
probs = lk.predict(post, prompts, select=yes_no).mean


class WordTokenizer:
    """Any object with a tokenizer's `encode` will do."""

    def encode(self, text: str, /, *, add_special_tokens: bool = True) -> list[int]:
        return [len(text)]


tokenizer_like: lk.transformers.Tokenizer = WordTokenizer()
yes_no = lk.transformers.answer_logits(WordTokenizer(), [" True", " False"])

# A sequence classifier, fitted on a subsample of a 10 000-example training set.
classifier = lk.transformers.SequenceClassification()
post = lk.transformers.laplace_lora(model, loader, task=classifier, n_data=10_000)
post = lk.transformers.laplace_lora(model, loader, task=lk.transformers.CausalLM(), n_data=None)
probs = lk.predict(post, prompts).mean

# The same classifier fit, spelled out.
post = lk.laplace(
    lk.transformers.sequence_classification(model),
    lk.data.from_loader(loader, prepare=lk.transformers.prepare_sequence_classification),
    likelihood=lk.Categorical(),
    subset=lk.subsets.trainable,
)
