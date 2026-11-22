r"""Laplace-LoRA on Hugging Face `transformers` models: causal LMs and sequence classifiers.

`laplace_lora` fits a Laplace approximation over the LoRA adapters of a peft model:

```python
task = lk.transformers.SequenceClassification()
post = lk.transformers.laplace_lora(model, batches, task=task, n_data=len(train))
probs = lk.predict(post, inputs).probs  # linearized over the K class logits
```

For a causal LM, a batch is a tokenizer's output, `{"input_ids", "attention_mask"}`, each
`[N, T]`. The model predicts token `t + 1` from the tokens up to `t`, so position `t`
counts when both tokens are real (`padding_mask`). Right padding then changes neither the
curvature nor the data-fit term. Masked positions still enter the Jacobian, and a
non-finite activation there leaks through, since `0 * NaN` is NaN. Left padding with eager
attention in float64 gives such NaNs, so pad on the right for the fit.

`transformers` loads models with SDPA attention by default, and the fit works with it.
The linearized predictive, the default of `lk.predict`, differentiates the model's
backward pass, which SDPA does not support, so it raises `CapabilityError`. Load the model
with `attn_implementation="eager"`, or predict with `lk.WeightSamples()`.

A causal LM's logits span the vocabulary. To predict among a few answers, such as
$P(\text{True})$ or a multiple-choice letter, `answer_logits` selects their logits at the
last real token. A sequence classifier (`SequenceClassification()`) has $K$ logits per
sequence to begin with.

Code reaches this package as `lk.transformers`, which imports it on first use.
"""

from importlib import metadata
from typing import Literal

import laplacekit as lk
from laplacekit_transformers.answers import Tokenizer, answer_logits
from laplacekit_transformers.lora import laplace_lora
from laplacekit_transformers.tasks import (
    CausalLM,
    SequenceClassification,
    causal_lm,
    last_token,
    padding_mask,
    prepare_causal_lm,
    prepare_sequence_classification,
    sequence_classification,
)

__all__ = [
    "CausalLM",
    "SequenceClassification",
    "Tokenizer",
    "answer_logits",
    "causal_lm",
    "health",
    "laplace_lora",
    "last_token",
    "padding_mask",
    "prepare_causal_lm",
    "prepare_sequence_classification",
    "sequence_classification",
]


_OPTIONAL = {
    "peft": "LoRA adapters for laplace_lora",
}


def health() -> tuple[lk.HealthCheck, ...]:
    """Check for `lk.health()` that `transformers` is installed, and `peft` if possible.

    Returns:
        One check per package. Each is `ok` with the version when the package is
        installed. Otherwise it is `error` for `transformers`, which is required, and
        `warning` for `peft`, which is optional.
    """

    checks = [_installed("transformers", "error", "required")]
    checks += [
        _installed(dist, "warning", f"optional, for {use}") for dist, use in _OPTIONAL.items()
    ]

    return tuple(checks)


def _installed(dist: str, missing: Literal["warning", "error"], why: str) -> lk.HealthCheck:
    try:
        return lk.HealthCheck(dist, "ok", metadata.version(dist))
    except metadata.PackageNotFoundError:
        return lk.HealthCheck(dist, missing, f"not installed ({why})")


plugin = lk.Plugin(supports=">=0.1a0,<0.2", health=health)
"""The record behind the `laplacekit.plugins` entry point."""
