"""Causal LMs and sequence classifiers as `ModelFn`s, with their batches and masks.

`CausalLM` and `SequenceClassification` name the two tasks. `causal_lm` and
`sequence_classification` wrap a model so that it returns its logits, and the `prepare_*`
functions turn tokenizer batches into a `Batch`. `padding_mask` removes from a causal
LM's loss the positions where the token or the next token is padding. `last_token`
selects the logits at each sequence's last real token.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from transformers.utils import ModelOutput

import laplacekit as lk
from laplacekit.core import (
    Inputs,
    Outputs,
    Params,
    is_transformed,
)

__all__ = [
    "CausalLM",
    "SequenceClassification",
    "causal_lm",
    "last_token",
    "padding_mask",
    "prepare_causal_lm",
    "prepare_sequence_classification",
    "sequence_classification",
]


_INPUTS = ("input_ids", "attention_mask")


@dataclass(frozen=True)
class CausalLM:
    """The task of `laplace_lora` for a causal LM (`AutoModelForCausalLM`).

    The model predicts each next token, with logits `[N, T, V]` over the vocabulary.
    `laplace_lora` then uses `causal_lm`, `prepare_causal_lm` and `padding_mask`.
    """


@dataclass(frozen=True)
class SequenceClassification:
    """The task of `laplace_lora` for a sequence classifier (`AutoModelForSequenceClassification`).

    The model predicts one of $K$ labels per sequence, with logits `[N, K]`. The batches
    carry `"labels"`, and `laplace_lora` uses `sequence_classification` and
    `prepare_sequence_classification`.
    """


def causal_lm(model: nn.Module, /) -> lk.core.ModelFn:
    """Wrap a Hugging Face causal LM as a `ModelFn` that returns the logits.

    Args:
        model: The causal LM, plain or peft-wrapped.

    Returns:
        A `ModelFn` whose `forward(params, inputs)` calls
        `model(**inputs, use_cache=False)` and returns its `.logits`, `[N, T, V]`. The
        inputs are `{"input_ids", "attention_mask"}`. A call raises `StructureError` when
        the inputs are not a mapping or the model does not return a `ModelOutput`.
    """

    call = lk.core.ModelFn.from_module(model)

    def forward(params: Params, inputs: Inputs) -> Outputs:
        # from_module passes a mapping's entries as keyword arguments, so the flag rides
        # along with the tensors; this inner call is never vmapped.
        kwargs = cast(Inputs, {**_mapping(inputs, "causal_lm"), "use_cache": False})
        out = call(params, kwargs)

        if not isinstance(out, ModelOutput):
            raise lk.StructureError(
                f"causal_lm needs a model that returns a transformers ModelOutput, got "
                f"{type(out).__name__}"
            )

        return out["logits"]

    return lk.core.ModelFn(forward, model)


def prepare_causal_lm(raw: Mapping[str, Tensor], generator: torch.Generator, /) -> lk.Batch:
    """Turn a tokenizer batch into a `Batch` for next-token prediction.

    Args:
        raw: The tokenizer's output. Keys other than `input_ids` and `attention_mask` are
            dropped.
        generator: Unused: nothing is drawn.

    Returns:
        A `Batch` with the inputs `input_ids` and `attention_mask`, and with `input_ids`
        shifted left by one as targets. The last column has no next token, so it holds a
        placeholder 0, which `padding_mask` removes.

    Raises:
        StructureError: `raw` lacks `input_ids` or `attention_mask`.
    """

    del generator
    missing = [k for k in _INPUTS if k not in raw]

    if missing:
        raise lk.StructureError(f"prepare_causal_lm needs {', '.join(missing)} in the batch")

    ids = raw["input_ids"]
    targets = torch.cat([ids[:, 1:], torch.zeros_like(ids[:, :1])], dim=1)
    inputs = {"input_ids": ids, "attention_mask": raw["attention_mask"]}

    return lk.Batch(inputs, targets)


def sequence_classification(model: nn.Module, /) -> lk.core.ModelFn:
    """Wrap a Hugging Face sequence classifier as a `ModelFn` that returns the logits.

    Args:
        model: The sequence classifier, plain or peft-wrapped.

    Returns:
        A `ModelFn` whose `forward(params, inputs)` calls `model(**inputs)` and returns
        its `.logits`, `[N, K]`. The inputs are a tokenizer's output. A call raises
        `StructureError` when the inputs are not a mapping, or the output is not a
        `ModelOutput` with 2-D logits `[N, K]`.
    """

    call = lk.core.ModelFn.from_module(model)

    def forward(params: Params, inputs: Inputs) -> Outputs:
        out = call(params, _mapping(inputs, "sequence_classification"))

        if not isinstance(out, ModelOutput) or "logits" not in out:
            raise lk.StructureError(
                f"sequence_classification needs a model that returns a transformers ModelOutput "
                f"with logits, got {type(out).__name__}"
            )

        logits = out["logits"]

        if not isinstance(logits, Tensor) or logits.ndim != 2:
            shape = tuple(logits.shape) if isinstance(logits, Tensor) else type(logits).__name__

            raise lk.StructureError(
                f"sequence_classification needs logits [N, K], got {shape}; a causal LM goes "
                f"with CausalLM() and answer_logits"
            )

        return logits

    return lk.core.ModelFn(forward, model)


def prepare_sequence_classification(
    raw: Mapping[str, Tensor], generator: torch.Generator, /
) -> lk.Batch:
    """Turn a tokenizer batch with `"labels"` into a `Batch` for sequence classification.

    Args:
        raw: The tokenizer's output plus `"labels"` `[N]`.
        generator: Unused: nothing is drawn.

    Returns:
        A `Batch` with the labels as targets. Every other entry (`input_ids`,
        `attention_mask`, `token_type_ids`, ...) becomes an input that the model receives
        as a keyword argument.

    Raises:
        StructureError: `raw` lacks `input_ids`, `attention_mask` or `labels`.
    """

    del generator
    missing = [k for k in (*_INPUTS, "labels") if k not in raw]

    if missing:
        raise lk.StructureError(
            f"prepare_sequence_classification needs {', '.join(missing)} in the batch"
        )

    inputs = {k: v for k, v in raw.items() if k != "labels"}

    return lk.Batch(inputs, raw["labels"])


@lk.core.serializable
def padding_mask(f: Outputs, inputs: Inputs, /) -> Tensor:
    """Mark the positions that count in a causal LM's loss, as a `mask_fn` for `lk.Categorical`.

    Args:
        f: The logits (unused).
        inputs: The inputs, a mapping with `attention_mask` `[N, T]`.

    Returns:
        A bool tensor `[N, T]`, true at position `t` when tokens `t` and `t + 1` are both
        real. The last position is always false.

    Raises:
        StructureError: `inputs` is not a mapping.
    """

    del f
    real = _mapping(inputs, "padding_mask")["attention_mask"].bool()
    both = real[:, :-1] & real[:, 1:]

    return torch.cat([both, torch.zeros_like(real[:, :1])], dim=1)


def last_token(f: Outputs, inputs: Inputs, /) -> Tensor:
    """The logits at each sequence's last real token, with left or right padding.

    Pass it as the `select` of `lk.predict` to predict the next token after each prompt.

    Args:
        f: The logits `[N, T, V]`.
        inputs: The inputs, a mapping with `attention_mask` `[N, T]`.

    Returns:
        The logits `[N, V]`.

    Raises:
        StructureError: `f` is not a tensor, `inputs` is not a mapping, or a row has no
            real token. The last check is skipped under `vmap`, because `lk.predict` has
            already run it on the whole batch.
    """

    if not isinstance(f, Tensor):
        raise lk.StructureError(f"last_token needs logits [N, T, V], got {type(f).__name__}")

    real = _mapping(inputs, "last_token")["attention_mask"].bool()

    # Under vmap the mask can't be read. `lk.predict` calls `select` on the whole batch
    # first, so the check has run by then.
    if not is_transformed(real):
        empty = (~real.any(dim=1)).nonzero().flatten().tolist()

        if empty:
            rows = ", ".join(str(i) for i in empty)
            noun = "row" if len(empty) == 1 else "rows"

            raise lk.StructureError(
                f"last_token needs a real token in every row; none in {noun} {rows}"
            )

    positions = torch.arange(real.shape[1], device=real.device)
    last = torch.where(real, positions, 0).amax(dim=1)
    index = last[:, None, None].expand(-1, 1, f.shape[-1])

    return f.gather(1, index).squeeze(1)


def _mapping(inputs: Inputs, what: str) -> Mapping[str, Any]:
    if not isinstance(inputs, Mapping):
        raise lk.StructureError(
            f"{what} needs inputs {{'input_ids', 'attention_mask'}}, got {type(inputs).__name__}"
        )

    return inputs
