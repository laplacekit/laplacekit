"""Select a causal LM's logits for a few answers, so the predictive covers only those."""

from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import torch

import laplacekit as lk
from laplacekit.core import (
    Inputs,
    Outputs,
)
from laplacekit_transformers.tasks import last_token

__all__ = ["Tokenizer", "answer_logits"]


@runtime_checkable
class Tokenizer(Protocol):
    """A tokenizer for `answer_logits`, which only needs its `encode` method.

    Every Hugging Face tokenizer has it, and so can any other object.
    """

    @abstractmethod
    def encode(self, text: str, /, *, add_special_tokens: bool = True) -> list[int]:
        """Encode `text` as token ids.

        Args:
            text: The text.
            add_special_tokens: Whether to add the tokenizer's special tokens.
                `answer_logits` passes `False`.

        Returns:
            The token ids of `text`.
        """
        ...


def answer_logits(
    tokenizer: Tokenizer, answers: Sequence[str], /
) -> Callable[[Outputs, Inputs], Outputs]:
    """Build a `select` for `lk.predict` that keeps the logits of `answers` at the last real token.

    With it, the linearized predictive covers only those logits instead of the whole
    vocabulary:

    ```python
    select = lk.transformers.answer_logits(tokenizer, [" True", " False"])
    probs = lk.predict(post, inputs, select=select).probs  # [N, 2]
    ```

    Args:
        tokenizer: A `Tokenizer`.
        answers: The answers, as a sequence of strings. Each is encoded as given, without
            special tokens, and must be exactly one token. A leading space is part of the
            answer, as in `" True"` for most BPE tokenizers.

    Returns:
        A function that maps causal LM logits `[N, T, V]` to `[N, len(answers)]`, in the
        order of `answers`, through `last_token`. Errors from `last_token` pass through.

    Raises:
        StructureError: `tokenizer` has no `encode` method, `answers` is a single string
            or empty, an answer encodes to zero or several tokens, or two answers encode
            to the same token. The messages of the last two name the tokens. All of these
            checks run before any call of the model.
    """

    if not isinstance(tokenizer, Tokenizer):
        raise lk.StructureError(
            f"answer_logits needs a tokenizer with encode(text, add_special_tokens=False), "
            f"got {type(tokenizer).__name__}"
        )

    if isinstance(answers, str):
        raise lk.StructureError(
            f"answer_logits needs a sequence of answers, got the string {answers!r}; pass "
            f"[{answers!r}] for one answer"
        )

    if not answers:
        raise lk.StructureError("answer_logits needs at least one answer")

    ids: list[int] = []
    seen: dict[int, str] = {}

    for answer in answers:
        tokens = list(tokenizer.encode(answer, add_special_tokens=False))

        if len(tokens) != 1:
            raise lk.StructureError(
                f"answer_logits needs answers of one token each; {answer!r} encodes to "
                f"{len(tokens)} tokens {tokens}"
            )

        (token,) = tokens

        if token in seen:
            raise lk.StructureError(
                f"answer_logits needs distinct answers; {seen[token]!r} and {answer!r} both "
                f"encode to token {token}"
            )

        seen[token] = answer
        ids.append(token)

    def select(f: Outputs, inputs: Inputs) -> Outputs:
        last = last_token(f, inputs)
        index = torch.tensor(ids, device=last.device)

        return last.index_select(-1, index)

    return select
