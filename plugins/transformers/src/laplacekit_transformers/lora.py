"""Laplace-LoRA (Yang et al., 2024) on a causal LM or a sequence classifier.

`laplace_lora` runs `lk.laplace` over the trainable parameters, which are the LoRA
adapters and any head in `modules_to_save`. For a causal LM, the default curvature is one
`DiagGGN(fisher="mc")`. For a sequence classifier, it is one `KFAC()` when the probe's
module hooks reach every trainable block, and one `DiagGGN(fisher="mc")` otherwise.
"""

from collections.abc import Iterable, Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn

import laplacekit as lk
from laplacekit_transformers.tasks import (
    CausalLM,
    SequenceClassification,
    causal_lm,
    padding_mask,
    prepare_causal_lm,
    prepare_sequence_classification,
    sequence_classification,
)

__all__ = ["laplace_lora"]


def laplace_lora(
    model: nn.Module,
    data: Iterable[Mapping[str, Tensor]],
    *,
    task: CausalLM | SequenceClassification = CausalLM(),
    n_data: int | None = None,
    hessian_approx: lk.core.HessianApprox[Any] | None = None,
    prior: lk.core.Prior = lk.priors.Isotropic(),  # noqa: B008  # immutable
    hyper_tuner: lk.core.HyperTuner | None = lk.MarginalLikelihood(),  # noqa: B008  # immutable
    generator: torch.Generator | None = None,
) -> lk.LaplacePosterior[Any, Any]:
    r"""Fit a Laplace approximation over the trainable parameters of a Hugging Face model.

    This is Laplace-LoRA (Yang et al., 2024, https://arxiv.org/abs/2308.13111) on a causal
    LM or a sequence classifier, where the trainable parameters are the LoRA adapters. It
    runs `lk.laplace` with `subset=lk.subsets.trainable`.

    With `task=CausalLM()`, it fits `causal_lm(model)` with `prepare_causal_lm` and
    `Categorical(mask_fn=padding_mask)`. The default `hessian_approx` is then one
    `DiagGGN(fisher="mc")` over all LoRA blocks. It takes one backward pass per batch, at
    next tokens drawn from the model, so it needs no Jacobian over the vocabulary, no
    per-example gradients and no `vmap`, and SDPA attention works.

    With `task=SequenceClassification()`, it fits `sequence_classification(model)` with
    `prepare_sequence_classification` and `Categorical()`, and the batches carry
    `"labels"`, `[N]`. A peft model trains its new head when the head is listed in
    `modules_to_save` (`["score"]` for GPT-2 and Llama classifiers). The default `hessian_approx`
    is then one `KFAC()` over all trainable blocks, as in Laplace-LoRA, when three
    conditions hold: the probe's module hooks reach each block's layer, each block holds
    its layer's weight, and no tensor is tied. Otherwise it is one `DiagGGN(fisher="mc")`.
    The per-block default of `lk.laplace` would give each small LoRA block the dense GGN,
    which is much slower on a large model.

    Before the fit, `laplace_lora` reads one batch. It checks that the logits are `[N, K]`
    with $K \ge 2$ and, for the default `hessian_approx`, runs a forward pass on two of the
    batch's examples to find the hooked layers. `data` is therefore iterated once more
    than by `lk.laplace`.

    Args:
        model: The model. The fit covers its trainable parameters (`requires_grad`), so
            load a saved adapter with `PeftModel.from_pretrained(..., is_trainable=True)`.
        data: Tokenizer batches, iterated once per pass.
        task: `CausalLM()` or `SequenceClassification()`.
        n_data: The number of examples, which is the size of the training set when `data`
            is a subset of it, as in `lk.data.from_loader`.
        hessian_approx: As in `lk.laplace`, with the defaults above.
        prior: As in `lk.laplace`.
        hyper_tuner: As in `lk.laplace`.
        generator: As in `lk.laplace`.

    Returns:
        The posterior over the trainable parameters, as from `lk.laplace`.

    Raises:
        StructureError: Before any data is read, when `task` is neither `CausalLM()` nor
            `SequenceClassification()` or the model has no trainable parameter. Later,
            when a batch lacks the keys its `prepare_*` function needs or the model's
            output is not a `ModelOutput`. With `SequenceClassification()`, also when the
            output has no logits, the logits of the first batch are not `[N, K]` for its
            N sequences, or $K < 2$. Otherwise as in `lk.laplace`.
        CapabilityError: As in `lk.laplace`.
        ValueError: The data has no batches, or as in `lk.laplace`.
        TypeError: `data` is a one-shot iterator, or as in `lk.laplace`.
    """

    if isinstance(task, CausalLM):
        fn = causal_lm(model)
    elif isinstance(task, SequenceClassification):
        fn = sequence_classification(model)
    else:
        raise lk.StructureError(
            f"laplace_lora needs task=CausalLM() or SequenceClassification(), got "
            f"{type(task).__name__}"
        )

    # `trainable` reads only the module, so it needs no probe.
    if not lk.subsets.trainable(fn, lk.core.Capabilities()):
        raise lk.StructureError(
            "laplace_lora found no trainable parameters; add LoRA adapters (peft), load a "
            "saved adapter with PeftModel.from_pretrained(..., is_trainable=True), or set "
            "requires_grad on the parameters to fit"
        )

    if isinstance(task, CausalLM):
        src = lk.data.from_loader(data, prepare=prepare_causal_lm, n_data=n_data)
        likelihood = lk.Categorical(mask_fn=padding_mask)

        # One method over all blocks: one backward per batch, not one per block.
        chosen = lk.DiagGGN(fisher="mc") if hessian_approx is None else hessian_approx
    else:
        src = lk.data.from_loader(data, prepare=prepare_sequence_classification, n_data=n_data)
        likelihood = lk.Categorical()
        first = _check_classifier(fn, src)
        chosen = (
            _classifier_default(fn, likelihood, first) if hessian_approx is None else hessian_approx
        )

    return lk.laplace(
        fn,
        src,
        likelihood,
        subset=lk.subsets.trainable,
        hessian_approx=chosen,
        prior=prior,
        hyper_tuner=hyper_tuner,
        generator=generator,
    )


def _check_classifier(fn: lk.core.ModelFn, src: lk.data.Data) -> lk.Batch:
    """Run one batch through `fn` before the fit and check for logits `[N, K]`.

    Returns the batch. Its `prepare` draws nothing, so a fixed generator reads it.
    """

    first = next(iter(src(torch.Generator())), None)

    if first is None:
        raise ValueError("laplace_lora: the data yielded no batches")

    with torch.no_grad():
        logits = cast(Tensor, fn({}, first.inputs))

    if logits.ndim != 2 or logits.shape[0] != first.size:
        raise lk.StructureError(
            f"sequence_classification needs logits [N, K] for N = {first.size} sequences, got "
            f"{tuple(logits.shape)}"
        )

    # One class has no curvature: the posterior would be the prior.
    if logits.shape[1] < 2:
        raise lk.StructureError(
            f"sequence_classification needs K >= 2 classes, got K = {logits.shape[1]}; set "
            f"num_labels >= 2 in the model config"
        )

    return first


def _classifier_default(
    fn: lk.core.ModelFn, likelihood: lk.Categorical, first: lk.Batch
) -> lk.core.HessianApprox[Any]:
    """`KFAC()` if the capabilities allow it on every trainable block, else `DiagGGN(fisher="mc")`.

    The probe's order pass, one forward pass on two examples without transforms, records
    which layers the module hooks reach. KFAC needs those hooks on each block's layer,
    each block to hold its layer's weight, and no tied tensor.
    """

    caps = lk.core.probe(fn, likelihood, first, params={})
    params = lk.subsets.trainable(fn, caps)
    blocks = lk.core.Layout.by_module(params).blocks
    kfac = lk.KFAC()
    req = kfac.requires

    hooked = all(b in caps.hooked and issubclass(caps.hooked_types[b], req.hooks) for b in blocks)
    weights = all(f"{b}.weight" in names for b, names in blocks.items())
    fits = req.satisfies(caps) and hooked and weights and lk.KFAC.supports(likelihood, caps, params)

    return kfac if fits else lk.DiagGGN(fisher="mc")
