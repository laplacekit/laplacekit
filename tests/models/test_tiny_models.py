"""End to end on tiny random Hugging Face and diffusers models, through the public API.

Models are built from configs, never downloaded. The probe results are the measured table
in the design spec (section 4): the fused CPU attention kernel has no forward-mode AD, Mamba's
reference scan breaks `vmap`, GPT-2's `Conv1D` is not an `nn.Linear`, and Mamba uses
`dt_proj.weight` without calling the module.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from diffusers import UNet2DConditionModel, UNet2DModel
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MambaConfig,
    MambaForCausalLM,
)
from transformers.pytorch_utils import Conv1D

import laplacekit as lk

pytestmark = [
    # functorch has no batching rule for the fused CPU attention kernel and says so on every
    # call; vmap still runs, through the slow fallback.
    pytest.mark.filterwarnings(
        "ignore:There is a performance drop because we have not yet implemented the batching "
        "rule:UserWarning"
    ),
    # peft 0.21 calls a transformers method that transformers 5 deprecated.
    pytest.mark.filterwarnings(
        "ignore:`is_gradient_checkpointing` is deprecated:FutureWarning:transformers"
    ),
]

N, T, VOCAB = 4, 6, 32


@dataclass(frozen=True)
class Case:
    model: nn.Module
    fn: lk.core.ModelFn
    likelihood: lk.Categorical | lk.Gaussian
    batch: lk.Batch


def _logits(model: nn.Module) -> lk.core.ModelFn:
    """Calls the module on a dict of inputs and returns `.logits`."""

    base = lk.core.ModelFn.from_module(model)

    return lk.core.ModelFn(lambda p, x: base(p, x).logits, model)


def _sample(model: nn.Module) -> lk.core.ModelFn:
    """Calls the module on a dict of inputs and returns `.sample`."""

    base = lk.core.ModelFn.from_module(model)

    return lk.core.ModelFn(lambda p, x: base(p, x).sample, model)


def _tokens(n: int = N, vocab: int = VOCAB) -> Tensor:
    return torch.randint(0, vocab, (n, T))


def _lm_case(model: nn.Module, vocab: int = VOCAB) -> Case:
    ids = _tokens(vocab=vocab)

    return Case(model.eval(), _logits(model), lk.Categorical(), lk.Batch({"input_ids": ids}, ids))


def _llama_config(attn: str) -> LlamaConfig:
    # The MLP blocks (32 x 160) exceed DENSE_MAX_PARAMS, so the default picks KFAC there.
    return LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=160,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attn_implementation=attn,
    )


def llama(attn: str = "eager") -> Case:
    torch.manual_seed(0)
    return _lm_case(LlamaForCausalLM(_llama_config(attn)))


def gpt2() -> Case:
    # Untied, so lm_head (32 x 160) is its own block and large enough for KFAC.
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=160,
        n_positions=32,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
        tie_word_embeddings=False,
    )

    return _lm_case(GPT2LMHeadModel(config), vocab=160)


def mamba() -> Case:
    torch.manual_seed(0)
    config = MambaConfig(
        vocab_size=VOCAB,
        hidden_size=16,
        state_size=4,
        num_hidden_layers=2,
        expand=2,
        conv_kernel=4,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
    )

    return _lm_case(MambaForCausalLM(config))


_UNET = dict(
    sample_size=8,
    in_channels=2,
    out_channels=2,
    layers_per_block=1,
    block_out_channels=(16, 32),
    norm_num_groups=8,
    attention_head_dim=8,
)


def _unet_case(model: nn.Module, inputs: dict[str, Tensor]) -> Case:
    targets = torch.randn(inputs["sample"].shape)

    return Case(
        model.eval(),
        _sample(model),
        lk.Gaussian(sigma=1.0, event_dims=3),
        lk.Batch(inputs, targets),
    )


def unet() -> Case:
    torch.manual_seed(0)
    model = UNet2DModel(
        **_UNET,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    )
    inputs = {"sample": torch.randn(N, 2, 8, 8), "timestep": torch.randint(0, 1000, (N,))}

    return _unet_case(model, inputs)


def unet_cond() -> Case:
    torch.manual_seed(0)
    model = UNet2DConditionModel(
        **_UNET,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=8,
    )
    inputs = {
        "sample": torch.randn(N, 2, 8, 8),
        "timestep": torch.randint(0, 1000, (N,)),
        "encoder_hidden_states": torch.randn(N, 3, 8),
    }

    return _unet_case(model, inputs)


CASES: dict[str, Callable[[], Case]] = {
    "llama-eager": llama,
    "llama-sdpa": lambda: llama("sdpa"),
    "gpt2": gpt2,
    "mamba": mamba,
    "unet": unet,
    "unet-cond": unet_cond,
}

# (vjp, jvp, vmap) from the spec's measured table. Eager attention has forward AD.
# vjp, vjp_twice, jvp, vmap. Fused attention has neither forward AD nor a double backward.
TABLE = {
    "llama-eager": (True, True, True, True),
    "llama-sdpa": (True, False, False, True),
    "gpt2": (True, False, False, True),
    "mamba": (True, True, True, False),
    "unet": (True, False, False, True),
    "unet-cond": (True, False, False, True),
}
_COLUMNS = ("vjp", "vjp_twice", "jvp", "vmap")


def _names(model: nn.Module, keep: Callable[[str, nn.Module], bool]) -> set[str]:
    return {n for n, m in model.named_modules() if keep(n, m)}


def _linear_or_conv2d(model: nn.Module) -> set[str]:
    return _names(model, lambda _, m: isinstance(m, nn.Linear | nn.Conv2d))


def _unhookable(model: nn.Module) -> set[str]:
    """GPT-2's `Conv1D` layers and Mamba's `dt_proj`, which KFAC hooks can't see."""

    return _names(model, lambda n, m: isinstance(m, Conv1D) or n.endswith(".dt_proj"))


@pytest.mark.parametrize("name", list(CASES))
def test_probe_matches_measured_table(name):
    case = CASES[name]()
    caps = lk.core.probe(case.fn, case.likelihood, case.batch)
    _, vjp_twice, jvp, _ = TABLE[name]

    assert (caps.vjp, caps.vjp_twice, caps.jvp, caps.vmap) == TABLE[name], caps.failures
    failed = {k for k, ok in zip(_COLUMNS, TABLE[name], strict=True) if not ok}

    assert set(caps.failures) == failed

    if not jvp:
        assert "forward AD" in caps.failures["jvp"]

    if not vjp_twice:
        assert "derivative for" in caps.failures["vjp_twice"]

    unhookable = _unhookable(case.model)

    assert not caps.hooked & unhookable
    # Every Linear and Conv2d is hooked, except Mamba's dt_proj.
    assert caps.hooked == _linear_or_conv2d(case.model) - unhookable


def test_unhookable_layers_exist():
    """The exclusions above are not vacuous."""

    assert len(_unhookable(gpt2().model)) == 8  # c_attn, c_proj, c_fc, mlp.c_proj per layer
    assert _unhookable(mamba().model) == {f"backbone.layers.{i}.mixer.dt_proj" for i in (0, 1)}


@pytest.mark.parametrize("name", list(CASES))
def test_default_curvature_puts_kfac_only_on_hooked_layers(name):
    case = CASES[name]()
    caps = lk.core.probe(case.fn, case.likelihood, case.batch)
    params = lk.subsets.all(case.fn, caps)
    chosen = lk.curvature.default(params, caps, likelihood=case.likelihood).by_block(
        lk.core.Layout.by_module(params).blocks
    )
    kfac = {b for b, c in chosen.items() if isinstance(c, lk.KFAC)}

    assert kfac, f"no block got KFAC: {chosen}"
    assert kfac <= caps.hooked
    assert kfac <= _linear_or_conv2d(case.model)

    unhookable = _unhookable(case.model) & set(chosen)

    assert not kfac & unhookable

    # With vmap the exact diagonal takes these layers; without it, the sampled Fisher does.
    fallback = lk.DiagGGN if caps.vmap else lk.DiagGGN

    for b in unhookable:
        assert isinstance(chosen[b], fallback | lk.DenseGGN), (b, chosen[b])

    if name == "gpt2":
        assert {b for b in unhookable if isinstance(chosen[b], lk.DiagGGN)}  # big Conv1D

    if name == "mamba":
        assert unhookable and all(isinstance(chosen[b], fallback) for b in unhookable)


def _lm_data(case: Case) -> list[tuple[dict[str, Tensor], Tensor]]:
    ids = case.batch.targets

    assert isinstance(ids, Tensor)

    return [({"input_ids": ids[i : i + 2]}, ids[i : i + 2]) for i in (0, 2)]


def _check_token_probs(d: Any, ids: Tensor) -> None:
    probs = d.probs

    assert probs.shape == (*ids.shape, VOCAB)
    assert torch.isfinite(probs).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(ids.shape))


def test_llama_last_layer_fit_and_predict():
    case = llama()
    post = lk.laplace(case.fn, _lm_data(case), likelihood=case.likelihood)
    ids = _tokens(2)

    assert '"lm_head.weight": (32, 32)' in repr(post)
    _check_token_probs(lk.predict(post, {"input_ids": ids}), ids)
    _check_token_probs(lk.predict(post, {"input_ids": ids}, lk.WeightSamples(n_samples=4)), ids)


def test_llama_default_curvature_on_all_params_fits():
    case = llama()
    post = lk.laplace(
        case.fn, _lm_data(case), likelihood=case.likelihood, subset=lk.subsets.all, hyper_tuner=None
    )

    assert isinstance(post.curvature.blocks["model.layers.0.mlp.up_proj"], lk.core.KronEigenbasis)
    assert all(torch.isfinite(p).all() for p in post.mean.values())
    ids = _tokens(2)
    _check_token_probs(lk.predict(post, {"input_ids": ids}), ids)


def test_llama_lora_fit_and_predict():
    torch.manual_seed(0)
    base = LlamaForCausalLM(_llama_config("eager"))
    lora = get_peft_model(base, LoraConfig(r=2, target_modules=["q_proj", "v_proj"]))
    case = _lm_case(lora)
    frozen = {n: p.detach().clone() for n, p in lora.named_parameters() if not p.requires_grad}

    post = lk.laplace(
        case.fn, _lm_data(case), likelihood=case.likelihood, subset=lk.subsets.trainable
    )

    assert set(post.mean) == {n for n, p in lora.named_parameters() if p.requires_grad}
    assert all("lora_" in n for n in post.mean) and len(post.mean) == 8
    ids = _tokens(2)
    _check_token_probs(lk.predict(post, {"input_ids": ids}), ids)

    for n, p in lora.named_parameters():
        if n in frozen:
            torch.testing.assert_close(p, frozen[n])


@pytest.mark.parametrize("approx", [None, lk.KFAC()], ids=["default", "kfac"])
def test_unet_last_layer_fit_and_predict(approx):
    case = unet()
    inputs = case.batch.inputs
    targets = case.batch.targets

    assert isinstance(inputs, dict) and isinstance(targets, Tensor)
    data = [({k: v[i : i + 2] for k, v in inputs.items()}, targets[i : i + 2]) for i in (0, 2)]

    post = lk.laplace(case.fn, data, likelihood=case.likelihood, hessian_approx=approx)

    assert '"conv_out.weight": (2, 16, 3, 3)' in repr(post)
    d = lk.predict(post, {k: v[:2] for k, v in inputs.items()})

    assert d.event_shape == (2, 8, 8) and d.batch_shape == (2,)
    assert torch.isfinite(d.mean).all() and torch.isfinite(d.variance).all()
    assert (d.variance > 0).all()


def test_mamba_tied_last_layer_fit_and_predict():
    """Mamba ties `lm_head.weight` to the embedding: one tensor, both uses in the posterior."""

    case = mamba()
    model = case.model

    assert model.lm_head.weight is model.backbone.embeddings.weight

    post = lk.laplace(case.fn, _lm_data(case), likelihood=case.likelihood)

    assert set(post.mean) == {"lm_head.weight"}
    # No vmap rules out DenseGGN and Diag, and the tie rules out KFAC.
    assert "DiagGGN(fisher='mc')" in repr(post) and "KFAC" not in repr(post)
    assert model.lm_head.weight is model.backbone.embeddings.weight

    ids = _tokens(2)

    # No vmap: the linearized predictive uses sampled output covariances.
    assert lk.Linearized().resolve(post).n_samples == 64
    _check_token_probs(lk.predict(post, {"input_ids": ids}), ids)

    _check_token_probs(lk.predict(post, {"input_ids": ids}, lk.WeightSamples(n_samples=4)), ids)
