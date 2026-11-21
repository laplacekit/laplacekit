"""`lk.reward` on a tiny random GPT-2 reward model with LoRA adapters (Laplace-LoRA), and
on a tiny random BERT reward model (no downloads)."""

import time
import warnings
from collections import Counter

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from torch import Tensor, nn
from torch.func import functional_call, jacrev
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    GPT2Config,
    GPT2ForSequenceClassification,
    PreTrainedTokenizerFast,
)

import laplacekit as lk

F64 = torch.float64
VOCAB = 32
PAD = 0


def tiny_reward_model(attn: str = "eager") -> nn.Module:
    """GPT-2 with one output (`num_labels=1`), hidden 16, 2 layers, 2 heads; LoRA r=2 on
    c_attn, and the `score` head trained in full (`modules_to_save`)."""

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB,
        n_embd=16,
        n_layer=2,
        n_head=2,
        n_positions=32,
        num_labels=1,
        pad_token_id=PAD,
        bos_token_id=PAD,
        eos_token_id=PAD,
        attn_implementation=attn,
    )
    lora = LoraConfig(
        r=2,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
        modules_to_save=["score"],
        init_lora_weights=False,
    )

    with warnings.catch_warnings():
        # peft 0.21.3 calls transformers' deprecated `is_gradient_checkpointing`.
        warnings.filterwarnings("ignore", "`is_gradient_checkpointing`", FutureWarning)
        model = get_peft_model(GPT2ForSequenceClassification(config), lora)

    return model.to(F64).eval()


def tokenizer() -> PreTrainedTokenizerFast:
    """Words `w1` ... `w31`, each one token; `[PAD]` is token 0."""

    words = {"[PAD]": PAD, **{f"w{i}": i for i in range(1, VOCAB)}}
    inner = Tokenizer(WordLevel(words, unk_token="[PAD]"))
    inner.pre_tokenizer = WhitespaceSplit()

    return PreTrainedTokenizerFast(tokenizer_object=inner, pad_token="[PAD]", unk_token="[PAD]")


def texts(lengths: list[int], seed: int) -> list[str]:
    g = torch.Generator().manual_seed(seed)

    return [" ".join(f"w{t}" for t in torch.randint(1, VOCAB, (n,), generator=g)) for n in lengths]


def preferences(n: int, seed: int) -> dict[str, dict[str, Tensor]]:
    """`n` pairs, both sides tokenized in one padded call, then split."""

    chosen = texts(list(range(2, 2 + n)), seed)
    rejected = texts(list(range(n + 1, 1, -1)), seed + 100)
    enc = tokenizer()(chosen + rejected, padding=True, return_tensors="pt")

    return {
        "chosen": {k: v[:n] for k, v in enc.items()},
        "rejected": {k: v[n:] for k, v in enc.items()},
    }


def loader() -> list[dict[str, dict[str, Tensor]]]:
    """Two batches, 7 pairs."""

    return [preferences(4, seed=1), preferences(3, seed=2)]


def fit(
    model: nn.Module, approx: lk.core.HessianApprox, batches: list | None = None
) -> lk.LaplacePosterior:
    fn = lk.reward.pairwise(lk.transformers.sequence_classification(model))
    data = lk.data.from_loader(
        loader() if batches is None else batches, prepare=lk.reward.prepare_preferences
    )

    return lk.laplace(
        fn,
        data,
        likelihood=lk.reward.bradley_terry(),
        subset=lk.subsets.trainable,
        hessian_approx=approx,
        prior=lk.priors.Isotropic(torch.tensor(1.0, dtype=F64)),
        hyper_tuner=None,
    )


def test_prepare_preferences_puts_chosen_first():
    raw = preferences(3, seed=0)
    batch = lk.reward.prepare_preferences(raw, torch.Generator().manual_seed(0))

    assert isinstance(batch.inputs, lk.reward.Pair)
    assert batch.inputs.first == raw["chosen"]
    assert batch.inputs.second == raw["rejected"]
    assert batch.size == 3
    assert batch.weights is None
    assert batch.targets is not None
    assert torch.equal(batch.targets, torch.zeros(3, dtype=torch.long))


def test_prepare_preferences_missing_key():
    raw = preferences(3, seed=0)
    gen = torch.Generator().manual_seed(0)

    with pytest.raises(lk.StructureError, match="'rejected'"):
        lk.reward.prepare_preferences({"chosen": raw["chosen"]}, gen)

    with pytest.raises(lk.StructureError, match="'chosen'"):
        lk.reward.prepare_preferences({"rejected": raw["rejected"], "prompt": {}}, gen)


def test_prepare_preferences_needs_tokenizer_outputs():
    raw = preferences(3, seed=0)
    gen = torch.Generator().manual_seed(0)

    with pytest.raises(lk.StructureError, match="'chosen'.*mapping of tensors, got list"):
        lk.reward.prepare_preferences({**raw, "chosen": ["w1 w2"]}, gen)

    with pytest.raises(lk.StructureError, match="'rejected'.*mapping of tensors, got str"):
        lk.reward.prepare_preferences({**raw, "rejected": {"input_ids": "w1"}}, gen)

    with pytest.raises(lk.StructureError, match="'chosen'.*empty"):
        lk.reward.prepare_preferences({**raw, "chosen": {}}, gen)


def test_lora_reward_diag_matches_dense():
    """Diag over the LoRA parameters is diag(Σᵢ Jᵢᵀ Hᵢ Jᵢ) + 1, with Jᵢ from `jacrev` of the
    pair rewards `[N, 2]` and Hᵢ = diag(p) − ppᵀ, p = softmax(r)."""

    model = tiny_reward_model()
    post = fit(model, lk.DiagGGN())

    params = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    names = list(post.mean)

    assert set(names) == set(params)
    assert any("lora_A" in n for n in names) and any("score" in n for n in names)

    def rewards(p: dict[str, Tensor], raw: dict[str, dict[str, Tensor]]) -> Tensor:
        def r(side: dict[str, Tensor]) -> Tensor:
            return functional_call(model, p, args=(), kwargs=side).logits

        return torch.cat([r(raw["chosen"]), r(raw["rejected"])], -1)

    want = torch.zeros(sum(params[n].numel() for n in names), dtype=F64)

    for raw in loader():
        n = raw["chosen"]["input_ids"].shape[0]
        jac = jacrev(rewards)(params, raw)
        j = torch.cat([jac[k].reshape(n, 2, -1) for k in names], -1)  # [N, 2, P]
        p = torch.softmax(rewards(params, raw), -1)
        h = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)

        want += torch.einsum("nkp,nkl,nlp->p", j, h, j)

    # The diagonal is keyed by block, one weight per block here.
    diag = post.precision().diagonal()
    got = torch.cat([diag[k.removesuffix(".weight")].reshape(-1) for k in names])

    assert int(post.n_data) == 7
    torch.testing.assert_close(got, want + 1)


def _calls(model: nn.Module, batches: list) -> tuple[int, dict[str, list[int]]]:
    """During a KFAC fit: the forwards of the GPT-2 model, and the batch size of each call
    to each LoRA layer."""

    forwards = 0
    sizes: dict[str, list[int]] = {}

    def count(module, args, kwargs, out):
        nonlocal forwards
        forwards += 1

    handles = [model.base_model.model.register_forward_hook(count, with_kwargs=True)]

    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            sizes[name] = []

            def record(module, args, out, name=name):
                sizes[name].append(args[0].shape[0])

            handles.append(module.register_forward_hook(record))

    try:
        post = fit(model, lk.KFAC(), batches)
    finally:
        for h in handles:
            h.remove()

    assert torch.isfinite(post.log_marglik())

    return forwards, sizes


def test_lora_reward_kfac_runs_and_hooks_once():
    """Each LoRA layer records one call per forward of the model: both sides of the pairs go
    through it together, `[2N, T]`, not one side at a time."""

    model = tiny_reward_model()
    post = fit(model, lk.KFAC())

    assert str(post).count("KFAC()") == 1
    assert set(post.mean) == {n for n, p in model.named_parameters() if p.requires_grad}
    assert torch.isfinite(post.log_marglik())

    forwards_one, one = _calls(model, loader()[:1])
    forwards_two, two = _calls(model, loader())
    extra = forwards_two - forwards_one

    assert len(one) == 2  # c_attn in each of the 2 layers
    assert extra >= 1

    # The second batch holds 3 pairs, so each call it adds sees 6 sequences.
    for name in one:
        assert Counter(two[name]) - Counter(one[name]) == {6: extra}
        assert len(two[name]) - len(one[name]) == extra


def test_reward_outputs_on_single_texts():
    model = tiny_reward_model()
    post = fit(model, lk.KFAC())
    single = tokenizer()(texts([3, 5, 2], seed=9), padding=True, return_tensors="pt")

    out = lk.predict_outputs(post, single)

    assert out.mean.shape == (3, 1)
    torch.testing.assert_close(out.mean, model(**single).logits)

    var = out.cov.per_input()

    assert var.shape == (3, 1, 1)
    assert bool((var > 0).all())


def tiny_bert_reward_model() -> nn.Module:
    """BERT with one output, hidden 32, 4 layers, 2 heads, eager attention. Its attention
    mask code branches on `padding_mask.all()`, so it has no vmap, as the docs' reward
    model."""

    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=32,
        num_labels=1,
        pad_token_id=PAD,
        attn_implementation="eager",
    )

    return BertForSequenceClassification(config).to(F64).eval()


def test_outputs_tiny_bert_matches_and_is_fast(monkeypatch):
    """Without vmap, `outputs` samples 64 draws J z_s. On a last-layer posterior they come
    from one recorded forward (a VJP of a VJP), not one forward per draw, and equal a
    per-input jacrev reference."""

    model = tiny_bert_reward_model()
    fn = lk.reward.pairwise(lk.transformers.sequence_classification(model))
    data = lk.data.from_loader(loader(), prepare=lk.reward.prepare_preferences)
    post = lk.laplace(
        fn,
        data,
        likelihood=lk.reward.bradley_terry(),
        subset=lk.subsets.named("classifier.weight"),
        hessian_approx=lk.KFAC(),
        prior=lk.priors.Isotropic(torch.tensor(1.0, dtype=F64)),
        hyper_tuner=None,
    )
    x = dict(tokenizer()(texts([3, 5, 2, 4], seed=9), padding=True, return_tensors="pt"))

    assert not post.caps.vmap

    # The posterior's noise, as `outputs` draws it: a spy keeps the last draw.
    noise = lk.LaplacePosterior.sample_centered
    drawn: list[dict[str, Tensor]] = []

    def spy(self, n, /, *, generator=None):
        drawn.append(noise(self, n, generator=generator))
        return drawn[-1]

    monkeypatch.setattr(lk.LaplacePosterior, "sample_centered", spy)

    calls = 0

    def count(*_: object) -> None:
        nonlocal calls
        calls += 1

    lk.predict_outputs(post, x)  # warm-up
    handle = model.register_forward_hook(count)
    start = time.perf_counter()
    out = lk.predict_outputs(post, x, generator=torch.Generator().manual_seed(0))
    seconds = time.perf_counter() - start
    handle.remove()

    # Reference: J_i z_s, with J_i from jacrev on input i alone and z_s the posterior's noise,
    # centred and rescaled as `outputs` documents.
    s = lk.core.DEFAULT_OUTPUT_SAMPLES
    z = drawn[-1]["classifier.weight"]
    z = (z - z.mean(0)) * (s / (s - 1)) ** 0.5
    w = post.mean["classifier.weight"]

    def reward(weight: Tensor, xi: dict[str, Tensor]) -> Tensor:
        return functional_call(model, {"classifier.weight": weight}, kwargs=xi).logits[0, 0]

    jac = torch.stack(
        [jacrev(reward)(w, {k: v[i : i + 1] for k, v in x.items()}) for i in range(4)]
    )
    expected = torch.einsum("spq,ipq->si", z, jac).unsqueeze(-1)

    assert isinstance(out.cov, lk.core.SampleCov)
    torch.testing.assert_close(out.cov.draws, expected, rtol=1e-5, atol=1e-12)

    # The old path, one JVP per draw, gives the same draws.
    old = lk.predict_outputs(
        post, x, strategy="forward-mode", generator=torch.Generator().manual_seed(0)
    )

    assert isinstance(old.cov, lk.core.SampleCov)
    torch.testing.assert_close(out.cov.draws, old.cov.draws, rtol=1e-5, atol=1e-12)

    # One forward for the mean and one recorded for the draws; per draw, a JVP would run
    # all 64 again. Measured 0.006 s; one JVP per draw took 0.13 s.
    assert calls <= 2
    assert seconds < 0.05
