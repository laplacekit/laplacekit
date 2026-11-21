"""laplacekit-transformers on tiny random Llama and GPT-2 models with LoRA adapters (no
downloads)."""

import glob
import warnings
from importlib import metadata

import laplacekit_transformers
import pytest
import torch
from laplacekit_transformers import (
    CausalLM,
    SequenceClassification,
    answer_logits,
    causal_lm,
    laplace_lora,
    last_token,
    padding_mask,
    prepare_causal_lm,
    prepare_sequence_classification,
    sequence_classification,
)
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from torch import Tensor, nn
from torch.func import functional_call, jacrev
from transformers import (
    BatchEncoding,
    GPT2Config,
    GPT2ForSequenceClassification,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)
from transformers.modeling_outputs import SequenceClassifierOutput

import laplacekit as lk

VOCAB = 64


def tiny_lora(
    vocab: int = VOCAB, dtype: torch.dtype = torch.float32, attn: str = "eager"
) -> nn.Module:
    """Llama with vocab `vocab`, hidden 16, 2 layers, 2 heads; LoRA r=2 on q_proj, v_proj.

    Eager attention by default, because the SDPA path branches on the mask and so doesn't
    vmap. LoRA's B starts random, so the A gradients are not all zero.
    """

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=vocab,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attn_implementation=attn,
    )
    lora = LoraConfig(r=2, target_modules=["q_proj", "v_proj"], init_lora_weights=False)

    with warnings.catch_warnings():
        # peft 0.21.3 calls transformers' deprecated `is_gradient_checkpointing`.
        warnings.filterwarnings("ignore", "`is_gradient_checkpointing`", FutureWarning)
        model = get_peft_model(LlamaForCausalLM(config), lora)

    return model.to(dtype).eval()


def lora_params(model: nn.Module) -> dict[str, Tensor]:
    return {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}


def sequences(lengths: list[int], vocab: int = VOCAB, seed: int = 1) -> list[Tensor]:
    g = torch.Generator().manual_seed(seed)

    return [torch.randint(0, vocab, (n,), generator=g) for n in lengths]


def right_padded(seqs: list[Tensor], width: int, seed: int = 2) -> dict[str, Tensor]:
    """A tokenizer-style batch, right-padded to `width` with random token ids."""

    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (len(seqs), width), generator=g)
    mask = torch.zeros(len(seqs), width, dtype=torch.long)

    for i, s in enumerate(seqs):
        ids[i, : len(s)] = s
        mask[i, : len(s)] = 1

    return {"input_ids": ids, "attention_mask": mask}


def left_padded(seqs: list[Tensor], width: int) -> dict[str, Tensor]:
    ids = torch.zeros(len(seqs), width, dtype=torch.long)
    mask = torch.zeros(len(seqs), width, dtype=torch.long)

    for i, s in enumerate(seqs):
        ids[i, width - len(s) :] = s
        mask[i, width - len(s) :] = 1

    return {"input_ids": ids, "attention_mask": mask}


def loader() -> list[dict[str, Tensor]]:
    """Two right-padded batches of sentences with different lengths, 5 sequences."""

    return [
        right_padded(sequences([5, 3, 6], seed=1), 6),
        right_padded(sequences([4, 6], seed=3), 7),
    ]


def row(post: lk.LaplacePosterior, key: str) -> str:
    """The value of one row of `print(post)`."""

    for line in str(post).splitlines():
        name, _, value = line.strip().partition(" = ")

        if name.strip() == key:
            return value

    raise KeyError(key)


# --- the pieces ---------------------------------------------------------------------------


def test_causal_lm_returns_the_logits():
    model = tiny_lora()
    batch = right_padded(sequences([5, 3]), 6)
    params = lora_params(model)
    fn = causal_lm(model)

    f = fn(params, batch)
    ref = model(**batch).logits

    assert fn.module is model
    assert isinstance(f, Tensor) and f.shape == (2, 6, VOCAB)
    torch.testing.assert_close(f, ref)

    # Only the passed parameters are variables.
    name = next(iter(params))
    weight = params[name].clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(fn({**params, name: weight}, batch).sum(), weight)

    assert grad.abs().sum() > 0


def test_prepare_shifts_targets_left():
    raw = right_padded(sequences([4, 2]), 5)
    batch = prepare_causal_lm(raw, torch.Generator().manual_seed(0))

    assert isinstance(batch.inputs, dict)
    assert set(batch.inputs) == {"input_ids", "attention_mask"}
    assert batch.inputs["input_ids"] is raw["input_ids"]
    assert batch.inputs["attention_mask"] is raw["attention_mask"]
    assert batch.weights is None

    y = batch.targets
    assert isinstance(y, Tensor)
    assert torch.equal(y[:, :-1], raw["input_ids"][:, 1:])
    assert torch.equal(y[:, -1], torch.zeros(2, dtype=y.dtype))


def test_prepare_needs_input_ids_and_attention_mask():
    with pytest.raises(lk.StructureError, match="attention_mask"):
        prepare_causal_lm(
            {"input_ids": torch.zeros(1, 3, dtype=torch.long)}, torch.Generator().manual_seed(0)
        )


def test_padding_mask_counts_positions_with_a_real_next_token():
    inputs = {
        "input_ids": torch.zeros(3, 4, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 1]]),
    }
    f = torch.zeros(3, 4, VOCAB)

    mask = padding_mask(f, inputs)

    expected = torch.tensor(
        [[True, True, False, False], [True, True, True, False], [False, True, True, False]]
    )

    assert mask.dtype == torch.bool
    assert torch.equal(mask, expected)


@pytest.mark.parametrize("pad", [right_padded, left_padded])
def test_last_token_picks_the_last_real_position(pad):
    seqs = sequences([3, 5, 1])
    inputs = pad(seqs, 5)
    f = torch.randn(3, 5, VOCAB)

    picked = last_token(f, inputs)

    real = inputs["attention_mask"].bool()
    expected = torch.stack([f[i][real[i]][-1] for i in range(3)])

    assert torch.equal(picked, expected)


def test_last_token_all_padding_raises():
    """A row with no real token has no last token: refused, naming the row."""

    inputs = right_padded(sequences([3, 5, 2]), 5)
    inputs["attention_mask"][1] = 0

    with pytest.raises(lk.StructureError, match=r"last_token.*row 1\b"):
        last_token(torch.randn(3, 5, VOCAB), inputs)

    inputs["attention_mask"][2] = 0

    with pytest.raises(lk.StructureError, match=r"none in rows 1, 2$"):
        last_token(torch.randn(3, 5, VOCAB), inputs)


def test_predict_refuses_an_all_padding_row_before_the_jacobian():
    """`lk.predict` calls `select` on the whole batch first, outside vmap, so it refuses."""

    model = tiny_lora(attn="eager")
    post = laplace_lora(model, loader(), hyper_tuner=None)
    prompts = right_padded(sequences([3, 5, 2], seed=4), 5)
    prompts["attention_mask"][2] = 0

    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))

    try:
        with pytest.raises(lk.StructureError, match=r"row 2\b"):
            lk.predict(post, prompts, select=last_token)
    finally:
        handle.remove()

    assert len(calls) == 1  # the mean forward only


# --- the front door ---------------------------------------------------------------------


def test_laplace_lora_fits():
    model = tiny_lora()
    base = {n: p.clone() for n, p in model.named_parameters() if not p.requires_grad}
    post = laplace_lora(model, loader())

    blocks = lk.core.Layout.by_module(lora_params(model)).blocks
    assert len(blocks) == 8
    assert set(post.mean) == set(lora_params(model))
    names = list(lora_params(model))
    shown = ", ".join(f'"{n}": {tuple(lora_params(model)[n].shape)}' for n in names[:3])

    assert row(post, "subset") == f"trainable {{{shown}, ...}} (8 tensors, 256 parameters)"
    assert row(post, "hessian_approx") == "DiagGGN(fisher='mc')"
    assert row(post, "curvature") == "BlockDiag[Diag] (8 blocks, 256 parameters)"

    assert torch.isfinite(post.log_marglik())

    # The frozen base is untouched.
    for n, p in model.named_parameters():
        if n in base:
            assert torch.equal(p, base[n])

    prompts = left_padded(sequences([3, 5, 2], seed=4), 5)
    pred = lk.predict(post, prompts, select=last_token)
    probs = pred.probs

    assert probs.shape == (3, VOCAB)
    assert (probs >= 0).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(3))


def test_laplace_lora_matches_the_empirical_fisher_by_hand():
    """Diagonal of the precision against per-sequence gradients of the unpadded sequences."""

    model = tiny_lora(dtype=torch.float64)
    batches = loader()
    post = laplace_lora(
        model,
        batches,
        hessian_approx=lk.DiagGGN(fisher="empirical"),
        prior=lk.priors.Isotropic(1.0),
        hyper_tuner=None,
    )

    params = dict((n, p) for n, p in model.named_parameters() if p.requires_grad)
    names = list(params)
    ef = {n: torch.zeros_like(p) for n, p in params.items()}
    total = torch.zeros((), dtype=torch.float64)
    count = 0

    for raw in batches:
        for ids, mask in zip(raw["input_ids"], raw["attention_mask"], strict=True):
            seq = ids[mask.bool()].unsqueeze(0)
            logits = model(input_ids=seq).logits[0, :-1]
            ll = torch.log_softmax(logits, -1).gather(-1, seq[0, 1:, None]).sum()
            grads = torch.autograd.grad(ll, [params[n] for n in names])

            for n, g in zip(names, grads, strict=True):
                ef[n] = ef[n] + g.detach() ** 2

            total = total + ll.detach()
            count += 1

    layout = post.layout
    expected = {b: 1 + layout.flatten(ef, b) for b in layout.blocks}

    assert int(post.n_data) == count == 5
    torch.testing.assert_close(post.precision().diagonal(), expected, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(post.data_fit[0], total, rtol=1e-8, atol=1e-8)


def test_laplace_lora_equals_the_spelled_out_call():
    model = tiny_lora()
    by_hand = lk.laplace(
        causal_lm(model),
        lk.data.from_loader(loader(), prepare=prepare_causal_lm),
        likelihood=lk.Categorical(mask_fn=padding_mask),
        subset=lk.subsets.trainable,
        hessian_approx=lk.DiagGGN(fisher="mc"),
        generator=torch.Generator().manual_seed(0),
    )
    post = laplace_lora(model, loader(), generator=torch.Generator().manual_seed(0))

    assert str(post) == str(by_hand)
    torch.testing.assert_close(post.precision().diagonal(), by_hand.precision().diagonal())


def test_one_empirical_fisher_equals_the_per_block_map():
    model = tiny_lora(dtype=torch.float64)
    blocks = lk.core.Layout.by_module(lora_params(model)).blocks
    rules = lk.per_block({glob.escape(b): lk.DiagGGN(fisher="empirical") for b in blocks})

    post = laplace_lora(
        model, loader(), hessian_approx=lk.DiagGGN(fisher="empirical"), hyper_tuner=None
    )
    by_block = laplace_lora(model, loader(), hessian_approx=rules, hyper_tuner=None)

    assert row(by_block, "hessian_approx") == repr(rules)
    torch.testing.assert_close(post.precision().diagonal(), by_block.precision().diagonal())
    torch.testing.assert_close(post.data_fit, by_block.data_fit)


def _forwards(model: nn.Module, n_batches: int, hessian_approx=None) -> int:
    count = 0

    def hook(module, args, kwargs, out):
        nonlocal count
        count += 1

    handle = model.register_forward_hook(hook, with_kwargs=True)

    try:
        laplace_lora(model, loader()[:n_batches], hessian_approx=hessian_approx, hyper_tuner=None)
    finally:
        handle.remove()

    return count


def test_default_adds_one_forward_per_batch_not_one_per_block():
    """One sampled-Fisher pass per batch over all LoRA blocks, not one per block."""

    model = tiny_lora()
    blocks = lk.core.Layout.by_module(lora_params(model)).blocks
    rules = lk.per_block({glob.escape(b): lk.DiagGGN(fisher="mc") for b in blocks})

    # Per batch: one forward, which the data-fit term reuses. The per-block map runs one
    # forward per block.
    assert _forwards(model, 2) - _forwards(model, 1) == 1
    assert _forwards(model, 2, rules) - _forwards(model, 1, rules) == len(blocks)


def test_default_fits_with_sdpa_attention():
    """The sampled Fisher needs only backward passes, so SDPA (no vmap) fits by default."""

    post = laplace_lora(tiny_lora(attn="sdpa"), loader(), hyper_tuner=None)

    assert not post.caps.vmap
    assert row(post, "hessian_approx") == "DiagGGN(fisher='mc')"
    assert torch.isfinite(post.log_marglik())


def test_sdpa_fits_but_the_linearized_predictive_refuses_before_any_forward():
    """SDPA has no double backward: `predict` raises CapabilityError naming eager attention."""

    model = tiny_lora(attn="sdpa")
    post = laplace_lora(model, loader(), hyper_tuner=None)

    assert post.caps.vjp and not post.caps.vjp_twice

    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))
    prompts = left_padded(sequences([3, 5, 2], seed=4), 5)

    try:
        with pytest.raises(lk.CapabilityError, match='(?s)vjp_twice.*attn_implementation="eager"'):
            lk.predict(post, prompts, select=last_token)
    finally:
        handle.remove()

    assert calls == []
    probs = lk.predict(post, prompts, lk.WeightSamples(n_samples=3), select=last_token).probs

    assert probs.shape == (3, VOCAB)


def test_laplace_lora_passes_an_explicit_approx_through():
    post = laplace_lora(tiny_lora(), loader(), hessian_approx=lk.DiagGGN(), hyper_tuner=None)

    assert row(post, "hessian_approx").startswith("DiagGGN(")


def test_laplace_lora_needs_trainable_parameters():
    model = tiny_lora()

    for p in model.parameters():
        p.requires_grad_(False)

    class Untouchable:
        def __iter__(self):
            raise AssertionError("data consumed")

    with pytest.raises(lk.StructureError, match="trainable"):
        laplace_lora(model, Untouchable())


# --- padding --------------------------------------------------------------------------------


def _stat(model: nn.Module, raw: dict[str, Tensor], lik: lk.Categorical) -> tuple[Tensor, Tensor]:
    """The summed `DenseGGN` blocks (each 32 x 32) stacked, and the data-fit term."""

    fit = lk.core.fit_curvature(
        causal_lm(model),
        lik,
        lora_params(model),
        lk.DenseGGN(),
        generator=torch.Generator().manual_seed(3),
    )
    stat = fit.update(fit.init(), prepare_causal_lm(raw, torch.Generator().manual_seed(0)))
    dense = torch.stack([stat.curvature.sum[b] for b in sorted(stat.curvature.sum)])

    assert stat.data_fit is not None

    return dense, stat.data_fit


def test_right_padding_changes_nothing():
    model = tiny_lora(dtype=torch.float64)
    lik = lk.Categorical(mask_fn=padding_mask)

    # Equal lengths, no padding, against the same batch right-padded with 3 random tokens.
    seqs = sequences([5, 5, 5])
    plain = right_padded(seqs, 5)
    padded = right_padded(seqs, 8)

    assert plain["attention_mask"].all() and not padded["attention_mask"].all()

    s1, fit1 = _stat(model, plain, lik)
    s2, fit2 = _stat(model, padded, lik)

    torch.testing.assert_close(s1, s2, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fit1, fit2, rtol=1e-5, atol=1e-5)

    # Mixed lengths in one padded batch: the sum over each sequence alone, unpadded.
    mixed = sequences([2, 6, 4], seed=5)
    s3, fit3 = _stat(model, right_padded(mixed, 7), lik)
    parts = [_stat(model, right_padded([s], len(s)), lik) for s in mixed]

    torch.testing.assert_close(s3, sum(p[0] for p in parts), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fit3, sum(p[1] for p in parts), rtol=1e-5, atol=1e-5)

    # Without the mask the padding shows.
    u1, _ = _stat(model, plain, lk.Categorical())
    u2, _ = _stat(model, padded, lk.Categorical())

    assert not torch.allclose(u1, u2, rtol=1e-3, atol=1e-3)


# --- large vocabularies -----------------------------------------------------------------------


def test_large_vocab_fits_without_output_hessian_and_predicts(monkeypatch):
    """Vocab 1000 and T 8 is an event of 8000 > MAX_DENSE_EVENT: the fit builds no dense
    output Hessian, and the predictive is exact through the factor or sampled."""

    def refuse(self, f):
        raise AssertionError("closed-form output Hessian built for an 8000-entry event")

    monkeypatch.setattr(lk.Categorical, "output_hessian", refuse)

    vocab = 1000
    model = tiny_lora(vocab)
    data = [right_padded(sequences([8, 6, 7], vocab=vocab), 8)]
    lik = lk.Categorical(mask_fn=padding_mask)

    # The core default picks sampled methods only.
    post = lk.laplace(
        causal_lm(model),
        lk.data.from_loader(data, prepare=prepare_causal_lm),
        likelihood=lik,
        subset=lk.subsets.trainable,
    )
    approx = row(post, "hessian_approx")

    assert post.caps.event_size == 8 * vocab
    # One DiagGGN(fisher="mc") over all 8 blocks; no exact DiagGGN, DenseGGN or KFAC.
    assert approx == "PerBlockCurvature({'*': DiagGGN(fisher='mc')})"
    assert "DenseGGN" not in approx and "KFAC" not in approx

    # The preset needs no output Hessian either.
    laplace_lora(model, data)

    # The full sequence is exact through the low-rank factor (E * P = 8000 * 256 entries per
    # input, within MAX_DENSE_EVENT ** 2 entries), read by Probit without [E, E] blocks.
    full_pred = lk.Linearized().resolve(post)

    assert full_pred.strategy == "forward-mode" and full_pred.n_samples == "exact"
    assert isinstance(lk.predict_outputs(post, data[0]).cov, lk.core.FactorCov)
    full = lk.predict(post, data[0])
    torch.testing.assert_close(full.probs.sum(-1), torch.ones(3, 8))

    last = lk.predict(post, data[0], select=last_token)
    torch.testing.assert_close(last.probs.sum(-1), torch.ones(3))

    # Sampled outputs: a VJP of a VJP per draw (eager attention has vjp_twice), read by MC
    # over the draws.
    sampled = lk.Linearized(n_samples=8)

    assert sampled.resolve(post).strategy == "reverse-mode"
    assert isinstance(lk.predict_outputs(post, data[0], n_samples=8).cov, lk.core.SampleCov)
    drawn = lk.predict(post, data[0], sampled)
    torch.testing.assert_close(drawn.probs.sum(-1), torch.ones(3, 8))


# --- sequence classification ------------------------------------------------------------------

CLASSES = 3
PAD = VOCAB - 1


def tiny_classifier(
    dtype: torch.dtype = torch.float64, attn: str = "eager", classes: int = CLASSES
) -> nn.Module:
    """GPT-2 sequence classifier, 3 classes, hidden 16, 2 layers, 2 heads, eager attention
    by default; LoRA r=2 on c_attn, and the `score` head trained in full (`modules_to_save`).
    """

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB,
        n_embd=16,
        n_layer=2,
        n_head=2,
        n_positions=32,
        num_labels=classes,
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
        warnings.filterwarnings("ignore", "`is_gradient_checkpointing`", FutureWarning)
        model = get_peft_model(GPT2ForSequenceClassification(config), lora)

    return model.to(dtype).eval()


def labelled(lengths: list[int], seed: int) -> dict[str, Tensor]:
    """A right-padded tokenizer batch (pad id `PAD`, never a real token) with labels."""

    g = torch.Generator().manual_seed(seed)
    width = max(lengths)
    ids = torch.full((len(lengths), width), PAD)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)

    for i, n in enumerate(lengths):
        ids[i, :n] = torch.randint(0, PAD, (n,), generator=g)
        mask[i, :n] = 1

    labels = torch.randint(0, CLASSES, (len(lengths),), generator=g)

    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def classifier_loader() -> list[dict[str, Tensor]]:
    """Two batches, 16 sequences."""

    return [labelled([5, 3, 6, 4, 2, 6, 5, 3], seed=1), labelled([4, 6, 2, 5, 3, 3, 6, 4], 2)]


def test_sequence_classifier_returns_the_logits():
    model = tiny_classifier()
    raw = labelled([5, 3, 6], seed=3)
    inputs = {k: raw[k] for k in ("input_ids", "attention_mask")}
    fn = sequence_classification(model)

    f = fn(lora_params(model), inputs)

    assert fn.module is model
    assert isinstance(f, Tensor) and f.shape == (3, CLASSES)
    torch.testing.assert_close(f, model(**inputs).logits)


def test_prepare_sequence_classification_splits_off_the_labels():
    raw = labelled([4, 2], seed=3)
    batch = prepare_sequence_classification(raw, torch.Generator().manual_seed(0))

    assert isinstance(batch.inputs, dict)
    assert set(batch.inputs) == {"input_ids", "attention_mask"}
    assert batch.inputs["input_ids"] is raw["input_ids"]
    assert batch.targets is raw["labels"]
    assert batch.weights is None


def test_missing_labels_raises():
    raw = labelled([4, 2], seed=3)
    del raw["labels"]

    with pytest.raises(lk.StructureError, match="labels"):
        prepare_sequence_classification(raw, torch.Generator().manual_seed(0))

    with pytest.raises(lk.StructureError, match="labels"):
        laplace_lora(tiny_classifier(), [raw], task=SequenceClassification())


def test_sequence_classification_matches_by_hand_fit():
    model = tiny_classifier()
    call = lk.core.ModelFn.from_module(model)

    def logits(params, inputs):
        return call(params, inputs).logits

    def prepare(raw, generator):
        return lk.Batch({k: raw[k] for k in ("input_ids", "attention_mask")}, raw["labels"])

    by_hand = lk.laplace(
        lk.core.ModelFn(logits, model),
        lk.data.from_loader(classifier_loader(), prepare=prepare),
        likelihood=lk.Categorical(),
        subset=lk.subsets.trainable,
        hessian_approx=lk.KFAC(),
        generator=torch.Generator().manual_seed(7),
    )
    task = SequenceClassification()
    gen = torch.Generator().manual_seed(7)
    post = laplace_lora(model, classifier_loader(), task=task, generator=gen)

    # LoRA A and B on two layers, and the score head, all with KFAC (as in Laplace-LoRA).
    assert set(post.mean) == set(lora_params(model))
    assert any("score" in n for n in post.mean)
    assert row(post, "hessian_approx") == row(by_hand, "hessian_approx") == "KFAC()"
    assert int(post.n_data) == 16

    torch.testing.assert_close(post.prior.prior_precision, by_hand.prior.prior_precision)
    torch.testing.assert_close(post.precision().diagonal(), by_hand.precision().diagonal())
    torch.testing.assert_close(post.log_marglik(), by_hand.log_marglik())


def test_sequence_classification_with_sdpa_fits_kfac():
    """SDPA attention doesn't vmap; KFAC needs only the hooks."""

    task = SequenceClassification()
    post = laplace_lora(
        tiny_classifier(attn="sdpa"), classifier_loader(), task=task, hyper_tuner=None
    )

    assert not post.caps.vmap
    assert row(post, "hessian_approx") == "KFAC()"
    assert torch.isfinite(post.log_marglik())


class BagOfEmbeddings(nn.Module):
    """A sequence classifier whose trainable head is a bare parameter of the root module,
    used without calling a layer: no module hook reaches it."""

    def __init__(self) -> None:
        super().__init__()

        torch.manual_seed(0)
        self.emb = nn.Embedding(VOCAB, 8, dtype=torch.float64).requires_grad_(False)
        self.head = nn.Parameter(torch.randn(CLASSES, 8, dtype=torch.float64))

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> SequenceClassifierOutput:
        h = (self.emb(input_ids) * attention_mask[..., None]).sum(1)

        return SequenceClassifierOutput(logits=h @ self.head.T)


def test_sequence_classification_without_hooks_falls_back_to_the_sampled_fisher():
    model = BagOfEmbeddings()
    task = SequenceClassification()
    gen = torch.Generator().manual_seed(3)
    post = laplace_lora(model, classifier_loader(), task=task, generator=gen)

    def prepare(raw, generator):
        return lk.Batch({k: raw[k] for k in ("input_ids", "attention_mask")}, raw["labels"])

    by_hand = lk.laplace(
        lk.core.ModelFn(lambda p, x: lk.core.ModelFn.from_module(model)(p, x).logits, model),
        lk.data.from_loader(classifier_loader(), prepare=prepare),
        likelihood=lk.Categorical(),
        subset=lk.subsets.trainable,
        hessian_approx=lk.DiagGGN(fisher="mc"),
        generator=torch.Generator().manual_seed(3),
    )

    assert post.caps.hooked == frozenset()
    assert row(post, "hessian_approx") == "DiagGGN(fisher='mc')"
    torch.testing.assert_close(post.precision().diagonal(), by_hand.precision().diagonal())


def test_sequence_classification_passes_an_explicit_approx_through():
    task = SequenceClassification()
    post = laplace_lora(
        tiny_classifier(), classifier_loader(), task=task, hessian_approx=lk.DiagGGN()
    )

    assert row(post, "hessian_approx").startswith("DiagGGN(")


def test_sequence_classification_linearized_is_exact():
    """K = 3 outputs: the exact per-input covariance J Sigma J^T, no Monte Carlo."""

    model = tiny_classifier()
    post = laplace_lora(model, classifier_loader(), task=SequenceClassification(), hyper_tuner=None)
    raw = labelled([5, 2, 6, 4], seed=5)
    inputs = {k: raw[k] for k in ("input_ids", "attention_mask")}

    resolved = lk.Linearized().resolve(post)

    assert resolved.n_samples == "exact" and resolved.strategy == "reverse-mode"

    # The reference: J from jacrev on the trainable parameters, Sigma column by column.
    names = list(post.mean)
    sizes = [post.mean[n].numel() for n in names]

    def unflat(v: Tensor) -> dict[str, Tensor]:
        parts = torch.split(v, sizes)
        return {n: t.reshape(post.mean[n].shape) for n, t in zip(names, parts, strict=True)}

    def flat(p: dict[str, Tensor]) -> Tensor:
        return torch.cat([p[n].reshape(-1) for n in names])

    def f(params: dict[str, Tensor]) -> Tensor:
        return functional_call(model, params, args=(), kwargs=inputs).logits

    jac = jacrev(f)(dict(post.mean))
    j = torch.cat([jac[n].reshape(4, CLASSES, -1) for n in names], dim=-1)
    eye = torch.eye(sum(sizes), dtype=torch.float64)
    sigma = torch.stack([flat(post.cov_mv(unflat(e))) for e in eye])
    cov = j @ sigma @ j.transpose(1, 2)

    out = lk.predict_outputs(post, inputs)

    assert isinstance(out.cov, lk.core.PerInputCov)
    torch.testing.assert_close(out.cov.per_input(), cov)

    # The predictive reads that covariance through the probit link.
    ref = lk.Probit()(
        post.likelihood, lk.core.OutputGaussian(f(dict(post.mean)), lk.core.PerInputCov(cov, (3,)))
    )
    torch.testing.assert_close(lk.predict(post, inputs).probs, ref.probs)

    # A tokenizer's output is a BatchEncoding, not a dict.
    torch.testing.assert_close(lk.predict(post, BatchEncoding(inputs)).probs, ref.probs)


def test_sequence_classification_refuses_a_causal_lm_before_the_fit():
    """Logits [N, T, V] are not [N, K]: refused after one forward, before the fit pass."""

    model = tiny_lora()
    raw = {**right_padded(sequences([5, 3]), 5), "labels": torch.tensor([0, 1])}
    calls = []
    handle = model.register_forward_hook(lambda *_: calls.append(1))

    try:
        with pytest.raises(lk.StructureError, match=r"\[N, K\].*\(2, 5, 64\)"):
            laplace_lora(model, [raw], task=SequenceClassification())
    finally:
        handle.remove()

    assert len(calls) == 1


def test_sequence_classification_refuses_one_label_before_the_fit():
    """Categorical over one class has no curvature: the posterior would be the prior."""

    raw = {**labelled([5, 3], seed=1), "labels": torch.tensor([0, 0])}

    with pytest.raises(lk.StructureError, match=r"K = 1; set num_labels"):
        laplace_lora(tiny_classifier(classes=1), [raw], task=SequenceClassification())


def test_laplace_lora_refuses_an_unknown_task():
    with pytest.raises(lk.StructureError, match=r"CausalLM\(\) or SequenceClassification\(\)"):
        laplace_lora(tiny_lora(), loader(), task="causal")  # ty: ignore[invalid-argument-type]


def test_size_sets_the_scale():
    causal = [right_padded(sequences([5] * 8, seed=s), 6) for s in (1, 2)]
    post = laplace_lora(tiny_lora(), causal, n_data=1000, hyper_tuner=None)

    assert float(post.n_data) == 1000

    task = SequenceClassification()
    post = laplace_lora(
        tiny_classifier(), classifier_loader(), task=task, n_data=1000, hyper_tuner=None
    )

    assert float(post.n_data) == 1000


def test_task_default_is_the_causal_lm():
    assert laplace_lora.__kwdefaults__["task"] == CausalLM()
    assert str(laplace_lora(tiny_lora(), loader(), task=CausalLM())) == str(
        laplace_lora(tiny_lora(), loader())
    )


def test_no_trainable_message_names_is_trainable():
    for model, task in [(tiny_lora(), CausalLM()), (tiny_classifier(), SequenceClassification())]:
        for p in model.parameters():
            p.requires_grad_(False)

        for approx in [None, lk.DiagGGN()]:
            with pytest.raises(lk.StructureError, match=r"is_trainable=True"):
                laplace_lora(model, [], task=task, hessian_approx=approx)


# --- answer tokens ----------------------------------------------------------------------------


class FakeTokenizer:
    """Encodes each listed word to its ids; records the calls."""

    def __init__(self, vocab: dict[str, list[int]]):
        self.vocab = vocab
        self.calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.calls.append((text, add_special_tokens))

        return self.vocab[text]


def test_answer_logits_gathers_last_real_token():
    tok = FakeTokenizer({" True": [7], " False": [3]})
    select = answer_logits(tok, [" True", " False"])

    assert tok.calls == [(" True", False), (" False", False)]

    inputs = right_padded(sequences([3, 5, 1]), 5)
    f = torch.randn(3, 5, VOCAB)

    picked = select(f, inputs)

    expected = torch.stack([f[0, 2, [7, 3]], f[1, 4, [7, 3]], f[2, 0, [7, 3]]])

    assert torch.equal(picked, expected)


def test_answer_logits_predicts_with_a_causal_lm():
    model = tiny_lora(dtype=torch.float64)
    post = laplace_lora(model, loader(), hyper_tuner=None)
    select = answer_logits(
        FakeTokenizer({"yes": [5], "no": [9], "maybe": [11]}), ["yes", "no", "maybe"]
    )
    prompts = right_padded(sequences([3, 5], seed=4), 5)

    probs = lk.predict(post, prompts, select=select).probs

    assert probs.shape == (2, 3)
    torch.testing.assert_close(probs.sum(-1), torch.ones(2, dtype=torch.float64))


def test_answer_logits_accepts_a_transformers_tokenizer():
    words = {"[UNK]": 0, "True": 1, "False": 2, "Maybe": 3}
    inner = Tokenizer(WordLevel(words, unk_token="[UNK]"))
    inner.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="[UNK]")

    select = answer_logits(tok, ["False", "Maybe"])
    f = torch.randn(1, 2, VOCAB)
    inputs = {"input_ids": torch.zeros(1, 2, dtype=torch.long), "attention_mask": torch.ones(1, 2)}

    assert torch.equal(select(f, inputs), f[0, 1, [2, 3]][None])

    with pytest.raises(lk.StructureError, match=r"'True False'.*\[1, 2\]"):
        answer_logits(tok, ["True False"])


def test_answer_logits_rejects_multi_token_answer():
    tok = FakeTokenizer({" True": [7], " Unsure": [12, 13]})

    with pytest.raises(lk.StructureError, match=r"' Unsure' encodes to 2 tokens \[12, 13\]"):
        answer_logits(tok, [" True", " Unsure"])


def test_answer_logits_rejects_repeated_answers():
    tok = FakeTokenizer({" True": [7], "True": [7], " False": [3]})

    with pytest.raises(lk.StructureError, match=r"' True' and ' True'.*token 7"):
        answer_logits(tok, [" True", " True"])

    with pytest.raises(lk.StructureError, match=r"' True' and 'True'.*token 7"):
        answer_logits(tok, [" True", " False", "True"])


def test_answer_logits_rejects_non_tokenizer():
    with pytest.raises(lk.StructureError, match="encode"):
        answer_logits(object(), [" True"])


def test_answer_logits_rejects_a_string_or_no_answers():
    tok = FakeTokenizer({"T": [1], "r": [2]})

    with pytest.raises(lk.StructureError, match="a sequence of answers"):
        answer_logits(tok, "Tr")

    with pytest.raises(lk.StructureError, match="at least one answer"):
        answer_logits(tok, [])


# --- health ----------------------------------------------------------------------------------


def _without(monkeypatch, missing):
    real = metadata.version

    def version(dist):
        if dist in missing:
            raise metadata.PackageNotFoundError(dist)

        return real(dist)

    monkeypatch.setattr(metadata, "version", version)


def test_health_all_installed_is_ok():
    checks = {c.name: c for c in laplacekit_transformers.health()}

    assert list(checks) == ["transformers", "peft"]
    assert all(c.status == "ok" for c in checks.values())
    assert metadata.version("transformers") in checks["transformers"].detail


def test_health_optional_dep_missing_is_warning(monkeypatch):
    _without(monkeypatch, {"peft"})
    checks = {c.name: c for c in laplacekit_transformers.health()}

    assert checks["transformers"].status == "ok"
    assert checks["peft"].status == "warning"
    assert "not installed" in checks["peft"].detail
    assert lk.HealthReport({"laplacekit_transformers": tuple(checks.values())}).ok


def test_health_transformers_missing_is_error(monkeypatch):
    _without(monkeypatch, {"transformers"})
    checks = {c.name: c for c in laplacekit_transformers.health()}

    assert checks["transformers"].status == "error"
    assert "not installed" in checks["transformers"].detail
