"""Tied input and output embeddings, the default in GPT-2 and Mamba configs.

A tied tensor is one parameter: whichever name the subset uses, every use of the tensor
is differentiated, so the posterior is the Laplace approximation on the shared tensor.
"""

import pytest
import torch
from torch import Tensor, nn
from torch.func import jacrev, vmap

import laplacekit as lk

VOCAB, D, T = 10, 6, 5


class TiedLM(nn.Module):
    def __init__(self, vocab: int = VOCAB, d: int = D) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.body = nn.Linear(d, d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight

    def forward(self, ids: Tensor) -> Tensor:
        return self.head(torch.tanh(self.body(self.emb(ids))))  # [N, T, V]


def _setup(vocab: int = VOCAB) -> tuple[TiedLM, Tensor, list[tuple[Tensor, Tensor]]]:
    torch.manual_seed(0)
    model = TiedLM(vocab).double()
    ids = torch.randint(0, vocab, (8, T))
    data = [(ids[:4], ids[:4]), (ids[4:], ids[4:])]

    return model, ids, data


def _explicit_ggn(model: TiedLM, ids: Tensor) -> Tensor:
    """(1/N) sum_i J_i^T H_i J_i, with the tie written out: W is both the embedding
    table and the head, and both uses are differentiated."""

    body_w, body_b = model.body.weight.detach(), model.body.bias.detach()

    def logits(w: Tensor, seq: Tensor) -> Tensor:
        h = torch.tanh(w[seq] @ body_w.T + body_b)
        return (h @ w.T).reshape(-1)  # [T * V]

    w = model.emb.weight.detach()
    jac = vmap(jacrev(logits), in_dims=(None, 0))(w, ids).reshape(ids.shape[0], T * VOCAB, -1)

    p = torch.softmax(vmap(logits, in_dims=(None, 0))(w, ids).reshape(-1, T, VOCAB), dim=-1)
    per_position = torch.diag_embed(p) - p.unsqueeze(-1) * p.unsqueeze(-2)  # [N, T, V, V]
    h = torch.stack([torch.block_diag(*blocks) for blocks in per_position])  # [N, TV, TV]

    return torch.einsum("nep,nef,nfq->pq", jac, h, jac) / ids.shape[0]


def _fold(model: TiedLM, data, name: str, approx) -> lk.core.BlockDiag:
    fn = lk.core.ModelFn.from_module(model)
    params = {name: dict(model.named_parameters(remove_duplicate=False))[name].detach()}
    fit = lk.core.fit_curvature(fn, lk.Categorical(), params, approx)
    stat = fit.init()

    for pair in data:
        stat = fit.update(stat, lk.Batch.from_pair(pair))

    return fit.finalize(stat)


@pytest.mark.parametrize("name", ["head.weight", "emb.weight"], ids=["head", "emb"])
def test_dense_ggn_and_diag_differentiate_every_use_of_a_tied_tensor(name):
    model, ids, data = _setup()
    want = _explicit_ggn(model, ids)
    block = name.split(".")[0]

    dense = _fold(model, data, name, lk.DenseGGN()).blocks[block].mat
    diag = _fold(model, data, name, lk.DiagGGN()).blocks[block].diag

    torch.testing.assert_close(dense, want)
    torch.testing.assert_close(diag, want.diagonal())


def test_kfac_refuses_a_tied_block_naming_the_aliases():
    model, _, data = _setup()

    with pytest.raises(
        lk.CapabilityError, match=r"emb\.weight.*head\.weight|head\.weight.*emb\.weight"
    ):
        lk.laplace(
            model, data, likelihood=lk.Categorical(), hessian_approx=lk.KFAC(), hyper_tuner=None
        )

    caps = lk.core.probe(
        lk.core.ModelFn.from_module(model), lk.Categorical(), lk.Batch.from_pair(data[0])
    )

    assert not lk.KFAC.supports(lk.Categorical(), caps, ("head.weight",))
    assert lk.KFAC.supports(lk.Categorical(), caps, ("body.weight", "body.bias"))


def test_default_routes_a_tied_head_past_kfac():
    # 700 x 6 is above DENSE_MAX_PARAMS, so an untied head would get KFAC.
    model, _, data = _setup(vocab=700)
    fn = lk.core.ModelFn.from_module(model)
    caps = lk.core.probe(fn, lk.Categorical(), lk.Batch.from_pair(data[0]))
    params = lk.subsets.last_layer(fn, caps)
    chosen = lk.curvature.default(params, caps, likelihood=lk.Categorical())

    assert "head" in caps.hooked
    assert chosen.by_block(["head"]) == {"head": lk.DiagGGN()}


@pytest.mark.parametrize("approx", [None, lk.DiagGGN(fisher="mc")], ids=["default", "sfd"])
def test_tied_last_layer_through_the_front_door(approx):
    model, ids, data = _setup()
    post = lk.laplace(
        model, data, likelihood=lk.Categorical(), hessian_approx=approx, hyper_tuner=None
    )

    assert set(post.mean) == {"head.weight"}

    d = lk.predict(post, ids[:2])

    assert d.probs.shape == (2, T, VOCAB) and torch.isfinite(d.probs).all()
    assert model.head.weight is model.emb.weight


def test_probed_names_are_literal():
    """Probed under one name of the tied tensor, fitted under the other: refused."""

    model, ids, _ = _setup()
    fn = lk.core.ModelFn.from_module(model)
    named = dict(model.named_parameters(remove_duplicate=False))
    caps = lk.core.probe(
        fn, lk.Categorical(), lk.Batch(ids[:4], ids[:4]), params={"emb.weight": named["emb.weight"]}
    )

    with pytest.raises(lk.CapabilityError, match="Not probed: head.weight"):
        lk.core.fit_curvature(
            fn,
            lk.Categorical(),
            {"head.weight": named["head.weight"]},
            lk.DiagGGN(),
            capabilities=caps,
        )
