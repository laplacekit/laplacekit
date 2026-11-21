"""`lk.reward`: pairs, the pairwise reward model and Bradley–Terry."""

import math
import re
from importlib import metadata

import pytest
import torch
from torch import Tensor, nn
from torch.func import functional_call, jacrev, vmap

import laplacekit as lk

F64 = torch.float64
N = 12


def _net(seed: int = 0, outputs: int = 1) -> nn.Module:
    torch.manual_seed(seed)

    return nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, outputs)).to(F64)


def _pairs(seed: int, n: int = N) -> tuple[Tensor, Tensor, Tensor]:
    g = torch.Generator().manual_seed(seed)
    first = torch.randn(n, 3, generator=g, dtype=F64)
    second = torch.randn(n, 3, generator=g, dtype=F64)
    labels = torch.randint(0, 2, (n,), generator=g)

    return first, second, labels


def _fit(net: nn.Module, approx: lk.core.HessianApprox | None = None) -> lk.LaplacePosterior:
    data = lk.data.from_loader([_pairs(1)], prepare=lk.reward.prepare_pairs)

    return lk.laplace(
        lk.reward.pairwise(net),
        data,
        likelihood=lk.reward.bradley_terry(),
        hessian_approx=approx if approx is not None else lk.DiagGGN(),
        subset=lk.subsets.all,
        prior=lk.priors.Isotropic(torch.tensor(1.0, dtype=F64)),
        hyper_tuner=None,
    )


class Counting(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()

        self.inner = inner
        self.calls = 0

    def forward(self, x: Tensor) -> Tensor:
        self.calls += 1

        return self.inner(x)


class TokenNet(nn.Module):
    """A tokenized reward model: `input_ids` and `attention_mask` to one reward per text."""

    def __init__(self) -> None:
        super().__init__()

        torch.manual_seed(0)
        self.emb = nn.Embedding(10, 4, dtype=F64)
        self.head = nn.Linear(4, 1, dtype=F64)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        h = self.emb(input_ids) * attention_mask.unsqueeze(-1)

        return self.head(h.sum(1))


def _tokens(n: int, length: int, seed: int) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 10, (n, length), generator=g)

    return {"input_ids": ids, "attention_mask": torch.ones(n, length, dtype=F64)}


def test_pair_is_a_pytree_and_prepare_pairs_builds_a_batch():
    first, second, labels = _pairs(0)
    batch = lk.reward.prepare_pairs((first, second, labels.int()), torch.Generator().manual_seed(0))

    assert isinstance(batch.inputs, lk.reward.Pair)
    assert batch.inputs.first is first and batch.inputs.second is second
    assert batch.size == N
    assert batch.targets is not None and batch.targets.dtype == torch.long

    # vmap sees both sides' tensors, one example at a time.
    diff = vmap(lambda p: p.first - p.second)(batch.inputs)

    torch.testing.assert_close(diff, first - second)


def test_pairwise_runs_inner_once():
    net = Counting(_net())
    fn = lk.reward.pairwise(net)
    first, second, _ = _pairs(0)
    params = dict(net.named_parameters())

    with torch.no_grad():
        got = fn(params, lk.reward.pair(first, second))
        calls = net.calls
        want = torch.cat([net(first), net(second)], -1)

    assert calls == 1
    assert fn.module is net
    assert isinstance(got, Tensor) and got.shape == (N, 2)
    torch.testing.assert_close(got, want)


def test_pairwise_takes_a_model_fn():
    net = _net()
    first, second, _ = _pairs(0)
    fn = lk.reward.pairwise(lk.core.ModelFn.from_module(net))

    with torch.no_grad():
        got = fn(dict(net.named_parameters()), lk.reward.pair(first, second))

        torch.testing.assert_close(got, torch.cat([net(first), net(second)], -1))

    assert fn.module is net


def test_single_items_get_one_reward_each():
    net = _net()
    x = _pairs(0)[0]

    with torch.no_grad():
        got = lk.reward.pairwise(net)(dict(net.named_parameters()), x)

        torch.testing.assert_close(got, net(x))


def test_plain_dict_with_first_second_is_items():
    """Detection is by type: a dict keyed "first" and "second" is one item per row."""

    class Both(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            self.w = nn.Linear(3, 1, dtype=F64)

        def forward(self, first: Tensor, second: Tensor) -> Tensor:
            return self.w(first * second)

    net = Both()
    first, second, _ = _pairs(0)

    with torch.no_grad():
        got = lk.reward.pairwise(net)(
            dict(net.named_parameters()), {"first": first, "second": second}
        )

        torch.testing.assert_close(got, net(first=first, second=second))

    assert isinstance(got, Tensor) and got.shape == (N, 1)


def test_more_than_one_value_per_item_raises():
    net = _net(outputs=2)
    fn = lk.reward.pairwise(net)
    first, second, _ = _pairs(0)

    with pytest.raises(lk.StructureError, match="a reward model returns one value per item"):
        fn(dict(net.named_parameters()), lk.reward.pair(first, second))

    with pytest.raises(lk.StructureError, match="a reward model returns one value per item"):
        fn(dict(net.named_parameters()), first)


@pytest.mark.parametrize(
    ("second", "key"),
    [
        (_tokens(4, 7, 1), "input_ids"),
        ({"input_ids": _tokens(4, 5, 1)["input_ids"]}, "attention_mask"),
    ],
    ids=["lengths", "keys"],
)
def test_mismatched_sides_raise_at_probe(second, key):
    net = TokenNet()
    fn = lk.reward.pairwise(net)
    batch = lk.Batch(lk.reward.pair(_tokens(4, 5, 0), second), torch.tensor([0, 1, 1, 0]))
    msg = rf"'{key}'.*tokenize both sides in one padded call"

    with pytest.raises(lk.StructureError, match=msg):
        lk.core.probe(fn, lk.reward.bradley_terry(), batch)

    with pytest.raises(lk.StructureError, match=msg):
        lk.laplace(fn, [batch], likelihood=lk.reward.bradley_terry(), hessian_approx=lk.DiagGGN())


def test_mismatched_tensor_sides_raise():
    net = _net()
    first, second, _ = _pairs(0)

    with pytest.raises(lk.StructureError, match=r"\(12, 3\) and \(11, 3\)"):
        lk.reward.pairwise(net)(dict(net.named_parameters()), lk.reward.pair(first, second[:-1]))


def test_tokenized_pairs_concatenate_key_by_key():
    net = TokenNet()
    first, second = _tokens(4, 5, 0), _tokens(4, 5, 1)

    with torch.no_grad():
        got = lk.reward.pairwise(net)(dict(net.named_parameters()), lk.reward.pair(first, second))

        torch.testing.assert_close(got, torch.cat([net(**first), net(**second)], -1))


def test_bradley_terry_is_the_softmax_over_the_two_rewards():
    lik = lk.reward.bradley_terry()
    r = torch.tensor([[1.5, -0.5], [0.0, 2.0]], dtype=F64)
    y = torch.tensor([0, 0])

    assert lik == lk.Categorical()
    torch.testing.assert_close(lik.log_prob(r, y), torch.sigmoid(r[:, 0] - r[:, 1]).log())


def test_pair_probit_is_probit_of_reward_difference():
    """`lk.predict` on pairs is σ(μ_d / √(1 + π v_d / 8)) on d = r₁ − r₂, with the
    covariance built here from `jacrev` and the Bradley–Terry Hessian."""

    net = _net()
    post = _fit(net, lk.DenseGGN(joint=True))

    names = [k for k, _ in net.named_parameters()]
    shapes = [p.shape for _, p in net.named_parameters()]
    sizes = [p.numel() for _, p in net.named_parameters()]
    theta = torch.cat([p.detach().reshape(-1) for _, p in net.named_parameters()])

    def rewards(flat: Tensor, first: Tensor, second: Tensor) -> Tensor:
        chunks = flat.split(sizes)
        p = {k: c.reshape(s) for k, c, s in zip(names, chunks, shapes, strict=True)}

        return torch.cat(
            [functional_call(net, p, (first,)), functional_call(net, p, (second,))], -1
        )

    first, second, _ = _pairs(1)
    f = rewards(theta, first, second)
    jac = jacrev(rewards)(theta, first, second)  # [N, 2, P]
    s = torch.softmax(f, -1)
    hess = torch.diag_embed(s) - s.unsqueeze(-1) * s.unsqueeze(-2)

    precision = torch.einsum("nkp,nkl,nlq->pq", jac, hess, jac) + torch.eye(
        theta.numel(), dtype=F64
    )
    cov = torch.linalg.inv(precision)

    a, b, _ = _pairs(2, n=7)
    mu = rewards(theta, a, b)
    d = (jacrev(rewards)(theta, a, b) * torch.tensor([1.0, -1.0], dtype=F64)[:, None]).sum(1)
    v_d = torch.einsum("np,pq,nq->n", d, cov, d)
    z = (mu[:, 0] - mu[:, 1]) / torch.sqrt(1 + math.pi / 8 * v_d)
    want = torch.stack([torch.sigmoid(z), torch.sigmoid(-z)], -1)

    got = lk.predict(post, lk.reward.pair(a, b))

    assert isinstance(got, torch.distributions.Categorical)
    torch.testing.assert_close(got.probs, want, rtol=1e-6, atol=1e-10)


def test_single_item_predict_points_to_outputs():
    post = _fit(_net())
    x = _pairs(2)[0]
    msg = "a Categorical over one class gives it probability 1; for a scalar output such as a "
    msg += "reward, use lk.predict_outputs(posterior, inputs)"

    with pytest.raises(lk.CapabilityError, match=re.escape(msg)):
        lk.predict(post, x)

    out = lk.predict_outputs(post, x)

    assert out.mean.shape == (N, 1)
    assert bool((out.cov.per_input() > 0).all())


def test_entry_point_loads_the_plugin_record():
    (ep,) = [
        ep
        for ep in metadata.entry_points(group="laplacekit.plugins")
        if ep.name == "laplacekit_reward"
    ]

    assert ep.load() is lk.reward.plugin
    assert ep.load() == lk.Plugin(supports=">=0.1a0,<0.2")


def test_plain_tuple_is_not_a_pair():
    """`Pair` is a named tuple, found by its class: a plain tuple goes to the inner model."""

    seen: list[object] = []

    def inner(params: lk.core.Params, inputs: object) -> Tensor:
        seen.append(inputs)

        return torch.zeros(N, 1, dtype=F64)

    first, second, _ = _pairs(0)
    fn = lk.reward.pairwise(lk.core.ModelFn(inner))

    got = fn({}, (first, second))

    assert isinstance(got, Tensor) and got.shape == (N, 1)
    assert type(seen[0]) is tuple


def test_prepare_pairs_names_what_is_wrong():
    first, second, labels = _pairs(0)
    gen = torch.Generator().manual_seed(0)

    with pytest.raises(lk.StructureError, match="got 2 entries"):
        lk.reward.prepare_pairs((first, second), gen)

    with pytest.raises(lk.StructureError, match="needs a label tensor, got list"):
        lk.reward.prepare_pairs((first, second, labels.tolist()), gen)


def test_empty_batch_is_not_an_index_error():
    net = _net()
    empty = torch.zeros(0, 3, dtype=F64)

    with torch.no_grad():
        got = lk.reward.pairwise(net)(dict(net.named_parameters()), lk.reward.pair(empty, empty))

    assert isinstance(got, Tensor) and got.shape == (0, 2)
