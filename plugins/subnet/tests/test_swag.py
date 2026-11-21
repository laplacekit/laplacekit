"""`lk.subnet.swag_variance`, `largest_variance` over a mapping and `rescaled_prior`.

The parity with laplace-torch is in `tests/parity/test_parity_subnet.py`.
"""

import copy

import pytest
import torch
from torch import nn

import laplacekit as lk
from laplacekit.core import Capabilities, ModelFn, StructureError

DT = torch.float64


def _data(n: int = 32) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(3)
    x = torch.randn(n, 3, generator=g, dtype=DT)

    return [(x[:16], torch.randint(0, 2, (16,), generator=g)), (x[16:], torch.zeros(16).long())]


def _bn_model() -> nn.Sequential:
    g = torch.Generator().manual_seed(1)
    model = nn.Sequential(nn.Linear(3, 5), nn.BatchNorm1d(5), nn.ReLU(), nn.Linear(5, 2)).to(DT)

    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=DT))

    return model


def test_matches_a_hand_written_swag_loop():
    """Every other epoch snapshotted (every=2), against SGD and running moments by hand."""

    model = _bn_model()
    data = _data()
    var = lk.subnet.swag_variance(
        model, data, likelihood=lk.Categorical(), n_snapshots=3, every=2, lr=0.05
    )

    net = copy.deepcopy(model).train()
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=3e-4)
    snaps = []

    for epoch in range(6):
        for x, y in data:
            opt.zero_grad()
            nn.functional.cross_entropy(net(x), y).backward()
            opt.step()

        if epoch % 2 == 0:
            snaps.append(torch.nn.utils.parameters_to_vector(net.parameters()).detach())

    stack = torch.stack(snaps)
    want = (stack.square().mean(0) - stack.mean(0).square()).clamp(min=1e-30)
    got = torch.cat([v.reshape(-1) for v in var.values()])

    assert list(var) == [name for name, _ in model.named_parameters()]
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-14)


def test_model_untouched():
    model = _bn_model().eval()
    params = {k: (p.detach().clone(), p._version) for k, p in model.named_parameters()}
    buffers = {k: b.clone() for k, b in model.named_buffers()}

    var = lk.subnet.swag_variance(model, _data(), likelihood=lk.Categorical(), n_snapshots=3)

    assert not model.training
    assert all(not m.training for m in model.modules())

    for k, p in model.named_parameters():
        torch.testing.assert_close(p, params[k][0], rtol=0, atol=0)
        assert p._version == params[k][1]
        assert var[k].data_ptr() != p.data_ptr()

    for k, b in model.named_buffers():
        torch.testing.assert_close(b, buffers[k], rtol=0, atol=0)

    # The copy did train: its weights moved between snapshots.
    assert float(var["0.weight"].max()) > 1e-12


def test_refusals():
    model = _bn_model()
    data = _data()
    lik = lk.Categorical()
    loss = nn.functional.cross_entropy

    with pytest.raises(ValueError, match="exactly one of likelihood= and loss="):
        lk.subnet.swag_variance(model, data, likelihood=lik, loss=loss)

    with pytest.raises(ValueError, match="exactly one of likelihood= and loss="):
        lk.subnet.swag_variance(model, data)

    with pytest.raises(ValueError, match="snapshots >= 1"):
        lk.subnet.swag_variance(model, data, likelihood=lik, n_snapshots=0)

    with pytest.raises(ValueError, match="every >= 1"):
        lk.subnet.swag_variance(model, data, loss=loss, every=0)

    fn = ModelFn.from_module(model)
    shapes = {k: torch.ones(p.shape, dtype=DT) for k, p in model.named_parameters()}

    unknown = {**shapes, "9.weight": torch.ones(2, dtype=DT)}

    with pytest.raises(StructureError, match=r"'9\.weight'.* not parameters of the module"):
        lk.subnet.largest_variance(unknown, 3)(fn, Capabilities())

    wrong = {**shapes, "0.bias": torch.ones(4, dtype=DT)}

    with pytest.raises(StructureError, match=r"'0\.bias' has shape \(4,\)"):
        lk.subnet.largest_variance(wrong, 3)(fn, Capabilities())


def test_largest_variance_takes_a_mapping():
    model = _bn_model()
    g = torch.Generator().manual_seed(5)
    var = {k: torch.rand(p.shape, generator=g, dtype=DT) for k, p in model.named_parameters()}
    flat = torch.cat([v.reshape(-1) for v in var.values()])
    want = sorted(flat.argsort(descending=True)[:7].tolist())

    sel = lk.subnet.largest_variance(var, 7)(ModelFn.from_module(model), Capabilities())
    sizes = [v.numel() for v in var.values()]
    offsets = dict(zip(var, [sum(sizes[:i]) for i in range(len(sizes))], strict=True))
    got = sorted(int(offsets[k]) + int(i) for k, idx in sel.index.items() for i in idx)

    assert got == want


def test_rescaled_prior():
    model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1)).to(DT)  # 8 + 3 = 11 entries
    model[1].weight.requires_grad_(False)  # frozen, still counted

    prior = lk.subnet.rescaled_prior(2.0, 4, model)

    assert isinstance(prior, lk.priors.Isotropic)
    assert prior.prior_precision.dtype == DT
    torch.testing.assert_close(prior.prior_precision, torch.tensor(2.0 * 4 / 11, dtype=DT))

    tensor = lk.subnet.rescaled_prior(torch.tensor(0.5, dtype=DT), 11, model)
    torch.testing.assert_close(tensor.prior_precision, torch.tensor(0.5, dtype=DT))
