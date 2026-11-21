"""`lk.subnet`: random, largest-magnitude and largest-variance subnetworks.

The index parity with laplace-torch is in `tests/parity/test_parity.py`.
"""

import copy

import pytest
import torch
from torch import nn

import laplacekit as lk
from laplacekit.core import Capabilities, ModelFn, Selection, StructureError

DT = torch.float64


def _regression() -> nn.Sequential:
    """151 parameters: 50 + 50 in the first layer, 50 + 1 in the second."""

    model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1)).to(DT)
    g = torch.Generator().manual_seed(0)  # leaves the global RNG alone

    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=DT))

    return model


def _flat(model: nn.Module, sel: Selection) -> list[int]:
    """The selected entries as indices into `parameters_to_vector(model.parameters())`."""

    offsets, start = {}, 0

    for name, p in model.named_parameters():
        offsets[name] = start
        start += p.numel()

    return sorted(int(offsets[name] + i) for name, idx in sel.index.items() for i in idx)


def _select(subset, model: nn.Module) -> Selection:
    out = subset(ModelFn.from_module(model), Capabilities())
    assert isinstance(out, Selection)

    return out


def _diag(model: nn.Module, params) -> lk.LaplacePosterior:
    return lk.laplace(
        model,
        [(torch.zeros(4, 1, dtype=DT), torch.zeros(4, 1, dtype=DT))],
        likelihood=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.3, dtype=DT))),
        subset=params,
        hessian_approx=lk.DiagGGN(),
        hyper_tuner=None,
    )


def test_magnitude_picks_the_largest_entries():
    model = _regression()
    flat = torch.nn.utils.parameters_to_vector(model.parameters()).detach().abs()
    want = sorted(flat.argsort(descending=True)[:30].tolist())

    assert _flat(model, _select(lk.subnet.largest_magnitude(30), model)) == want


def test_magnitude_ties_go_to_the_lower_index():
    model = _regression()

    with torch.no_grad():
        model[0].bias.zero_()  # entries 50..99 of the flat vector tie at 0
        model[2].weight.zero_()  # entries 100..149 tie at 0 too

    nonzero = 50 + 1  # the first weight and the last bias
    picked = _flat(model, _select(lk.subnet.largest_magnitude(nonzero + 3), model))

    assert picked == [*range(50), 50, 51, 52, 150]


def test_magnitude_and_largest_variance_pass_check_subset():
    model = _regression()
    post = _diag(model, lk.subsets.all)

    lk.testing.check_subset(lk.subnet.largest_magnitude(30), model, Capabilities())
    lk.testing.check_subset(lk.subnet.largest_variance(post, 30), model, Capabilities())


def test_largest_variance_picks_the_largest_marginal_variances():
    model = _regression()
    post = _diag(model, lk.subsets.all)
    var = torch.cat([post.variance[n].reshape(-1) for n in post.mean])
    want = sorted(var.argsort(descending=True, stable=True)[:30].tolist())  # ties: lower index

    assert _flat(model, _select(lk.subnet.largest_variance(post, 30), model)) == want


def test_random_is_seeded_and_sized():
    model = _regression()
    p = sum(q.numel() for q in model.parameters())

    def pick(generator=None):
        return _flat(model, _select(lk.subnet.random(30, generator=generator), model))

    one = pick(torch.Generator().manual_seed(0))
    two = pick(torch.Generator().manual_seed(0))
    other = pick(torch.Generator().manual_seed(1))

    assert one == two
    assert other != one

    # No generator: torch's global RNG, drawn when `random` is called.
    torch.manual_seed(2)
    subset = lk.subnet.random(30)
    torch.manual_seed(2)

    assert _flat(model, _select(subset, model)) == pick()
    assert _flat(model, _select(subset, model)) == _flat(model, _select(subset, model))
    assert len(one) == len(set(one)) == 30
    assert 0 <= one[0] and one[-1] < p

    lk.testing.check_subset(lk.subnet.random(30), model, Capabilities())


def test_random_and_magnitude_stay_within_of():
    model = _regression()

    for subset in (
        lk.subnet.random(5, subset=lk.subsets.named("2.*")),
        lk.subnet.largest_magnitude(5, subset=lk.subsets.named("2.*")),
    ):
        sel = _select(subset, model)

        assert set(sel.index) <= {"2.weight", "2.bias"}
        assert sum(i.numel() for i in sel.index.values()) == 5
        assert sel.base["2.weight"].data_ptr() == model[2].weight.data_ptr()


@pytest.mark.parametrize("n", [0, 152])
@pytest.mark.parametrize("name", ["random", "largest_magnitude"])
def test_n_out_of_range_raises(name, n):
    make = getattr(lk.subnet, name)

    with pytest.raises(ValueError, match=rf"subnet\.{name}: n must be in \[1, 151\]"):
        _select(make(n), _regression())


def test_largest_variance_n_out_of_range_raises():
    model = _regression()
    post = _diag(model, lk.subsets.all)

    with pytest.raises(ValueError, match=r"subnet\.largest_variance: n must be in \[1, 151\]"):
        _select(lk.subnet.largest_variance(post, 152), model)


def test_of_returning_selection_raises():
    def picked(fn: ModelFn, caps: Capabilities, /) -> Selection:
        return _select(lk.subnet.largest_magnitude(3), _regression())

    for subset in (
        lk.subnet.random(2, subset=picked),
        lk.subnet.largest_magnitude(2, subset=picked),
    ):
        with pytest.raises(StructureError, match="Selection"):
            subset(ModelFn.from_module(_regression()), Capabilities())


def test_largest_variance_over_a_selection_raises():
    model = _regression()
    post = _diag(model, _select(lk.subnet.largest_magnitude(5), model))

    with pytest.raises(StructureError, match=r"subnet\.largest_variance: .*Selection"):
        _select(lk.subnet.largest_variance(post, 3), model)


def test_needs_a_module():
    model = ModelFn(lambda p, x: x, None)

    for subset in (lk.subnet.random(2), lk.subnet.largest_magnitude(2)):
        with pytest.raises(StructureError, match="needs a module"):
            subset(model, Capabilities())
