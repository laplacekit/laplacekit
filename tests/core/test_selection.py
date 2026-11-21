"""Index-set subnetworks: `Selection` against the full model, `functional_call` and `torch.func`."""

import pytest
import torch
from torch import Tensor, nn
from torch.func import functional_call, grad, hessian, jacrev, jvp, vmap

from laplacekit.core import (
    ModelFn,
    Selection,
    StructureError,
    select,
)
from laplacekit.core.subsets import _SELECTED_SUFFIX

F64 = torch.float64


def _net() -> nn.Sequential:
    torch.manual_seed(0)

    return nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).to(F64)


def _x() -> Tensor:
    return torch.randn(5, 3, generator=torch.Generator().manual_seed(1), dtype=F64)


def _masks(net: nn.Module) -> dict[str, Tensor]:
    """Given out of the module's order, to check the selection follows `params`."""

    g = torch.Generator().manual_seed(2)
    w = torch.rand(net.get_parameter("0.weight").shape, generator=g) < 0.4
    w[0, 0] = True

    return {"2.bias": torch.tensor([False, True]), "0.weight": w}


def _selection(net: nn.Module) -> Selection:
    return Selection.from_masks(dict(net.named_parameters()), _masks(net))


def test_from_masks_round_trip() -> None:
    net = _net()
    masks = _masks(net)
    params = dict(net.named_parameters())
    sel = Selection.from_masks(params, {**masks, "2.weight": torch.zeros(2, 4, dtype=torch.bool)})

    assert list(sel.base) == ["0.weight", "2.bias"]
    assert list(sel.index) == ["0.weight", "2.bias"]
    assert all(i.dtype == torch.int64 for i in sel.index.values())

    for name, p in sel.base.items():
        assert p.data_ptr() == params[name].data_ptr() and not p.requires_grad

    back = sel.masks()

    assert set(back) == set(masks)
    assert all(torch.equal(back[n], masks[n]) for n in masks)

    expected = {f"{n}{_SELECTED_SUFFIX}": params[n].detach()[masks[n]] for n in sel.base}

    assert list(sel.params) == ["0.weight[idx]", "2.bias[idx]"]
    assert all(torch.equal(sel.params[n], expected[n]) for n in expected)
    assert list(sel.selected_index) == list(sel.params)
    assert torch.equal(sel.selected_index["2.bias[idx]"], torch.tensor([1]))


def test_params_are_copies() -> None:
    net = _net()
    sel = _selection(net)
    before = net[0].weight.detach().clone()

    sel.params["0.weight[idx]"].add_(1.0)

    assert torch.equal(net[0].weight.detach(), before)


def test_apply_matches_full_model() -> None:
    net = _net()
    x = _x()
    fn = ModelFn.from_module(net)
    sel = _selection(net)
    sub = sel.apply(fn)

    assert sub.module is net
    torch.testing.assert_close(sub(sel.params, x), net(x).detach())

    # Move the second selected entry of 0.weight; only that entry of the full weight moves.
    values = {n: v.clone() for n, v in sel.params.items()}
    values["0.weight[idx]"][1] += 0.5

    rows, cols = _masks(net)["0.weight"].nonzero(as_tuple=True)
    edited = net[0].weight.detach().clone()
    edited[rows[1], cols[1]] += 0.5
    expected = functional_call(net, {"0.weight": edited}, (x,))

    torch.testing.assert_close(sub(values, x), expected.detach())


def test_select_passes_params_through_and_applies_a_selection() -> None:
    net = _net()
    x = _x()
    fn = ModelFn.from_module(net)
    plain = dict(net.named_parameters())

    same_fn, same_params = select(fn, plain)

    assert same_fn is fn and same_params is plain

    sel = _selection(net)
    sub_fn, sub_params = select(fn, sel)

    assert list(sub_params) == list(sel.params)
    torch.testing.assert_close(sub_fn(sub_params, x), net(x).detach())


def _full_loss(net: nn.Module, x: Tensor, name: str):
    def loss(full: Tensor) -> Tensor:
        return functional_call(net, {name: full}, (x,)).pow(2).sum()

    return loss


def test_apply_gradients_match_masked_full_gradient() -> None:
    net = _net()
    x = _x()
    masks = _masks(net)
    sel = _selection(net)
    sub = sel.apply(ModelFn.from_module(net))

    grads = grad(lambda v: sub(v, x).pow(2).sum())(sel.params)

    for name, mask in masks.items():
        full = net.get_parameter(name).detach()
        expected = jacrev(_full_loss(net, x, name))(full)[mask]

        torch.testing.assert_close(grads[f"{name}{_SELECTED_SUFFIX}"], expected)


def test_apply_under_vmap_and_jvp() -> None:
    net = _net()
    x = _x()
    sel = _selection(net)
    sub = sel.apply(ModelFn.from_module(net))
    name = "0.weight"
    values = sel.params[f"{name}{_SELECTED_SUFFIX}"]

    # vmap over a batch of selected values equals calling once per value.
    g = torch.Generator().manual_seed(3)
    batch = values + 0.1 * torch.randn(4, values.numel(), generator=g, dtype=F64)
    mapped = vmap(lambda v: sub({f"{name}{_SELECTED_SUFFIX}": v}, x))(batch)
    looped = torch.stack([sub({f"{name}{_SELECTED_SUFFIX}": v}, x) for v in batch])

    torch.testing.assert_close(mapped, looped)

    # jvp in the selected values equals the full model's jvp along the tangent placed at the
    # mask.
    tangent = torch.randn(values.numel(), generator=g, dtype=F64)
    _, out = jvp(lambda v: sub({f"{name}{_SELECTED_SUFFIX}": v}, x), (values,), (tangent,))

    full = net.get_parameter(name).detach()
    placed = torch.zeros_like(full)
    placed[_masks(net)[name]] = tangent
    _, expected = jvp(lambda w: functional_call(net, {name: w}, (x,)), (full,), (placed,))

    torch.testing.assert_close(out, expected)


def test_apply_hessian_matches_restricted_full_hessian() -> None:
    """Also checks the value checks in `apply` hold under `hessian`, and nothing is
    written in place."""

    net = _net()
    x = _x()
    name = "0.weight"
    mask = _masks(net)[name]
    sel = _selection(net)
    sub = sel.apply(ModelFn.from_module(net))
    versions = {n: t._version for n, t in sel.base.items()}
    before = net[0].weight.detach().clone()

    def loss(v: Tensor) -> Tensor:
        return sub({f"{name}{_SELECTED_SUFFIX}": v}, x).pow(2).sum()

    h = hessian(loss)(sel.params[f"{name}{_SELECTED_SUFFIX}"])

    full = net.get_parameter(name).detach()
    flat = mask.reshape(-1)
    ref = hessian(_full_loss(net, x, name))(full).reshape(full.numel(), full.numel())

    torch.testing.assert_close(h, ref[flat][:, flat])
    assert {n: t._version for n, t in sel.base.items()} == versions
    assert torch.equal(net[0].weight.detach(), before)


def test_selection_unknown_name_raises() -> None:
    net = _net()
    sel = _selection(net)
    sub = sel.apply(ModelFn.from_module(net))

    with pytest.raises(StructureError, match=r"'2\.weight\[idx\]'"):
        sub({"2.weight[idx]": torch.zeros(3, dtype=F64)}, _x())

    with pytest.raises(StructureError, match="'head.weight'"):
        Selection.from_masks(dict(net.named_parameters()), {"head.weight": torch.ones(2, 4) > 0})


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (
            torch.zeros(2, dtype=F64),
            r"'2\.bias\[idx\]' needs shape \(1,\) and dtype torch\.float64, "
            r"got \(2,\) and torch\.float64",
        ),
        (torch.zeros(0, dtype=F64), r"'2\.bias\[idx\]' needs shape \(1,\) .* got \(0,\)"),
        (
            torch.zeros(1, dtype=torch.float32),
            r"'2\.bias\[idx\]' needs .* dtype torch\.float64, got \(1,\) and torch\.float32",
        ),
    ],
    ids=["too long", "too short", "dtype"],
)
def test_apply_checks_the_selected_values(value: Tensor, match: str) -> None:
    net = _net()
    sel = _selection(net)
    sub = sel.apply(ModelFn.from_module(net))

    with pytest.raises(StructureError, match=match):
        sub({"2.bias[idx]": value}, _x())


def test_apply_checks_the_selected_values_under_vmap() -> None:
    """vmap sees the per-example shape: a batch of correct values passes, a batch of long
    ones is refused."""

    net = _net()
    x = _x()
    sel = _selection(net)
    sub = sel.apply(ModelFn.from_module(net))

    good = torch.zeros(3, 1, dtype=F64)
    mapped = vmap(lambda v: sub({"2.bias[idx]": v}, x))(good)
    looped = torch.stack([sub({"2.bias[idx]": v}, x) for v in good])

    torch.testing.assert_close(mapped, looped)

    with pytest.raises(StructureError, match=r"got \(2,\)"):
        vmap(lambda v: sub({"2.bias[idx]": v}, x))(torch.zeros(3, 2, dtype=F64))


def test_selections_compare_by_identity() -> None:
    sel = _selection(_net())

    assert sel == sel
    assert sel != _selection(_net())
    assert isinstance(hash(sel), int)


def test_apply_refuses_a_selected_name_given_twice() -> None:
    net = _net()
    sel = _selection(net)
    sub = sel.apply(ModelFn.from_module(net))
    both = {**sel.params, "0.weight": net[0].weight.detach()}

    with pytest.raises(StructureError, match=r"both '0\.weight' and '0\.weight\[idx\]'"):
        sub(both, _x())


@pytest.mark.parametrize(
    ("masks", "match"),
    [
        ({"0.weight": torch.ones(3, 4, dtype=torch.bool)}, r"shape \(3, 4\), .* \(4, 3\)"),
        ({"0.weight": torch.ones(4, 3)}, "bool"),
        ({"0.weight": torch.zeros(4, 3, dtype=torch.bool)}, "select no entry"),
    ],
    ids=["shape", "dtype", "empty"],
)
def test_from_masks_bad_mask_raises(masks: dict[str, Tensor], match: str) -> None:
    with pytest.raises(StructureError, match=match):
        Selection.from_masks(dict(_net().named_parameters()), masks)


@pytest.mark.parametrize(
    ("index", "match"),
    [
        ({"2.bias": torch.tensor([1])}, "same names"),
        ({"0.weight": torch.tensor([1.0])}, "int64"),
        ({"0.weight": torch.tensor([], dtype=torch.int64)}, "non-empty"),
        ({"0.weight": torch.tensor([2, 1])}, "sorted"),
        ({"0.weight": torch.tensor([1, 1])}, "sorted"),
        ({"0.weight": torch.tensor([12])}, "12 entries"),
    ],
    ids=["names", "dtype", "empty", "unsorted", "repeated", "range"],
)
def test_selection_checks_its_index(index: dict[str, Tensor], match: str) -> None:
    base = {"0.weight": _net()[0].weight.detach()}

    with pytest.raises(StructureError, match=match):
        Selection(base, index)


def test_empty_selection_raises() -> None:
    with pytest.raises(StructureError, match="no entries selected"):
        Selection({}, {})
