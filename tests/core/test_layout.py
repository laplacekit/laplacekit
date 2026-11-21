import torch
from torch import nn

from laplacekit.core import ROOT_BLOCK, Layout


def _params() -> dict[str, torch.Tensor]:
    return {
        "fc.weight": torch.randn(3, 2),
        "fc.bias": torch.randn(3),
        "out.weight": torch.randn(1, 3),
    }


def test_by_module_blocks() -> None:
    lay = Layout.by_module(_params())

    assert dict(lay.blocks) == {"fc": ("fc.weight", "fc.bias"), "out": ("out.weight",)}
    assert lay.shapes["fc.weight"] == (3, 2)


def test_flatten() -> None:
    p = _params()
    lay = Layout.by_module(p)
    v = lay.flatten(p, "fc")

    assert v.shape == (9,)
    assert torch.equal(v, torch.cat([p["fc.weight"].flatten(), p["fc.bias"]]))
    assert lay.size("fc") == 9 and lay.size("out") == 3


def test_roundtrip() -> None:
    p = _params()
    lay = Layout.by_module(p)
    back = lay.unflatten(lay.flatten(p, "fc"), "fc")

    assert set(back) == {"fc.weight", "fc.bias"}

    for k, v in back.items():
        assert torch.equal(v, p[k])


def test_flatten_all() -> None:
    p = _params()
    lay = Layout.by_module(p)
    flat = lay.flatten_all(p)

    assert set(flat) == {"fc", "out"}
    assert torch.equal(flat["out"], p["out.weight"].flatten())


def test_no_dot_names_form_the_root_block() -> None:
    lay = Layout.by_module({"w": torch.zeros(2), "fc.weight": torch.zeros(1), "b": torch.zeros(3)})

    assert dict(lay.blocks) == {ROOT_BLOCK: ("w", "b"), "fc": ("fc.weight",)}


def test_root_linear_is_one_block() -> None:
    lay = Layout.by_module(dict(nn.Linear(3, 2).named_parameters()))

    assert ROOT_BLOCK == "<root>"
    assert dict(lay.blocks) == {"<root>": ("weight", "bias")}


def test_restrict_keeps_only_the_given_blocks() -> None:
    params = {
        "a.weight": torch.zeros(2, 3),
        "a.bias": torch.zeros(2),
        "b.weight": torch.zeros(4),
    }
    sub = Layout.by_module(params).restrict(["b"])

    assert sub == Layout.by_module({"b.weight": params["b.weight"]})


def test_n_params_counts_every_entry() -> None:
    params = {"fc.weight": torch.zeros(3, 2), "fc.bias": torch.zeros(3), "w": torch.zeros(4, 5)}
    layout = Layout.by_module(params)

    assert layout.n_params == 6 + 3 + 20
    assert layout.restrict(["fc"]).n_params == 9


def test_flatten_and_unflatten_with_a_leading_axis() -> None:
    """`lead=1` flattens n parameter sets at once, row i the flat vector of set i;
    `unflatten` keeps the leading axes of its vectors."""

    sets = [_params() for _ in range(4)]
    batched = {k: torch.stack([s[k] for s in sets]) for k in sets[0]}
    lay = Layout.by_module(sets[0])

    vecs = lay.flatten(batched, "fc", lead=1)
    want = torch.stack([torch.cat([s["fc.weight"].reshape(-1), s["fc.bias"]]) for s in sets])

    assert torch.equal(vecs, want)
    assert torch.equal(
        lay.flatten_all(batched, lead=1)["out"], batched["out.weight"].reshape(4, -1)
    )

    back = lay.unflatten(vecs, "fc")

    assert back["fc.weight"].shape == (4, 3, 2)

    for k, v in back.items():
        assert torch.equal(v, batched[k])
