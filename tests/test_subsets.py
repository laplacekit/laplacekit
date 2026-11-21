import json
from pathlib import Path

import pytest
import torch
from torch import nn

from laplacekit import subsets, testing
from laplacekit.core import Capabilities, CapabilityError, ModelFn, Selection, StructureError


def _mlp() -> nn.Sequential:
    return nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 1))


def test_last_layer_uses_module_order():
    net = _mlp()
    caps = Capabilities(module_order=("0", "2"))
    out = subsets.last_layer(ModelFn.from_module(net), caps)

    assert set(out) == {"2.weight", "2.bias"}
    assert out["2.weight"] is net[2].weight


def test_last_layer_maps_root_block_to_the_root_module():
    net = nn.Linear(2, 3)
    out = subsets.last_layer(ModelFn.from_module(net), Capabilities(module_order=("<root>",)))

    assert set(out) == {"weight", "bias"}
    assert out["weight"] is net.weight


def test_last_layer_without_module_order_raises():
    with pytest.raises(CapabilityError, match="no parameterized module"):
        subsets.last_layer(ModelFn.from_module(_mlp()), Capabilities())


def test_trainable_skips_frozen():
    net = _mlp()
    net[0].weight.requires_grad_(False)
    out = subsets.trainable(ModelFn.from_module(net), Capabilities())

    assert set(out) == {"0.bias", "2.weight", "2.bias"}


def test_all_lists_tied_tensor_once():
    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(5, 3)
            self.head = nn.Linear(3, 5, bias=False)
            self.head.weight = self.emb.weight

    net = Tied()
    out = subsets.all(ModelFn.from_module(net), Capabilities())

    assert len(out) == 1
    assert next(iter(out.values())) is net.emb.weight


def test_named_globs_and_identity():
    net = _mlp()
    out = subsets.named("0.*")(ModelFn.from_module(net), Capabilities())

    assert set(out) == {"0.weight", "0.bias"}
    assert out["0.bias"] is net[0].bias


def test_named_no_match_raises():
    with pytest.raises(StructureError, match="no parameter matches"):
        subsets.named("9.*")(ModelFn.from_module(_mlp()), Capabilities())


@pytest.mark.parametrize(
    "subset", [subsets.all, subsets.trainable, subsets.last_layer, subsets.named("*")]
)
def test_needs_module(subset):
    model = ModelFn(lambda p, x: x, None)

    with pytest.raises(StructureError, match="needs a module"):
        subset(model, Capabilities(module_order=("0",)))


def test_last_layer_takes_only_direct_parameters():
    class Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_proj_weight = nn.Parameter(torch.ones(2, 2))
            self.out_proj = nn.Linear(2, 2)

    net = nn.Sequential(Attn())
    out = subsets.last_layer(ModelFn.from_module(net), Capabilities(module_order=("0",)))

    assert set(out) == {"0.in_proj_weight"}


def test_last_layer_keeps_tied_head():
    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(5, 3)
            self.head = nn.Linear(3, 5, bias=False)
            self.head.weight = self.embed.weight

    net = Tied()
    caps = Capabilities(module_order=("embed", "head"))
    out = subsets.last_layer(ModelFn.from_module(net), caps)

    assert set(out) == {"head.weight"}
    assert out["head.weight"] is net.embed.weight


def test_last_layer_does_not_match_name_prefix():
    net = nn.Sequential(*[nn.Linear(2, 2) for _ in range(11)])
    out = subsets.last_layer(ModelFn.from_module(net), Capabilities(module_order=("10", "1")))

    assert set(out) == {"1.weight", "1.bias"}


REF = json.loads((Path(__file__).parent / "parity" / "reference.json").read_text())
DT = torch.float64


def _regression() -> nn.Sequential:
    model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1)).to(DT)
    model.load_state_dict({k: torch.tensor(v, dtype=DT) for k, v in REF["state_dict"].items()})

    return model


def _select(subset, model: nn.Module) -> Selection:
    out = subset(ModelFn.from_module(model), Capabilities())
    assert isinstance(out, Selection)

    return out


def test_from_mask_matches_the_masks():
    model = _regression()
    mask = torch.zeros(50, 1, dtype=torch.bool)
    mask[[3, 7]] = True
    sel = _select(
        subsets.from_mask({"0.weight": mask, "2.bias": torch.ones(1, dtype=torch.bool)}), model
    )

    assert list(sel.index) == ["0.weight", "2.bias"]
    assert sel.index["0.weight"].tolist() == [3, 7]
    assert sel.index["2.bias"].tolist() == [0]

    testing.check_subset(subsets.from_mask({"0.weight": mask}), model, Capabilities())


def test_from_mask_moves_a_tied_alias_to_the_listed_name():
    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(5, 3)
            self.head = nn.Linear(3, 5, bias=False)
            self.head.weight = self.emb.weight

    net = Tied()
    mask = torch.zeros(5, 3, dtype=torch.bool)
    mask[0, 1] = True
    sel = _select(subsets.from_mask({"head.weight": mask}), net)

    assert list(sel.index) == ["emb.weight"]
    assert sel.index["emb.weight"].tolist() == [1]


def test_from_mask_two_masks_on_one_tied_tensor_raise():
    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(5, 3)
            self.head = nn.Linear(3, 5, bias=False)
            self.head.weight = self.emb.weight

    mask = torch.ones(5, 3, dtype=torch.bool)

    with pytest.raises(StructureError, match="emb.weight"):
        _select(subsets.from_mask({"head.weight": mask, "emb.weight": mask}), Tied())


def test_from_mask_unknown_name_raises():
    with pytest.raises(StructureError, match="9.weight"):
        _select(subsets.from_mask({"9.weight": torch.ones(1, dtype=torch.bool)}), _mlp())


def test_strategies_are_not_in_subsets():
    """Random, magnitude and variance selection live in the `laplacekit-subnet` plugin."""

    for name in ("random", "largest_magnitude", "largest_variance"):
        assert not hasattr(subsets, name), name
