import pytest
import torch
from torch import Tensor

from laplacekit.core.types import Batch, Capabilities, Requirements


def test_batch_size_tensor() -> None:
    assert Batch(torch.zeros(4, 3)).size == 4


def test_batch_size_mapping() -> None:
    assert Batch({"ids": torch.zeros(5, 7)}).size == 5


def test_inputs_as_dict() -> None:
    d = {"ids": torch.zeros(5, 7)}
    assert Batch(d).inputs_as(dict) is d


def test_inputs_as_wrong_type_raises() -> None:
    with pytest.raises(TypeError):
        Batch({"ids": torch.zeros(5, 7)}).inputs_as(Tensor)


def test_inputs_as_predicate() -> None:
    def is_tensor(x: object) -> bool:
        return isinstance(x, Tensor)

    x = torch.zeros(2)

    assert Batch(x).inputs_as(is_tensor) is x

    with pytest.raises(TypeError):
        Batch({"a": x}).inputs_as(is_tensor)


def test_from_pair() -> None:
    x, y = torch.zeros(3, 2), torch.ones(3)
    b = Batch.from_pair((x, y))

    assert b.inputs is x and b.targets is y and b.weights is None


def test_requirements_defaults() -> None:
    r = Requirements()

    assert r.additive_stat and r.weights and not r.vjp and r.one_of == ()
    assert r.input_type is None


def test_capabilities_defaults() -> None:
    c = Capabilities()
    assert not c.vjp and c.hooked == frozenset() and c.failures == {}


def test_hooked_is_derived_from_hooked_types() -> None:
    from torch import nn

    from laplacekit.core import CapabilityError, Requirements

    caps = Capabilities(hooked_types={"fc": nn.Linear})

    assert caps.hooked == frozenset({"fc"})

    with pytest.raises(CapabilityError, match="emb"):
        caps.check(Requirements(hooks=(nn.Linear,)), "KFAC", blocks=["fc", "emb"])


def test_hook_error_names_a_failed_order_pass() -> None:
    from torch import nn

    from laplacekit.core import CapabilityError, Requirements

    caps = Capabilities(failures={"module_order": "RuntimeError: boom"})

    with pytest.raises(CapabilityError, match="forward-order pass failed.*boom"):
        caps.check(Requirements(hooks=(nn.Linear,)), "KFAC", blocks=["fc"])


def test_batch_size_of_a_scalar_leaf_raises_value_error() -> None:
    with pytest.raises(ValueError, match="0-d"):
        _ = Batch({"t": torch.tensor(3.0), "x": torch.zeros(4, 2)}).size
