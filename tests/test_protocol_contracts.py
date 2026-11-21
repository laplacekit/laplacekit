"""A class that inherits a protocol must implement its required methods, or it can't be built."""

import inspect
import typing

import pytest
import torch

import laplacekit as lk
import laplacekit.core as core

PROTOCOLS = [
    obj
    for name in core.__all__
    if inspect.isclass(obj := getattr(core, name)) and typing.is_protocol(obj)
]
ABSTRACT = [p for p in PROTOCOLS if p.__abstractmethods__]


def test_the_required_methods_are_abstract():
    assert core.Likelihood.__abstractmethods__ == {"log_prob"}
    assert len(ABSTRACT) >= 20, [p.__name__ for p in ABSTRACT]


@pytest.mark.parametrize("protocol", ABSTRACT, ids=lambda p: p.__name__)
def test_a_subclass_missing_a_method_cannot_be_built(protocol):
    missing = type(f"Incomplete{protocol.__name__}", (protocol,), {})

    with pytest.raises(TypeError, match="abstract method"):
        missing()


def test_a_complete_subclass_and_a_plain_class_both_work():
    class Inherited(lk.core.Likelihood):
        def log_prob(self, f, y):
            return -(f - y).square().sum(-1)

    class Plain:
        def log_prob(self, f, y):
            return -(f - y).square().sum(-1)

    f = torch.zeros(3, 2)

    for lik in (Inherited(), Plain()):
        assert isinstance(lik, lk.core.Likelihood)
        assert lik.log_prob(f, f).shape == (3,)
