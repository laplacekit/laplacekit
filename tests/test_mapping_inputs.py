"""Inputs may be any `Mapping`, not only a `dict`: a `UserDict` or a transformers
`BatchEncoding` (a tokenizer's output) works wherever a dict does."""

from collections import UserDict

import pytest
import torch
from torch import Tensor, nn
from transformers import BatchEncoding

import laplacekit as lk
import laplacekit.function_space as fs
from laplacekit.core import Batch, probe


class Encoding(UserDict):
    """A mapping that is not a dict and not a torch pytree node."""


class TwoInputs(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc = nn.Linear(3, 4)
        self.head = nn.Linear(4, 3)

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.head(torch.tanh(self.fc(a) + b))


def data() -> tuple[TwoInputs, dict[str, Tensor], Tensor]:
    torch.manual_seed(0)
    model = TwoInputs().double().eval()
    x = {"a": torch.randn(12, 3, dtype=torch.float64), "b": torch.randn(12, 4, dtype=torch.float64)}
    y = torch.arange(12) % 3

    return model, x, y


MAPPINGS = [Encoding, BatchEncoding]


def fit(model: nn.Module, x: dict[str, Tensor], y: Tensor) -> lk.LaplacePosterior:
    return lk.laplace(model, [(x, y)], likelihood=lk.Categorical(), hyper_tuner=None)


@pytest.mark.parametrize("wrap", MAPPINGS, ids=["UserDict", "BatchEncoding"])
def test_predict_and_outputs_take_any_mapping(wrap):
    model, x, y = data()
    post = fit(model, x, y)
    x_test = {k: v[:5] for k, v in x.items()}

    want = lk.predict_outputs(post, x_test, generator=torch.Generator().manual_seed(0))
    got = lk.predict_outputs(post, wrap(x_test), generator=torch.Generator().manual_seed(0))

    torch.testing.assert_close(got.mean, model(**x_test))
    torch.testing.assert_close(got.mean, want.mean)
    torch.testing.assert_close(got.cov.diagonal(), want.cov.diagonal())

    probs = lk.predict(post, wrap(x_test), generator=torch.Generator().manual_seed(0)).probs

    torch.testing.assert_close(
        probs, lk.predict(post, x_test, generator=torch.Generator().manual_seed(0)).probs
    )

    # `select` sees the inputs too.
    picked = lk.predict(post, wrap(x_test), select=lambda f, inp: f[:, :2]).probs

    assert picked.shape == (5, 2)

    method = lk.WeightSamples(n_samples=4)
    draws = lk.predict(post, wrap(x_test), method, generator=torch.Generator().manual_seed(0)).probs

    torch.testing.assert_close(
        draws, lk.predict(post, x_test, method, generator=torch.Generator().manual_seed(0)).probs
    )


@pytest.mark.parametrize("wrap", MAPPINGS, ids=["UserDict", "BatchEncoding"])
def test_laplace_takes_batches_of_any_mapping(wrap):
    model, x, y = data()

    want = fit(model, x, y)
    got = fit(model, wrap(x), y)

    assert repr(got) == repr(want)
    torch.testing.assert_close(got.precision().diagonal(), want.precision().diagonal())


@pytest.mark.parametrize("wrap", MAPPINGS, ids=["UserDict", "BatchEncoding"])
def test_batch_and_probe_take_any_mapping(wrap):
    model, x, y = data()
    batch = Batch(wrap(x), y)

    assert batch.size == 12
    assert isinstance(batch.inputs, dict)

    fn = lk.core.ModelFn.from_module(model)
    caps = probe(fn, lk.Categorical(), batch, params=dict(model.head.named_parameters()))

    assert caps.vjp and caps.jvp and caps.vmap


def test_nested_mappings_are_plain_dicts():
    model, x, y = data()

    def forward(params, inputs):
        return lk.core.ModelFn.from_module(model)(params, inputs["x"])

    fn = lk.core.ModelFn(forward, model)
    post = lk.laplace(fn, [({"x": x}, y)], likelihood=lk.Categorical(), hyper_tuner=None)
    x_test = {k: v[:4] for k, v in x.items()}

    got = lk.predict_outputs(post, {"x": Encoding(x_test)})
    want = lk.predict_outputs(post, {"x": x_test})

    torch.testing.assert_close(got.cov.diagonal(), want.cov.diagonal())


def test_function_space_takes_any_mapping():
    model, x, y = data()
    post = fit(model, x, y)
    gp = fs.laplace(
        model,
        [(Encoding(x), y)],
        likelihood=lk.Categorical(),
        method=fs.SubsetOfData(n_subset=6),
        prior=post.prior,
        subset=dict(post.mean),
    )
    x_test = {k: v[:4] for k, v in x.items()}

    got = gp.at(BatchEncoding(x_test))
    want = gp.at(x_test)

    torch.testing.assert_close(got.mean, want.mean)
    torch.testing.assert_close(got.cov.diagonal(), want.cov.diagonal())
