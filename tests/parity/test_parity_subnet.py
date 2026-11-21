"""Match laplace-torch's `fit_diagonal_swag_var` and `LargestVarianceSWAGSubnetMask`.

The loaders are full-batch and unshuffled, as in `make_reference.py`: 5 snapshots, one per
epoch, at lr 0.01 with laplace-torch's other defaults. The smallest variances (about 1e-15
to 1e-8) come from `E[θ²] − E[θ]²` and lose about four digits to cancellation in float64,
hence the absolute tolerance.
"""

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import laplacekit as lk
from laplacekit.core import Capabilities, ModelFn, Selection

REF = json.loads(Path(__file__).with_name("reference.json").read_text())
DT = torch.float64


def _tensor(x: object) -> torch.Tensor:
    return torch.tensor(x, dtype=DT)


def _classifier() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(2, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.Linear(20, 4, bias=False),
    )


def _regressor() -> nn.Sequential:
    return nn.Sequential(nn.Linear(1, 50), nn.Tanh(), nn.Linear(50, 1))


def _flat(model: nn.Module, sel: Selection) -> list[int]:
    """The selected entries as indices into `parameters_to_vector(model.parameters())`."""

    offsets, start = {}, 0

    for name, p in model.named_parameters():
        offsets[name] = start
        start += p.numel()

    return sorted(int(offsets[name] + i) for name, idx in sel.index.items() for i in idx)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_parity_swag(task):
    ref = REF["swag"][task]
    model = _classifier() if task == "classification" else _regressor()
    model.to(DT).load_state_dict({k: _tensor(v) for k, v in ref["state_dict"].items()})
    model.eval()

    if task == "classification":
        cls = REF["classification"]
        data = [(_tensor(cls["X"]), torch.tensor(cls["Y"]))]
        var = lk.subnet.swag_variance(model, data, likelihood=lk.Categorical(), n_snapshots=5)
    else:
        data = [(_tensor(REF["X"]), _tensor(REF["Y"]))]
        var = lk.subnet.swag_variance(model, data, loss=torch.nn.functional.mse_loss, n_snapshots=5)

    flat = torch.cat([var[name].reshape(-1) for name, _ in model.named_parameters()])

    assert [tuple(var[n].shape) for n in var] == [tuple(p.shape) for p in model.parameters()]
    torch.testing.assert_close(flat, _tensor(ref["variances"]), rtol=1e-4, atol=1e-14)

    sel = lk.subnet.largest_variance(var, 30)(ModelFn.from_module(model), Capabilities())

    assert isinstance(sel, Selection)
    assert _flat(model, sel) == ref["indices"]
