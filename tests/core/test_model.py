import torch
from torch import Tensor, nn

from laplacekit.core.model import ModelFn


def test_linear_matches_module() -> None:
    m = nn.Linear(3, 2)
    x = torch.randn(4, 3)
    params = dict(m.named_parameters())

    assert torch.equal(ModelFn.from_module(m)(params, x), m(x))


def test_zero_params_give_zeros() -> None:
    m = nn.Linear(3, 2)
    zeros = {k: torch.zeros_like(v) for k, v in m.named_parameters()}
    out = ModelFn.from_module(m)(zeros, torch.randn(4, 3))

    assert torch.equal(out, torch.zeros(4, 2))
    assert all(p.abs().sum() > 0 for p in m.parameters() if p.numel() > 1)


def test_mapping_inputs_become_kwargs() -> None:
    class Two(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.tensor(2.0))

        def forward(self, a: Tensor, b: Tensor) -> Tensor:
            return self.w * a + b

    m = Two()
    a, b = torch.ones(3), torch.full((3,), 5.0)
    out = ModelFn.from_module(m)(dict(m.named_parameters()), {"a": a, "b": b})

    assert torch.equal(out, m(a, b))


def test_dict_outputs_pass_through() -> None:
    class Heads(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(()))

        def forward(self, x: Tensor) -> dict[str, Tensor]:
            return {"cls": self.w * x, "box": x + 1}

    m = Heads()
    x = torch.randn(2)
    out = ModelFn.from_module(m)(dict(m.named_parameters()), x)

    assert isinstance(out, dict) and set(out) == {"cls", "box"}
    assert torch.equal(out["box"], x + 1)


def test_from_module_treats_unpassed_parameters_as_constants() -> None:
    import torch
    from torch import nn

    from laplacekit.core import ModelFn

    m = nn.Linear(2, 3)
    fn = ModelFn.from_module(m)
    x = torch.randn(4, 2)
    w = m.weight.detach().clone().requires_grad_()
    out = fn({"weight": w}, x)

    out.sum().backward()

    assert w.grad is not None
    assert m.bias.grad is None
