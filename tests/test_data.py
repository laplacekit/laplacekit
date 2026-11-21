import pytest
import torch
from torch import Tensor

import laplacekit as lk
from laplacekit.core import Batch, generator_for_batch, spawn_generator, tree_leaves


def _raw() -> list[Tensor]:
    return [torch.arange(3.0), torch.arange(5.0), torch.arange(2.0)]


def test_prepare_is_called_per_item() -> None:
    seen: list[Tensor] = []

    def prepare(raw: Tensor, generator: torch.Generator) -> Batch:
        seen.append(raw)

        return Batch(raw.unsqueeze(1), raw.unsqueeze(1) * 2)

    data = lk.data.from_loader(_raw(), prepare=prepare)
    batches = list(data(torch.Generator().manual_seed(0)))

    assert len(seen) == 3
    assert [b.size for b in batches] == [3, 5, 2]

    for raw, b in zip(_raw(), batches, strict=True):
        assert isinstance(b.inputs, Tensor)
        torch.testing.assert_close(b.inputs, raw.unsqueeze(1))
        torch.testing.assert_close(b.targets, raw.unsqueeze(1) * 2)

    # Re-iterable: a second pass prepares every item again.
    list(data(torch.Generator().manual_seed(0)))

    assert len(seen) == 6


def test_prepare_generators_come_from_the_pass_and_the_count() -> None:
    seeds: list[int] = []

    def prepare(raw: Tensor, generator: torch.Generator) -> Batch:
        seeds.append(generator.initial_seed())

        return Batch(raw.unsqueeze(1))

    data = lk.data.from_loader(_raw(), prepare=prepare)

    for pass_seed in (7, 8):
        seeds.clear()
        list(data(torch.Generator().manual_seed(pass_seed)))

        pass_gen = spawn_generator(torch.Generator().manual_seed(pass_seed))
        expected = [generator_for_batch(pass_gen, count).initial_seed() for count in (0, 3, 8)]

        assert seeds == expected
        assert len(set(seeds)) == 3


def test_prepare_draws_repeat_and_ignore_the_global_rng() -> None:
    def prepare(raw: Tensor, generator: torch.Generator) -> Batch:
        return Batch(raw.unsqueeze(1), torch.randn(raw.shape[0], 1, generator=generator))

    data = lk.data.from_loader(_raw(), prepare=prepare)

    torch.manual_seed(0)
    a = [b.targets for b in data(torch.Generator().manual_seed(1))]
    torch.manual_seed(1)
    b = [b.targets for b in data(torch.Generator().manual_seed(1))]
    c = [b.targets for b in data(torch.Generator().manual_seed(2))]

    for x, y, z in zip(a, b, c, strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
        assert not torch.allclose(x, z)


def test_prepare_must_return_a_batch() -> None:
    data = lk.data.from_loader(_raw(), prepare=lambda raw, generator: raw)

    with pytest.raises(TypeError, match="prepare must return a Batch"):
        list(data(torch.Generator().manual_seed(0)))


def test_laplace_with_prepare() -> None:
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 3)
    x = torch.randn(12, 2)
    y = torch.randint(0, 3, (12,))
    raw = [(x[i : i + 4], y[i : i + 4]) for i in range(0, 12, 4)]

    def prepare(item: tuple[Tensor, Tensor], generator: torch.Generator) -> Batch:
        return Batch(item[0], item[1])

    a = lk.laplace(model, lk.data.from_loader(raw, prepare=prepare), likelihood=lk.Categorical())
    b = lk.laplace(model, raw, likelihood=lk.Categorical())

    torch.testing.assert_close(tree_leaves(a.precision()), tree_leaves(b.precision()))
    torch.testing.assert_close(a.n_data, b.n_data)
