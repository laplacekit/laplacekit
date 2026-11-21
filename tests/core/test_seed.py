import contextlib

import pytest
import torch

from laplacekit.core._seed import _batch_seeds, _Seed


def test_fold_in_differs_by_data() -> None:
    k = _Seed(0)
    assert k.fold_in(1).value != k.fold_in(2).value


def test_same_inputs_same_seed() -> None:
    assert _Seed(0).fold_in(1) == _Seed(0).fold_in(1)


def test_split_distinct() -> None:
    ks = _Seed(7).split(3)

    assert len(ks) == 3
    assert len({k.value for k in ks}) == 3


def test_value_is_taken_modulo_two_to_the_64() -> None:
    assert _Seed(-1).value == (1 << 64) - 1
    assert _Seed(1 << 64) == _Seed(0)


def test_generator_reproducible() -> None:
    a = torch.randn(3, generator=_Seed(5).generator())
    b = torch.randn(3, generator=_Seed(5).generator())

    assert torch.equal(a, b)
    assert _Seed(5).generator().device.type == "cpu"


def test_draw_follows_the_global_rng_or_the_generator() -> None:
    torch.manual_seed(0)
    a = _Seed.draw()
    torch.manual_seed(0)
    b = _Seed.draw()
    c = _Seed.draw()

    assert a == b != c

    g, h = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    before = torch.get_rng_state()

    assert _Seed.draw(g) == _Seed.draw(h)
    assert torch.equal(torch.get_rng_state(), before)  # the global RNG is untouched
    assert _Seed.draw(g) != _Seed.draw(torch.Generator().manual_seed(1))  # g advanced


def test_batch_seeds_fold_the_count_then_split() -> None:
    s = _Seed(3)
    prep, upd = _batch_seeds(s, 4)

    assert (prep, upd) == s.fold_in(4).split(2)
    assert prep != upd
    assert _batch_seeds(s, 0) != _batch_seeds(s, 4)


def test_torch_rng_reproducible_and_restores_global_state() -> None:
    k = _Seed(3)
    before = torch.get_rng_state()

    with k.torch_rng():
        a = torch.rand(1)

    assert torch.equal(torch.get_rng_state(), before)

    with k.torch_rng():
        b = torch.rand(1)

    assert torch.equal(a, b)


class _FakeAccelerator:
    """Stands in for `torch.accelerator`: the current device index, switchable."""

    def __init__(self) -> None:
        self.current = 0

    def current_accelerator(self) -> None:  # what the real `fork_rng` asks on CPU
        return None

    def current_device_index(self) -> int:
        return self.current

    @contextlib.contextmanager
    def device_index(self, index: int, /):
        before, self.current = self.current, index
        yield
        self.current = before


class _SeedOnlyModule:
    """A device module with only `manual_seed` (like `torch.mps`): it seeds the current
    device of the accelerator."""

    def __init__(self, acc: _FakeAccelerator) -> None:
        self.acc, self.seeded = acc, []

    def manual_seed(self, seed: int) -> None:
        self.seeded.append((self.acc.current, seed))


def _fake_accelerator(monkeypatch: pytest.MonkeyPatch) -> tuple[_SeedOnlyModule, list]:
    acc = _FakeAccelerator()
    fake, forks = _SeedOnlyModule(acc), []
    real_fork = torch.random.fork_rng

    def fork_rng(devices, device_type="cuda", **kw):
        forks.append((device_type, list(devices)))
        return real_fork(devices=[])

    monkeypatch.setattr(torch.random, "fork_rng", fork_rng)
    monkeypatch.setattr(torch, "get_device_module", lambda t: fake)
    monkeypatch.setattr(torch, "accelerator", acc)

    return fake, forks


def test_torch_rng_forks_and_seeds_only_the_sampled_device(monkeypatch) -> None:
    fake, forks = _fake_accelerator(monkeypatch)
    k = _Seed(3)
    seed = k.value % (1 << 63)

    with k.torch_rng(torch.device("cuda", 1)):
        a = torch.rand(3)

    with k.torch_rng(torch.device("cuda")):  # no index: the current device
        b = torch.rand(3)

    with k.torch_rng(torch.device("mps")):  # a backend whose module has only manual_seed
        m = torch.rand(3)

    with k.torch_rng(torch.device("cpu")):
        c = torch.rand(3)

    with k.torch_rng():
        d = torch.rand(3)

    assert forks == [("cuda", [1]), ("cuda", [0]), ("mps", [0]), ("cuda", []), ("cuda", [])]
    assert fake.seeded == [(1, seed), (0, seed), (0, seed)]

    for t in (b, m, c, d):  # the CPU generator is seeded the same way every time
        torch.testing.assert_close(t, a)


def test_draw_works_under_transforms() -> None:
    """Guards the private `torch._C._DisableFuncTorch` that `_Seed.draw` uses."""

    from torch.func import grad, vmap

    seeds: list[_Seed] = []

    def f(x: torch.Tensor) -> torch.Tensor:
        seeds.append(_Seed.draw(torch.Generator().manual_seed(0)))
        return (x * x).sum()

    vmap(f)(torch.ones(3, 2))
    grad(f)(torch.ones(2))

    assert seeds == [_Seed.draw(torch.Generator().manual_seed(0))] * 2


def _fake_rank(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: rank)


def test_batch_seeds_fold_in_the_rank(monkeypatch) -> None:
    s = _Seed(3)
    alone = _batch_seeds(s, 4)
    per_rank = []

    for rank in (0, 1):
        _fake_rank(monkeypatch, rank)
        per_rank.append(_batch_seeds(s, 4))

    prep0, upd0 = per_rank[0]
    prep1, upd1 = per_rank[1]

    assert prep0 != prep1 and upd0 != upd1
    assert per_rank[0] != alone


def test_a_sampled_fit_draws_differently_per_rank(monkeypatch) -> None:
    import laplacekit as lk

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 3))

    params = {k: v.detach() for k, v in model.named_parameters()}
    fn = lk.core.ModelFn.from_module(model)
    batch = lk.Batch(torch.randn(6, 3), torch.randint(0, 3, (6,)))

    def diag(rank: int) -> torch.Tensor:
        _fake_rank(monkeypatch, rank)
        g = torch.Generator().manual_seed(0)
        fit = lk.core.fit_curvature(
            fn, lk.Categorical(), params, lk.DiagGGN(fisher="mc"), generator=g
        )
        op = fit.finalize(fit.update(fit.init(), batch))

        return torch.cat([b.diag for b in op.blocks.values()])

    assert not torch.allclose(diag(0), diag(1))
    torch.testing.assert_close(diag(1), diag(1), rtol=0, atol=0)
