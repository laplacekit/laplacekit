"""`all_reduce` across processes: two gloo ranks give what one process gives over all data.

Each test spawns two ranks. A rank writes its results to a file; the parent compares them
with a single-process fit. An exception in a rank fails the test with its traceback.
"""

import socket
import time
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor, nn

import laplacekit as lk
from laplacekit.core import all_reduce, tree_leaves, tree_map

WORLD = 2
TIMEOUT = 120.0  # seconds for all ranks together


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(rank: int, worker: Any, port: int, out: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD
    )

    try:
        result = worker(rank)
    finally:
        dist.destroy_process_group()

    torch.save(result, Path(out) / f"rank{rank}.pt")


def _spawn(worker: Any, tmp_path: Path) -> list[Any]:
    """Run `worker(rank)` on each rank under a gloo group; return the ranks' results."""

    ctx = mp.spawn(_entry, (worker, _free_port(), str(tmp_path)), nprocs=WORLD, join=False)
    deadline = time.monotonic() + TIMEOUT

    while not ctx.join(timeout=1.0):  # raises with the rank's traceback when one fails
        if time.monotonic() > deadline:
            for p in ctx.processes:
                p.kill()

            pytest.fail(f"the ranks did not finish within {TIMEOUT:.0f} s")

    return [torch.load(tmp_path / f"rank{r}.pt") for r in range(WORLD)]


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 5), nn.Tanh(), nn.Linear(5, 3)).double()


def _batches() -> list[lk.Batch]:
    """Five batches of ten."""

    g = torch.Generator().manual_seed(1)
    x = torch.randn(50, 2, generator=g, dtype=torch.float64)
    y = torch.randint(0, 3, (50,), generator=g)

    return [lk.Batch(x[i : i + 10], y[i : i + 10]) for i in range(0, 50, 10)]


def _features(op: Any) -> list[Tensor]:
    """The diagonal, products with two fixed vectors, and the log-determinant of `op + I`.

    These don't depend on the sign of eigenvectors, unlike the operator's leaves. The
    curvature alone is (near) singular, so its own log-determinant is round-off.
    """

    g = torch.Generator().manual_seed(2)
    diag = op.diagonal()
    one = torch.ones((), dtype=torch.float64)
    out = [*tree_leaves(diag), op.add_isotropic(one).logdet()]

    for _ in range(2):
        v = tree_map(lambda d: torch.randn(d.shape, generator=g, dtype=d.dtype), diag)
        out += tree_leaves(op.mv(v))

    return [t.detach() for t in out]


class _TwoPassDiag:
    """A staged method: Diag twice, summed, so each pass is reduced across the ranks."""

    @property
    def requires(self) -> lk.core.Requirements:
        return lk.DiagGGN().requires

    def plan(self, params: Any, ctx: Any) -> Any:
        def second(first: Any) -> Any:
            def add(again: Any) -> Any:
                blocks = {
                    b: lk.core.Diag(op.diag + again.blocks[b].diag)
                    for b, op in first.blocks.items()
                }

                return lk.core.Done(lk.core.BlockDiag(blocks))

            return lk.core.chain(lk.DiagGGN(), then=add)

        return lk.core.chain(lk.DiagGGN(), then=second)


def _methods() -> dict[str, Any]:
    return {"diag": lk.DiagGGN(), "kfac": lk.KFAC(), "staged": _TwoPassDiag()}


def _by_hand(approx: Any, batches: list[lk.Batch]) -> dict[str, Any]:
    """The step loop from the docs, with all parameters, on `batches`."""

    model = _model()
    fn = lk.core.ModelFn.from_module(model)
    lik = lk.Categorical()
    params = {k: v.detach() for k, v in model.named_parameters()}
    caps = lk.core.probe(fn, lik, _batches()[0], params=params)
    step = lk.core.fit_curvature(fn, lik, params, approx, generator=None, capabilities=caps).start()

    while isinstance(step, lk.core.Running):
        stat = step.init()

        for batch in batches:
            stat = step.update(stat, batch)

        step = step.finish(lk.distributed.all_reduce(stat))  # the spec's by-hand loop

    last = step.stats[-1]

    return {
        "features": _features(step.result),
        "count": last.count,
        "data_fit": last.data_fit,
        "passes": len(step.stats),
    }


def _close(got: dict[str, Any], want: dict[str, Any]) -> None:
    assert got["passes"] == want["passes"]
    assert torch.equal(got["count"], want["count"])
    torch.testing.assert_close(got["data_fit"], want["data_fit"], rtol=1e-6, atol=1e-6)

    for a, b in zip(got["features"], want["features"], strict=True):
        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


def _identical(results: list[dict[str, Any]]) -> None:
    """Every rank holds bit-for-bit the same result."""

    first = results[0]

    for other in results[1:]:
        assert torch.equal(other["count"], first["count"])
        assert torch.equal(other["data_fit"], first["data_fit"])

        for a, b in zip(other["features"], first["features"], strict=True):
            assert torch.equal(a, b)


def _uneven(rank: int) -> dict[str, Any]:
    batches = _batches()
    shard = batches[:3] if rank == 0 else batches[3:]
    out: dict[str, Any] = {k: _by_hand(m, shard) for k, m in _methods().items()}

    # The rank's own contribution keeps its graph; the other ranks' are constants.
    model = _model()
    fn = lk.core.ModelFn.from_module(model)
    params = {k: v.detach().requires_grad_() for k, v in model.named_parameters()}
    fit = lk.core.fit_curvature(fn, lk.Categorical(), params, lk.DiagGGN())
    stat = fit.init()

    for batch in shard:
        stat = fit.update(stat, batch)

    reduced = all_reduce(stat)
    local = torch.autograd.grad(
        sum(t.sum() for t in tree_leaves(stat.curvature)), [*params.values()], retain_graph=True
    )
    total = torch.autograd.grad(
        sum(t.sum() for t in tree_leaves(reduced.curvature)), [*params.values()]
    )
    out["grads"] = (local, total)

    return out


def test_all_reduce_matches_single_process(tmp_path):
    results = _spawn(_uneven, tmp_path)

    for name, approx in _methods().items():
        want = _by_hand(approx, _batches())
        _close(results[0][name], want)
        _identical([r[name] for r in results])

    assert int(results[0]["diag"]["count"]) == 50

    for r in results:
        local, total = r["grads"]

        for a, b in zip(local, total, strict=True):
            torch.testing.assert_close(a, b)


def _one_empty(rank: int) -> dict[str, Any]:
    shard = _batches() if rank == 0 else []

    return {k: _by_hand(m, shard) for k, m in _methods().items()}


def test_rank_without_batches(tmp_path):
    results = _spawn(_one_empty, tmp_path)

    for name, approx in _methods().items():
        want = _by_hand(approx, _batches())
        _close(results[1][name], want)
        _identical([r[name] for r in results])


def _front(batches: list[lk.Batch]) -> dict[str, Any]:
    post = lk.laplace(_model(), batches, likelihood=lk.Categorical(), subset=lk.subsets.all)

    return {
        "n_data": post.n_data,
        "data_fit": post.data_fit,
        "prec": post.prior.prior_precision,
        "features": _features(post.precision()),
    }


def _front_rank(rank: int) -> dict[str, Any]:
    batches = _batches()

    return _front(batches[:3] if rank == 0 else batches[3:])


def test_laplace_front_door_two_ranks(tmp_path):
    results = _spawn(_front_rank, tmp_path)
    want = _front(_batches())

    assert float(want["n_data"]) == 50.0
    assert len(want["data_fit"]) == 1

    for got in results:
        assert torch.equal(got["n_data"], want["n_data"])
        torch.testing.assert_close(got["data_fit"], want["data_fit"], rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(got["prec"], want["prec"], rtol=1e-6, atol=1e-6)

        for a, b in zip(got["features"], want["features"], strict=True):
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-6)


def test_dist_reexports_all_reduce():
    import laplacekit.core

    assert lk.distributed.all_reduce is laplacekit.core.all_reduce
    assert "distributed" in lk.__all__
