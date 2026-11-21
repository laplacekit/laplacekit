"""Diagonal SWAG variances (Maddox et al. 2019), which `largest_variance` can pick from."""

import copy
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn

import laplacekit as lk
import laplacekit.core as core
from laplacekit.core import (
    Batch,
    Likelihood,
    ModelFn,
    Outputs,
    Params,
    Targets,
)

__all__ = ["swag_variance"]


def swag_variance(
    model: nn.Module,
    data: lk.data.Data | Iterable[Any],
    /,
    *,
    likelihood: Likelihood | None = None,
    loss: Callable[[Outputs, Targets], Tensor] | None = None,
    n_snapshots: int = 40,
    every: int = 1,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 3e-4,
    min_var: float = 1e-30,
    generator: torch.Generator | None = None,
) -> Params:
    r"""Diagonal SWAG variances of `model`'s parameters (Maddox et al. 2019).

    A deep copy of `model` trains in train mode with SGD for `n_snapshots * every` epochs.
    After every `every`-th epoch, starting with the first, the running mean and mean of
    squares of its parameters take a snapshot. Over the $T$ snapshots
    $\theta_1, \dots, \theta_T$, the variance of each entry is

    $$\operatorname{Var}[\theta] = \max\Big(\frac{1}{T} \sum_{t=1}^T \theta_t^2 -
    \Big(\frac{1}{T} \sum_{t=1}^T \theta_t\Big)^2, v_{\min}\Big),$$

    where $v_{\min}$ is `min_var`. This is laplace-torch's `fit_diagonal_swag_var`, with
    the same defaults.

    Entries clamped at `min_var` tie in `largest_variance`, which breaks ties at the cut
    toward the lower index. laplace-torch's unstable `argsort` may pick other entries
    among them.

    Args:
        model: The network. It is not changed.
        data: Training data with targets, as an `lk.data.Data` or a re-iterable loader
            that `lk.data.from_loader` wraps.
        likelihood: The likelihood whose batch mean of $-\log p(y \mid f)$ is the loss,
            weighted by the batch's weights when it has them. Give exactly one of
            `likelihood` and `loss`.
        loss: A loss called as `loss(outputs, targets)`.
        n_snapshots: The number of snapshots, at least 1.
        every: The number of epochs between snapshots, at least 1.
        lr: The SGD learning rate.
        momentum: The SGD momentum.
        weight_decay: The SGD weight decay.
        min_var: The lower clamp on each variance.
        generator: Seeds the data. One seed is drawn from it, or from torch's global RNG
            when `None`, so `torch.manual_seed` makes the run repeatable. Every epoch reads
            `data(g)` with the same CPU generator `g` seeded from it, so a shuffling source
            shuffles differently each epoch.

    Returns:
        A mapping from each parameter name of `model` to its variances, shaped like the
        parameter.

    Raises:
        ValueError: Not exactly one of `likelihood` and `loss` is given, or `n_snapshots` or
            `every` is below 1. During training, when a batch has no targets, or has
            weights while `loss` is given.
        TypeError: `data` is its own iterator (a generator, for example), or, during
            training, an item is neither a `Batch` nor an `(inputs, targets)` pair.
        StructureError: During training, `likelihood`'s `log_prob` refuses a batch's
            outputs or targets, returns a scalar, or gives a bad mask (see
            `laplacekit.core.log_prob`).
    """

    if (likelihood is None) == (loss is None):
        raise ValueError("subnet.swag_variance needs exactly one of likelihood= and loss=")

    for name, value in (("n_snapshots", n_snapshots), ("every", every)):
        if value < 1:
            raise ValueError(f"subnet.swag_variance needs {name} >= 1, got {value}")

    src = data if isinstance(data, lk.data.Data) else lk.data.from_loader(data)
    data_generator = lk.core.spawn_generator(generator)

    net = copy.deepcopy(model).train()
    fn = ModelFn.from_module(net)
    params = dict(net.named_parameters())
    optimizer = torch.optim.SGD(
        params.values(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    def objective(batch: Batch) -> Tensor:
        if batch.targets is None:
            raise ValueError("subnet.swag_variance needs data with targets, but a batch has none")

        f = fn(params, batch.inputs)

        if loss is not None:
            if batch.weights is not None:
                raise ValueError(
                    "subnet.swag_variance: loss= takes no example weights, but a batch has some"
                )

            return loss(f, batch.targets)

        assert likelihood is not None
        log_p = core.log_prob(likelihood, f, batch.targets, inputs=batch.inputs)

        if batch.weights is None:
            return -log_p.mean()

        return -(batch.weights * log_p).sum() / batch.weights.sum()

    dtypes = {k: core.stat_dtype(p.dtype) for k, p in params.items()}
    mean = {k: torch.zeros_like(p, dtype=dtypes[k]) for k, p in params.items()}
    sq_mean = {k: torch.zeros_like(p, dtype=dtypes[k]) for k, p in params.items()}
    taken = 0

    for epoch in range(n_snapshots * every):
        for batch in src(data_generator):
            optimizer.zero_grad()

            with torch.enable_grad():
                objective(batch).backward()

            optimizer.step()

        if epoch % every != 0:
            continue

        old, new = taken / (taken + 1), 1 / (taken + 1)

        with torch.no_grad():
            for k, p in params.items():
                theta = p.detach().to(dtypes[k])
                mean[k] = mean[k] * old + theta * new
                sq_mean[k] = sq_mean[k] * old + theta**2 * new

        taken += 1

    return {k: torch.clamp(sq_mean[k] - mean[k] ** 2, min=min_var) for k in params}
