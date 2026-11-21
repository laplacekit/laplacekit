"""Data for the posterior: a callable that yields batches, once per pass.

`lk.laplace` and the objectives take the same forms: a `Data` source, or a re-iterable
(a list, a `DataLoader`) of `(inputs, targets)` pairs or `Batch`es. `as_data` turns each
into a `Data` source and refuses the rest with a `TypeError` that says what to pass.
"""

from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch.utils.data import DataLoader, Dataset

from laplacekit.core import Batch, as_pytree, generator_for_batch, spawn_generator, tree_leaves

__all__ = ["Data", "as_data", "from_loader"]


@runtime_checkable
class Data(Protocol):
    """A data source: `data(generator)` returns the batches of one pass."""

    @property
    @abstractmethod
    def n_data(self) -> int | None:
        """The number of examples over all processes.

        Returns:
            The count, or `None` when unknown. `lk.laplace` then scales by the examples
            the fit saw.
        """
        ...

    @abstractmethod
    def __call__(self, generator: torch.Generator, /) -> Iterable[Batch]:
        """The batches of one pass.

        Args:
            generator: A fresh CPU generator for this pass, for sources that draw random
                numbers (a shuffle, say). The fit derives it from its seed and the pass
                number.

        Returns:
            The batches, each a `Batch`.
        """
        ...


def _as_batch(item: Any) -> Batch:
    if isinstance(item, Batch):
        return item

    if isinstance(item, (tuple, list)) and len(cast(Sequence[Any], item)) == 2:
        pair = cast(Sequence[Any], item)
        return Batch.from_pair((pair[0], pair[1]))

    raise TypeError(
        f"expected a Batch or an (inputs, targets) pair, got {type(cast(object, item)).__name__}"
    )


class _Loader:
    def __init__(
        self,
        loader: Iterable[Any],
        prepare: Callable[[Any, torch.Generator], Batch] | None,
        n_data: int | None,
    ) -> None:
        self.loader = loader
        self.prepare = prepare
        self.n_data = n_data

    def __call__(self, generator: torch.Generator, /) -> Iterator[Batch]:
        pass_gen = spawn_generator(generator) if self.prepare is not None else None
        count = 0

        for item in self.loader:
            if pass_gen is None or self.prepare is None:
                batch = _as_batch(item)
            else:
                own = generator_for_batch(pass_gen, count, device=_device_of(item))
                batch = self.prepare(item, own)

                if not isinstance(batch, Batch):
                    raise TypeError(
                        f"prepare must return a Batch, got {type(cast(object, batch)).__name__}"
                    )

            count += batch.size

            yield batch


def from_loader(
    loader: Iterable[Any],
    /,
    prepare: Callable[[Any, torch.Generator], Batch] | None = None,
    n_data: int | None = None,
) -> Data:
    """Wrap an iterable of raw items, one batch each.

    Args:
        loader: A re-iterable of items, such as a list or a `DataLoader`. It is iterated
            once per pass.
        prepare: `None`: each item is a `Batch` or an `(inputs, targets)` pair, and the
            pass's generator is unused. Otherwise each item becomes
            `prepare(item, generator)`, with a fresh generator for that item on the device
            of the item's first tensor (the CPU when it has none). It is
            `lk.core.generator_for_batch(pass_gen, count)`, where `pass_gen` is
            `lk.core.spawn_generator` of the pass's generator and `count` the examples
            before the item in this pass, so it draws the same numbers on every rerun.
        n_data: The number of examples over all processes: on several ranks, pass the
            global count, not this rank's. `None` means unknown.

    Returns:
        The `Data` source.

    Raises:
        TypeError: `loader` is its own iterator (a generator, say), so it can't be iterated
            again. During a pass: an item is neither a `Batch` nor a pair, or `prepare`
            returns something other than a `Batch`.
    """

    _check_data(loader, "data")

    if iter(loader) is loader:
        raise TypeError(
            f"data is iterated once per pass, so it must be re-iterable: got a "
            f"{type(loader).__name__}; pass a list, a DataLoader, or a `lk.data` source"
        )

    return _Loader(loader, prepare, n_data)


def _device_of(item: Any) -> torch.device:
    """The device of the item's first tensor, or the CPU when it has none."""

    leaf = next((t for t in tree_leaves(as_pytree(item)) if isinstance(t, torch.Tensor)), None)

    return leaf.device if leaf is not None else torch.device("cpu")


def _is_single_batch(data: object) -> bool:
    """Whether `data` is one `(inputs, targets)` pair of tensors instead of a list of them."""

    if not isinstance(data, tuple) or len(cast(tuple[Any, ...], data)) != 2:
        return False

    first = cast(tuple[Any, ...], data)[0]

    return isinstance(first, torch.Tensor) or (
        isinstance(first, Mapping)
        and any(isinstance(v, torch.Tensor) for v in cast(Mapping[Any, Any], first).values())
    )


def _check_data(data: object, what: str, /) -> None:
    """Refuse the data forms that look right but are not a re-iterable of batches.

    Raises:
        TypeError: `data` is one `(inputs, targets)` pair, a tensor, or a `Dataset`.
    """

    if isinstance(data, Batch) or _is_single_batch(data):
        raise TypeError(
            f"{what} must be a re-iterable of batches, got one batch; wrap it in a list: "
            "[(inputs, targets)]"
        )

    if isinstance(data, torch.Tensor):
        raise TypeError(
            f"{what} must be a re-iterable of (inputs, targets) batches, got a Tensor; "
            "pass [(inputs, targets)]"
        )

    if isinstance(data, Dataset) and not isinstance(data, DataLoader | Sequence):
        raise TypeError(
            f"{what} must be a re-iterable of batches, got a {type(data).__name__}, which "
            "yields single examples; wrap it in a DataLoader: "
            "torch.utils.data.DataLoader(dataset, batch_size=...)"
        )

    if not isinstance(data, Iterable):
        raise TypeError(
            f"{what} must be a Data source or a re-iterable of (inputs, targets) batches, "
            f"got {type(data).__name__}"
        )


def as_data(data: "Data | Iterable[Any]", /, *, what: str = "data") -> Data:
    """The `Data` source for any data form `lk.laplace` accepts.

    Args:
        data: A `Data` source, or a re-iterable (a list, a `DataLoader`) of
            `(inputs, targets)` pairs or `Batch`es.
        what: The argument's name in error messages.

    Returns:
        `data` itself when it is a `Data` source, else `from_loader(data)`.

    Raises:
        TypeError: `data` is one `(inputs, targets)` pair (pass `[(inputs, targets)]`),
            a `Dataset` (wrap it in a `DataLoader`), a tensor, a one-shot iterator, or not
            iterable at all.
    """

    if isinstance(data, Data):
        return data

    _check_data(data, what)

    return from_loader(data)
