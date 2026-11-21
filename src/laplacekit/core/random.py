"""Generators for code that takes `generator=None` and hands generators on.

A function that draws takes an optional `torch.Generator`, as torch's own functions do.
When it hands randomness on (to a data pass, a fit, a batch), it draws one seed and
seeds fresh generators from it:

```python
def my_fit(model, data, /, *, generator=None):
    g = lk.core.spawn_generator(generator)  # one draw; None means torch's global RNG
    for batch in data(lk.core.spawn_generator(g)):
        ...
```

A `Data` source that draws per batch uses `generator_for_batch`, so that each batch's
draws depend on the pass and on the examples before the batch, and not on how many
batches came before it.
"""

import torch

from laplacekit.core._seed import _batch_seeds, _Seed


def spawn_generator(
    generator: torch.Generator | None = None,
    /,
    *,
    device: torch.device | str | None = None,
) -> torch.Generator:
    """Draw one seed and return a fresh generator seeded with it.

    `torch.manual_seed` makes the result repeatable when `generator` is `None`. Each call
    advances the source by one draw, so two calls give generators with different seeds.

    ```python
    g = lk.core.spawn_generator()  # from torch's global RNG
    h = lk.core.spawn_generator(torch.Generator().manual_seed(0), device="cuda")
    ```

    Args:
        generator: The source of the seed. It advances by one draw. `None` means torch's
            global RNG. It may live on any device.
        device: The device of the new generator. `None` means the CPU.

    Returns:
        A new `torch.Generator`. Its `initial_seed()` is the seed that was drawn, at most
        $2^{63} - 1$.
    """

    return _Seed.draw(generator).generator(device)


def generator_for_batch(
    generator: torch.Generator,
    count: int,
    /,
    *,
    device: torch.device | str | None = None,
) -> torch.Generator:
    """Give the batch that starts at example `count` of a pass its own generator.

    A `Data` source calls it once per batch with the pass's generator:

    ```python
    def __call__(self, generator, /):  # a Data source
        count = 0
        for raw in self.loader:
            g = lk.core.generator_for_batch(generator, count)
            ...
            count += len(raw)
    ```

    One seed is drawn from a copy of `generator`'s state, so `generator` itself does not
    advance, and every batch of a pass starts from the same seed. The new generator
    comes from that seed, from `count` and, under `torch.distributed`, from the process
    rank, the way the fit derives the generator of each batch's `update`. So a batch at
    the same place gets the same draws however the pass was cut into batches before it,
    and ranks with equal shards draw differently. Generators in different states give
    different batch generators, even unseeded ones.

    Args:
        generator: The pass's generator. Only its current state counts, and it does not
            change.
        count: The number of examples in the pass before the batch.
        device: The device of the new generator. `None` means the CPU.

    Returns:
        A new `torch.Generator`. The same state, `count`, rank and process group give one
        with the same seed.
    """

    copy = torch.Generator(device=generator.device)
    copy.set_state(generator.get_state())
    prepare, _ = _batch_seeds(_Seed.draw(copy), count)

    return prepare.generator(device)
