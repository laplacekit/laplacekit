"""Seeds behind the `generator` arguments. Private.

A top-level call draws one seed, from its `generator` or from torch's global RNG, and
derives the rest from it: the fit folds in the pass index and the examples before each
batch, then splits for `prepare` and `update`; `PerBlockCurvature` folds in each member's position;
under `torch.distributed`, the batch seed folds in the rank. A batch's draws then depend
only on the seed, its pass and the examples before it, and a member's draws only on its
position, not on the other members.
"""

import dataclasses
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager

import torch
import torch.distributed as dist

from laplacekit.core._functorch import outside_transforms

_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15


def _mix(z: int) -> int:
    """SplitMix64 finalizer."""

    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK

    return z ^ (z >> 31)


@dataclasses.dataclass(frozen=True)
class _Seed:
    """A 64-bit seed that derives others and seeds torch generators.

    Args:
        value: The seed, taken modulo $2^{64}$.
    """

    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value & _MASK)

    @staticmethod
    def draw(generator: torch.Generator | None = None, /) -> "_Seed":
        """One seed drawn from `generator`, or from torch's global RNG when `None`.

        It works under `torch.func` transforms: the seed is one draw for the whole call.

        Args:
            generator: The generator to draw from. It advances by one draw.

        Returns:
            A new seed.
        """

        device = generator.device if generator is not None else torch.device("cpu")
        low, high = -(1 << 63), (1 << 63) - 1

        with outside_transforms():
            t = torch.randint(low, high, (), dtype=torch.int64, generator=generator, device=device)

            return _Seed(int(t.item()))

    def fold_in(self, data: int, /) -> "_Seed":
        """A new seed from this one and `data`. The same inputs give the same seed.

        Args:
            data: Any integer.

        Returns:
            The derived seed.
        """

        return _Seed(_mix(self.value ^ ((data * _GOLDEN) & _MASK)))

    def split(self, n: int, /) -> tuple["_Seed", ...]:
        """`n` independent seeds, the i-th equal to `fold_in(i)`.

        Args:
            n: Number of seeds.

        Returns:
            The seeds.
        """

        return tuple(self.fold_in(i) for i in range(n))

    def generator(self, device: torch.device | str | None = None, /) -> torch.Generator:
        """A fresh generator seeded from this seed.

        Args:
            device: The generator's device. `None` means the CPU.

        Returns:
            A new `torch.Generator`.
        """

        g = torch.Generator(device=device if device is not None else "cpu")

        return g.manual_seed(self.value % (1 << 63))

    def torch_rng(self, device: torch.device | None = None, /) -> AbstractContextManager[None]:
        """Seed torch's global RNG from this seed inside the block, restoring it after.

        For code that can't take a generator, such as `Distribution.sample()`. The CPU
        generator is always seeded. No other device is touched unless `device` names one.

        Args:
            device: The accelerator the draws are made on. Its generator is forked and
                seeded too. `None` or a CPU device seeds only the CPU generator.

        Returns:
            A context manager. On exit, the forked generators get their old state back.
        """

        return self._torch_rng(device)

    @contextmanager
    def _torch_rng(self, device: torch.device | None) -> Generator[None]:
        seed = self.value % (1 << 63)

        # Not `torch.manual_seed`: it seeds every accelerator without restoring them, and
        # its lazy device calls dominate the cost of a CPU draw.
        if device is None or device.type == "cpu":
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed)
                yield

            return

        # The device-generic accelerator API picks the device; the device module only
        # seeds its current one, which every backend (cuda, xpu, mps) supports.
        acc = torch.accelerator
        index = device.index if device.index is not None else acc.current_device_index()

        with torch.random.fork_rng(devices=[index], device_type=device.type):
            torch.default_generator.manual_seed(seed)

            with acc.device_index(index):
                torch.get_device_module(device.type).manual_seed(seed)

            yield


def _batch_seeds(seed: _Seed, count: int, /) -> tuple[_Seed, _Seed]:
    """The `(prepare, update)` seeds of the batch that starts at example `count`.

    `seed` is the pass's seed. Under `torch.distributed` the rank is folded in, so ranks
    with equal shards draw differently.

    Args:
        seed: The pass's seed.
        count: Examples seen in this pass before the batch.

    Returns:
        Two independent seeds.
    """

    s = seed.fold_in(count)

    if dist.is_available() and dist.is_initialized():
        s = s.fold_in(dist.get_rank())

    prepare, update = s.split(2)

    return prepare, update
