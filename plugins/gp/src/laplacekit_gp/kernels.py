"""Kernels for function-space priors: RBF, Matérn, periodic, and their sums and products.

A kernel maps inputs $x$ `[n, ...]` and $y$ `[m, ...]` to the `[n, m]` matrix
$k(x_i, y_j)$, with each row flattened after the first axis. The hyperparameters are tensor
leaves, so `lk.tune` and autograd can change them.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import field
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from laplacekit.core import StructureError, pytree, static

__all__ = ["RBF", "Kernel", "Matern", "Periodic", "Product", "Sum"]


@runtime_checkable
class Kernel(Protocol):
    """A covariance function $k$, evaluated between two batches of inputs."""

    @abstractmethod
    def __call__(self, x: Tensor, y: Tensor, /) -> Tensor:
        """The kernel matrix $k(x_i, y_j)$ for every pair of rows.

        Args:
            x: Inputs `[n, ...]`.
            y: Inputs `[m, ...]`, with as many entries per row as `x`.

        Returns:
            The kernel matrix, `[n, m]`.

        Raises:
            StructureError: Implementations may raise it when `x` or `y` does not fit the
                kernel.
        """
        ...


# --- helpers --------------------------------------------------------------------------


def _one() -> Tensor:
    return torch.tensor(1.0)


def _check_leaf(name: str, t: Tensor, *, vector: bool = False) -> None:
    """Raise unless `t` is a positive tensor: 0-dim, or 1-dim when `vector`."""

    if not isinstance(t, Tensor) or t.ndim > (1 if vector else 0):
        shape = "a 0-dim or 1-dim tensor" if vector else "a 0-dim tensor"
        got = f"shape {list(t.shape)}" if isinstance(t, Tensor) else type(t).__name__

        raise StructureError(f"the kernel's {name} must be {shape}, got {got}")

    if not bool((t > 0).all()):
        raise ValueError(f"the kernel's {name} must be positive, got {t}")


def _flat(x: Tensor, what: str) -> Tensor:
    if not isinstance(x, Tensor) or x.ndim == 0:
        got = "a 0-dim tensor" if isinstance(x, Tensor) else type(x).__name__

        raise StructureError(
            f"kernels take {what} as a tensor with a leading example axis, got {got}"
        )

    return x.reshape(x.shape[0], -1)


def _pair(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Flatten `x` and `y` to `[n, D]` and `[m, D]`, and check that they share D."""

    xf, yf = _flat(x, "x"), _flat(y, "y")

    if xf.shape[1] != yf.shape[1]:
        raise StructureError(
            f"kernel inputs differ in entries per example: x has {xf.shape[1]}, y has {yf.shape[1]}"
        )

    return xf, yf


def _distance(x: Tensor, y: Tensor, lengthscale: Tensor | None) -> Tensor:
    """‖(x_i − y_j) / lengthscale‖ for every pair, `[n, m]`. A None lengthscale means no scaling."""

    xf, yf = _pair(x, y)

    if lengthscale is None:
        return _cdist(xf, yf)

    if lengthscale.ndim == 1 and lengthscale.shape[0] != xf.shape[1]:
        raise StructureError(
            f"the kernel's lengthscale has {lengthscale.shape[0]} entries; it needs one, or "
            f"one per flattened input entry ({xf.shape[1]})"
        )

    return _cdist(xf / lengthscale, yf / lengthscale)


def _cdist(x: Tensor, y: Tensor) -> Tensor:
    # Exact differences: the matmul form loses digits, and its distance at x == y is
    # not exactly zero. `cdist` gives a zero gradient there.
    return torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist")


class _Algebra(ABC):
    """`+` and `*` for kernels."""

    @abstractmethod
    def __call__(self, x: Tensor, y: Tensor, /) -> Tensor: ...

    def __add__(self, other: Kernel, /) -> "Sum":
        return Sum(self, other)

    def __mul__(self, other: Kernel, /) -> "Product":
        return Product(self, other)


# --- kernels --------------------------------------------------------------------------


@pytree
class RBF(_Algebra):
    r"""The RBF kernel $k(x, y) = \sigma^2 \exp(-\tfrac{1}{2} \lVert (x - y) / \ell \rVert^2)$.

    A lengthscale $\ell$ with one entry per flattened input entry divides entrywise.

    Args:
        lengthscale: The lengthscale $\ell$, a positive 0-dim tensor, or a 1-dim one with
            one entry per flattened input entry.
        variance: The variance $\sigma^2$, a positive 0-dim tensor.

    Raises:
        StructureError: `lengthscale` or `variance` is not a tensor of the allowed shape.
        ValueError: An entry of `lengthscale` or `variance` is not positive.
    """

    lengthscale: Tensor
    variance: Tensor = field(default_factory=_one)

    def __post_init__(self) -> None:
        _check_leaf("lengthscale", self.lengthscale, vector=True)
        _check_leaf("variance", self.variance)

    def __call__(self, x: Tensor, y: Tensor, /) -> Tensor:
        """The kernel matrix, `[n, m]`.

        Args:
            x: Inputs `[n, ...]`.
            y: Inputs `[m, ...]`, with as many entries per row as `x`.

        Returns:
            The kernel matrix, `[n, m]`.

        Raises:
            StructureError: `x` or `y` is not a tensor with a leading example axis, they
                differ in entries per row, or a 1-dim `lengthscale` has another length.
        """

        r = _distance(x, y, self.lengthscale)

        return self.variance * torch.exp(-0.5 * r**2)


def _matern_half(s: Tensor) -> Tensor:
    return torch.exp(-s)


def _matern_three_halves(s: Tensor) -> Tensor:
    a = math.sqrt(3) * s

    return (1 + a) * torch.exp(-a)


def _matern_five_halves(s: Tensor) -> Tensor:
    a = math.sqrt(5) * s

    return (1 + a + a**2 / 3) * torch.exp(-a)


_MATERN: dict[str, Callable[[Tensor], Tensor]] = {
    "1/2": _matern_half,
    "3/2": _matern_three_halves,
    "5/2": _matern_five_halves,
}


@pytree
class Matern(_Algebra):
    r"""The Matérn kernel with smoothness $\nu$, on $s = \lVert (x - y) / \ell \rVert$.

    $$k(x, y) = \sigma^2 \begin{cases}
    e^{-s} & \nu = 1/2 \text{ (the exponential kernel)}, \\
    (1 + \sqrt{3} s) \, e^{-\sqrt{3} s} & \nu = 3/2, \\
    (1 + \sqrt{5} s + 5 s^2 / 3) \, e^{-\sqrt{5} s} & \nu = 5/2.
    \end{cases}$$

    Args:
        nu: The smoothness $\nu$, one of `"1/2"`, `"3/2"` and `"5/2"`. It is a string
            because a `Literal` cannot hold floats.
        lengthscale: The lengthscale $\ell$, as in `RBF`.
        variance: The variance $\sigma^2$, as in `RBF`.

    Raises:
        ValueError: `nu` is another value, or an entry of `lengthscale` or `variance` is
            not positive.
        StructureError: `lengthscale` or `variance` has a shape `RBF` refuses.
    """

    nu: Literal["1/2", "3/2", "5/2"] = static()
    lengthscale: Tensor
    variance: Tensor = field(default_factory=_one)

    def __post_init__(self) -> None:
        if self.nu not in _MATERN:
            raise ValueError(f'Matern needs nu "1/2", "3/2" or "5/2", got {self.nu!r}')

        _check_leaf("lengthscale", self.lengthscale, vector=True)
        _check_leaf("variance", self.variance)

    def __call__(self, x: Tensor, y: Tensor, /) -> Tensor:
        """The kernel matrix, `[n, m]`.

        Args:
            x: Inputs `[n, ...]`.
            y: Inputs `[m, ...]`, with as many entries per row as `x`.

        Returns:
            The kernel matrix, `[n, m]`.

        Raises:
            StructureError: As for `RBF.__call__`.
        """

        s = _distance(x, y, self.lengthscale)

        return self.variance * _MATERN[self.nu](s)


@pytree
class Periodic(_Algebra):
    r"""The periodic kernel, a product over the flattened input entries $d$.

    $$k(x, y) = \sigma^2 \prod_d \exp\!\Big(-\frac{2}{\ell^2}
    \sin^2\!\big(\pi \, |x_d - y_d| / p\big)\Big).$$

    With one entry per example, this is the 1-D kernel of Rasmussen and Williams (2006),
    Eq. 4.31. With more entries, it multiplies unit-variance 1-D kernels over the entries
    and then $\sigma^2$ once, which keeps it positive semidefinite.

    Args:
        period: The period $p$, a positive 0-dim tensor.
        lengthscale: The lengthscale $\ell$, a positive 0-dim tensor.
        variance: The variance $\sigma^2$, a positive 0-dim tensor.

    Raises:
        StructureError: `period`, `lengthscale` or `variance` is not a 0-dim tensor.
        ValueError: One of them is not positive.
    """

    period: Tensor
    lengthscale: Tensor
    variance: Tensor = field(default_factory=_one)

    def __post_init__(self) -> None:
        _check_leaf("period", self.period)
        _check_leaf("lengthscale", self.lengthscale)
        _check_leaf("variance", self.variance)

    def __call__(self, x: Tensor, y: Tensor, /) -> Tensor:
        """The kernel matrix, `[n, m]`.

        Args:
            x: Inputs `[n, ...]`.
            y: Inputs `[m, ...]`, with as many entries per row as `x`.

        Returns:
            The kernel matrix, `[n, m]`.

        Raises:
            StructureError: `x` or `y` is not a tensor with a leading example axis, or they
                differ in entries per row.
        """

        xf, yf = _pair(x, y)

        # sin² is even, so the signed difference gives the same value as |x_d − y_d|
        # and a gradient that is defined at x_d == y_d.
        diff = xf[:, None, :] - yf[None, :, :]
        s = torch.sin(math.pi * diff / self.period) / self.lengthscale

        return self.variance * torch.exp(-2 * (s**2).sum(-1))


@pytree
class Sum(_Algebra):
    """The sum kernel $a(x, y) + b(x, y)$, also written `a + b`.

    Args:
        a: The first kernel.
        b: The second kernel.
    """

    a: Kernel
    b: Kernel

    def __call__(self, x: Tensor, y: Tensor, /) -> Tensor:
        """The kernel matrix, `[n, m]`. Errors from `a` or `b` pass through.

        Args:
            x: Inputs `[n, ...]`.
            y: Inputs `[m, ...]`, with as many entries per row as `x`.

        Returns:
            The kernel matrix, `[n, m]`.
        """

        return self.a(x, y) + self.b(x, y)


@pytree
class Product(_Algebra):
    r"""The product kernel $a(x, y) \, b(x, y)$, entrywise, also written `a * b`.

    Args:
        a: The first kernel.
        b: The second kernel.
    """

    a: Kernel
    b: Kernel

    def __call__(self, x: Tensor, y: Tensor, /) -> Tensor:
        """The kernel matrix, `[n, m]`. Errors from `a` or `b` pass through.

        Args:
            x: Inputs `[n, ...]`.
            y: Inputs `[m, ...]`, with as many entries per row as `x`.

        Returns:
            The kernel matrix, `[n, m]`.
        """

        return self.a(x, y) * self.b(x, y)
