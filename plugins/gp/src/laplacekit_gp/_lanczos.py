"""Lanczos tridiagonalization of a symmetric positive definite matrix.

The module also gives the inverse root that the tridiagonal matrix yields (Golub and Van
Loan, section 10.1).
"""

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = ["lanczos", "pinv_root"]

_BREAKDOWN = 1e-12
"""Lanczos stops when β falls below this fraction of the norm of T so far."""


def lanczos(mv: Callable[[Tensor], Tensor], v0: Tensor, steps: int) -> tuple[Tensor, Tensor]:
    """Q `[n, k]` with orthonormal columns, and the tridiagonal T = Qᵀ K Q `[k, k]`.

    `mv` applies K, and `v0`, `[n]`, is the start vector. Each new vector is
    orthogonalized twice against all earlier ones. Everything runs in float64, and `mv`
    receives float64 vectors. Lanczos stops after `steps` steps, or earlier when β, the
    norm of the next residual, falls below 1e-12 times the norm of T. The Krylov space is
    then invariant, so k ≤ `steps`. Raises `ValueError` when `v0` is zero or `steps` is
    below 1.
    """

    if steps < 1:
        raise ValueError(f"lanczos needs steps >= 1, got {steps}")

    v = v0.to(torch.float64)
    norm = v.norm()

    if not bool(norm > 0):
        raise ValueError("lanczos needs a nonzero start vector")

    qs = [v / norm]
    alphas: list[Tensor] = []
    betas: list[Tensor] = []

    for j in range(steps):
        w = mv(qs[j]).to(torch.float64)
        alphas.append(qs[j] @ w)
        q = torch.stack(qs, dim=1)

        # Full reorthogonalization, twice ("twice is enough").
        w = w - q @ (q.T @ w)
        w = w - q @ (q.T @ w)

        beta = w.norm()
        t_norm = torch.stack(alphas).square().sum() + 2 * sum(b.square() for b in betas)

        if j == steps - 1 or bool(beta < _BREAKDOWN * t_norm.sqrt()):
            break

        betas.append(beta)
        qs.append(w / beta)

    k = len(alphas)
    t = torch.diag(torch.stack(alphas))

    if k > 1:
        off = torch.stack(betas[: k - 1])
        t = t + torch.diag(off, 1) + torch.diag(off, -1)

    return torch.stack(qs[:k], dim=1), t


def pinv_root(q: Tensor, t: Tensor) -> Tensor:
    """L = Q U_T D_T^{-1/2} from T = U_T D_T U_Tᵀ, so that L Lᵀ = Q T⁺ Qᵀ.

    Eigenvalues of T at most eps · k · max are dropped, with eps from T's dtype. A nearly
    singular T then loses columns instead of dividing by a tiny pivot.
    """

    evals, evecs = torch.linalg.eigh(t)
    keep = evals > torch.finfo(t.dtype).eps * t.shape[0] * evals.max()

    return q @ evecs[:, keep] / evals[keep].sqrt()
