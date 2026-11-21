"""Core value types."""

import dataclasses
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Final, TypeIs, cast

from torch import Tensor, nn

from laplacekit.core._pytree import as_pytree, pytree, static, tree_leaves

type PyTree = Tensor | Mapping[str, PyTree] | Sequence[PyTree]
type Params = Mapping[str, Tensor]
type Inputs = PyTree
type Outputs = PyTree
type Targets = PyTree

ROOT_BLOCK: Final = "<root>"
"""Block name of the root module's own parameters. Not a valid module path."""


@pytree
class Batch:
    """What `prepare` produces and `update` consumes.

    An example is one entry along the leading (batch) dimension of every input
    leaf: a sequence is one example, and per-token terms are summed within it.
    Inputs shared by the whole batch (a scalar timestep) are expanded to the
    batch dimension by `prepare`. Losses that don't decompose per example
    (pairwise, listwise) are out of scope. A pair, or any group scored together, is one
    example when its log-likelihood depends only on its own outputs.

    A mapping that torch's pytree treats as a leaf (a `UserDict`, a tokenizer's
    `BatchEncoding`) is stored as a plain `dict`, in `inputs` and `targets`, so `vmap`
    sees its tensors.

    Args:
        inputs: Whatever `forward` takes. Code that knows its input type narrows it with
            `batch.inputs_as(MyType)` or `batch.inputs_as(is_my_type)`.
        targets: Targets, or `None` when the batch has none.
        weights: Per-example weights. `None` means all ones.
    """

    inputs: Inputs
    targets: Targets | None = None
    weights: Tensor | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", as_pytree(self.inputs))
        object.__setattr__(self, "targets", as_pytree(self.targets))

    @property
    def size(self) -> int:
        """Number of examples: the leading dimension of the first input leaf.

        Raises:
            ValueError: `inputs` has no tensor leaves, or its first leaf is 0-dim (expand
                shared inputs such as a timestep to the batch size).
        """

        leaves = tree_leaves(self.inputs)

        if not leaves:
            raise ValueError("Batch.inputs has no tensor leaves")

        if leaves[0].ndim == 0:
            raise ValueError(
                "Batch.inputs' first leaf is 0-d, so it has no batch dimension; expand shared "
                "inputs (e.g. a timestep) to the batch size"
            )

        return int(leaves[0].shape[0])

    def inputs_as[T](self, check: type[T] | Callable[[Inputs], TypeIs[T]], /) -> T:
        """`inputs`, checked at runtime.

        Args:
            check: A plain class, or a `TypeIs` predicate for generic types such as
                `dict[str, Tensor]` (`isinstance` can't check those).

        Returns:
            `inputs`, typed as `T`.

        Raises:
            TypeError: `inputs` fails `check`.
        """

        ok = isinstance(self.inputs, check) if isinstance(check, type) else check(self.inputs)

        if not ok:
            name = check.__name__ if hasattr(check, "__name__") else repr(check)
            raise TypeError(f"Batch.inputs is {type(self.inputs).__name__}, expected {name}")

        return cast(T, self.inputs)

    @staticmethod
    def from_pair(pair: tuple[Inputs, Targets]) -> "Batch":
        """Batch from an `(inputs, targets)` pair, as most data loaders yield.

        Args:
            pair: Inputs and targets.

        Returns:
            A batch without weights.
        """

        return Batch(inputs=pair[0], targets=pair[1])


@dataclasses.dataclass(frozen=True)
class Requirements:
    """What a method needs from the model beyond one backward pass, and from the likelihood.

    `missing` and `Capabilities.check` cover the model; `check_likelihood` covers the
    likelihood needs.

    Args:
        vjp: Needs `torch.func.vjp`.
        vjp_twice: Needs a VJP of a VJP: $J t$ from reverse mode alone, which
            differentiates the model's backward.
        jvp: Needs `torch.func.jvp` (forward-mode AD).
        vmap: Needs `torch.func.vmap`.
        hooks: Module types whose hooks must fire. Hook-based methods also need the
            subset's params to be the module's own (live) parameters, which the probe
            checks.
        needs_targets: Needs targets in every batch.
        needs_sample: The likelihood draws targets (`HasSample` or `HasDistribution`).
        needs_distribution: The likelihood has `HasDistribution`.
        needs_closed_form_hessian: Needs the closed-form output Hessian
            (`HasOutputHessian`), as a method with `fisher="exact"` does.
        needs_output_hessian: Needs an output Hessian: the closed form
            (`HasOutputHessian`) or the MC Fisher (sampling).
        late_scale: True: the statistic is linear in a `LateLikelihood`'s
            `curvature_scale`, so its hyperparameters can be applied after the fit. False
            (e.g. the empirical Fisher, which is quadratic in the score):
            `check_likelihood` refuses a `LateLikelihood` with a late hyperparameter, one
            whose `unit()` is another object.
        additive_stat: True: the statistic is a sum, so `mix` can scale it. False: it
            still merges under `+` (e.g. a top-M subset), so `all_reduce` works, but it
            can't be mixed.
        weights: The method honours `Batch.weights`.
        input_type: Required type of `Batch.inputs`, e.g. `Tensor` for backends that can't
            take mappings. `None` accepts any.
        one_of: Alternatives; at least one must hold.
    """

    vjp: bool = False
    vjp_twice: bool = False
    jvp: bool = False
    vmap: bool = False
    hooks: tuple[type[nn.Module], ...] = ()
    needs_targets: bool = False
    needs_sample: bool = False
    needs_distribution: bool = False
    needs_output_hessian: bool = False
    needs_closed_form_hessian: bool = False
    late_scale: bool = True
    additive_stat: bool = True
    weights: bool = True
    input_type: type | None = None
    one_of: tuple["Requirements", ...] = ()

    def missing(self, caps: "Capabilities") -> list[str]:
        """Capabilities this requirement lacks.

        Covers `vjp`, `vjp_twice`, `jvp`, `vmap` and `one_of`. Hooks are checked by
        `Capabilities.check`.

        Args:
            caps: What the probe found.

        Returns:
            Names of the missing capabilities, empty when satisfied. An unmet `one_of`
            adds one entry listing what each alternative lacks.
        """

        out: list[str] = []

        if self.vjp and not caps.vjp:
            out.append("vjp")

        if self.vjp_twice and not caps.vjp_twice:
            out.append("vjp_twice")

        if self.jvp and not caps.jvp:
            out.append("jvp")

        if self.vmap and not caps.vmap:
            out.append("vmap")

        if self.one_of and not any(not r.missing(caps) for r in self.one_of):
            out.append(
                "one of (" + " | ".join("+".join(r.missing(caps)) for r in self.one_of) + ")"
            )

        return out

    def satisfies(self, caps: "Capabilities") -> bool:
        """Whether `caps` has everything `missing` checks.

        Args:
            caps: What the probe found.

        Returns:
            True when `missing(caps)` is empty.
        """

        return not self.missing(caps)

    def check_batch(self, batch: Batch, what: str, /) -> None:
        """Check `batch` against `needs_targets`, `input_type` and `weights`.

        The fit calls this on each batch before the method sees it.

        Args:
            batch: The batch to check.
            what: Name of the method, used in the error message.

        Raises:
            CapabilityError: The batch has no targets but `needs_targets` is set, its
                inputs are not of `input_type`, or it has weights but `weights` is False.
        """

        if self.needs_targets and batch.targets is None:
            raise CapabilityError(f"{what} needs targets, but the batch has none")

        if self.input_type is not None and not isinstance(batch.inputs, self.input_type):
            raise CapabilityError(
                f"{what} needs {self.input_type.__name__} inputs, got {type(batch.inputs).__name__}"
            )

        if not self.weights and batch.weights is not None:
            raise CapabilityError(f"{what} does not support per-example weights")


NOT_PROBED: Final = "not probed"
"""The failure of a capability that was never tried (no probe, or no parameters)."""

TRANSFORMS: Final = ("vjp", "vjp_twice", "jvp", "vmap")
"""The capabilities the probe tries."""


@pytree
class Capabilities:
    """What the probe found. Each capability is independent of the others.

    Args:
        vjp: `torch.func.vjp` works.
        vjp_twice: The VJP of a VJP works and agrees with the VJP. Fails on kernels
            without a double backward (PyTorch's fused attention) or a custom
            `autograd.Function` that has none.
        jvp: `torch.func.jvp` works.
        vmap: `torch.func.vmap` works.
        hooked_types: Module type of each Linear and Conv2d module whose forward hook
            fired in the probe.
        module_order: Parameterized modules: the root module first, as `"<root>"`
            (`ROOT_BLOCK`), when it owns parameters, then the others in the order their
            forward finished.
        tied: Each name of a parameter tensor that has several names (tied weights),
            mapped to all of them.
        event_size: Output entries per example (summed over output leaves). `None` when
            not probed.
        failures: Each failed capability mapped to the failing op.
        probed: Parameter names the vjp, jvp and vmap checks covered. `None` means every
            parameter. The names are literal: a tied tensor probed under one name and
            fitted under another is refused.
        selected: Probed names that end in `"[idx]"`: the `"fc.weight[idx]"` parameters
            of a `Selection`.
    """

    vjp: bool = static(default=False)
    vjp_twice: bool = static(default=False)
    jvp: bool = static(default=False)
    vmap: bool = static(default=False)
    hooked_types: Mapping[str, type[nn.Module]] = static(default_factory=dict[str, type[nn.Module]])
    module_order: tuple[str, ...] = static(default=())
    tied: Mapping[str, tuple[str, ...]] = static(default_factory=dict[str, tuple[str, ...]])
    event_size: int | None = static(default=None)
    failures: Mapping[str, str] = static(default_factory=dict[str, str])
    probed: frozenset[str] | None = static(default=None)
    selected: frozenset[str] = static(default=frozenset())

    @staticmethod
    def unprobed() -> "Capabilities":
        """No capabilities, each marked not probed: what a fit without `caps` assumes.

        Returns:
            Capabilities whose `failures` map each of `TRANSFORMS` to `NOT_PROBED`.
        """

        return Capabilities(failures=dict.fromkeys(TRANSFORMS, NOT_PROBED))

    def was_probed(self, name: str, /) -> bool:
        """Whether a capability was tried.

        Args:
            name: Capability name, e.g. `"vjp"`.

        Returns:
            False when `failures` marks `name` as `NOT_PROBED`, as `unprobed()` and a probe
            with `subset={}` do.
        """

        return self.failures.get(name) != NOT_PROBED

    @property
    def hooked(self) -> frozenset[str]:
        """Names of the modules in `hooked_types`."""

        return frozenset(self.hooked_types)

    @property
    def selected_blocks(self) -> frozenset[str]:
        """Block names (module paths, or `ROOT_BLOCK`) of the `selected` parameters."""

        return frozenset(n.rpartition(".")[0] or ROOT_BLOCK for n in self.selected)

    def check(self, req: Requirements, what: str, blocks: Iterable[str] | None = None) -> None:
        """Check that these capabilities meet `req`.

        Hook-based methods are refused first, with or without `blocks`, when parameters
        were `selected`: hooks see a module's own parameters, not some of their entries.

        Args:
            req: What the method needs.
            what: Name of the method, used in the error message.
            blocks: Default-layout block names, i.e. module paths. With `req.hooks`, each
                block's module must have been reached by hooks and be of a required type.
                `None` skips that check.

        Raises:
            CapabilityError: Naming `what` and the missing capabilities, and whether they
                were not probed or failed. Also raised when `req.hooks` is set and
                parameters were `selected`, or a block in `blocks` was not reached by
                hooks or is not of a required type.
        """

        if req.hooks and self.selected:
            raise CapabilityError(
                f"{what} needs module hooks on the module's own parameters, but blocks "
                f"{sorted(self.selected_blocks)} hold index selections (Selection); use DiagGGN, "
                "DenseGGN or LowRankGGN"
            )

        missing = req.missing(self)

        if missing and not any(self.was_probed(m) for m in missing):
            raise CapabilityError(
                f"{what} needs {'; '.join(missing)}; capabilities not probed; pass "
                "caps=probe(model, likelihood, batch, params=params) to fit_curvature"
            )

        if missing:
            detail = "; ".join(
                f"{m} ({self.failures[m]})" if m in self.failures else m for m in missing
            )
            raise CapabilityError(f"{what} needs {detail}, which this model does not support")

        if req.hooks and blocks is not None:
            bad = [
                b
                for b in blocks
                if b not in self.hooked or not issubclass(self.hooked_types[b], req.hooks)
            ]

            if bad:
                types = ", ".join(t.__name__ for t in req.hooks)
                why = (
                    f"the probe's forward-order pass failed ({self.failures['module_order']})"
                    if "module_order" in self.failures
                    else "weight used without calling the module, or another layer type"
                )

                raise CapabilityError(
                    f"{what} needs module hooks on {types} layers, but these blocks are not "
                    f"reached by hooks ({why}): {', '.join(bad)}"
                )


class CapabilityError(RuntimeError):
    """A requested method needs something the model or likelihood doesn't support.

    Raised before fitting, naming what is missing and what works instead.
    """


class StructureError(ValueError):
    """Shapes or structures don't fit together, such as a prior and a curvature.

    The message names both sides.
    """
