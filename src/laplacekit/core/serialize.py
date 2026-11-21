"""Save and load posteriors, priors, likelihoods: any `@pytree` object.

```python
lk.save(post, "post.pt")
post = lk.load("post.pt", model)
```

The file is a plain description, written with `torch.save` and read with
`torch.load(weights_only=True)`: the class of each object by name, its options as plain
data (numbers, strings, tuples, dicts) and its tensors. Nothing is unpickled, and loading
calls no function from the file: objects are rebuilt field by field, without their
constructors. A class must be a `@pytree` class (or one registered with `serializable`)
that is already imported, or one that laplacekit or a plugin registered under the
`laplacekit.plugins` entry point defines. A function must be registered with
`serializable`. The model is not saved: `load` takes it, as `load_state_dict` takes a
module. The file keeps a fingerprint of the module (each parameter and buffer by name,
shape, dtype and a hash of its bytes), and `load` checks the model against it.
"""

import dataclasses
import hashlib
import importlib
import os
from collections.abc import Callable, Mapping
from importlib import metadata
from typing import IO, Any, cast

import torch
from torch import Tensor, nn

from laplacekit.core._pytree import PYTREE_CLASSES
from laplacekit.core.layout import Layout
from laplacekit.core.likelihoods import Likelihood
from laplacekit.core.model import ModelFn
from laplacekit.core.posteriors import LaplacePosterior, param_versions
from laplacekit.core.subsets import Selection, select
from laplacekit.core.types import StructureError

FORMAT = "laplacekit/4"

_MAX_DEPTH = 200
"""How deeply `load` follows nested entries; deeper files are malformed."""

_FUNCTIONS: dict[str, Callable[..., Any]] = {}


def serializable[T](obj: T, /) -> T:
    """Let `lk.save` write `obj` by name, and `lk.load` find it again.

    `@pytree` classes are registered already. Use it for a module-level function stored
    in an object, such as a likelihood's `mask_fn`, or for a plain frozen dataclass.

    ```python
    @lk.core.serializable
    def pad_mask(f, inputs):
        return inputs != 0
    ```

    Args:
        obj: A module-level function or a frozen dataclass.

    Returns:
        `obj` itself, so it works as a decorator.

    Raises:
        TypeError: `obj` is a lambda, a local function, or neither a function nor a
            dataclass.
    """

    qual = getattr(obj, "__qualname__", "")
    key = f"{getattr(obj, '__module__', '')}:{qual}"

    if "<" in qual:
        raise TypeError(f"serializable needs a module-level object, got {qual}")

    if isinstance(obj, type) and dataclasses.is_dataclass(obj):
        PYTREE_CLASSES[key] = obj
    elif callable(obj):
        _FUNCTIONS[key] = obj
    else:
        raise TypeError(f"serializable takes a function or a dataclass, got {type(obj).__name__}")

    return obj


serializable(Layout)


def _trusted_roots() -> set[str]:
    roots = {"laplacekit"}

    for ep in metadata.entry_points(group="laplacekit.plugins"):
        roots.add(ep.module.split(".")[0])

    return roots


def _key(obj: Any) -> str:
    return f"{obj.__module__}:{obj.__qualname__}"


def _fingerprint(module: nn.Module | None) -> list[list[str]] | None:
    """Each parameter and buffer as [name, shape, dtype, sha256 of its bytes]."""

    if module is None:
        return None

    out = []

    for name, t in [*module.named_parameters(), *module.named_buffers()]:
        if t.is_meta:
            digest = "meta"
        else:
            raw = t.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            digest = hashlib.sha256(raw.numpy().tobytes()).hexdigest()

        out.append([name, str(tuple(t.shape)), str(t.dtype).removeprefix("torch."), digest])

    return out


def _check_fingerprint(saved: Any, module: nn.Module | None) -> None:
    """The model is the saved one, as `load_state_dict(strict=True)` checks a module."""

    if saved is None:
        return

    if not isinstance(saved, list) or not all(
        isinstance(row, list) and len(row) == 4 and all(isinstance(s, str) for s in row)
        for row in saved
    ):
        raise StructureError("load: malformed model fingerprint")

    live = {row[0]: row for row in _fingerprint(module) or []}
    file = {row[0]: row for row in saved}

    for name in file.keys() - live.keys():
        raise StructureError(f"load: the model has no {name!r}, which the saved one had")

    for name in live.keys() - file.keys():
        raise StructureError(f"load: the model has {name!r}, which the saved one did not")

    what = ("shape", "dtype", "values")

    # Shapes and dtypes first, so another architecture is named as such.
    for i, label in enumerate(what, 1):
        for name, row in file.items():
            if row[i] == live[name][i] or "meta" in (row[i], live[name][i]):
                continue

            hint = (
                "load the weights it was fitted with first (model.load_state_dict(...))"
                if label == "values"
                else "another architecture?"
            )
            raise StructureError(
                f"load: the model's {name!r} has other {label} than when saved "
                f"({row[i][:16]} saved, {live[name][i][:16]} now); {hint}"
            )


class _Writer:
    def __init__(self) -> None:
        self.tensors: dict[str, Tensor] = {}

    def __call__(self, obj: Any, where: str, in_likelihood: bool) -> Any:
        if obj is None or isinstance(obj, bool | int | float | str):
            return obj

        if isinstance(obj, Tensor):
            leaf = obj.detach().cpu()

            # torch.save writes a view's whole storage; a copy keeps the file to the leaf
            if leaf.untyped_storage().nbytes() > leaf.nbytes:
                leaf = leaf.clone()

            key = str(len(self.tensors))
            self.tensors[key] = leaf

            return {"tensor": key}

        if isinstance(obj, ModelFn):
            return {"model": _fingerprint(obj.module)}

        if isinstance(obj, torch.dtype):
            return {"dtype": str(obj).removeprefix("torch.")}

        if isinstance(obj, type) and issubclass(obj, nn.Module):
            if not obj.__module__.startswith("torch.nn."):
                raise TypeError(f"save: {where} is the module class {obj.__name__}, not torch.nn's")

            return {"module_type": _key(obj)}

        if isinstance(obj, tuple | list | frozenset):
            kind = {tuple: "tuple", list: "list", frozenset: "frozenset"}[type(obj)]
            return {kind: [self(v, f"{where}[{i}]", in_likelihood) for i, v in enumerate(obj)]}

        if isinstance(obj, Mapping):
            return {
                "dict": [
                    [self(k, where, in_likelihood), self(v, f"{where}[{k!r}]", in_likelihood)]
                    for k, v in obj.items()
                ]
            }

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            cls = type(obj)

            if PYTREE_CLASSES.get(_key(cls)) is not cls:
                raise TypeError(
                    f"save: {where} is a {cls.__name__}, which is not a registered class; "
                    "decorate it with @lk.core.pytree or lk.core.serializable"
                )

            lik = in_likelihood or where == "likelihood" or where.endswith(".likelihood")
            fields = {
                f.name: self(getattr(obj, f.name), f"{where}.{f.name}".lstrip("."), lik)
                for f in dataclasses.fields(obj)
            }

            return {"class": _key(cls), "fields": fields}

        if callable(obj) and _FUNCTIONS.get(_key(obj)) is obj:
            return {"function": _key(obj)}

        if callable(obj) and in_likelihood:
            return {"missing": where}

        raise TypeError(
            f"save: {where} is a {type(obj).__name__} that can't be written; register a "
            "module-level function with lk.core.serializable"
        )


def save(obj: object, f: str | os.PathLike[str] | IO[bytes], /) -> None:
    """Write a posterior (or any `@pytree` object) to a file, without its model.

    ```python
    lk.save(post, "post.pt")
    ```

    A function inside the likelihood that is not registered with
    `lk.core.serializable` (a lambda as `mask_fn`, say) is left out; `lk.load` then
    needs the likelihood back as `likelihood=`.

    Args:
        obj: The object to save, usually a posterior from `lk.laplace`.
        f: A path or a binary file object.

    Raises:
        TypeError: The object holds a class that is not a registered `@pytree` class or
            a function outside the likelihood that is not registered. The message names
            the field.
    """

    writer = _Writer()
    spec = writer(obj, "", False)

    torch.save({"format": FORMAT, "spec": spec, "tensors": writer.tensors}, f)


class _Reader:
    def __init__(self, model: ModelFn, tensors: dict[str, Tensor], device: torch.device) -> None:
        self.model = model
        self.tensors = tensors
        self.device = device
        self.roots = _trusted_roots()
        self.missing: list[str] = []
        self.checked = False

    def _class(self, key: str) -> type:
        """A registered class: imported in this process, or in a trusted package."""

        module, _, qual = key.partition(":")

        if key not in PYTREE_CLASSES:
            if module.split(".")[0] not in self.roots:
                raise StructureError(
                    f"load: the file names {key}, outside laplacekit and its registered "
                    "plugins; import the module that defines it first"
                )

            try:
                importlib.import_module(module)  # a trusted package registers its classes
            except ImportError as e:
                raise StructureError(f"load: the file names {key}, which can't be imported") from e

        cls = PYTREE_CLASSES.get(key)

        if cls is None or cls.__module__ != module or cls.__qualname__ != qual:
            raise StructureError(f"load: {key} is not a registered class")

        return cls

    def __call__(self, spec: Any, depth: int = 0) -> Any:
        if depth > _MAX_DEPTH:
            raise StructureError(f"load: entries nest deeper than {_MAX_DEPTH}")

        if spec is None or isinstance(spec, bool | int | float | str):
            return spec

        shape = {"class", "fields"} if isinstance(spec, dict) and "class" in spec else None

        if not isinstance(spec, dict) or (set(spec) != shape if shape else len(spec) != 1):
            raise StructureError(f"load: malformed entry {type(spec).__name__}")

        kind = "class" if shape else next(iter(spec))
        value = spec[kind]
        down = depth + 1

        if kind == "model":
            if not self.checked:
                _check_fingerprint(value, self.model.module)
                self.checked = True

            return self.model

        if kind in ("tuple", "list", "frozenset"):
            items = [self(v, down) for v in _list(value, kind)]
            return {"tuple": tuple, "list": list, "frozenset": frozenset}[kind](items)

        if kind == "dict":
            pairs = [_list(pair, "dict pair") for pair in _list(value, kind)]

            if any(len(pair) != 2 for pair in pairs):
                raise StructureError("load: malformed dict entry")

            return {self(k, down): self(v, down) for k, v in pairs}

        if kind == "class":
            return self._object(_str(value, kind), spec["fields"], down)

        value = _str(value, kind)

        if kind == "tensor":
            tensor = self.tensors.get(value)

            if not isinstance(tensor, Tensor):
                raise StructureError(f"load: the file has no tensor {value!r}")

            return tensor.to(self.device, copy=True)

        if kind == "dtype":
            dtype = getattr(torch, value, None)

            if not isinstance(dtype, torch.dtype):
                raise StructureError(f"load: unknown dtype {value!r}")

            return dtype

        if kind == "module_type":
            return _module_type(value)

        if kind == "function":
            return self._function(value)

        if kind == "missing":
            self.missing.append(value)
            return None

        raise StructureError(f"load: unknown entry {kind!r}")

    def _function(self, key: str) -> Callable[..., Any]:
        fn = _FUNCTIONS.get(key)
        module = key.partition(":")[0]

        if fn is None and module.split(".")[0] in self.roots:
            try:
                importlib.import_module(module)
            except ImportError as e:
                raise StructureError(f"load: the file names {key}, which can't be imported") from e

            fn = _FUNCTIONS.get(key)

        if fn is None:
            raise StructureError(
                f"load: the function {key} is not registered with lk.core.serializable"
            )

        return fn

    def _object(self, key: str, fields: Any, depth: int) -> Any:
        cls = self._class(key)
        names = {f.name for f in dataclasses.fields(cast(Any, cls))}

        if not isinstance(fields, dict) or set(fields) != names:
            raise StructureError(f"load: {key} has other fields than the file's")

        obj = object.__new__(cls)

        for name, sub in fields.items():
            object.__setattr__(obj, name, self(sub, depth))

        return obj


def _list(value: Any, what: str) -> list[Any]:
    if not isinstance(value, list):
        raise StructureError(f"load: malformed {what} entry, {type(value).__name__}")

    return value


def _str(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise StructureError(f"load: malformed {what} entry, {type(value).__name__}")

    return value


def _module_type(key: str) -> type[nn.Module]:
    module, _, name = key.partition(":")

    if not module.startswith("torch.nn."):
        raise StructureError(f"load: {key} is not a torch.nn module class")

    try:
        cls = getattr(importlib.import_module(module), name, None)
    except ImportError as e:
        raise StructureError(f"load: {key} is not a torch.nn module class") from e

    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise StructureError(f"load: {key} is not a torch.nn module class")

    return cls


def _selected_model(model: ModelFn, index: Mapping[str, Tensor]) -> ModelFn:
    """The model that scatters the selected entries back, from the saved indices."""

    module = model.module
    params = dict(module.named_parameters()) if module is not None else {}
    masks: dict[str, Tensor] = {}

    for name, idx in index.items():
        src = name.removesuffix("[idx]")

        if src not in params:
            raise StructureError(f"load: the model has no parameter {src!r} of the selection")

        size = params[src].numel()
        ok = (
            isinstance(idx, Tensor)
            and idx.dtype == torch.int64
            and idx.ndim == 1
            and (idx.numel() == 0 or (int(idx.min()) >= 0 and int(idx.max()) < size))
        )

        if not ok:
            raise StructureError(
                f"load: the saved index of {src!r} is not a 1-d int64 tensor of entries "
                f"in [0, {size}); another model?"
            )

        flat = torch.zeros(size, dtype=torch.bool, device=params[src].device)
        masks[src] = flat.index_fill(0, idx.to(flat.device), True).reshape(params[src].shape)

    selected, _ = select(model, Selection.from_masks(params, masks))

    return selected


def _check_weights(post: LaplacePosterior[Any, Any], module: nn.Module | None) -> None:
    """The model holds the fitted weights: names, shapes and values, as `strict=True`."""

    if module is None:
        return

    params = dict(module.named_parameters())

    for name, mean in post.mean.items():
        src = name.removesuffix("[idx]")

        if src not in params:
            raise StructureError(f"load: the model has no parameter {src!r}; another model?")

        live = params[src].detach()

        if name in post.index:
            live = live.reshape(-1)[post.index[name].to(live.device)]

        if live.shape != mean.shape:
            raise StructureError(
                f"load: {name} is {tuple(mean.shape)} in the file, {tuple(live.shape)} in the "
                "model; another architecture?"
            )

        if not mean.is_meta and not torch.equal(live.to(mean), mean):
            raise StructureError(
                f"load: the model's {name} differs from the fitted one; load the trained "
                "weights first (model.load_state_dict(...))"
            )


def _check_shapes(post: LaplacePosterior[Any, Any]) -> None:
    """The curvature acts on vectors of the mean's layout, as a fitted one does."""

    if not isinstance(post.layout, Layout) or post.layout != Layout.by_module(post.mean):
        raise StructureError("load: the saved layout does not match the saved mean")

    if not isinstance(post.n_data, Tensor) or post.n_data.numel() < 1:
        raise StructureError("load: malformed n_data")

    zeros = {k: torch.zeros_like(v) for k, v in post.mean.items()}

    try:
        vec = post._vec(zeros)
        out = post.curvature.mv(vec)
        shapes_in = [v.shape for v in post._leaves(vec)]
        shapes_out = [v.shape for v in post._leaves(out)]
    except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
        raise StructureError(f"load: the saved curvature does not fit the mean ({e})") from e

    if shapes_in != shapes_out:
        raise StructureError(
            f"load: the saved curvature maps {shapes_in} to {shapes_out}; malformed file"
        )


def _attach(obj: Any, model: ModelFn) -> Any:
    """Each posterior: the selection's model, the weight check and fresh version counters."""

    if not isinstance(obj, LaplacePosterior):
        return obj

    if not isinstance(obj.mean, Mapping) or not isinstance(obj.index, Mapping):
        raise StructureError("load: malformed posterior")

    fn = _selected_model(model, obj.index) if obj.index else model
    _check_weights(obj, model.module)
    _check_shapes(obj)

    out = obj
    object.__setattr__(out, "model", fn)
    object.__setattr__(out, "prior", _attach(obj.prior, model))
    object.__setattr__(out, "versions", param_versions(fn, obj.mean))

    return out


def load(
    f: str | os.PathLike[str] | IO[bytes],
    model: nn.Module | ModelFn,
    /,
    *,
    likelihood: Any = None,
    map_location: torch.device | str | None = None,
) -> Any:
    """Read a file written by `save`, attaching `model`.

    ```python
    model.load_state_dict(torch.load("model.pt"))
    post = lk.load("post.pt", model)
    ```

    Pass the same kind of model as the fit: a module for `lk.laplace(module, ...)`, or
    the `ModelFn` the fit took (such as `lk.transformers.causal_lm(model)`). A posterior
    over a `Selection` gets its scattering model back from the saved indices. The model
    must be the saved one: the same parameters and buffers, with the same shapes, dtypes
    and values, as `load_state_dict(strict=True)` checks a module.

    Args:
        f: A path or a binary file object.
        model: The model, a module or a `ModelFn`.
        likelihood: The likelihood to use instead of the saved one, of the same class
            and with the same tensors (a tuned sigma, say). Needed when the saved
            likelihood held a function that `save` could not write (a lambda as
            `mask_fn`, say).
        map_location: The device of the loaded tensors. `None` means the device of the
            model's first parameter (the CPU when it has none).

    Returns:
        The saved object, on `model`. A posterior records the model's current version
        counters, so later in-place changes are caught.

    Raises:
        ValueError: The file is not a `save` file.
        StructureError: The file is malformed, or names a class or function outside
            laplacekit and its registered plugins; the model differs from the saved one
            (a parameter or buffer missing, added, or of another shape, dtype or value);
            the saved likelihood lacks a function and no `likelihood` is given; or
            `likelihood` is of another class or has other tensors than the saved one.
    """

    blob: Any = torch.load(f, weights_only=True)

    if not isinstance(blob, dict) or blob.get("format") != FORMAT:
        found = blob.get("format") if isinstance(blob, dict) else type(blob).__name__
        raise ValueError(f"load: not a {FORMAT} file (format {found!r})")

    if set(blob) != {"format", "spec", "tensors"} or not isinstance(blob["tensors"], dict):
        raise ValueError(f"load: a malformed {FORMAT} file (keys {sorted(map(str, blob))})")

    fn = model if isinstance(model, ModelFn) else ModelFn.from_module(model)
    first = next(fn.module.parameters(), None) if fn.module is not None else None

    if map_location is not None:
        device = torch.device(map_location)
    else:
        device = first.device if first is not None else torch.device("cpu")

    reader = _Reader(fn, blob["tensors"], device)

    try:
        obj = reader(blob["spec"])
    except (KeyError, TypeError, AttributeError, IndexError, RecursionError) as e:
        raise StructureError(f"load: malformed file ({type(e).__name__}: {e})") from e

    if not reader.checked and fn.module is not None and _holds_model(obj):
        raise StructureError("load: malformed file, without the model's fingerprint")

    if likelihood is not None and isinstance(obj, LaplacePosterior):
        _check_likelihood(likelihood, obj.likelihood)
        object.__setattr__(obj, "likelihood", likelihood)
    elif reader.missing:
        raise StructureError(
            f"load: the saved likelihood had functions save could not write "
            f"({', '.join(reader.missing)}); pass it back: lk.load(f, model, likelihood=...)"
        )

    return _attach(obj, fn)


def _holds_model(obj: Any) -> bool:
    return isinstance(obj, LaplacePosterior)


def _check_likelihood(given: Any, saved: Any) -> None:
    """`given` replaces `saved`: the same class, and no tuned value lost."""

    if not isinstance(given, Likelihood):
        raise StructureError(f"load: likelihood= takes a Likelihood, got {type(given).__name__}")

    if type(given) is not type(saved):
        raise StructureError(
            f"load: likelihood= is a {type(given).__name__}, the saved one a "
            f"{type(saved).__name__}; pass one of the saved class"
        )

    if not dataclasses.is_dataclass(saved):
        return

    for field in dataclasses.fields(saved):
        old, new = getattr(saved, field.name), getattr(given, field.name)

        if isinstance(old, Tensor) and not (
            isinstance(new, Tensor)
            and new.shape == old.shape
            and torch.equal(new.detach().cpu().to(old.dtype), old.detach().cpu())
        ):
            raise StructureError(
                f"load: likelihood= has another {field.name} than the saved one "
                f"({old.tolist()} saved), so the saved (tuned) value would be lost; pass "
                f"copy.replace(likelihood, {field.name}=...) with the saved value"
            )
