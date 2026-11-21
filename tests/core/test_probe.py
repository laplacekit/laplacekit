import copy
import re
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd.function import once_differentiable
from torch.distributions import Distribution, Normal

from laplacekit.core import (
    Batch,
    Capabilities,
    CapabilityError,
    HasOutputCheck,
    ModelFn,
    Requirements,
    Selection,
    StructureError,
    fit_curvature,
    is_transformed,
    probe,
)
from laplacekit.curvature import DenseGGN


class Gauss:
    def log_prob(self, f: Tensor, y: Tensor, /) -> Tensor:
        return self.distribution(f).log_prob(y)

    def distribution(self, f: Tensor, /) -> Distribution:
        return Normal(f, 1.0)


def _batch(shape: tuple[int, ...] = (4, 3)) -> Batch:
    return Batch(inputs=torch.randn(*shape), targets=torch.randn(shape[0], 2))


def test_mlp_supports_everything() -> None:
    m = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    caps = probe(ModelFn.from_module(m), Gauss(), _batch())

    assert caps.vjp and caps.vjp_twice and caps.jvp and caps.vmap
    assert caps.module_order == ("0", "2")
    assert caps.hooked == frozenset({"0", "2"})
    assert not caps.failures


def test_inplace_breaks_vmap() -> None:
    class InPlace(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 2)

        def forward(self, x: Tensor) -> Tensor:
            h = self.lin.bias * 1.0  # not batched under vmap
            h.add_(self.lin(x).sum(0))  # in-place write of a per-example value

            return h.expand(x.shape[0], 2)

    caps = probe(ModelFn.from_module(InPlace()), Gauss(), _batch())

    assert caps.vjp
    assert not caps.vmap
    assert caps.failures["vmap"].startswith("RuntimeError")
    assert "broadcast shape" in caps.failures["vmap"]
    unlabeled = probe(ModelFn.from_module(InPlace()), Gauss(), Batch(inputs=torch.randn(4, 3)))

    assert not unlabeled.vmap


class Attn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, x: Tensor) -> Tensor:
        h = self.lin(x).reshape(x.shape[0], 1, 2)
        return F.scaled_dot_product_attention(h, h, h).reshape(x.shape[0], 2)


@pytest.mark.filterwarnings("ignore:There is a performance drop")
def test_sdpa_has_no_jvp() -> None:
    caps = probe(ModelFn.from_module(Attn()), Gauss(), _batch())

    assert not caps.jvp
    assert "jvp" in caps.failures

    with pytest.raises(CapabilityError, match=r"(?s)Linearized.*jvp"):
        caps.check(Requirements(jvp=True), "Linearized(strategy='forward-mode')")


@pytest.mark.filterwarnings("ignore:There is a performance drop")
def test_sdpa_has_no_double_vjp() -> None:
    """SDPA's backward has no derivative, so J t from two VJPs fails while one VJP works."""

    caps = probe(ModelFn.from_module(Attn()), Gauss(), _batch())

    assert caps.vjp
    assert not caps.vjp_twice
    assert "not implemented" in caps.failures["vjp_twice"]

    with pytest.raises(CapabilityError, match="vjp_twice"):
        caps.check(Requirements(vjp_twice=True), "sampled outputs")


class _Square(torch.autograd.Function):
    """x^2 with a backward that is not differentiable itself."""

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return x * x

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Tensor], output: Tensor) -> None:
        ctx.save_for_backward(inputs[0])

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, g: Tensor) -> Tensor:
        (x,) = ctx.saved_tensors

        return 2 * x * g


class _OnceDifferentiable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, x: Tensor) -> Tensor:
        return _Square.apply(self.lin(x))


def test_once_differentiable_function_has_no_double_vjp() -> None:
    """Its double VJP silently returns zeros; the probe compares it with the plain VJP."""

    caps = probe(ModelFn.from_module(_OnceDifferentiable()), Gauss(), _batch())

    assert caps.vjp
    assert not caps.vjp_twice
    assert "vjp_twice" in caps.failures


def test_mlp_has_double_vjp() -> None:
    m = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))

    assert probe(ModelFn.from_module(m), Gauss(), _batch()).vjp_twice
    assert not probe(ModelFn.from_module(m), Gauss(), _batch(), params={}).vjp_twice


@pytest.mark.filterwarnings("ignore:There is a performance drop")
def test_one_of_satisfied_by_vjp() -> None:
    caps = probe(ModelFn.from_module(Attn()), Gauss(), _batch())
    req = Requirements(one_of=(Requirements(jvp=True), Requirements(vjp=True)))

    assert req.satisfies(caps)
    caps.check(req, "x")


def test_probe_leaves_module_untouched() -> None:
    m = nn.Sequential(nn.Linear(3, 5), nn.Linear(5, 2))
    m[1].weight.requires_grad_(False)
    before = {k: (v.detach().clone(), v.requires_grad) for k, v in m.named_parameters()}
    probe(ModelFn.from_module(m), Gauss(), _batch())

    for k, v in m.named_parameters():
        assert torch.equal(v, before[k][0])
        assert v.requires_grad == before[k][1]

    assert not any(h for mod in m.modules() for h in mod._forward_hooks)


def test_unlabeled_batch_still_probes_vmap() -> None:
    m = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    caps = probe(ModelFn.from_module(m), Gauss(), Batch(inputs=torch.randn(4, 3)))

    assert caps.vmap
    assert "vmap" not in caps.failures


def test_hooks_removed_when_order_pass_raises() -> None:
    class Flaky(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 2)
            self.calls = 0

        def forward(self, x: Tensor) -> Tensor:
            out = self.lin(x)

            if not torch.is_grad_enabled() and not torch._C._functorch.is_functorch_wrapped_tensor(
                x
            ):
                raise RuntimeError("boom")

            return out

    m = Flaky()
    caps = probe(ModelFn.from_module(m), Gauss(), _batch())

    assert caps.failures["module_order"].startswith("RuntimeError: boom")
    assert not any(mod._forward_hooks for mod in m.modules())


def test_probe_emits_no_jit_future_warning() -> None:
    import warnings

    m = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe(ModelFn.from_module(m), Gauss(), _batch())

    assert not [w for w in caught if issubclass(w.category, FutureWarning)]


class _FunctionalProj(nn.Module):
    """Uses `proj.weight` through `F.linear`, so no hook on `proj` fires."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)
        self.proj = nn.Linear(4, 2)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(torch.tanh(self.fc(x)), self.proj.weight, self.proj.bias)


def test_hooked_skips_layer_used_without_call() -> None:
    caps = probe(ModelFn.from_module(_FunctionalProj()), Gauss(), _batch())

    assert caps.hooked == frozenset({"fc"})
    assert "proj" not in caps.hooked

    req = Requirements(hooks=(nn.Linear,))
    caps.check(req, "KFAC", blocks=["fc"])

    with pytest.raises(CapabilityError, match="KFAC") as e:
        caps.check(req, "KFAC", blocks=["fc", "proj"])

    assert "proj" in str(e.value) and "fc" not in str(e.value).replace("KFAC", "")


def test_hooked_only_linear_and_conv_with_custom_param() -> None:
    class Mixed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 2)
            self.scale = nn.Parameter(torch.ones(2))
            self.norm = nn.LayerNorm(2)

        def forward(self, x: Tensor) -> Tensor:
            return self.norm(self.lin(x)) * self.scale

    caps = probe(ModelFn.from_module(Mixed()), Gauss(), _batch())

    assert caps.hooked == frozenset({"lin"})

    with pytest.raises(CapabilityError) as e:
        caps.check(Requirements(hooks=(nn.Linear,)), "KFAC", blocks=["lin", "scale", "norm"])

    assert "scale" in str(e.value) and "norm" in str(e.value)


def test_hooked_conv2d_and_type_filter() -> None:
    m = nn.Sequential(nn.Conv2d(1, 2, 3), nn.Flatten(), nn.Linear(2 * 2 * 2, 2))
    caps = probe(ModelFn.from_module(m), Gauss(), _batch((4, 1, 4, 4)))

    assert caps.hooked == frozenset({"0", "2"})

    caps.check(Requirements(hooks=(nn.Linear, nn.Conv2d)), "KFAC", blocks=["0", "2"])

    with pytest.raises(CapabilityError, match="0"):
        caps.check(Requirements(hooks=(nn.Linear,)), "KFAC", blocks=["0", "2"])


def test_check_without_blocks_ignores_hooks() -> None:
    caps = probe(ModelFn.from_module(_FunctionalProj()), Gauss(), _batch())

    caps.check(Requirements(hooks=(nn.Linear,)), "KFAC")


class _NoDerivatives(torch.autograd.Function):
    """Identity whose backward and forward-mode derivatives raise."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x: Tensor) -> Tensor:
        return x.clone()

    @staticmethod
    def setup_context(ctx: object, inputs: object, output: object) -> None:
        pass

    @staticmethod
    def backward(ctx: object, g: Tensor) -> Tensor:
        raise RuntimeError("frozen body: no backward")

    @staticmethod
    def jvp(ctx: object, t: Tensor) -> Tensor:
        raise RuntimeError("frozen body: no forward mode")


class _FrozenBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Linear(3, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: Tensor) -> Tensor:
        h = _NoDerivatives.apply(self.body(x))

        return self.head(h)


def test_probe_subset_skips_frozen() -> None:
    m = _FrozenBody()
    fn = ModelFn.from_module(m)
    head = {k: v for k, v in m.named_parameters() if k.startswith("head.")}

    sub = probe(fn, Gauss(), _batch(), params=head)

    assert sub.vjp and sub.jvp
    assert sub.probed == frozenset({"head.weight", "head.bias"})

    every = probe(fn, Gauss(), _batch())

    assert not every.vjp and not every.jvp
    assert "frozen body" in every.failures["vjp"]
    assert "frozen body" in every.failures["jvp"]
    assert every.probed is None


def test_probe_empty_params_order_only() -> None:
    m = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    full = probe(ModelFn.from_module(m), Gauss(), _batch())
    caps = probe(ModelFn.from_module(m), Gauss(), _batch(), params={})

    assert caps.module_order == full.module_order == ("0", "2")
    assert caps.event_size == full.event_size == 2
    assert caps.hooked_types == full.hooked_types
    assert not caps.vjp and not caps.jvp and not caps.vmap

    for name in ("vjp", "jvp", "vmap"):
        assert caps.failures[name] == "not probed"

    assert caps.probed == frozenset()


def test_fit_curvature_rejects_caps_for_other_params() -> None:
    m = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    fn = ModelFn.from_module(m)
    params = {k: v.detach() for k, v in m.named_parameters()}
    head = {k: v for k, v in params.items() if k.startswith("2.")}
    caps = probe(fn, Gauss(), _batch(), params=head)

    fit_curvature(fn, Gauss(), head, DenseGGN(fisher="mc"), capabilities=caps)

    with pytest.raises(CapabilityError, match="probed for other parameters"):
        fit_curvature(fn, Gauss(), params, DenseGGN(fisher="mc"), capabilities=caps)

    with pytest.raises(CapabilityError, match="probed for other parameters"):
        fit_curvature(
            fn,
            Gauss(),
            head,
            DenseGGN(fisher="mc"),
            capabilities=probe(fn, Gauss(), _batch(), params={}),
        )


def test_probe_records_root_parameters_as_root_block() -> None:
    caps = probe(ModelFn.from_module(nn.Linear(3, 2)), Gauss(), _batch())

    assert caps.module_order == ("<root>",)


class _RootParam(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(3))
        self.head = nn.Linear(3, 2)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x + self.pos)


def test_root_parameters_come_first_in_the_order() -> None:
    caps = probe(ModelFn.from_module(_RootParam()), Gauss(), _batch())

    assert caps.module_order == ("<root>", "head")
    assert caps.hooked == frozenset({"head"})


def test_probe_records_selected_and_transforms() -> None:
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    params = dict(m.named_parameters())
    masks = {"2.weight": torch.rand(2, 5) < 0.5, "0.bias": torch.rand(5) < 0.5}
    masks["2.weight"][0, 0] = masks["0.bias"][0] = True
    sel = Selection.from_masks(params, masks)

    caps = probe(sel.apply(ModelFn.from_module(m)), Gauss(), _batch(), params=sel.params)

    assert caps.vjp and caps.jvp and caps.vmap
    assert caps.selected == frozenset({"0.bias[idx]", "2.weight[idx]"})
    assert caps.selected_blocks == frozenset({"0", "2"})

    # The module's own parameters, or no module, select nothing.
    own = probe(ModelFn.from_module(m), Gauss(), _batch(), params={"2.weight": params["2.weight"]})

    assert own.selected == frozenset()

    def forward(p, x):
        return x @ p["w"]

    bare = probe(ModelFn(forward), Gauss(), _batch(), params={"w": torch.randn(3, 2)})

    assert bare.selected == frozenset()


def test_probe_does_not_count_by_hand_extra_parameters_as_selected() -> None:
    """A by-hand `ModelFn` with a parameter the module doesn't own (a temperature) selects
    nothing, so a hook method is not told about index selections."""

    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))

    def forward(p, x):
        return torch.func.functional_call(m, {k: v for k, v in p.items() if k != "T"}, x) / p["T"]

    params = {**dict(m.named_parameters()), "T": torch.tensor(2.0)}
    caps = probe(ModelFn(forward, m), Gauss(), _batch(), params=params)

    assert caps.selected == frozenset()

    with pytest.raises(CapabilityError) as info:
        caps.check(Requirements(hooks=(nn.Linear,)), "KFAC", blocks={"<root>": ["T"]})

    assert "Selection" not in str(info.value)


def test_check_refuses_hooks_on_selected_blocks() -> None:
    caps = Capabilities(vjp=True, vmap=True, selected=frozenset({"2.weight[idx]", "w[idx]"}))
    hooks = Requirements(hooks=(nn.Linear,))

    for blocks in (None, {"2": ["2.weight[idx]"]}):
        with pytest.raises(
            CapabilityError,
            match=r"KFAC needs module hooks.*\['2', '<root>'\] hold index selections",
        ):
            caps.check(hooks, "KFAC", blocks=blocks)

    # Hook-free methods, and hooks without selections, pass.
    caps.check(Requirements(vjp=True, vmap=True), "Diag", blocks={"2": ["2.weight[idx]"]})
    Capabilities().check(hooks, "KFAC")


class _Counted(nn.Module):
    """A small MLP that counts its forwards, plain or under a transform."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
        self.calls = 0

    def forward(self, x: Tensor) -> Tensor:
        self.calls += 1

        return self.net(x)


class Checked(Gauss):
    """A plugin-style likelihood with an output check: public API only."""

    def __init__(self, width: int = 2) -> None:
        self.width = width
        self.seen: list[tuple[tuple[int, ...], bool, bool]] = []

    def check_outputs(self, f: Tensor, /) -> None:
        wrapped = torch._C._functorch.is_functorch_wrapped_tensor(f)
        self.seen.append((tuple(f.shape), wrapped, torch.is_grad_enabled()))

        if f.shape[-1] != self.width:
            raise StructureError(f"Checked needs {self.width} outputs, got {f.shape[-1]}")


def test_output_check_is_a_core_protocol() -> None:
    assert isinstance(Checked(), HasOutputCheck)
    assert not isinstance(Gauss(), HasOutputCheck)


@pytest.mark.parametrize("targets", [True, False], ids=["targets", "no-targets"])
@pytest.mark.parametrize("params", ["all", "head", "none"])
def test_probe_checks_the_plain_forward_output(targets: bool, params: str) -> None:
    """Once per probe, on the order pass's output: two examples, no transform, no grad."""

    m = _Counted()
    fn = ModelFn.from_module(m)
    batch = _batch() if targets else Batch(inputs=torch.randn(4, 3))
    chosen = {
        "all": None,
        "head": {k: v.detach() for k, v in m.named_parameters() if k.startswith("net.2.")},
        "none": {},
    }[params]
    lik = Checked()

    caps = probe(fn, lik, batch, params=chosen)

    assert lik.seen == [((2, 2), False, False)]
    assert params == "none" or caps.vmap


@pytest.mark.parametrize("targets", [True, False], ids=["targets", "no-targets"])
@pytest.mark.parametrize("error", [StructureError, ValueError])
def test_probe_raises_the_output_check_before_any_transform(targets: bool, error: type) -> None:
    """The check's error propagates, whatever its type, after the order pass's one forward."""

    class Refusing(Gauss):
        def check_outputs(self, f: Tensor, /) -> None:
            raise error("bad outputs")

    m = _Counted()
    batch = _batch() if targets else Batch(inputs=torch.randn(4, 3))

    with pytest.raises(error, match="bad outputs"):
        probe(ModelFn.from_module(m), Refusing(), batch)

    assert m.calls == 1


def test_output_check_adds_no_forward_to_the_probe() -> None:
    plain, checked = _Counted(), _Counted()
    checked.load_state_dict(plain.state_dict())

    for targets in (True, False):
        batch = _batch() if targets else Batch(inputs=torch.randn(4, 3))
        plain.calls = checked.calls = 0

        probe(ModelFn.from_module(plain), Gauss(), batch)
        probe(ModelFn.from_module(checked), Checked(), batch)

        assert checked.calls == plain.calls > 0


def test_probe_checks_each_leaf_of_a_likelihood_tree() -> None:
    class Two(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a, self.b = nn.Linear(3, 2), nn.Linear(3, 4)

        def forward(self, x: Tensor) -> dict[str, Tensor]:
            return {"a": self.a(x), "b": self.b(x)}

    a, b = Checked(2), Checked(3)
    batch = Batch(inputs=torch.randn(4, 3))

    with pytest.raises(StructureError, match="Checked needs 3 outputs, got 4"):
        probe(ModelFn.from_module(Two()), {"a": a, "b": b}, batch)

    assert a.seen == [((2, 2), False, False)]
    assert b.seen == [((2, 4), False, False)]


def test_probe_skips_the_check_when_the_forward_fails() -> None:
    """A failed order pass is a recorded failure; there is no output to check."""

    class Broken(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(3, 2)

        def forward(self, x: Tensor) -> Tensor:
            raise RuntimeError("boom")

    lik = Checked()
    caps = probe(ModelFn.from_module(Broken()), lik, _batch())

    assert caps.failures["event_size"].startswith("RuntimeError: boom")
    assert lik.seen == []


@pytest.mark.parametrize("targets", [True, False], ids=["targets", "no-targets"])
def test_probe_refuses_a_likelihood_tree_over_plain_outputs(targets: bool) -> None:
    m = _Counted()
    batch = _batch() if targets else Batch(inputs=torch.randn(4, 3))

    with pytest.raises(StructureError, match=r"keys \['a'\].*got Tensor"):
        probe(ModelFn.from_module(m), {"a": Gauss()}, batch)

    assert m.calls == 1


def test_probe_refuses_a_likelihood_tree_with_a_missing_output_key() -> None:
    class One(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(3, 2)

        def forward(self, x: Tensor) -> dict[str, Tensor]:
            return {"a": self.a(x), "extra": x}

    with pytest.raises(StructureError, match=r"'b'.*outputs have keys \['a', 'extra'\]"):
        probe(ModelFn.from_module(One()), {"a": Gauss(), "b": Gauss()}, Batch(torch.randn(4, 3)))

    # Extra output keys without a likelihood are fine.
    probe(ModelFn.from_module(One()), {"a": Checked()}, Batch(torch.randn(4, 3)))


class _DictOut(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(3, 2)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        return {"a": self.a(x)}


@pytest.mark.parametrize("lik_kind", ["Gaussian", "Categorical", "Bernoulli", "plugin"])
def test_probe_dict_outputs_message(lik_kind: str) -> None:
    """A dict reaching a tensor likelihood is the caller's mistake, named as such, not a
    model that can't be vmapped."""

    from laplacekit.likelihoods import Bernoulli, Categorical, Gaussian

    lik: Any = {
        "Gaussian": Gaussian(),
        "Categorical": Categorical(),
        "Bernoulli": Bernoulli(),
        "plugin": Gauss(),
    }[lik_kind]
    y = torch.zeros(4, dtype=torch.long) if lik_kind == "Categorical" else torch.zeros(4, 2)
    name = "Gauss" if lik_kind == "plugin" else lik_kind

    # torch's Normal refuses a dict with a ValueError; it keeps its type, with a note.
    error = ValueError if lik_kind == "plugin" else StructureError

    with pytest.raises(error) as info:
        probe(ModelFn.from_module(_DictOut()), lik, Batch(torch.randn(4, 3), y))

    text = "\n".join([str(info.value), *getattr(info.value, "__notes__", [])])

    assert re.search(rf"{name}.*outputs of type dict", text)


def test_probe_dict_outputs_message_from_the_output_check() -> None:
    """An output check that trips over a dict's missing tensor attributes says so too."""

    from laplacekit.likelihoods import Gaussian

    lik = copy.replace(Gaussian(), sigma=torch.as_tensor(torch.ones(2)))

    with pytest.raises(StructureError, match=r"Gaussian.*outputs of type dict"):
        probe(ModelFn.from_module(_DictOut()), lik, Batch(torch.randn(4, 3)))


def test_probe_mapping_likelihood_missing_key() -> None:
    """Targets of a likelihood mapping need its keys, as the outputs do."""

    class Two(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a, self.b = nn.Linear(3, 2), nn.Linear(3, 2)

        def forward(self, x: Tensor) -> dict[str, Tensor]:
            return {"a": self.a(x), "b": self.b(x)}

    lik = {"a": Gauss(), "b": Gauss()}
    fn = ModelFn.from_module(Two())

    with pytest.raises(StructureError, match=r"'b'.*targets have keys \['a'\]"):
        probe(fn, lik, Batch(torch.randn(4, 3), {"a": torch.randn(4, 2)}))

    with pytest.raises(StructureError, match=r"keys \['a', 'b'\].*targets.*got Tensor"):
        probe(fn, lik, Batch(torch.randn(4, 3), torch.randn(4, 2)))

    # Complete targets probe every transform.
    full = {"a": torch.randn(4, 2), "b": torch.randn(4, 2)}
    caps = probe(fn, lik, Batch(torch.randn(4, 3), full))

    assert caps.vmap and not caps.failures


class _DictGauss:
    """A plugin likelihood over dict outputs `{"a": f}`; refuses targets above `limit`."""

    def __init__(self, limit: float = float("inf")) -> None:
        self.limit = limit

    def log_prob(self, f: dict[str, Tensor], y: Tensor, /) -> Tensor:
        if not is_transformed(y) and bool((y.abs() > self.limit).any()):
            raise ValueError(f"_DictGauss targets must lie within {self.limit}")

        return Normal(f["a"], 1.0).log_prob(y).sum(-1)


def test_probe_accepts_a_likelihood_over_dict_outputs() -> None:
    caps = probe(ModelFn.from_module(_DictOut()), _DictGauss(), _batch())

    assert caps.vjp and caps.vmap and not caps.failures


def test_probe_surfaces_a_value_error_from_a_dict_output_likelihood() -> None:
    """A failure that isn't about the output type is not reworded as one."""

    batch = Batch(torch.randn(4, 3), 100 * torch.ones(4, 2))

    with pytest.raises(ValueError, match="within 10") as info:
        probe(ModelFn.from_module(_DictOut()), _DictGauss(limit=10.0), batch)

    assert "output type" not in str(info.value)
    assert info.value.__notes__ == [
        "raised by _DictGauss.log_prob in the probe, on the model's outputs of type dict"
    ]
