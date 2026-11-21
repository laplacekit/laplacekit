import copy
from collections import OrderedDict, UserDict
from collections.abc import Mapping

import pytest
import torch
from torch import Tensor
from torch.func import functional_call, grad, vmap

from laplacekit.core._pytree import as_pytree, pytree, static
from laplacekit.core.tree import tree_leaves, tree_map


@pytree
class Diag:
    diag: Tensor

    def add(self, other: "Diag") -> "Diag":
        return Diag(self.diag + other.diag)

    def logdet(self) -> Tensor:
        return self.diag.log().sum()


@pytree
class Isotropic:
    prior_precision: Tensor

    def precision_like(self, op: Diag) -> Diag:
        return Diag(self.prior_precision.expand_as(op.diag))


@pytree
class WeightPosterior:
    mean: Mapping[str, Tensor]
    curvature: Diag
    prior: Isotropic
    n_data: Tensor
    name: str = static(default="post")

    def precision(self) -> Diag:
        return Diag(self.n_data * self.curvature.diag).add(
            self.prior.precision_like(self.curvature)
        )

    def neg_logdet(self) -> Tensor:
        return -0.5 * self.precision().logdet()


P = 5


def make_post() -> WeightPosterior:
    torch.manual_seed(0)
    return WeightPosterior(
        mean={"w": torch.randn(P)},
        curvature=Diag(torch.rand(P) + 0.1),
        prior=Isotropic(torch.tensor(1.0)),
        n_data=torch.tensor(100.0),
    )


def test_copy_replace() -> None:
    post = make_post()
    new = copy.replace(post, prior=Isotropic(torch.tensor(2.0)))

    assert new.prior.prior_precision.item() == 2.0
    assert post.prior.prior_precision.item() == 1.0
    assert new.name == "post"


def test_leaves_exclude_statics() -> None:
    leaves = tree_leaves(make_post())
    assert [tuple(x.shape) for x in leaves] == [(P,), (P,), (), ()]


def test_grad_wrt_leaf() -> None:
    post = make_post()
    g = grad(lambda pr: copy.replace(post, prior=Isotropic(pr)).neg_logdet())(torch.tensor(1.0))

    assert g.shape == () and torch.isfinite(g)


def test_grad_wrt_whole_object() -> None:
    g = grad(lambda p: p.neg_logdet())(make_post())

    assert isinstance(g, WeightPosterior)
    assert g.curvature.diag.shape == (P,)


def test_vmap_over_leaf() -> None:
    post = make_post()
    out = vmap(lambda pr: copy.replace(post, prior=Isotropic(pr)).neg_logdet())(
        torch.tensor([0.1, 1.0, 10.0])
    )

    assert out.shape == (3,)


def test_vmap_over_stacked_objects() -> None:
    post = make_post()
    stacked = tree_map(lambda x: torch.stack([x, x * 2]), post)
    out = vmap(lambda p: p.neg_logdet())(stacked)

    assert out.shape == (2,)


def _setup() -> tuple[dict[str, Tensor], Tensor, "torch.nn.Module"]:
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    params = {k: v.detach() for k, v in model.named_parameters()}

    return params, torch.randn(32, 3), model


def _val_loss_fn() -> tuple["object", dict[str, Tensor]]:
    params, X, model = _setup()

    def fit_diag(params: dict[str, Tensor]) -> Diag:
        def f(p: dict[str, Tensor], x: Tensor) -> Tensor:
            return functional_call(model, p, (x.unsqueeze(0),)).squeeze()

        per_ex = vmap(grad(f), in_dims=(None, 0))(params, X)
        flat = torch.cat([g.flatten(1) for g in per_ex.values()], dim=1)

        return Diag((flat**2).mean(0))

    def val_loss(log_prec: Tensor, params: dict[str, Tensor]) -> Tensor:
        flat_mean = torch.cat([v.flatten() for v in params.values()])
        p = WeightPosterior(
            {"all": flat_mean}, fit_diag(params), Isotropic(log_prec.exp()), torch.tensor(32.0)
        )

        return p.neg_logdet() + 0.5 * p.prior.prior_precision * (flat_mean**2).sum()

    return val_loss, params


def test_grad_through_diag_fit_wrt_log_prior_precision() -> None:
    val_loss, params = _val_loss_fn()
    g = grad(val_loss)(torch.tensor(0.0), params)

    assert g.shape == () and torch.isfinite(g)


def test_grad_through_diag_fit_wrt_map_params() -> None:
    val_loss, params = _val_loss_fn()
    gs = grad(val_loss, argnums=1)(torch.tensor(0.0), params)

    assert set(gs) == set(params)
    assert all(torch.isfinite(g).all() for g in gs.values())


class _Plain(dict):
    """A dict subclass torch's pytree does not know: a leaf to it."""


def test_as_pytree_turns_leaf_mappings_into_dicts_at_any_depth() -> None:
    a, b, c = torch.ones(2), torch.zeros(2), torch.arange(2.0)
    tree = (UserDict(a=a, inner=_Plain(b=b)), [OrderedDict(c=UserDict(c=c))])

    # The reference: torch's own pytree sees one leaf per unknown mapping.
    assert len(tree_leaves(tree)) == 2

    out = as_pytree(tree)

    assert type(out) is tuple and type(out[0]) is dict and type(out[0]["inner"]) is dict
    assert type(out[1][0]) is OrderedDict and type(out[1][0]["c"]) is dict
    assert [x.data_ptr() for x in tree_leaves(out)] == [a.data_ptr(), b.data_ptr(), c.data_ptr()]

    # vmap now finds the tensors.
    doubled = vmap(lambda t: tree_map(lambda x: 2 * x, t))(out)

    torch.testing.assert_close(doubled[0]["inner"]["b"], 2 * b)


def test_as_pytree_returns_a_plain_tree_as_it_is() -> None:
    tree = {"a": torch.ones(2), "b": (torch.zeros(1),)}

    assert as_pytree(tree) is tree
    assert as_pytree(None) is None


def test_static_kw_only_fields() -> None:
    """A keyword-only static field keeps its place in the treedef, and replace and
    equality see it."""

    import dataclasses

    @pytree
    class Holder:
        x: float
        tag: str = static(default="a")
        seed: int = static(default=0, kw_only=True)

    h = Holder(1.0, "b", seed=7)
    back = tree_map(lambda t: t + 0, h)  # flatten, then unflatten

    assert [f.name for f in dataclasses.fields(Holder) if f.kw_only] == ["seed"]
    assert back.seed == 7 and back.tag == "b" and back.x == 1.0
    assert copy.replace(h, tag="c").seed == 7
    assert copy.replace(h, seed=8).seed == 8
    assert Holder(h.x, "b", seed=7) == h and Holder(h.x, "b", seed=8) != h

    with pytest.raises(TypeError):
        Holder(h.x, "b", 7)  # ty: ignore[too-many-positional-arguments]
