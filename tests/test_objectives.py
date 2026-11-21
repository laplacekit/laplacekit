"""Brier score and expected calibration error, against torchmetrics and hand-computed numbers."""

import copy
import dataclasses
import json
import re
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn
from torch.distributions import Bernoulli, Categorical, Distribution, Independent

import laplacekit as lk
from laplacekit.core import Batch, CapabilityError, Requirements, StructureError

F64 = torch.float64
REF = json.loads((Path(__file__).parent / "parity" / "reference.json").read_text())["metrics"]


@dataclasses.dataclass(frozen=True)
class _Stored:
    """A predictive that returns the probabilities passed as inputs, ignoring the posterior."""

    binary: bool = False
    independent: bool = False

    @property
    def requires(self) -> Requirements:
        return Requirements()

    def __call__(self, posterior: Any, inputs: Any, /) -> Distribution:
        if not self.binary:
            return Categorical(probs=inputs)

        d = Bernoulli(probs=inputs)

        return Independent(d, 1) if self.independent else d


def _batches(probs: str, labels: str, split: int = 3) -> list[Batch]:
    """The stored arrays, in `split` batches, so pooling across batches is exercised."""

    p = torch.tensor(REF[probs], dtype=F64)
    y = torch.tensor(REF[labels])

    return [Batch(a, b) for a, b in zip(p.chunk(split), y.chunk(split), strict=True)]


@pytest.mark.parametrize("suffix", ["", "_saturated"])
def test_ece_matches_torchmetrics(suffix: str) -> None:
    batches = _batches("probs" + suffix, "labels")
    got = lk.objectives.ece(batches, predictive=_Stored())(None)

    assert got.ndim == 0
    assert not got.requires_grad

    # torchmetrics casts multi-class confidences to float32.
    torch.testing.assert_close(got.item(), REF["ece" + suffix], rtol=1e-6, atol=0.0)


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("suffix", ["", "_saturated"])
def test_binary_ece_matches_torchmetrics(suffix: str, independent: bool) -> None:
    batches = _batches("binary_probs" + suffix, "binary_labels")
    pred = _Stored(binary=True, independent=independent)
    got = lk.objectives.ece(batches, predictive=pred)(None)

    # torchmetrics' binary path stays in float64.
    torch.testing.assert_close(got.item(), REF["binary_ece" + suffix], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("suffix", ["", "_saturated"])
def test_brier_matches_reference(suffix: str) -> None:
    batches = _batches("probs" + suffix, "labels")
    got = lk.objectives.brier(batches, predictive=_Stored())(None)

    assert got.ndim == 0

    torch.testing.assert_close(got.item(), REF["brier" + suffix], rtol=1e-10, atol=0.0)


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("suffix", ["", "_saturated"])
def test_binary_brier_matches_reference(suffix: str, independent: bool) -> None:
    batches = _batches("binary_probs" + suffix, "binary_labels")
    pred = _Stored(binary=True, independent=independent)
    got = lk.objectives.brier(batches, predictive=pred)(None)

    torch.testing.assert_close(got.item(), REF["binary_brier" + suffix], rtol=1e-10, atol=0.0)


def _ece(probs: list[float], labels: list[int], n_bins: int) -> float:
    batch = Batch(torch.tensor(probs, dtype=F64), torch.tensor(labels))

    return lk.objectives.ece([batch], predictive=_Stored(binary=True), n_bins=n_bins)(None).item()


def test_ece_by_hand() -> None:
    # Two bins, [0, 0.5) and [0.5, 1), plus the bin of confidence 1.0.
    # Both in bin 0: |1/2 - 1/4| = 0.25.
    assert _ece([0.2, 0.3], [0, 1], 2) == pytest.approx(0.25, abs=1e-12)

    # The same bin sums give the same value: ece is flat along such moves.
    assert _ece([0.1, 0.4], [0, 1], 2) == pytest.approx(0.25, abs=1e-12)

    # Crossing the edge at 0.5: |0 - 0.2| / 2 + |1 - 0.6| / 2.
    assert _ece([0.2, 0.6], [0, 1], 2) == pytest.approx(0.3, abs=1e-12)

    # 0.5 is the left edge of bin 1: |0 - 0.2| / 2 + |1 - 0.5| / 2.
    assert _ece([0.2, 0.5], [0, 1], 2) == pytest.approx(0.35, abs=1e-12)


def test_confidence_one_forms_its_own_bin() -> None:
    # One bin [0, 1), and 1.0 apart: |1 - 0.5| / 2 + |0 - 1| / 2. Clamping 1.0 into the
    # last bin would give |1/2 - 3/4| = 0.25.
    assert _ece([0.5, 1.0], [1, 0], 1) == pytest.approx(0.75, abs=1e-12)


def test_ece_needs_a_bin() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        lk.objectives.ece([], n_bins=0)


def _fit(lik: Any = None, approx: Any = None) -> tuple[Any, list[Batch]]:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(2, 6), nn.Tanh(), nn.Linear(6, 3)).to(F64)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(40, 2, generator=g, dtype=F64)

    lik = lk.Categorical() if lik is None else lik

    if isinstance(lik, lk.Categorical):
        y: Tensor = torch.randint(0, 3, (40,), generator=g)
    else:
        y = torch.randn(40, 3, generator=g, dtype=F64)

    batches = [Batch(x[:20], y[:20]), Batch(x[20:], y[20:])]
    post = lk.laplace(
        model, batches, likelihood=lik, hessian_approx=approx or lk.DiagGGN(), hyper_tuner=None
    )

    return post, batches


def _prec(p: Any) -> Tensor:
    return p.prior.prior_precision


def test_brier_passes_check_objective() -> None:
    post, batches = _fit()

    lk.testing.check_objective(lk.objectives.brier(batches), post)


def test_tune_on_ece_names_grid_search() -> None:
    post, batches = _fit()
    obj = lk.objectives.ece(batches)

    with pytest.raises(ValueError, match="lk.GridSearch"):
        lk.tune(post, lk.Minimize(obj))

    values = [torch.tensor(v, dtype=F64) for v in (1e-3, 1e-1, 1e1, 1e3)]
    best = lk.tune(post, lk.GridSearch(obj, values=values))
    scores = [
        obj(dataclasses.replace(post, prior=dataclasses.replace(post.prior, prior_precision=v)))
        for v in values
    ]

    assert best.prior.prior_precision == values[int(torch.stack(scores).argmin())]


def test_ece_is_detached_on_a_posterior() -> None:
    post, batches = _fit()
    prec = post.prior.prior_precision.clone().requires_grad_()
    posterior = dataclasses.replace(
        post, prior=dataclasses.replace(post.prior, prior_precision=prec)
    )

    assert not lk.objectives.ece(batches)(posterior).requires_grad
    assert lk.objectives.brier(batches)(posterior).requires_grad


OBJECTIVES = [("brier", lk.objectives.brier), ("ece", lk.objectives.ece)]


@pytest.mark.parametrize(("name", "make"), OBJECTIVES)
def test_gaussian_predictive_refused(name: str, make: Any) -> None:
    post, batches = _fit(
        lik=copy.replace(lk.Gaussian(), sigma=torch.as_tensor(torch.tensor(0.5, dtype=F64)))
    )
    obj = make(batches)

    with pytest.raises(
        CapabilityError, match=f"{name} needs a Categorical or Bernoulli predictive, got Normal"
    ):
        obj(post)


@pytest.mark.parametrize(("name", "make"), OBJECTIVES)
def test_batches_need_targets(name: str, make: Any) -> None:
    post, batches = _fit()
    obj = make([Batch(batches[0].inputs)])

    with pytest.raises(ValueError, match=f"{name} needs batches with targets"):
        obj(post)


def test_ece_needs_a_batch() -> None:
    with pytest.raises(ValueError, match="at least one batch"):
        lk.objectives.ece([], predictive=_Stored())(None)


def _one(probs: Tensor, y: Tensor) -> list[Batch]:
    return [Batch(probs, y)]


_P = torch.tensor(REF["probs"], dtype=F64)
_Y = torch.tensor(REF["labels"])
_BP = torch.tensor(REF["binary_probs"], dtype=F64)


@pytest.mark.parametrize(("name", "make"), OBJECTIVES)
@pytest.mark.parametrize(
    ("probs", "y", "binary", "want", "got"),
    [
        # a binary classifier's [N, 1] probabilities against [N] labels
        (_BP[:, :1], _BP[:, 0].round().long(), True, "(200, 1)", "(200,)"),
        # class indices with a trailing axis
        (_P, _Y[:, None], False, "(200,)", "(200, 1)"),
        # one-hot targets where class indices go
        (_P, torch.nn.functional.one_hot(_Y, 4), False, "(200,)", "(200, 4)"),
    ],
)
def test_mis_shaped_targets_refused(
    name: str, make: Any, probs: Tensor, y: Tensor, binary: bool, want: str, got: str
) -> None:
    obj = make(_one(probs, y), predictive=_Stored(binary=binary))

    with pytest.raises(StructureError, match=rf"{name}.*{re.escape(want)}.*{re.escape(got)}"):
        obj(None)


@pytest.mark.parametrize(("name", "make"), OBJECTIVES)
@pytest.mark.parametrize(
    ("y", "match"),
    [
        (torch.full((200,), 4), r"class indices in \[0, 4\)"),
        (torch.full((200,), -1), r"class indices in \[0, 4\)"),
        (_Y.to(F64), "integer class indices"),
    ],
)
def test_bad_class_indices_refused(name: str, make: Any, y: Tensor, match: str) -> None:
    obj = make(_one(_P, y), predictive=_Stored())

    with pytest.raises(ValueError, match=f"{name} needs {match}"):
        obj(None)


@pytest.mark.parametrize(("name", "make"), OBJECTIVES)
def test_bernoulli_targets_outside_unit_interval_refused(name: str, make: Any) -> None:
    y = torch.full(_BP.shape, 2.0, dtype=F64)
    obj = make(_one(_BP, y), predictive=_Stored(binary=True))

    with pytest.raises(ValueError, match=rf"{name} needs Bernoulli targets in \[0, 1\]"):
        obj(None)


@pytest.mark.parametrize(
    ("probs", "y", "binary", "want", "got"),
    [
        (_P, _Y[:, None], False, "(200,)", "(200, 1)"),
        (_BP[:, :1], _BP[:, 0].round(), True, "(200, 1)", "(200,)"),
    ],
)
def test_predictive_nll_refuses_mis_shaped_targets(
    probs: Tensor, y: Tensor, binary: bool, want: str, got: str
) -> None:
    obj = lk.objectives.predictive_nll(_one(probs, y), predictive=_Stored(binary=binary))

    with pytest.raises(
        StructureError, match=rf"predictive_nll.*{re.escape(want)}.*{re.escape(got)}"
    ):
        obj(None)


def test_predictive_nll_refuses_mis_shaped_targets_on_a_posterior() -> None:
    post, batches = _fit()
    b = batches[0]
    targets = b.targets
    assert isinstance(targets, Tensor)

    with pytest.raises(StructureError, match=r"\(20,\).*\(20, 1\)"):
        lk.objectives.predictive_nll([Batch(b.inputs, targets[:, None])])(post)


def test_predictive_nll_accepts_matching_targets() -> None:
    obj = lk.objectives.predictive_nll(_one(_P, _Y), predictive=_Stored())
    want = -Categorical(probs=_P).log_prob(_Y).mean()

    torch.testing.assert_close(obj(None), want)


_LENGTHS = torch.tensor([5, 3, 1, 4])


def _padding(f: Tensor, inputs: Any) -> Tensor:
    """True at the first `_LENGTHS[i]` positions of sequence i: the rest is padding."""

    return torch.arange(f.shape[1]) < _LENGTHS[:, None]


@dataclasses.dataclass(frozen=True)
class _SeqPosterior:
    """A function posterior whose outputs at `inputs` (probabilities) are their logs."""

    likelihood: Any
    model: Any = None
    requires: Requirements = dataclasses.field(default_factory=Requirements)

    def at(self, inputs: Tensor, /) -> lk.core.OutputGaussian:
        return lk.core.OutputGaussian(inputs.log(), cast(Any, None))


def test_predictive_nll_masks() -> None:
    """Padded positions drop out: the mean over sequences of the summed log-probs of the
    real positions, as `core.log_prob` with inputs."""

    g = torch.Generator().manual_seed(2)
    probs = torch.rand(4, 5, 3, generator=g, dtype=F64).softmax(-1)
    y = torch.randint(0, 3, (4, 5), generator=g)
    keep = torch.tensor(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=torch.bool,
    )

    lp = torch.stack(
        [torch.stack([probs[i, t, y[i, t]].log() for t in range(5)]) for i in range(4)]
    )
    want = -(lp * keep).sum() / 4

    posterior = _SeqPosterior(lk.Categorical(mask_fn=_padding))
    got = lk.objectives.predictive_nll([Batch(probs, y)], predictive=_Stored())(posterior)

    torch.testing.assert_close(got, want)
    assert not torch.allclose(got, -lp.sum() / 4)

    # Without a mask every position counts, as before.
    plain = _SeqPosterior(lk.Categorical())
    got = lk.objectives.predictive_nll([Batch(probs, y)], predictive=_Stored())(plain)

    torch.testing.assert_close(got, -lp.sum() / 4)


def test_predictive_nll_masks_on_a_posterior() -> None:
    """A weight posterior: the mask reads the outputs at the mean and the inputs."""

    def first_positive(f: Tensor, inputs: Tensor) -> Tensor:
        return inputs[:, 0] > 0

    post, batches = _fit(lk.Categorical(mask_fn=first_positive))
    pred = lk.Linearized()

    total = torch.zeros((), dtype=F64)

    for b in batches:
        x, y = b.inputs, cast(Tensor, b.targets)
        lp = lk.predict(post, x, pred).log_prob(y)
        total = total - lp[x[:, 0] > 0].sum()

    got = lk.objectives.predictive_nll(batches, predictive=pred)(post)

    torch.testing.assert_close(got, total / 40)


@dataclasses.dataclass(frozen=True)
class _CountingPosterior(_SeqPosterior):
    """Counts its `at` calls."""

    calls: list[int] = dataclasses.field(default_factory=list)

    def at(self, inputs: Tensor, /) -> lk.core.OutputGaussian:
        self.calls.append(1)

        return super().at(inputs)


@pytest.mark.parametrize(
    "lik",
    [lk.Categorical(), lk.likelihoods.from_log_prob(lambda f, y: f.sum(-1))],
    ids=["categorical", "log-prob"],
)
def test_predictive_nll_skips_the_mask_read_without_a_mask(lik: Any) -> None:
    """A built-in without `mask_fn` can't mask, so the posterior is not evaluated for one."""

    posterior = _CountingPosterior(lik)
    probs = torch.full((4, 5, 3), 1 / 3, dtype=F64)
    y = torch.zeros(4, 5, dtype=torch.long)

    lk.objectives.predictive_nll([Batch(probs, y)], predictive=_Stored())(posterior)

    assert posterior.calls == []

    masked = _CountingPosterior(lk.Categorical(mask_fn=_padding))
    lk.objectives.predictive_nll([Batch(probs, y)], predictive=_Stored())(masked)

    assert masked.calls == [1]
