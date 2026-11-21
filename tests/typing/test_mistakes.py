"""Mistakes the type checker must catch. Each marked line must fail, and no other line may."""

import typing

import torch
from _fixtures import train_loader, train_my_resnet, val_batches
from torch.distributions import Distribution

import laplacekit as lk
from laplacekit.core import BlockDiag, Inputs, Joint, Likelihood, Outputs
from laplacekit.core import Dense as DenseOp
from laplacekit.core import Diag as DiagOp

model = train_my_resnet()
fn = lk.core.ModelFn.from_module(model)
lik = lk.Categorical()
params: lk.core.Params = {}
fit = lk.core.fit_curvature(fn, lik, params, lk.DiagGGN())
stat = fit.init()
post = fit.posterior(
    fit.finalize(stat), prior=lk.priors.Isotropic(torch.tensor(1.0)), n_data=torch.tensor(1.0)
)
obj = lk.objectives.predictive_nll(val_batches)

lk.tune(post, obj)  # EXPECTED ERROR
lk.tune(post, lk.GridSearch(obj, values="1e-3"))  # EXPECTED ERROR
iso = lk.priors.Isotropic(torch.tensor(1.0))
lk.laplace(model, train_loader, likelihood=lik, hessian_approx=iso)  # EXPECTED ERROR
lk.Linearized(strategy="jvq")  # EXPECTED ERROR
lk.laplace(model, train_loader, likelihood=object())  # EXPECTED ERROR


@lk.core.pytree
class SomeFunctionPosterior:
    model: lk.core.ModelFn = lk.core.static()
    likelihood: Likelihood = lk.core.static()

    @property
    def requires(self) -> lk.core.Requirements:
        return lk.core.Requirements()

    def at(self, inputs: Inputs, /) -> lk.core.OutputGaussian:
        raise NotImplementedError


gp = SomeFunctionPosterior(fn, lik)
_ok: Distribution = lk.predict(gp, torch.zeros(1), lk.Linearized())  # fine: any posterior
lk.predict(gp, torch.zeros(1), lk.WeightSamples())  # EXPECTED ERROR
lk.laplace(model, train_loader, likelihood=lik, prior=gp)  # EXPECTED ERROR
lk.priors.KronSum.of(gp)  # EXPECTED ERROR
lk.priors.KronSum(post)  # EXPECTED ERROR


@lk.core.pytree
class TwoPass:
    @property
    def requires(self) -> lk.core.Requirements:
        return lk.core.Requirements()

    def plan(self, params: lk.core.Params, ctx: lk.core.FitContext, /) -> lk.core.Pass[DiagOp]:
        raise NotImplementedError


staged = lk.core.fit_curvature(fn, lik, params, TwoPass())
staged.init()  # EXPECTED ERROR

# a wrong continuation: the first pass yields a BlockDiag[Diag], `then` expects a str
lk.core.chain(lk.DiagGGN(), then=lambda s: lk.core.Done(s.upper()))  # EXPECTED ERROR

# a rule that maps to something other than a weight approximation
lk.per_block({"*": object()})  # EXPECTED ERROR

# a prior where the subset goes, and a prior where the objective goes
lk.laplace(model, train_loader, likelihood=lik, subset=iso)  # EXPECTED ERROR
lk.laplace(model, train_loader, likelihood=lik, hyper_tuner=iso)  # EXPECTED ERROR
lk.objectives.predictive_nll(val_batches, predictive=iso)  # EXPECTED ERROR

# a Selection is not Params: it goes through `lk.core.select` first
sel = lk.core.Selection.from_masks(params, {})
lk.core.fit_curvature(fn, lik, sel, lk.DiagGGN())  # EXPECTED ERROR

# a Selection is not a sequence of parameters
lk.core.Selection.from_masks(params, {})[0]  # EXPECTED ERROR

# a subnetwork size is a count, not a fraction; variance needs a weight-space Gaussian
lk.subnet.largest_magnitude(0.1)  # EXPECTED ERROR
lk.subnet.largest_variance(gp, 10)  # EXPECTED ERROR
lk.subnet.largest_variance(10, post)  # EXPECTED ERROR


# a likelihood needs log_prob; a distribution alone is not enough
class OnlyDistribution:
    def distribution(self, f: Outputs, /) -> Distribution:
        raise NotImplementedError


lk.laplace(model, train_loader, likelihood=OnlyDistribution())  # EXPECTED ERROR
lk.DiagGGN(fisher="observed")  # EXPECTED ERROR
lk.DenseGGN(joint="full")  # EXPECTED ERROR
lk.DenseHessian(observed=True)  # EXPECTED ERROR
lk.KFAC(fisher="ef")  # EXPECTED ERROR
lk.Bridge(fallback="softmax")  # EXPECTED ERROR
lk.LowRankGGN()  # EXPECTED ERROR
lk.LowRankGGN(rank=8, fisher="ggn")  # EXPECTED ERROR

# an approximation named by a string, and `last_token` called where it should be passed
lk.transformers.laplace_lora(model, [], hessian_approx="ef")  # EXPECTED ERROR
prompt = torch.zeros(1)
lk.predict(post, prompt, select=lk.transformers.last_token(prompt))  # EXPECTED ERROR

# a task named by a string, a size as a word, and a model where the tokenizer goes
lk.transformers.laplace_lora(model, [], task="classification")  # EXPECTED ERROR
lk.transformers.laplace_lora(model, [], n_data="all")  # EXPECTED ERROR
lk.transformers.answer_logits(model, [" True"])  # EXPECTED ERROR

# `size` is a count, not a word
lk.diffusers.laplace(model, object(), [], n_data="all")  # EXPECTED ERROR

# a joint operator added to a single-block one, and a joint over block-keyed vectors
joint = Joint(DenseOp(torch.eye(2)), (("fc", 2),))
joint.add_same(DenseOp(torch.eye(2)))  # EXPECTED ERROR
Joint(BlockDiag({"fc": DenseOp(torch.eye(2))}), (("fc", 2),))  # EXPECTED ERROR

# output samples: a kind named after laplace-torch's pred_type, not ours
lk.sample_outputs(post, torch.zeros(1), 8, kind="glm")  # EXPECTED ERROR

# a checkpoint loaded without a template, and with the arguments swapped
lk.load("post.pt")  # EXPECTED ERROR
lk.load(post, "post.pt")  # EXPECTED ERROR


# a sample average that returns the mean outputs, not a distribution over targets
class MeanOnly:
    def average(self, f: torch.Tensor, /) -> torch.Tensor:
        return f.mean(0)


_avg: lk.core.HasSampleAverage = MeanOnly()  # EXPECTED ERROR

# the event axes as a string
lk.Bernoulli(event_dims="1")  # EXPECTED ERROR

# the number of bins as a string
lk.objectives.ece(val_batches, n_bins="15")  # EXPECTED ERROR

# a natural head through a link it doesn't have
lk.heteroscedastic.NaturalHead(link="relu")  # EXPECTED ERROR


# an output check sees the outputs only, not the targets
class ChecksWithTargets:
    def check_outputs(self, f: Outputs, y: Outputs, /) -> None:
        raise NotImplementedError


_check: lk.core.HasOutputCheck = ChecksWithTargets()  # EXPECTED ERROR

# the pairwise model passed as the likelihood
lk.laplace(model, train_loader, likelihood=lk.reward.pairwise(model))  # EXPECTED ERROR

# a pair of numbers, not of inputs
lk.reward.pair(1, 2)  # EXPECTED ERROR

# raw texts where tokenizer outputs go
texts = {"chosen": ["a"], "rejected": ["b"]}
lk.reward.prepare_preferences(texts, torch.Generator().manual_seed(0))  # EXPECTED ERROR

# a seed where a generator goes: seed one with torch.Generator().manual_seed(0)
lk.sample_outputs(post, val_batches[0].inputs, 8, generator=0)  # EXPECTED ERROR

# a batch passed where the inputs go
lk.predict(post, val_batches[0])  # EXPECTED ERROR

# laplace-torch's name for the per-output GPs, not one of the structures
lk.function_space.SubsetOfData(10, structure="independent_outputs")  # EXPECTED ERROR

# a function-space method passed to the weight-space front door
ella = lk.gp.ELLA(lk.gp.from_data(100), 20)
lk.laplace(model, train_loader, likelihood=lik, hessian_approx=ella)  # EXPECTED ERROR

# laplace-torch's name for every output, not one of ours
lk.gp.ELLA(lk.gp.from_data(100), 20, outputs="full")  # EXPECTED ERROR

# laplace-torch picks inducing classes from labels; ours are "predicted" or "random"
lk.gp.VaLLA(lk.gp.from_data(50), coords="labels")  # EXPECTED ERROR

# VaLLA's training loop given an ELLA posterior
nystrom_gp = lk.function_space.laplace(model, train_loader, lik, method=ella)
lk.gp.fit_variational(nystrom_gp, train_loader, n_steps=100)  # EXPECTED ERROR

# Matérn's smoothness as a float; ours are the strings "1/2", "3/2" and "5/2"
lk.gp.Matern(2.5, torch.tensor(1.0))  # EXPECTED ERROR

# a float lengthscale; kernel hyperparameters are tensors
lk.gp.RBF(1.0)  # EXPECTED ERROR

# a function-space prior passed to the weight-space front door; FSP-Laplace is lk.gp.fsp_laplace
fsp_prior = lk.gp.FunctionPrior(lk.gp.RBF(torch.tensor(1.0)))
lk.laplace(model, train_loader, likelihood=lik, prior=fsp_prior)  # EXPECTED ERROR

# laplace-torch's prior_structure strings: lk.marglik.train takes the prior class instead
marglik_opt = torch.optim.Adam(model.parameters())
lk.marglik.train(
    model,
    train_loader,
    likelihood=lik,
    optimizer=marglik_opt,
    epochs=1,
    prior="layerwise",  # EXPECTED ERROR
)

# the noise of a classifier's posterior: a Categorical has no sigma
cat_post = lk.laplace(model, train_loader, lk.Categorical())
_ = cat_post.likelihood.sigma  # EXPECTED ERROR

# a start value for a learned sigma goes on the tuner, not on the likelihood
lk.Gaussian(0.3, learned=True)  # EXPECTED ERROR


# A plugin class that inherits a protocol but leaves out a required method: the editor
# marks it where it is built, and on the class line itself when the class is `@final`.
class ForgotLogProb(lk.core.Likelihood):
    pass


ForgotLogProb()  # EXPECTED ERROR


@typing.final
class FinalForgotLogProb(lk.core.Likelihood):  # EXPECTED ERROR
    pass
