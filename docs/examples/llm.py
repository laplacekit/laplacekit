# %% [markdown]
# ---
# title: Language models
# jupyter: python3
# ---
#
# ::: {.callout-tip title="In short"}
# ```{.python}
# task = lk.transformers.SequenceClassification()
# post = lk.transformers.laplace_lora(model, batches, task=task, n_data=len(train))  # KFAC, tuned prior
# probs = lk.predict(post, inputs).probs  # linearized over the K class logits
# nll = lk.objectives.predictive_nll([(x_val, y_val)])
# post = lk.tune(post, lk.GridSearch(nll, values=lams))  # prior by validation NLL
# ```
# :::
#
# The goal of this page is to find out whether a Laplace approximation tells us
# when a fine-tuned language model is wrong. We take a published LoRA adapter
# that turns `SmolLM2-360M-Instruct` into a classifier of the emotion in a
# sentence, and fit a Laplace approximation to the adapter and the
# classification head. This is Laplace-LoRA, from Yang et al.
# (2024)[^yang2024]. Then we corrupt the test sentences with typos, dropped
# words and swapped words, at increasing severity, and ask whether the Laplace
# predictive separates the wrong predictions from the right ones better than the
# MAP model's own confidence does. Apart from one short section at the end,
# nothing on this page is trained: the Laplace approximation is post hoc.
#
# Recall that LoRA (Hu et al., 2022)[^hu2022] freezes each pretrained weight
# matrix $W_0$ and learns a low-rank update,
#
# $$
#   W = W_0 + \frac{\alpha}{r} B A , \qquad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k} ,
# $$
#
# with a small rank $r$ and a fixed scale $\alpha$. Let $\theta$ be all the $A$s
# and $B$s, together with the weights of the classification head, and
# $\theta_*$ their published values. The Laplace approximation over $\theta$ is
#
# $$
#   p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\theta \mid \theta_*, \Sigma) , \qquad \Sigma^{-1} = \lambda I + N G ,
# $$
#
# where $\lambda$ is the prior precision, $N$ the number of training sentences,
# and $G$ the generalized Gauss-Newton matrix (GGN) averaged over them. Even at
# $r = 8$, $\theta$ has 824,960 entries here, so we can't store $G$. Like
# Laplace-LoRA, we use KFAC (Martens & Grosse, 2015)[^martens2015]: one block
# per weight matrix, each block a Kronecker product of two small matrices.
#
# Since $G$ is an average, we can estimate it on a few sentences and still
# multiply it by the full $N$. The prior precision $\lambda$ is the other knob
# that sets the width of the posterior. By default the marginal likelihood picks
# it. Later on, we also pick it by the predictive NLL of held-out sentences and
# compare the two.
#
# The classifier has $K = 6$ logits $f_\theta(x)$ per sentence. Linearizing
# them around $\theta_*$ makes them Gaussian,
#
# $$
#   f(x) \sim \mathcal{N}\big(f_{\theta_*}(x), \; J(x) \Sigma J(x)^\top\big) ,
# $$
#
# with $J(x)$ the $K \times P$ Jacobian with respect to $\theta$. The probit
# approximation (MacKay, 1992)[^mackay1992] then gives the class probabilities.
# It centres the logits, $g = f_{\theta_*}(x) - \frac{1}{K} \sum_k
# f_{\theta_*}(x)_k$, and shrinks each one by its own variance $s_k^2$:
#
# $$
#   p(y \mid x, \mathcal{D}) \approx \mathrm{softmax}\Big(\frac{g_k}{\sqrt{1 + \pi s_k^2 / 8}}\Big)_y .
# $$
#
# The covariance $J \Sigma J^\top$ takes $K$ backward passes per sentence, so
# $K$ sets the cost. A causal LM predicts a token out of its whole vocabulary,
# $49{,}152$ of them for SmolLM2, which is why we only ever linearize the few
# logits that answer the question. For the classifier these are its 6 logits;
# for a causal LM, at the end of the page, they are the logits of the answer
# tokens.

# %%
import random
import warnings

import datasets
import huggingface_hub
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

import laplacekit as lk

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
datasets.disable_progress_bars()
huggingface_hub.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)  # deprecation notices from transformers

# %% [markdown]
# ## Model and data
#
# The adapter is `Abdelrahman-1/smollm2-emotion-classifier` on the Hugging Face
# Hub, pinned to one revision, like its base model. It has rank-8 LoRA updates
# with $\alpha = 16$ on the query and value projections of all 32 layers, and a
# new linear head, `score`, from the 960 features of the last real token to the
# 6 classes. `is_trainable=True` marks the adapter and the head as the
# parameters to fit. The model needs the tokenizer's padding token to find that
# last real token.
#
# :::{.callout-note}
# We load the base model with `attn_implementation="eager"`. The default
# attention in `transformers` (SDPA) works for the fit, but the linearized
# predictive differentiates through its backward pass, which SDPA can't do.
# With SDPA, `lk.predict` raises a `CapabilityError` before it runs the model,
# and the message names both ways out: eager attention, or the
# `lk.WeightSamples` predictive.
# :::

# %%
BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
BASE_REV = "a10cc1512eabd3dde888204e902eca88bddb4951"
ADAPTER = "Abdelrahman-1/smollm2-emotion-classifier"
ADAPTER_REV = "c302741c3f2f89a22504f88fb7b6ad27e6ad1319"

tok = AutoTokenizer.from_pretrained(ADAPTER, revision=ADAPTER_REV)
base = AutoModelForSequenceClassification.from_pretrained(
    BASE, revision=BASE_REV, num_labels=6, dtype=torch.float32, attn_implementation="eager"
)
base.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(base, ADAPTER, revision=ADAPTER_REV, is_trainable=True)
_ = model.eval()

# %% [markdown]
# The data are `dair-ai/emotion` (Saravia et al., 2018)[^saravia2018]: short
# English messages from Twitter, each labelled with one of six emotions.

# %%
DATA_REV = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
emotion = load_dataset("dair-ai/emotion", revision=DATA_REV)
train = emotion["train"]
classes = train.features["label"].names

print({split: len(rows) for split, rows in emotion.items()})
print(classes)

# %% [markdown]
# We keep the sentences of at most 10 words in every split, which keeps the fit
# and the predictive fast on a CPU. The fit uses 32 of them from the training
# split, in two batches of 16. Each batch is the tokenizer's output with the
# labels added. The [Quickstart](../quickstart/llm.py) fits on 32 training
# sentences of any length instead, so its $\lambda$ differs a little from the
# one below.


# %%
def short(split, n):
    rows = emotion[split].shuffle(seed=0).filter(lambda r: len(r["text"].split()) <= 10)
    rows = rows.select(range(n))

    return list(rows["text"]), torch.tensor(list(rows["label"]))


train_texts, y_train = short("train", 32)

batches = [
    {
        **tok(train_texts[i : i + 16], return_tensors="pt", padding=True),
        "labels": y_train[i : i + 16],
    }
    for i in range(0, 32, 16)
]

# %% [markdown]
# ## Fitting the adapter
#
# `lk.transformers.laplace_lora` wraps the model so that it returns the logits,
# turns each batch into inputs and labels, and fits the trainable parameters
# with a categorical likelihood. For a LoRA sequence classifier its default
# curvature is KFAC on every block. The 32 sentences give the per-sentence
# average $G$, and `n_data=len(train)` sets $N = 16{,}000$.
# The marginal likelihood picks $\lambda$.

# %%
task = lk.transformers.SequenceClassification()
post = lk.transformers.laplace_lora(model, batches, task=task, n_data=len(train))
print(post)

# %% [markdown]
# The posterior has 129 blocks: 128 LoRA matrices and the head. The marginal
# likelihood chose $\lambda \approx 15.1$.
#
# ## When is the model wrong?
#
# On clean test sentences the MAP model is right most of the time, so there are
# few errors to detect. To get more of them, we shift the inputs, in the spirit
# of CIFAR-10-C (Hendrycks & Dietterich, 2019)[^hendrycks2019]. For a rate
# $\rho$, we drop each word with probability $\rho$, swap two neighbouring
# letters in each remaining word of three or more letters with probability
# $\rho$, and swap each pair of neighbouring words with probability $\rho$. We
# use $\rho = 0$, $0.2$ and $0.4$ on the same 32 test sentences, with a random
# generator seeded at 0 for each rate. Here is one sentence at each rate.


# %%
test_texts, y_test = short("test", 32)


def corrupt(text, rate, rng):
    """Drop words, swap two letters in a word, and swap neighbouring words, each at `rate`."""

    words = [w for w in text.split() if rng.random() >= rate] or text.split()[:1]

    for i, w in enumerate(words):
        if len(w) > 2 and rng.random() < rate:
            j = rng.randrange(len(w) - 1)
            words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2 :]

    for i in range(len(words) - 1):
        if rng.random() < rate:
            words[i], words[i + 1] = words[i + 1], words[i]

    return " ".join(words)


shifted = {}

for rate in [0.0, 0.2, 0.4]:
    rng = random.Random(0)
    shifted[rate] = [corrupt(t, rate, rng) for t in test_texts]

for rate, texts in shifted.items():
    print(f"{rate:.1f}  {texts[1]}")

# %% [markdown]
# For each rate, we compare the MAP model with the Laplace predictive at the
# marginal-likelihood $\lambda$. Besides the accuracy, the NLL and the mean
# confidence $\max_y p(y \mid x)$, two numbers ask whether the confidence ranks
# the errors. The first is the AUROC of error detection. With the uncertainty
# $u(x) = 1 - \max_y p(y \mid x)$,
#
# $$
#   \mathrm{AUROC} = \Pr\big(u(x) > u(x') \mid x \text{ wrong}, \; x' \text{ right}\big) ,
# $$
#
# with ties counting one half: 0.5 is chance and 1 a perfect ranking. The second
# is the selective accuracy, the accuracy on the 75% most confident sentences,
# i.e. after abstaining on the least confident quarter. Both depend only on how
# the confidences are ordered, not on their values.


# %%
def auroc(score, wrong):
    """P(score of a wrong prediction > score of a right one), ties counting one half."""

    pos, neg = score[wrong], score[~wrong]
    greater = (pos[:, None] > neg[None]).float() + 0.5 * (pos[:, None] == neg[None]).float()

    return greater.mean()


def metrics(p, y, keep=0.75):
    conf, pred = p.max(-1)
    wrong = pred != y
    kept = conf.argsort(descending=True)[: int(keep * len(y))]

    return {
        "acc": (~wrong).float().mean(),
        "NLL": -p.log()[range(len(y)), y].mean(),
        "conf": conf.mean(),
        "AUROC": auroc(-conf, wrong),
        "acc@75%": (~wrong[kept]).float().mean(),
    }


probs = {}

for rate, texts in shifted.items():
    x = tok(texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        p_map = model(**x).logits.softmax(-1)

    probs[rate] = (p_map, lk.predict(post, x).probs)

print(f"{'rate':<5} {'':<8}" + "".join(f"{k:>9}" for k in metrics(p_map, y_test)))

for rate, (p_map, p_lap) in probs.items():
    for name, p in [("MAP", p_map), ("Laplace", p_lap)]:
        row = "".join(f"{float(v):9.3f}" for v in metrics(p, y_test).values())
        print(f"{rate:<5.1f} {name:<8}{row}")

# %% [markdown]
# The MAP accuracy falls from 0.875 to 0.750 and 0.562 as the rate grows. The
# Laplace predictive has the same accuracy, except at $\rho = 0.2$, where it gets
# one more sentence right (0.781). The MAP confidence follows the accuracy on
# clean and mildly corrupted sentences (0.857 and 0.744) and runs ahead of it at
# $\rho = 0.4$ (0.649). The Laplace predictive is less confident than its
# accuracy at every rate, from 0.483 down to 0.342. Its NLL is worse than the MAP
# one at $\rho = 0$ and $0.2$ (0.827 against 0.343, and 1.135 against 0.894) and
# better only at $\rho = 0.4$ (1.422 against 1.521).
#
# The ranking is a different story at each rate. On clean sentences the Laplace
# AUROC is lower, 0.750 against 0.830, and at $\rho = 0.2$ it is 0.760 against
# 0.792. At $\rho = 0.4$ the MAP AUROC is 0.504, i.e. chance, while the Laplace
# one is 0.635. The selective accuracy is the same at the first two rates, and
# 0.625 against 0.542 at $\rho = 0.4$.
#
# With 32 sentences, how much of that is noise? We resample the sentences with
# replacement 1,000 times, recompute both AUROCs on each resample, and take the
# 5% and 95% quantiles of the difference. Resamples without a wrong or without a
# right prediction are skipped.

# %%
gen = torch.Generator().manual_seed(0)
draws = torch.randint(len(y_test), (1000, len(y_test)), generator=gen)

for rate, (p_map, p_lap) in probs.items():
    diffs = []

    for i in draws:
        conf_map, pred_map = p_map[i].max(-1)
        conf_lap, pred_lap = p_lap[i].max(-1)
        wrong_map, wrong_lap = pred_map != y_test[i], pred_lap != y_test[i]

        if all(w.any() and not w.all() for w in (wrong_map, wrong_lap)):
            diffs.append(auroc(-conf_lap, wrong_lap) - auroc(-conf_map, wrong_map))

    lo, hi = torch.stack(diffs).quantile(torch.tensor([0.05, 0.95]))
    print(f"rate {rate:.1f}: AUROC Laplace - MAP, 90% interval [{lo:+.3f}, {hi:+.3f}]")

# %% [markdown]
# At $\rho = 0$ and $0.2$, the intervals contain zero and lean negative: there,
# the Laplace uncertainty ranks the errors no better than the MAP confidence, and
# perhaps worse. At $\rho = 0.4$, the interval lies above zero. Under the
# heaviest corruption, the MAP confidence no longer tells wrong predictions from
# right ones, and the Laplace uncertainty still does, to some degree. With 32
# sentences, the size of that gap is not pinned down: the interval runs from
# +0.029 to +0.251.
#
# ## Choosing the prior precision
#
# So far, $\lambda$ came from the marginal likelihood. Changing $\lambda$ needs no
# refit, so we can also choose it by the predictive NLL of held-out sentences.
# We take 16 sentences from the validation split, again of at most 10 words.
# `lk.objectives.predictive_nll` turns them into an objective, and `lk.GridSearch`
# keeps the $\lambda$ with the lowest value. We print each value as the grid goes,
# and the MAP model's NLL for reference.

# %%
val_texts, y_val = short("validation", 16)
x_val = tok(val_texts, return_tensors="pt", padding=True)
nll = lk.objectives.predictive_nll([(x_val, y_val)])


def val_nll(p):
    value = nll(p)
    print(f"lambda = {float(p.prior.prior_precision):8.1f}   validation NLL = {float(value):.3f}")

    return value


lams = [post.prior.prior_precision, *torch.tensor([1000.0, 10000.0])]
post_val = lk.tune(post, lk.GridSearch(val_nll, values=lams))

with torch.no_grad():
    p_val = model(**x_val).logits.softmax(-1)

print(f"MAP                 validation NLL = {float(-p_val.log()[range(16), y_val].mean()):.3f}")

# %% [markdown]
# The validation NLL falls at every step up the grid, and the largest value,
# $\lambda = 10{,}000$, wins. The MAP model is lower still, at 0.242. It is the
# limit $\lambda \to \infty$, where $\Sigma \to 0$, so a longer grid would only
# move closer to it.
#
# The table below puts the three predictives side by side on the clean test
# sentences: the MAP model, and the Laplace predictive at the marginal-likelihood
# and at the validation $\lambda$. AUROC is the error-detection AUROC from
# above.

# %%
ood = [
    "the capital of france is paris",
    "add two cups of flour and stir for a minute",
    "def f(x): return x + 1",
    "the meeting has moved to room 204",
    "der zug fährt um acht uhr ab",
    "photosynthesis converts light into chemical energy",
]
x_ood = tok(ood, return_tensors="pt", padding=True)
x_both = tok(test_texts + ood, return_tensors="pt", padding=True)  # one call at the new lambda

with torch.no_grad():
    p_ood = model(**x_ood).logits.softmax(-1)

p_both = lk.predict(post_val, x_both).probs

results = {
    "MAP": ("-", probs[0.0][0], p_ood),
    "marglik": (post.prior.prior_precision, probs[0.0][1], lk.predict(post, x_ood).probs),
    "validation": (
        post_val.prior.prior_precision,
        p_both[: len(test_texts)],
        p_both[len(test_texts) :],
    ),
}


def entropy(p):
    return -(p * p.clamp_min(1e-30).log()).sum(-1)


print(f"{'':<11} {'lambda':>8} {'NLL':>6} {'acc':>6} {'AUROC':>6}")

for name, (lam, p_test, _) in results.items():
    m = metrics(p_test, y_test)
    lam = lam if isinstance(lam, str) else f"{float(lam):.1f}"
    print(f"{name:<11} {lam:>8} {m['NLL']:6.3f} {m['acc']:6.3f} {m['AUROC']:6.3f}")

# %% [markdown]
# We add six sentences that carry no emotion at all, a small far-OOD set. For
# each predictive, we report the mean entropy $-\sum_y p(y \mid x) \log p(y \mid
# x)$, which is at most $\log 6 \approx 1.79$, and the mean largest probability,
# on the clean test sentences and on the six. The last two columns ask whether
# the uncertainty separates the six from the test sentences: the AUROC of
# telling them apart by the largest probability and by the entropy, computed as
# above with the six as the positives.

# %%
is_ood = torch.cat(
    [torch.zeros(len(test_texts), dtype=torch.bool), torch.ones(len(ood), dtype=torch.bool)]
)

print(
    f"{'':<11} {'H test':>7} {'H OOD':>7} {'p test':>7} {'p OOD':>7} {'AUROC p':>8} {'AUROC H':>8}"
)

for name, (_, p_test, p_far) in results.items():
    h_test, h_far = entropy(p_test), entropy(p_far)
    max_test, max_far = p_test.max(-1).values, p_far.max(-1).values
    by_p = auroc(-torch.cat([max_test, max_far]), is_ood)
    by_h = auroc(torch.cat([h_test, h_far]), is_ood)
    print(
        f"{name:<11} {h_test.mean():7.3f} {h_far.mean():7.3f} {max_test.mean():7.3f} "
        f"{max_far.mean():7.3f} {by_p:8.3f} {by_h:8.3f}"
    )

# %%
print(f"{'sentence':<52} {'MAP':>13} {'marglik':>13} {'validation':>13}")

for i, text in enumerate(ood):
    cells = [f"{classes[p[i].argmax()]:>7} {p[i].max():.2f}" for _, _, p in results.values()]
    print(f"{text:<52} {cells[0]:>13} {cells[1]:>13} {cells[2]:>13}")

# %% [markdown]
# At the validation $\lambda$, the Laplace predictive is the MAP one to within
# about 0.02, on the test sentences and on the six without emotion. At the
# marginal-likelihood $\lambda$, it is much less sure of those six: their largest
# probability drops from 0.584 to 0.261 on average, and their entropy rises from
# 1.233 to 1.751, close to the maximum. Sentence by sentence, the MAP model calls
# five of them joy, with probabilities up to 0.74, and the marginal-likelihood
# predictive keeps the classes and lowers every probability to between 0.21 and
# 0.30.
#
# It is also less sure of the test sentences, though: their mean largest
# probability drops from 0.857 to 0.483 and their entropy rises from 0.447 to
# 1.426. So the gap in mean confidence between the two sets shrinks, from 0.273
# to 0.222. What improves is the ordering. Telling the six apart from the test
# sentences by the largest probability gives an AUROC of 0.938 against 0.896 for
# MAP, and by the entropy 0.979 against 0.943. With only six far-OOD sentences,
# that is a small sample, so take the gain as a hint. On the clean test
# sentences, the same predictive has more than twice the MAP NLL.
#
# So on this adapter, the marginal-likelihood $\lambda$ separates the sentences
# without emotion a little better and gives up NLL on the test sentences, and
# the validation $\lambda$ stays with the MAP model on both. The validation sentences come from the same
# distribution as the training data, so their NLL rewards a sharp predictive,
# and the MAP model is already sharp.
#
# `laplace_lora` tunes $\lambda$ on the marginal likelihood by default, which
# here costs NLL on the clean test sentences: 0.827, against 0.343 for MAP. When
# that matters more than the far-OOD sentences, we fix $\lambda$ in the call,
# or search the validation NLL `nll` from above with `lk.Minimize` instead of
# `lk.GridSearch`:
#
# ```python
# post = lk.transformers.laplace_lora(
#     model, batches, task=task, n_data=len(train), prior=lk.priors.Isotropic(λ), hyper_tuner=None
# )
# post = lk.tune(post, lk.Minimize(nll))
# ```
#
# ## True or false?
#
# A causal LM can classify too, by reading the answer off its next-token
# logits. To ask whether a statement is true, we append "True or false?
# Answer:" and compare the logits of the tokens " True" and " False" at the last
# position, which gives P(True) (Kadavath et al., 2022)[^kadavath2022]. These
# two logits play the role of the classifier's $K$ logits above, with $K = 2$,
# and `lk.transformers.answer_logits` picks them out of the $49{,}152$.
#
# We need an adapter to apply Laplace to, so we add rank-8 LoRA updates to the
# query and value projections of the same base model, loaded this time as a
# causal LM, and train them on 24 statements, half of them true. It uses the
# same tokenizer as the classifier.
#
# ::: {.callout-important}
# This fine-tune is the one step on this page that is not post hoc. It takes
# six epochs over the 24 statements.
# :::

# %%
torch.manual_seed(0)
lm = AutoModelForCausalLM.from_pretrained(
    BASE, revision=BASE_REV, dtype=torch.float32, attn_implementation="eager"
)
lm = get_peft_model(lm, LoraConfig(r=8, target_modules=["q_proj", "v_proj"]))

facts = [
    ("Paris is the capital of France.", True),
    ("Paris is the capital of Spain.", False),
    ("Berlin is the capital of Germany.", True),
    ("Berlin is the capital of Italy.", False),
    ("Tokyo is the capital of Japan.", True),
    ("Tokyo is the capital of Egypt.", False),
    ("Cheese is made from milk.", True),
    ("Cheese is made from sand.", False),
    ("Glass is made from sand.", True),
    ("Glass is made from milk.", False),
    ("A week has seven days.", True),
    ("A week has twelve days.", False),
    ("Fish swim in the sea.", True),
    ("Fish grow on trees.", False),
    ("Bees make honey.", True),
    ("Cows make honey.", False),
    ("Water boils at one hundred degrees.", True),
    ("Water boils at ten degrees.", False),
    ("The sun rises in the east.", True),
    ("The sun rises in the west.", False),
    ("Ice is colder than steam.", True),
    ("Steam is colder than ice.", False),
    ("A day has twenty four hours.", True),
    ("A day has six hours.", False),
]


def encode(statements):
    return tok(
        [f"{s} True or false? Answer:" for s in statements], return_tensors="pt", padding=True
    )


def labels(rows):
    return torch.tensor([0 if true else 1 for _, true in rows])  # 0 is " True", 1 is " False"


yes_no = lk.transformers.answer_logits(tok, [" True", " False"])
tf_batches = [
    (encode([s for s, _ in facts[i : i + 8]]), labels(facts[i : i + 8]))
    for i in range(0, len(facts), 8)
]

# %% [markdown]
# The loss is the cross-entropy over the two answer logits only, as in
# Laplace-LoRA.

# %%
opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=2e-3)
lm.train()

for _ in range(6):
    for x, y in tf_batches:
        loss = torch.nn.functional.cross_entropy(yes_no(lm(**x).logits, x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()

lm.eval()
print("final loss:", round(loss.item(), 3))

# %% [markdown]
# For the Laplace approximation, the likelihood has to be the one we trained
# with. `laplace_lora` with its default `task=CausalLM()` puts it on every next
# token of the text, so we call `lk.laplace` ourselves. The model is the causal
# LM (`lk.transformers.causal_lm`) followed by `yes_no`, which gives two logits
# per prompt; the likelihood is categorical over them; the parameters are the
# trainable ones; and the curvature is KFAC, as before. The 24 statements are
# the whole training set, so $N = 24$.

# %%
causal = lk.transformers.causal_lm(lm)
answers = lk.core.ModelFn(lambda p, x: yes_no(causal(p, x), x), lm)

tf_post = lk.laplace(
    answers,
    tf_batches,
    likelihood=lk.Categorical(),
    subset=lk.subsets.trainable,
    hessian_approx=lk.KFAC(),
)
print(tf_post)

# %% [markdown]
# We test on 16 new statements, again half true, and on four that the model
# can't check: an opinion, a forecast, a matter of taste and a private fact.
# Since `answers` already returns the two logits, `lk.predict` needs no
# `select`.

# %%
held = [
    ("Madrid is the capital of Spain.", True),
    ("Madrid is the capital of France.", False),
    ("Butter is made from milk.", True),
    ("Butter is made from wood.", False),
    ("An hour has sixty minutes.", True),
    ("An hour has nine minutes.", False),
    ("Birds fly in the sky.", True),
    ("Birds swim under the ground.", False),
    ("Rome is the capital of Italy.", True),
    ("Rome is the capital of Japan.", False),
    ("Snow is cold.", True),
    ("Snow is hot.", False),
    ("Dogs have four legs.", True),
    ("Dogs have ten legs.", False),
    ("The moon orbits the Earth.", True),
    ("The Earth orbits the moon.", False),
]
unknown = [
    "The number seven is lucky.",
    "It will rain in Oslo tomorrow.",
    "Blue is the best colour.",
    "My neighbour owns a red bicycle.",
]
statements = [s for s, _ in held] + unknown
x_tf = encode(statements)

with torch.no_grad():
    p_map = yes_no(lm(**x_tf).logits, x_tf).softmax(-1)

p_lap = lk.predict(tf_post, x_tf).probs

print(f"{'statement':<36} {'P(True) MAP':>12} {'Laplace':>8}")

for s, a, b in zip(statements, p_map[:, 0], p_lap[:, 0], strict=True):
    print(f"{s:<36} {a:12.2f} {b:8.2f}")

n, y_held = len(held), labels(held)

for name, p in [("MAP", p_map), ("Laplace", p_lap)]:
    nll_held = -p[:n].log()[range(n), y_held].mean()
    acc = (p[:n].argmax(-1) == y_held).float().mean()
    print(f"{name:<8} NLL {float(nll_held):.3f}, accuracy {float(acc):.3f} on the 16 statements")

# %% [markdown]
# The MAP model gets 13 of the 16 new statements right, and is sometimes very
# sure when it is wrong: it gives "The Earth orbits the moon." a P(True) of
# 0.92. On the four statements it can't check, it is just as sure, with 0.04
# for rain in Oslo tomorrow.
#
# The Laplace predictive pulls every P(True) towards one half, and the most
# confident ones the furthest: 0.92 becomes 0.60 and 0.04 becomes 0.38. It
# doesn't treat the unknown statements differently from the others, though, and
# at the marginal-likelihood $\lambda \approx 0.97$ it raises the NLL on the 16
# statements from 0.506 to 0.634, with the same accuracy. As on the clean
# emotion sentences, the marginal-likelihood $\lambda$ costs NLL, and the
# validation recipe above applies when held-out NLL is what matters.
#
# ## Wrap-up
#
# On a published LoRA classifier, Laplace-LoRA is one call,
# `lk.transformers.laplace_lora(..., task=lk.transformers.SequenceClassification(), n_data=...)`,
# with KFAC by default and a fit on 32 sentences. The linearized predictive
# works on the class logits only, and on a causal LM `answer_logits` reduces the
# vocabulary to the answers first.
#
# On this adapter, the Laplace predictive at the marginal-likelihood $\lambda$ is
# less confident than its accuracy at every corruption rate, so its NLL is worse
# except under heavy corruption. Its uncertainty ranks the errors no better than
# the MAP confidence on clean and mildly corrupted sentences, and better under
# heavy corruption, where the MAP confidence is at chance. On six sentences
# without emotion, its uncertainty also separates them from the test sentences
# a little better than the MAP one. Choosing $\lambda$ by the validation NLL
# sends it towards the MAP model, which gives that up.
#
# [^yang2024]: Yang, A. X., Robeyns, M., Wang, X., & Aitchison, L. (2024). Bayesian low-rank adaptation for large language models. *ICLR*. <https://arxiv.org/abs/2308.13111>
#
# [^hu2022]: Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. *ICLR*. <https://arxiv.org/abs/2106.09685>
#
# [^martens2015]: Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML*. <https://arxiv.org/abs/1503.05671>
#
# [^mackay1992]: MacKay, D. J. C. (1992). The evidence framework applied to classification networks. *Neural Computation*, 4(5), 720–736. <https://doi.org/10.1162/neco.1992.4.5.720>
#
# [^saravia2018]: Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). CARER: Contextualized affect representations for emotion recognition. *EMNLP*. <https://aclanthology.org/D18-1404/>. The dataset card allows educational and research use only; these docs are non-commercial.
#
# [^hendrycks2019]: Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. *ICLR*. <https://arxiv.org/abs/1903.12261>
#
# [^kadavath2022]: Kadavath, S., et al. (2022). Language models (mostly) know what they know. <https://arxiv.org/abs/2207.05221>
