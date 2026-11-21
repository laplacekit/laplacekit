# %% [markdown]
# ---
# title: Language models
# jupyter: python3
# ---
#
# We take a language model fine-tuned with LoRA to classify the emotion of a
# sentence, and put a Laplace approximation on its adapters. The model is
# `SmolLM2-360M-Instruct` with rank-8 adapters and a 6-class head, trained on
# the `dair-ai/emotion` dataset.

# %%
import warnings

import datasets
import huggingface_hub
import torch
import transformers
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import laplacekit as lk

transformers.logging.set_verbosity_error()
transformers.utils.logging.disable_progress_bar()
datasets.disable_progress_bars()
huggingface_hub.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)  # deprecation notices from transformers

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
# `is_trainable=True` marks the adapters and the head as the parameters to fit.
# Eager attention lets the linearized predictive differentiate through the
# model, and the model needs the tokenizer's padding token to find the last
# real token of each sentence.
#
# The batches are tokenizer outputs with the labels added. We fit on 32 of the
# 16,000 training sentences and pass the full size as `n_data=`.

# %%
DATA_REV = "cab853a1dbdf4c42c2b3ef2173804746df8825fe"
train = load_dataset("dair-ai/emotion", revision=DATA_REV, split="train")
classes = train.features["label"].names
sub = train.shuffle(seed=0).select(range(32))

batches = [
    {**tok(b["text"], return_tensors="pt", padding=True), "labels": torch.tensor(b["label"])}
    for b in sub.iter(batch_size=16)
]

task = lk.transformers.SequenceClassification()
post = lk.transformers.laplace_lora(model, batches, task=task, n_data=len(train))
print("prior precision:", float(post.prior.prior_precision))

# %% [markdown]
# Two sentences like the training data, and one that has no emotion at all.

# %%
texts = [
    "i feel so happy that my sister is coming home",
    "i am terrified of what the doctor will say tomorrow",
    "the capital of france is paris",
]
inputs = tok(texts, return_tensors="pt", padding=True)

with torch.no_grad():
    p_map = model(**inputs).logits.softmax(-1)

p_lap = lk.predict(post, inputs).probs

for text, a, b in zip(texts, p_map, p_lap, strict=True):
    print(text)
    print(f"  MAP     {classes[a.argmax()]:<8} {a.max():.2f}")
    print(f"  Laplace {classes[b.argmax()]:<8} {b.max():.2f}")

# %% [markdown]
# MAP and Laplace pick the same classes. The Laplace predictive is less sure on
# all three, and on the sentence with no emotion it drops below one half. It
# linearizes the model on the 6 class logits only, never on the 49,152 logits of
# the vocabulary, so their covariance is computed exactly, without sampling.
#
# ::: {.callout-note collapse="true" title="The math"}
# Let $\theta$ be the LoRA adapters and the head, and $\theta_*$ their trained
# values. The Laplace approximation is $\mathcal{N}(\theta \mid \theta_*,
# \Sigma)$ with $\Sigma^{-1} = \lambda I + N G$, where $N = 16{,}000$ and $G$ is
# a KFAC approximation of the GGN, averaged over the 32 sentences. The
# linearized model makes the $K = 6$ logits Gaussian,
#
# $$
#   f(x) \sim \mathcal{N}\big(f_{\theta_*}(x), \; J(x) \Sigma J(x)^\top\big) ,
# $$
#
# with $J(x)$ the $K \times P$ Jacobian with respect to the adapters. Computing
# $J(x) \Sigma J(x)^\top$ costs $K$ backward passes per sentence. On the
# vocabulary of a causal LM, $K$ would be $49{,}152$, which is why we read the
# answer from 6 logits. The probit approximation then gives the class
# probabilities.
# :::
#
# The full tutorial: [Language models](../examples/llm.py)
