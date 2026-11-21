"""The `+` lines of the Home page's diff tabs run as written."""

import re
import warnings
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from transformers import BatchEncoding, GPT2Config, GPT2LMHeadModel

INDEX = Path(__file__).parents[1] / "docs" / "index.qmd"
VOCAB = 32


def tabs() -> dict[str, list[str]]:
    """Each tab's title, with the code of its `diff` block's `+` lines."""

    text = INDEX.read_text()
    found = {}

    for title, block in re.findall(r"^## (.+?)\n+```diff\n(.*?)^```", text, re.M | re.S):
        found[title] = [line[2:] for line in block.splitlines() if line.startswith("+")]

    return found


def run(title: str, namespace: dict) -> dict:
    code = "\n".join(tabs()[title])
    exec(compile(code, f"{INDEX.name}: {title}", "exec"), namespace)

    return namespace


def train(model: nn.Module, x: Tensor, y: Tensor, loss) -> nn.Module:
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    for _ in range(50):
        opt.zero_grad()
        loss(model(x), y).backward()
        opt.step()

    return model.eval()


def test_home_has_the_three_tabs():
    found = tabs()

    assert list(found) == ["Classifier", "Regression", "Language model"]

    for lines in found.values():
        assert lines[0] == "import laplacekit as lk"


def test_classifier_tab():
    torch.manual_seed(0)
    y = torch.arange(60) % 3
    x = torch.randn(60, 2) + 3 * nn.functional.one_hot(y, 3)[:, :2].float()
    model = train(
        nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 3)),
        x,
        y,
        nn.functional.cross_entropy,
    )
    x_test = torch.randn(5, 2)

    ns = run("Classifier", {"model": model, "train_loader": [(x, y)], "x_test": x_test})
    probs = ns["probs"]

    assert probs.shape == (5, 3)
    assert (probs >= 0).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(5))


def test_regression_tab():
    torch.manual_seed(0)
    x = torch.linspace(-2, 2, 40)[:, None]
    y = torch.sin(2 * x) + 0.1 * torch.randn(40, 1)
    model = train(
        nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1)),
        x,
        y,
        nn.functional.mse_loss,
    )
    x_test = torch.linspace(-3, 3, 7)[:, None]

    ns = run("Regression", {"model": model, "train_loader": [(x, y)], "x_test": x_test})

    assert ns["y_pred"].shape == (7, 1)
    assert (ns["y_std"] > 0).all()


def right_padded(lengths: list[int], width: int, seed: int) -> BatchEncoding:
    """A tokenizer's output (a `BatchEncoding`, not a dict), right-padded to `width` with
    random token ids."""

    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (len(lengths), width), generator=g)
    mask = torch.zeros(len(lengths), width, dtype=torch.long)

    for i, n in enumerate(lengths):
        mask[i, :n] = 1

    return BatchEncoding({"input_ids": ids, "attention_mask": mask})


@pytest.fixture
def lora_gpt2() -> nn.Module:
    """GPT-2 with vocab 32, 2 layers, 2 heads; LoRA r=2 on c_attn. Eager attention."""

    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=VOCAB,
        n_positions=16,
        n_embd=16,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=0,
        attn_implementation="eager",
    )
    lora = LoraConfig(r=2, target_modules=["c_attn"], fan_in_fan_out=True)

    with warnings.catch_warnings():
        # peft 0.21.3 calls transformers' deprecated `is_gradient_checkpointing`.
        warnings.filterwarnings("ignore", "`is_gradient_checkpointing`", FutureWarning)
        model = get_peft_model(GPT2LMHeadModel(config), lora)

    return model.eval()


def test_language_model_tab(lora_gpt2):
    train_loader = [right_padded([5, 3, 6], 6, seed=1), right_padded([4, 6], 7, seed=2)]
    inputs = right_padded([3, 5, 2], 5, seed=3)

    ns = run("Language model", {"model": lora_gpt2, "train_loader": train_loader, "inputs": inputs})
    probs = ns["probs"]

    assert probs.shape == (3, VOCAB)
    assert (probs >= 0).all()
    torch.testing.assert_close(probs.sum(-1), torch.ones(3))


def test_citation_matches_the_readme():
    def bibtex(path):
        found = re.findall(r"```bibtex\n(.*?)```", path.read_text(), re.S)
        assert len(found) == 1
        return found[0]

    assert bibtex(INDEX) == bibtex(INDEX.parents[1] / "README.md")
