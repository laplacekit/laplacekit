# %% [markdown]
# ---
# title: Protocols in one page
# jupyter: python3
# ---
#
# Every extension point in **laplacekit** is a *protocol*: a class that lists the
# methods a plugin must have. Write your class as a subclass of it, and your
# editor marks a missing or misspelled method in red while you type.
#
# ## Inherit the protocol
#
# The likelihood protocol, `lk.core.Likelihood`, requires one method, `log_prob`.
# A Student-t likelihood subclasses it and implements that method:

# %%
import torch
from torch import nn

import laplacekit as lk


class StudentT(lk.core.Likelihood):
    def __init__(self, df: float = 3.0, scale: float = 0.3) -> None:
        self.df, self.scale = df, scale

    def log_prob(self, f: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return (
            torch.distributions.StudentT(self.df, f, self.scale)
            .log_prob(y)
            .sum(-1)
        )


# %% [markdown]
# It goes wherever a built-in likelihood goes:

# %%
torch.manual_seed(0)
x = torch.linspace(-3, 3, 64).unsqueeze(-1)
y = torch.sin(x) + 0.2 * torch.randn_like(x)

model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

for _ in range(500):
    opt.zero_grad()
    nn.functional.mse_loss(model(x), y).backward()
    opt.step()

post = lk.laplace(model, [(x, y)], StudentT())
out = lk.predict_outputs(post, x[:3])
out.mean.squeeze(-1), out.stddev.squeeze(-1)

# %% [markdown]
# ## A missing method is red in the editor
#
# Leave `log_prob` out, or misspell it, and the class is incomplete:
#
# ```python
# class StudentT(lk.core.Likelihood):
#     def logprob(self, f, y): ...
#
# StudentT()  # Cannot instantiate abstract class "StudentT"
# ```
#
# PyCharm marks the class itself. VS Code (Pylance) and `ty` mark the line that
# builds it; add `@typing.final` above the class to get the mark on the class
# line too. If you run the code anyway, Python refuses to build the object:
# `TypeError: Can't instantiate abstract class StudentT without an implementation
# for abstract method 'log_prob'`.
#
# ::: {.callout-note}
# ## Turn checking on in VS Code
# Pylance checks types only when asked. One line in your plugin's
# `pyproject.toml` turns it on for everyone who opens the project:
#
# ```toml
# [tool.pyright]
# typeCheckingMode = "standard"
# ```
# :::
#
# A class without the base still works: **laplacekit** only looks at its
# methods. But then the editor can only check it where you pass it in, and a
# missing method shows up later, when it is needed.
#
# ## Optional methods add capabilities
#
# Some features need more than the required method. Each one is its own small
# protocol, named `Has...`, with one method. You add the method when you want
# the feature. `lk.predict` gives a distribution over targets, so it needs the
# likelihood's distribution (`HasDistribution`). Our class has none, and the
# error says so before any computation:

# %%
try:
    lk.predict(post, x[:3])
except lk.CapabilityError as e:
    print(e)

# %% [markdown]
# [A likelihood](likelihood.py) lists every optional method a likelihood can
# have, and [a curvature](curvature.py) and [a predictive](predictive.py) list
# theirs.
#
# ## Check the behavior
#
# The editor checks that the methods exist and have the right signatures. The
# `lk.testing.check_*` functions check what they compute, such as `log_prob`
# returning one value per example. [Testing](testing.qmd) has them all:

# %%
lk.testing.check_likelihood(StudentT(), model(x).double().detach(), y.double())
