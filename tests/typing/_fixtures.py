"""Stand-ins for user objects the scenarios assume exist."""

from collections.abc import Iterable

import torch
from torch import Tensor, nn

import laplacekit as lk


def train_my_resnet() -> nn.Module:
    raise NotImplementedError


train_loader: Iterable[tuple[Tensor, Tensor]] = []
val_batches: list[lk.Batch] = []
x_test: Tensor = torch.zeros(1)
y_test: Tensor = torch.zeros(1)
image_loader: list[Tensor] = []
