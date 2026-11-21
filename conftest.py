"""Every test starts from `torch.manual_seed(0)`.

Unseeded draws (`generator=None`) come from torch's global RNG, so a test's numbers
would otherwise depend on the tests that ran before it.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    torch.manual_seed(0)
