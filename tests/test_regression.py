"""Pinned results of representative computations (tests/regression.py): a dependency update that changes a number fails here."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from regression import DATA, compute, differences  # noqa: E402


@pytest.mark.slow
def test_representative_results_unchanged():
    R = compute()
    assert differences(R, np.load(DATA)) == []
