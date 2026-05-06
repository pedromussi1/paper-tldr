"""Tests for src/eval.py validation paths.

The full compute_metrics pipeline is not run in CI because BERTScore loads
a ~1.6 GB DeBERTa model on first call. End-to-end metric correctness is
exercised by the baseline + qlora eval runs and tracked in W&B.
"""
from __future__ import annotations

import pytest

from src.eval import compute_metrics


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        compute_metrics(predictions=["a", "b"], references=["a"])
