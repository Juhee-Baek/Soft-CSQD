"""Internal random-state helpers used to keep stochastic steps reproducible."""

from __future__ import annotations

import numpy as np

MAX_UINT32 = np.iinfo(np.uint32).max


def make_rng(seed: int | None = 42) -> np.random.Generator:
    """Create a generator without ever delegating to NumPy's entropy default."""

    return np.random.default_rng(42 if seed is None else seed)


def derive_seed(rng: np.random.Generator) -> int:
    """Derive a child seed from the run-level generator."""

    return int(rng.integers(0, MAX_UINT32, dtype=np.uint32))


def child_rng(rng: np.random.Generator) -> tuple[np.random.Generator, int]:
    seed = derive_seed(rng)
    return np.random.default_rng(seed), seed
