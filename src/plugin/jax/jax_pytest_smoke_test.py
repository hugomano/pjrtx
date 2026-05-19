#!/usr/bin/env python3
"""Pytest-shaped local conformance checks for the PjRTx JAX plugin."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from pjrtx_harness import configure_pjrtx, run_cpu_vs_pjrtx


def test_elementwise_and_cache_trace() -> None:
    jax, jnp = configure_pjrtx(trace=True)
    x = np.arange(12, dtype=np.float32).reshape(3, 4) / np.float32(7.0)
    y = np.linspace(0.5, 1.7, 12, dtype=np.float32).reshape(3, 4)
    run_cpu_vs_pjrtx(
        jax,
        "pytest_elementwise_and_cache_trace",
        lambda a, b: jnp.tanh(a + b) * jnp.sqrt(b),
        x,
        y,
    )


def test_reduction_and_matmul() -> None:
    jax, jnp = configure_pjrtx(trace=True)
    lhs = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(5.0)
    rhs = np.arange(12, dtype=np.float32).reshape(4, 3) / np.float32(6.0)
    run_cpu_vs_pjrtx(jax, "pytest_matmul", lambda x, y: x @ y, lhs, rhs)
    run_cpu_vs_pjrtx(jax, "pytest_reduce_sum", lambda x: jnp.sum(x * x, axis=-1), lhs)


def test_gather_scatter_smoke() -> None:
    jax, _ = configure_pjrtx(trace=True)
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    rows = np.array([2, 0], dtype=np.int32)
    cols = np.array([3, 1], dtype=np.int32)
    updates = np.array([5.0, 7.0], dtype=np.float32)
    run_cpu_vs_pjrtx(jax, "pytest_point_gather", lambda a, i, j: a[i, j], x, rows, cols)
    run_cpu_vs_pjrtx(jax, "pytest_point_scatter_add", lambda a, i, j, u: a.at[i, j].add(u), x, rows, cols, updates)


def main() -> int:
    import pytest

    return pytest.main([__file__])


if __name__ == "__main__":
    raise SystemExit(main())
