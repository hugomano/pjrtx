#!/usr/bin/env python3
"""CPU-vs-PjRTx JAX correctness suite for the currently lowered fast path."""

from __future__ import annotations

import os
import pathlib
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from jax_plugin_smoke import import_jax, plugin_path, register_plugin


ArrayFn = Callable[..., Any]


def assert_close(name: str, got: Any, want: Any) -> None:
    got_np = np.asarray(got)
    want_np = np.asarray(want)
    if got_np.dtype.kind in {"f", "c"} or want_np.dtype.kind in {"f", "c"}:
        np.testing.assert_allclose(got_np, want_np, rtol=2.0e-4, atol=2.0e-4, err_msg=name)
    else:
        np.testing.assert_array_equal(got_np, want_np, err_msg=name)


def run_case(jax: Any, name: str, fn: ArrayFn, *inputs: np.ndarray) -> None:
    cpu_fn = jax.jit(fn, backend="cpu")
    want = jax.device_get(cpu_fn(*inputs))

    pjrtx_fn = jax.jit(fn, backend="pjrtx")
    got_first = jax.device_get(pjrtx_fn(*inputs))
    got_second = jax.device_get(pjrtx_fn(*inputs))

    assert_close(f"{name}/first", got_first, want)
    assert_close(f"{name}/second", got_second, want)


def main() -> int:
    jax, jnp = import_jax()

    os.environ.setdefault("PJRTX_BACKEND", "metal_mlx")
    register_plugin(plugin_path())
    jax.config.update("jax_platforms", "pjrtx,cpu")
    jax.config.update("jax_platform_name", "pjrtx")
    if not jax.devices("cpu"):
        raise RuntimeError("JAX CPU backend is required for the PjRTx op suite")

    f32_a = (np.arange(12, dtype=np.float32).reshape(3, 4) - np.float32(3.0)) / np.float32(7.0)
    f32_b = np.linspace(0.5, 1.7, 12, dtype=np.float32).reshape(3, 4)
    mat_lhs = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(5.0)
    mat_rhs = np.arange(12, dtype=np.float32).reshape(4, 3) / np.float32(6.0)
    i32_a = np.arange(8, dtype=np.int32)
    i32_b = np.arange(10, 18, dtype=np.int32)
    i32_shift = np.full((8,), 1, dtype=np.int32)
    gather_indices = np.array([2, -1], dtype=np.int32)
    column_indices = np.array([3, 1], dtype=np.int32)
    floor = np.full((3, 4), 0.25, dtype=np.float32)
    tiny = np.full((3, 4), 0.01, dtype=np.float32)
    clip_lo = np.full((3, 4), -0.25, dtype=np.float32)
    clip_hi = np.full((3, 4), 1.0, dtype=np.float32)

    run_case(
        jax,
        "elementwise_float_chain",
        lambda x, y, floor_, tiny_: jnp.tanh((x + y) * jax.lax.rsqrt(jnp.maximum(y, floor_))) - jnp.exp(x * tiny_),
        f32_a,
        f32_b,
        floor,
        tiny,
    )
    run_case(jax, "reshape_transpose_broadcast", lambda x: jnp.broadcast_to(jnp.reshape(x, (2, 6)), (2, 2, 6)).transpose(1, 2, 0), f32_a)
    run_case(jax, "reduce_sum", lambda x: jnp.sum(x * x, axis=-1, keepdims=True), f32_a)
    run_case(jax, "matmul", lambda x, y: x @ y, mat_lhs, mat_rhs)
    run_case(jax, "clip", lambda x, lo, hi: jnp.maximum(jnp.minimum(x, hi), lo), f32_a, clip_lo, clip_hi)
    run_case(jax, "bitwise_int", lambda x, y, shift: (x & y) ^ (x << shift), i32_a, i32_b, i32_shift)
    run_case(jax, "sort", lambda x: jnp.sort(x, axis=1), f32_a)
    run_case(jax, "axis0_gather", lambda x, i: x[i], f32_a, gather_indices)
    run_case(jax, "axis1_gather", lambda x, i: x[:, i], f32_a, column_indices)

    print(f"JAX backend: {jax.default_backend()}")
    print("PjRTx JAX op suite: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
