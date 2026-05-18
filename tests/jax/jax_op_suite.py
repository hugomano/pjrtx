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
    if isinstance(got, (tuple, list)) or isinstance(want, (tuple, list)):
        if not isinstance(got, type(want)) or len(got) != len(want):
            raise AssertionError(f"{name}: pytree shape mismatch")
        for index, (got_item, want_item) in enumerate(zip(got, want, strict=True)):
            assert_close(f"{name}[{index}]", got_item, want_item)
        return
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
    f32_cube = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) - np.float32(8.0)) / np.float32(5.0)
    mat_lhs = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(5.0)
    mat_rhs = np.arange(12, dtype=np.float32).reshape(4, 3) / np.float32(6.0)
    i32_a = np.arange(8, dtype=np.int32)
    i32_b = np.arange(10, 18, dtype=np.int32)
    i32_shift = np.full((8,), 1, dtype=np.int32)
    gather_indices = np.array([2, -1], dtype=np.int32)
    column_indices = np.array([3, 1], dtype=np.int32)
    row_indices = np.array([2, 0], dtype=np.int32)
    point_column_indices = np.array([3, 1], dtype=np.int32)
    point_updates = np.array([5.0, 7.0], dtype=np.float32)
    batched_gather_indices = np.array([[2, 0], [1, 2]], dtype=np.int32)
    row_window_updates = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    batched_scatter_updates = (np.arange(16, dtype=np.float32).reshape(2, 2, 4) + np.float32(5.0)) / np.float32(3.0)
    scatter_base = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float32)
    scatter_indices = np.array([1, -1], dtype=np.int32)
    scatter_updates = np.array([5.0, 7.0], dtype=np.float32)
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
    run_case(jax, "reduce_max", lambda x: jnp.max(x, axis=1), f32_cube)
    run_case(jax, "reduce_sum_all", lambda x: jnp.sum(x), f32_cube)
    run_case(jax, "reduce_sum_multi_axis", lambda x: jnp.sum(x, axis=(0, 2)), f32_cube)
    run_case(jax, "matmul", lambda x, y: x @ y, mat_lhs, mat_rhs)
    run_case(jax, "clip", lambda x, lo, hi: jnp.maximum(jnp.minimum(x, hi), lo), f32_a, clip_lo, clip_hi)
    run_case(jax, "bitwise_int", lambda x, y, shift: (x & y) ^ (x << shift), i32_a, i32_b, i32_shift)
    run_case(jax, "sort", lambda x: jnp.sort(x, axis=1), f32_a)
    run_case(jax, "sort_descending", lambda x: jnp.sort(x, axis=1, descending=True), f32_a)
    run_case(jax, "argsort", lambda x: jnp.argsort(x, axis=1), f32_a)
    if os.environ.get("PJRTX_DUMP_TOPK"):
        print(jax.jit(lambda x: jax.lax.top_k(x, 2), backend="cpu").lower(f32_a).compiler_ir(dialect="stablehlo"))
    run_case(jax, "top_k", lambda x: jax.lax.top_k(x, 2), f32_a)
    run_case(jax, "axis0_gather", lambda x, i: x[i], f32_a, gather_indices)
    run_case(jax, "axis1_gather", lambda x, i: x[:, i], f32_a, column_indices)
    run_case(jax, "point_gather", lambda x, i, j: x[i, j], f32_a, row_indices, point_column_indices)
    batched_gather_dnums = jax.lax.GatherDimensionNumbers(
        offset_dims=(2,),
        collapsed_slice_dims=(1,),
        operand_batching_dims=(0,),
        start_indices_batching_dims=(0,),
        start_index_map=(1,),
    )
    run_case(
        jax,
        "batched_gather",
        lambda x, i: jax.lax.gather(
            x,
            i[..., None],
            dimension_numbers=batched_gather_dnums,
            slice_sizes=(1, 1, 4),
            mode="promise_in_bounds",
        ),
        f32_cube,
        batched_gather_indices,
    )
    run_case(jax, "take_along_axis_gather", lambda x, i: jnp.take_along_axis(x, i[..., None], axis=1), f32_cube, batched_gather_indices)
    run_case(jax, "scatter_set", lambda x, i, y: x.at[i].set(y), scatter_base, scatter_indices, scatter_updates)
    run_case(jax, "scatter_add", lambda x, i, y: x.at[i].add(y), scatter_base, scatter_indices, scatter_updates)
    run_case(jax, "point_scatter_set", lambda x, i, j, y: x.at[i, j].set(y), f32_a, row_indices, point_column_indices, point_updates)
    run_case(jax, "point_scatter_add", lambda x, i, j, y: x.at[i, j].add(y), f32_a, row_indices, point_column_indices, point_updates)
    run_case(jax, "window_scatter_set", lambda x, i, y: x.at[i, :2].set(y), f32_a, row_indices, row_window_updates)
    run_case(jax, "window_scatter_add", lambda x, i, y: x.at[i, :2].add(y), f32_a, row_indices, row_window_updates)
    batched_scatter_dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(2,),
        inserted_window_dims=(1,),
        scatter_dims_to_operand_dims=(1,),
        operand_batching_dims=(0,),
        scatter_indices_batching_dims=(0,),
    )
    run_case(
        jax,
        "batched_scatter_set",
        lambda x, i, y: jax.lax.scatter(
            x,
            i[..., None],
            y,
            dimension_numbers=batched_scatter_dnums,
            indices_are_sorted=False,
            unique_indices=False,
            mode="promise_in_bounds",
        ),
        f32_cube,
        batched_gather_indices,
        batched_scatter_updates,
    )
    run_case(
        jax,
        "batched_scatter_add",
        lambda x, i, y: jax.lax.scatter_add(
            x,
            i[..., None],
            y,
            dimension_numbers=batched_scatter_dnums,
            indices_are_sorted=False,
            unique_indices=False,
            mode="promise_in_bounds",
        ),
        f32_cube,
        batched_gather_indices,
        batched_scatter_updates,
    )
    run_case(jax, "interior_pad", lambda x: jax.lax.pad(x, np.float32(0.0), [(1, 1, 1), (0, 2, 0)]), f32_a)

    print(f"JAX backend: {jax.default_backend()}")
    print("PjRTx JAX op suite: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
