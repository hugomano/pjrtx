#!/usr/bin/env python3
"""CPU-vs-PjRTx JAX correctness suite for the currently lowered fast path."""

from __future__ import annotations

import os
import pathlib
import re
import sys
import tempfile
from collections.abc import Callable
from typing import Any

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from jax_plugin_smoke import import_jax, plugin_path, register_plugin


ArrayFn = Callable[..., Any]
TRACE_KV_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")


def capture_stderr(fn: Callable[[], Any]) -> tuple[str, Any]:
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    try:
        with tempfile.TemporaryFile() as trace_file:
            os.dup2(trace_file.fileno(), stderr_fd)
            try:
                result = fn()
            finally:
                os.dup2(saved_fd, stderr_fd)
            trace_file.seek(0)
            trace_text = trace_file.read().decode("utf-8", errors="replace")
    finally:
        os.close(saved_fd)
    return trace_text, result


def parse_pjrtx_traces(trace_text: str) -> list[dict[str, str]]:
    traces: list[dict[str, str]] = []
    for line in trace_text.splitlines():
        if not line.startswith("pjrtx_trace "):
            continue
        traces.append({key: value for key, value in TRACE_KV_RE.findall(line)})
    return traces


def trace_int(trace: dict[str, str], key: str) -> int:
    if key not in trace:
        raise AssertionError(f"trace line missing {key}: {trace}")
    return int(trace[key])


def api_call_traces(traces: list[dict[str, str]], pjrt_name: str) -> list[dict[str, str]]:
    return [
        trace
        for trace in traces
        if trace.get("event") == "pjrt_api_call" and trace.get("name") == pjrt_name
    ]


def assert_fast_path_trace(name: str, trace_text: str, *, expect_compile: bool) -> None:
    if os.environ.get("PJRTX_TRACE") != "1":
        return
    traces = parse_pjrtx_traces(trace_text)
    if not traces:
        raise AssertionError(f"{name}: PJRTX_TRACE=1 did not produce pjrtx_trace lines")
    compile_traces = api_call_traces(traces, "PJRT_Client_Compile")
    execute_traces = api_call_traces(traces, "PJRT_LoadedExecutable_Execute")
    if expect_compile and not compile_traces:
        raise AssertionError(f"{name}: trace did not include PJRT_Client_Compile")
    if not execute_traces:
        raise AssertionError(f"{name}: trace did not include PJRT_LoadedExecutable_Execute")
    for trace in traces:
        if trace.get("event") != "pjrt_api_call":
            continue
        if trace_int(trace, "failed") != 0 or trace_int(trace, "error_code") != 0:
            raise AssertionError(f"{name}: PJRT API call failed: {trace}")


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
    first_trace, got_first = capture_stderr(lambda: jax.device_get(pjrtx_fn(*inputs)))
    second_trace, got_second = capture_stderr(lambda: jax.device_get(pjrtx_fn(*inputs)))

    assert_close(f"{name}/first", got_first, want)
    assert_close(f"{name}/second", got_second, want)
    assert_fast_path_trace(f"{name}/first", first_trace, expect_compile=True)
    assert_fast_path_trace(f"{name}/second", second_trace, expect_compile=False)


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
    argmax_values = np.array([[1.0, 7.0, 7.0, 3.0], [-2.0, 5.0, 4.0, 5.0]], dtype=np.float32)
    f16_reduce = np.arange(1, 9, dtype=np.float16).reshape(2, 4)
    mat_lhs = np.arange(8, dtype=np.float32).reshape(2, 4) / np.float32(5.0)
    mat_rhs = np.arange(12, dtype=np.float32).reshape(4, 3) / np.float32(6.0)
    mat_lhs_f16 = mat_lhs.astype(np.float16)
    mat_rhs_f16 = mat_rhs.astype(np.float16)
    q_heads = (np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5) - np.float32(11.0)) / np.float32(7.0)
    k_cache = (np.arange(2 * 6 * 5, dtype=np.float32).reshape(2, 6, 5) - np.float32(9.0)) / np.float32(11.0)
    attn_probs = (np.arange(2 * 3 * 4 * 6, dtype=np.float32).reshape(2, 3, 4, 6) + np.float32(1.0)) / np.float32(17.0)
    v_cache = (np.arange(2 * 6 * 5, dtype=np.float32).reshape(2, 6, 5) - np.float32(4.0)) / np.float32(13.0)
    i32_a = np.arange(8, dtype=np.int32)
    i32_b = np.arange(10, 18, dtype=np.int32)
    i32_shift = np.full((8,), 1, dtype=np.int32)
    i32_negative = np.array([-8, -4, -1, 0, 1, 4, 8, 16], dtype=np.int32)
    pred_a = np.array([True, False, True, False], dtype=np.bool_)
    pred_b = np.array([True, True, False, False], dtype=np.bool_)
    u32_bits = np.array([0, 1, 2, 3, 7, 8, 0x80000000], dtype=np.uint32)
    u32_float_bits = np.array([0x3F800000, 0xC0000000], dtype=np.uint32)
    rng_key_data = np.array([0, 7, 0, 0], dtype=np.uint32)
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
    conv_lhs = np.arange(4, dtype=np.float32).reshape(1, 1, 4) + np.float32(1.0)
    conv_rhs = np.array([[[1.0, 2.0]]], dtype=np.float32)
    conv2d_lhs = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) + np.float32(1.0)
    conv2d_rhs = np.ones((1, 1, 2, 2), dtype=np.float32)
    grouped_conv_lhs = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2) + np.float32(1.0)
    grouped_conv_rhs = np.array([[[[10.0]]], [[[100.0]]]], dtype=np.float32)
    conv3d_lhs = np.arange(27, dtype=np.float32).reshape(1, 1, 3, 3, 3) + np.float32(1.0)
    conv3d_rhs = np.ones((1, 1, 2, 2, 2), dtype=np.float32)
    fft_input = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    complex_real = np.array([1.0, -2.0, 3.5], dtype=np.float32)
    complex_imag = np.array([0.5, 4.0, -1.5], dtype=np.float32)
    complex_values = np.array([1.0 + 0.5j, -2.0 + 4.0j, 3.5 - 1.5j], dtype=np.complex64)
    spd_2x2 = np.array([[4.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    triangular_lhs = np.array([[2.0, 0.0], [1.0, 3.0]], dtype=np.float32)
    triangular_rhs = np.array([[2.0, 4.0], [7.0, 13.0]], dtype=np.float32)
    while_scalar = np.array(0.0, dtype=np.float32)

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
    run_case(jax, "reduce_min", lambda x: jnp.min(x, axis=1), f32_cube)
    run_case(jax, "reduce_sum_all", lambda x: jnp.sum(x), f32_cube)
    run_case(jax, "reduce_sum_multi_axis", lambda x: jnp.sum(x, axis=(0, 2)), f32_cube)
    run_case(jax, "reduce_sum_f16", lambda x: jnp.sum(x, axis=1), f16_reduce)
    run_case(jax, "reduce_max_f16", lambda x: jnp.max(x, axis=1), f16_reduce)
    run_case(jax, "reduce_min_f16", lambda x: jnp.min(x, axis=1), f16_reduce)
    run_case(
        jax,
        "while_f32_lt_add",
        lambda x: jax.lax.while_loop(
            lambda state: state < np.float32(4.0),
            lambda state: state + np.float32(1.0),
            x,
        ),
        while_scalar,
    )
    run_case(
        jax,
        "while_f32_gt_subtract",
        lambda x: jax.lax.while_loop(
            lambda state: state > np.float32(0.0),
            lambda state: state - np.float32(1.0),
            x,
        ),
        np.array(5.0, dtype=np.float32),
    )

    def gated_delta_net_bf16(x: Any) -> Any:
        gate = jnp.asarray(0.5, dtype=jnp.bfloat16)
        delta = jnp.asarray(0.25, dtype=jnp.bfloat16)
        limit = jnp.asarray(1.0, dtype=jnp.bfloat16)
        step = (gate * delta).astype(jnp.bfloat16)

        return jax.lax.while_loop(
            lambda state: state < limit,
            lambda state: (state + step).astype(jnp.bfloat16),
            x.astype(jnp.bfloat16),
        )

    run_case(
        jax,
        "gated_delta_net_bf16_while",
        gated_delta_net_bf16,
        np.array(0.0, dtype=np.float32),
    )

    def gated_delta_net_bf16_body_step(x: Any) -> Any:
        gate = jnp.asarray(0.5, dtype=jnp.bfloat16)
        delta = jnp.asarray(0.25, dtype=jnp.bfloat16)
        limit = jnp.asarray(1.0, dtype=jnp.bfloat16)

        def cond(carry: tuple[Any, Any, Any, Any]) -> Any:
            _, _, limit_, state = carry
            return state < limit_

        def body(carry: tuple[Any, Any, Any, Any]) -> tuple[Any, Any, Any, Any]:
            gate_, delta_, limit_, state = carry
            step = (gate_ * delta_).astype(jnp.bfloat16)
            return gate_, delta_, limit_, (state + step).astype(jnp.bfloat16)

        _, _, _, state = jax.lax.while_loop(cond, body, (gate, delta, limit, x.astype(jnp.bfloat16)))
        return state

    run_case(
        jax,
        "gated_delta_net_bf16_body_step_while",
        gated_delta_net_bf16_body_step,
        np.array(0.0, dtype=np.float32),
    )

    def reduce_argmax_axis1(x: Any) -> tuple[Any, Any]:
        indices = jnp.broadcast_to(jnp.arange(x.shape[1], dtype=jnp.int32), x.shape)

        def reducer(left: tuple[Any, Any], right: tuple[Any, Any]) -> tuple[Any, Any]:
            v_l, i_l = left
            v_r, i_r = right
            keep_left = (v_l > v_r) | ((v_l == v_r) & (i_l < i_r))
            return jnp.where(keep_left, v_l, v_r), jnp.where(keep_left, i_l, i_r)

        return jax.lax.reduce((x, indices), (np.float32(-np.inf), np.int32(0)), reducer, dimensions=(1,))

    run_case(jax, "reduce_argmax_axis1", reduce_argmax_axis1, argmax_values)
    run_case(
        jax,
        "reduce_window_sum",
        lambda x: jax.lax.reduce_window(x, np.float32(0.0), jax.lax.add, (1, 2), (1, 1), ((0, 0), (1, 0))),
        f32_a,
    )
    run_case(
        jax,
        "reduce_window_max",
        lambda x: jax.lax.reduce_window(x, np.float32(-np.inf), jax.lax.max, (1, 2), (1, 1), "VALID"),
        f32_a,
    )
    run_case(
        jax,
        "reduce_window_sum_f16",
        lambda x: jax.lax.reduce_window(x, np.float16(0.0), jax.lax.add, (1, 2), (1, 1), "VALID"),
        f16_reduce,
    )
    run_case(
        jax,
        "reduce_window_max_f16",
        lambda x: jax.lax.reduce_window(x, np.float16(-np.inf), jax.lax.max, (1, 2), (1, 1), "VALID"),
        f16_reduce,
    )
    def reduce_window_argmax_last(x: Any) -> tuple[Any, Any]:
        indices = jnp.broadcast_to(jnp.arange(x.shape[1], dtype=jnp.int32), x.shape)

        def reducer(left: tuple[Any, Any], right: tuple[Any, Any]) -> tuple[Any, Any]:
            v_l, i_l = left
            v_r, i_r = right
            keep_left = (v_l > v_r) | ((v_l == v_r) & (i_l < i_r))
            return jnp.where(keep_left, v_l, v_r), jnp.where(keep_left, i_l, i_r)

        return jax.lax.reduce_window(
            (x, indices),
            (np.float32(-np.inf), np.int32(0)),
            reducer,
            (1, 2),
            (1, 1),
            "VALID",
        )

    run_case(jax, "reduce_window_argmax_last", reduce_window_argmax_last, f32_a)
    def random_bits_u32(key_data: Any) -> Any:
        _, bits = jax.lax.rng_bit_generator(
            key_data,
            (2, 4),
            dtype=jnp.uint32,
            algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY,
        )
        return bits

    def random_bits_u16_odd(key_data: Any) -> Any:
        _, bits = jax.lax.rng_bit_generator(
            key_data,
            (5,),
            dtype=jnp.uint16,
            algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY,
        )
        return bits

    def random_bits_u8_odd_2d(key_data: Any) -> Any:
        _, bits = jax.lax.rng_bit_generator(
            key_data,
            (3, 3),
            dtype=jnp.uint8,
            algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY,
        )
        return bits

    run_case(jax, "random_bits_u32", random_bits_u32, rng_key_data)
    run_case(jax, "random_bits_u16_odd", random_bits_u16_odd, rng_key_data)
    run_case(jax, "random_bits_u8_odd_2d", random_bits_u8_odd_2d, rng_key_data)
    run_case(jax, "matmul", lambda x, y: x @ y, mat_lhs, mat_rhs)
    run_case(jax, "matmul_f16", lambda x, y: x @ y, mat_lhs_f16, mat_rhs_f16)
    run_case(jax, "attention_scores_rhs_contract_last", lambda q, k: jnp.einsum("bhqd,bkd->bhqk", q, k), q_heads, k_cache)
    run_case(jax, "attention_values_rhs_second_last", lambda p, v: jnp.einsum("bhqk,bkd->bhqd", p, v), attn_probs, v_cache)
    run_case(
        jax,
        "convolution_1d_ncw_oic",
        lambda x, w: jax.lax.conv_general_dilated(
            x,
            w,
            window_strides=(1,),
            padding="VALID",
            dimension_numbers=("NCH", "OIH", "NCH"),
        ),
        conv_lhs,
        conv_rhs,
    )
    run_case(
        jax,
        "convolution_2d_nchw_oihw",
        lambda x, w: jax.lax.conv_general_dilated(
            x,
            w,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        ),
        conv2d_lhs,
        conv2d_rhs,
    )
    run_case(
        jax,
        "grouped_convolution_2d_nchw_oihw",
        lambda x, w: jax.lax.conv_general_dilated(
            x,
            w,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            feature_group_count=2,
        ),
        grouped_conv_lhs,
        grouped_conv_rhs,
    )
    run_case(
        jax,
        "convolution_3d_ncdhw_oidhw",
        lambda x, w: jax.lax.conv_general_dilated(
            x,
            w,
            window_strides=(1, 1, 1),
            padding="VALID",
            dimension_numbers=("NCDHW", "OIDHW", "NCDHW"),
        ),
        conv3d_lhs,
        conv3d_rhs,
    )
    run_case(jax, "rfft", lambda x: jnp.fft.rfft(x), fft_input)
    run_case(jax, "complex_real_imag", lambda x, y: jnp.real(jax.lax.complex(x, y)) + jnp.imag(jax.lax.complex(x, y)), complex_real, complex_imag)
    run_case(jax, "complex_abs", lambda x: jnp.abs(x), complex_values)
    run_case(jax, "cholesky", lambda x: jax.lax.linalg.cholesky(x, symmetrize_input=False), spd_2x2)
    run_case(
        jax,
        "triangular_solve_left_lower",
        lambda a, b: jax.lax.linalg.triangular_solve(
            a,
            b,
            left_side=True,
            lower=True,
            transpose_a=False,
            conjugate_a=False,
            unit_diagonal=False,
        ),
        triangular_lhs,
        triangular_rhs,
    )
    run_case(jax, "clip", lambda x, lo, hi: jnp.maximum(jnp.minimum(x, hi), lo), f32_a, clip_lo, clip_hi)
    run_case(jax, "bitwise_int", lambda x, y, shift: (x & y) ^ (x << shift), i32_a, i32_b, i32_shift)
    run_case(jax, "zml_float_unary_math", lambda x: jnp.sin(x) + jnp.cos(x) + jnp.log1p(jnp.abs(x)) + jax.nn.sigmoid(x), f32_a)
    run_case(jax, "zml_rounding_sign_math", lambda x: jnp.floor(x) + jnp.ceil(x) + jnp.sign(x), f32_a)
    run_case(jax, "zml_power_remainder", lambda x, y: jnp.power(jnp.abs(x) + np.float32(1.0), y) + jnp.remainder(x, y), f32_a, f32_b)
    run_case(jax, "zml_shift_right_arithmetic", lambda x, shift: x >> shift, i32_negative, i32_shift)
    run_case(jax, "zml_shift_right_logical", lambda x, shift: x >> shift, u32_bits, np.full(u32_bits.shape, 1, dtype=np.uint32))
    run_case(jax, "zml_predicate_logic", lambda x, y: jnp.logical_not(x & y) | (x ^ y), pred_a, pred_b)
    run_case(jax, "zml_is_finite", lambda x: jnp.isfinite(x), np.array([0.0, np.inf, -np.inf, np.nan], dtype=np.float32))
    run_case(jax, "cbrt", lambda x: jnp.cbrt(x), f32_a)
    run_case(jax, "round_nearest_afz", lambda x: jax.lax.round(x, rounding_method=jax.lax.RoundingMethod.AWAY_FROM_ZERO), f32_a)
    run_case(jax, "real_on_real", lambda x: jnp.real(x), f32_a)
    run_case(jax, "imag_on_real", lambda x: jnp.imag(x), f32_a)
    run_case(jax, "popcnt", lambda x: jax.lax.population_count(x), u32_bits)
    run_case(jax, "count_leading_zeros", lambda x: jax.lax.clz(x), u32_bits)
    run_case(jax, "bitcast_convert_u32_to_f32", lambda x: jax.lax.bitcast_convert_type(x, np.float32), u32_float_bits)
    run_case(jax, "sort", lambda x: jnp.sort(x, axis=1), f32_a)
    run_case(jax, "sort_descending", lambda x: jnp.sort(x, axis=1, descending=True), f32_a)
    run_case(jax, "argsort", lambda x: jnp.argsort(x, axis=1), f32_a)
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
