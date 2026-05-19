#!/usr/bin/env python3
"""Run a tiny Llama-like JAX forward pass through the PjRTx plugin."""

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


TRACE_KV_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")
DEFAULT_RESIDENT_CONSTANT_OVERHEAD_BUDGET = 128
DEFAULT_STEADY_EXECUTE_BUDGET_US = 200_000
EXECUTE_COUNT = 3


def capture_stderr(call: Callable[[], Any]) -> tuple[str, Any]:
    sys.stderr.flush()
    saved_stderr = os.dup(2)
    with tempfile.TemporaryFile() as trace_file:
        try:
            os.dup2(trace_file.fileno(), 2)
            result = call()
            sys.stderr.flush()
        finally:
            os.dup2(saved_stderr, 2)
            os.close(saved_stderr)
        trace_file.seek(0)
        trace_text = trace_file.read().decode("utf-8", errors="replace")
    if trace_text:
        sys.stderr.write(trace_text)
        sys.stderr.flush()
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


def env_int(name: str, default: int) -> int:
    text = os.environ.get(name)
    if text is None or text == "":
        return default
    try:
        return int(text)
    except ValueError as err:
        raise AssertionError(f"{name} must be an integer, got {text!r}") from err


def fixture_budget(name: str, suffix: str, default: int) -> int:
    specific = f"PJRTX_{name.upper()}_{suffix}"
    generic = f"PJRTX_{suffix}"
    return env_int(specific, env_int(generic, default))


def assert_resident_constant_trace(
    trace_text: str,
    activation_bytes: int,
    min_resident_constant_bytes: int,
    output_bytes: int,
    execute_count: int,
) -> dict[str, int]:
    traces = parse_pjrtx_traces(trace_text)
    compile_traces = [trace for trace in traces if trace.get("event") == "compile"]
    compile_cache_traces = [trace for trace in traces if trace.get("event") == "compile_cache"]
    h2d_traces = [trace for trace in traces if trace.get("event") == "h2d"]
    execute_traces = [trace for trace in traces if trace.get("event") == "execute"]
    d2h_traces = [trace for trace in traces if trace.get("event") == "d2h"]
    if not compile_traces:
        raise AssertionError("PjRTx trace did not include a compile event")
    if not compile_cache_traces:
        raise AssertionError("PjRTx trace did not include a compile_cache event")
    if len(execute_traces) < execute_count:
        raise AssertionError("PjRTx trace did not include repeated execute events")

    compile_trace = compile_traces[-1]
    compile_cache_trace = compile_cache_traces[-1]
    checked_executes = execute_traces[-execute_count:]

    if trace_int(compile_trace, "backend_executable") != 1:
        raise AssertionError(f"llama-like program did not compile to a backend executable: {compile_trace}")
    if trace_int(compile_trace, "unlowered") != 0:
        raise AssertionError(f"llama-like program left instructions unlowered: {compile_trace}")
    if trace_int(compile_trace, "backend_devices") != 1:
        raise AssertionError(f"llama-like program did not report one backend device: {compile_trace}")
    cache_hit = trace_int(compile_trace, "cache_hit")
    cache_hits_total = trace_int(compile_trace, "cache_hits_total")
    cache_misses_total = trace_int(compile_trace, "cache_misses_total")
    backend_cache_reuse = trace_int(compile_trace, "backend_cache_reuse")
    backend_cache_compile_samples = trace_int(compile_cache_trace, "backend_cache_compile_samples")
    backend_cache_compile_us_total = trace_int(compile_cache_trace, "backend_cache_compile_us_total")
    backend_cache_compile_us_peak = trace_int(compile_cache_trace, "backend_cache_compile_us_peak")
    if backend_cache_compile_samples <= 0 or backend_cache_compile_us_total <= 0 or backend_cache_compile_us_peak <= 0:
        raise AssertionError(f"compile cache trace did not report executable cache compile latency: {compile_cache_trace}")
    if trace_int(compile_cache_trace, "cache_pressure_failed") != 0:
        raise AssertionError(f"compile resident-constant cache pressure should be satisfied: {compile_cache_trace}")
    if trace_int(compile_cache_trace, "cache_trim_bytes") < 0 or trace_int(compile_cache_trace, "cache_trim_evictions") < 0:
        raise AssertionError(f"compile cache pressure trace reported invalid trim counters: {compile_cache_trace}")

    resident_constants = trace_int(compile_trace, "resident_constants")
    resident_constant_bytes = trace_int(compile_trace, "resident_constant_bytes")
    if resident_constants <= 0 or resident_constant_bytes <= 0:
        raise AssertionError(f"llama-like program did not report resident constants: {compile_trace}")
    if resident_constant_bytes < min_resident_constant_bytes:
        raise AssertionError(
            "resident constant bytes did not cover closed-over weights: "
            f"got={resident_constant_bytes} want_at_least={min_resident_constant_bytes}"
        )
    program_values = trace_int(compile_trace, "program_values")
    program_nodes = trace_int(compile_trace, "program_nodes")
    program_edges = trace_int(compile_trace, "program_edges")
    schedule_items = trace_int(compile_trace, "schedule_items")
    subprograms = trace_int(compile_trace, "subprograms")
    control_flows = trace_int(compile_trace, "control_flows")
    fusion_groups = trace_int(compile_trace, "fusion_groups")
    materialization_boundaries = trace_int(compile_trace, "materialization_boundaries")
    planned_releases = trace_int(compile_trace, "planned_releases")
    planned_release_bytes = trace_int(compile_trace, "planned_release_bytes")
    peak_live_values = trace_int(compile_trace, "peak_live_values")
    peak_live_bytes = trace_int(compile_trace, "peak_live_bytes")
    if program_values <= 0 or program_nodes <= 0 or schedule_items <= 0:
        raise AssertionError(f"compile trace did not report a populated backend program: {compile_trace}")
    if program_edges <= 0:
        raise AssertionError(f"backend program did not report dependency edges: {compile_trace}")
    if fusion_groups <= 0:
        raise AssertionError(f"backend program did not report fusible groups: {compile_trace}")
    if materialization_boundaries <= 0:
        raise AssertionError(f"backend program did not report output materialization boundaries: {compile_trace}")
    if planned_releases <= 0:
        raise AssertionError(f"backend program did not report planned releases: {compile_trace}")
    if planned_release_bytes <= 0:
        raise AssertionError(f"backend program did not report planned release bytes: {compile_trace}")
    if peak_live_values <= 0 or peak_live_values > program_values:
        raise AssertionError(f"backend program reported invalid peak live value count: {compile_trace}")
    if peak_live_bytes <= 0:
        raise AssertionError(f"backend program reported invalid peak live byte count: {compile_trace}")

    if len(h2d_traces) != execute_count:
        raise AssertionError(
            f"expected {execute_count} activation H2D transfers, got {len(h2d_traces)}: {h2d_traces}"
        )
    activation_upload_bytes = 0
    for trace in h2d_traces:
        transfer_bytes = trace_int(trace, "bytes")
        activation_upload_bytes += transfer_bytes
        if transfer_bytes != activation_bytes:
            raise AssertionError(f"unexpected execute-time H2D size; expected activation only: {trace}")

    if len(d2h_traces) != execute_count:
        raise AssertionError(
            f"expected {execute_count} output D2H transfers, got {len(d2h_traces)}: {d2h_traces}"
        )
    output_readback_bytes = 0
    for trace in d2h_traces:
        transfer_bytes = trace_int(trace, "bytes")
        output_readback_bytes += transfer_bytes
        if transfer_bytes != output_bytes:
            raise AssertionError(f"unexpected D2H size; expected output only: {trace}")

    previous_borrows = 0
    previous_released_intermediates = 0
    execute_elapsed_us = 0
    first_execute_elapsed_us = 0
    last_execute_elapsed_us = 0
    steady_execute_elapsed_us = 0
    steady_execute_count = 0
    final_fusion_group_executes = 0
    final_materialization_evals = 0
    final_materialization_buffers = 0
    final_released_intermediates = 0
    for expected_execute_count, trace in enumerate(checked_executes, start=1):
        if trace_int(trace, "backend_executes") != expected_execute_count:
            raise AssertionError(f"execute did not advance backend execute count: {trace}")
        if trace_int(trace, "last_device_index") != 0 or trace_int(trace, "last_local_hardware_id") != 0:
            raise AssertionError(f"execute dispatched to an unexpected backend device slot: {trace}")
        if trace_int(trace, "resident_constants") != resident_constants:
            raise AssertionError(f"resident constant count changed across execute: {trace}")
        if trace_int(trace, "resident_constant_bytes") != resident_constant_bytes:
            raise AssertionError(f"resident constant bytes changed across execute: {trace}")
        if trace_int(trace, "cache_pressure_failed") != 0:
            raise AssertionError(f"execute output allocation pressure should be satisfied: {trace}")
        if trace_int(trace, "backend_completion_pending") != 0:
            raise AssertionError(f"MLX execute should report completed backend dispatches: {trace}")
        if trace_int(trace, "cache_trim_bytes") < 0 or trace_int(trace, "cache_trim_evictions") < 0:
            raise AssertionError(f"execute cache pressure trace reported invalid trim counters: {trace}")
        fusion_group_executes = trace_int(trace, "fusion_group_executes")
        if fusion_group_executes != fusion_groups * expected_execute_count:
            raise AssertionError(f"execute did not run each fusion group exactly once: {trace}")
        materialization_evals = trace_int(trace, "materialization_evals")
        if materialization_evals != expected_execute_count:
            raise AssertionError(f"execute did not materialize each PJRT output batch: {trace}")
        materialization_buffers = trace_int(trace, "materialization_buffers")
        if materialization_buffers != materialization_boundaries * expected_execute_count:
            raise AssertionError(f"execute did not eval each materialized output buffer: {trace}")
        released_intermediates = trace_int(trace, "released_intermediates")
        if released_intermediates != planned_releases * expected_execute_count:
            raise AssertionError(f"execute releases did not match compile-time liveness plan: {trace}")
        if released_intermediates <= previous_released_intermediates:
            raise AssertionError(f"execute did not release backend intermediates: {trace}")
        previous_released_intermediates = released_intermediates
        borrows = trace_int(trace, "borrowed_constant_nodes")
        if borrows <= previous_borrows:
            raise AssertionError(f"execute did not borrow resident constants again: {trace}")
        previous_borrows = borrows
        final_fusion_group_executes = fusion_group_executes
        final_materialization_evals = materialization_evals
        final_materialization_buffers = materialization_buffers
        final_released_intermediates = released_intermediates
        elapsed_us = trace_int(trace, "elapsed_us")
        execute_elapsed_us += elapsed_us
        if expected_execute_count == 1:
            first_execute_elapsed_us = elapsed_us
        else:
            steady_execute_elapsed_us += elapsed_us
            steady_execute_count += 1
        last_execute_elapsed_us = elapsed_us

    return {
        "activation_upload_bytes": activation_upload_bytes,
        "cache_hit": cache_hit,
        "cache_hits_total": cache_hits_total,
        "cache_misses_total": cache_misses_total,
        "backend_cache_reuse": backend_cache_reuse,
        "backend_cache_compile_samples": backend_cache_compile_samples,
        "backend_cache_compile_us_total": backend_cache_compile_us_total,
        "backend_cache_compile_us_peak": backend_cache_compile_us_peak,
        "compile_elapsed_us": trace_int(compile_trace, "elapsed_us"),
        "execute_elapsed_us": execute_elapsed_us,
        "first_execute_elapsed_us": first_execute_elapsed_us,
        "fusion_group_execute_count": final_fusion_group_executes,
        "last_execute_elapsed_us": last_execute_elapsed_us,
        "materialization_eval_buffer_count": final_materialization_buffers,
        "materialization_eval_count": final_materialization_evals,
        "output_readback_bytes": output_readback_bytes,
        "program_edge_count": program_edges,
        "program_fusion_group_count": fusion_groups,
        "program_materialization_boundary_count": materialization_boundaries,
        "program_node_count": program_nodes,
        "program_peak_live_bytes": peak_live_bytes,
        "program_peak_live_value_count": peak_live_values,
        "program_planned_release_bytes": planned_release_bytes,
        "program_planned_release_count": planned_releases,
        "program_schedule_item_count": schedule_items,
        "program_subprogram_count": subprograms,
        "program_control_flow_count": control_flows,
        "program_value_count": program_values,
        "released_intermediate_count": final_released_intermediates,
        "resident_constants": resident_constants,
        "resident_constant_bytes": resident_constant_bytes,
        "steady_execute_count": steady_execute_count,
        "steady_execute_elapsed_us": steady_execute_elapsed_us,
    }


def assert_cache_hit_trace(
    trace_text: str,
    activation_bytes: int,
    output_bytes: int,
    baseline: dict[str, int],
) -> dict[str, int]:
    traces = parse_pjrtx_traces(trace_text)
    compile_traces = [trace for trace in traces if trace.get("event") == "compile"]
    compile_cache_traces = [trace for trace in traces if trace.get("event") == "compile_cache"]
    h2d_traces = [trace for trace in traces if trace.get("event") == "h2d"]
    execute_traces = [trace for trace in traces if trace.get("event") == "execute"]
    d2h_traces = [trace for trace in traces if trace.get("event") == "d2h"]
    if not compile_traces:
        raise AssertionError("cache probe did not include a compile event")
    if not compile_cache_traces:
        raise AssertionError("cache probe did not include a compile_cache event")
    compile_trace = compile_traces[-1]
    compile_cache_trace = compile_cache_traces[-1]
    if trace_int(compile_trace, "cache_hit") != 1:
        raise AssertionError(f"cache probe did not hit executable fingerprint cache: {compile_trace}")
    if trace_int(compile_trace, "backend_cache_reuse") != 1:
        raise AssertionError(f"cache probe did not reuse backend executable: {compile_trace}")
    if trace_int(compile_trace, "backend_executable") != 1 or trace_int(compile_trace, "unlowered") != 0:
        raise AssertionError(f"cache probe did not stay on backend executable fast path: {compile_trace}")
    if trace_int(compile_trace, "backend_devices") != 1:
        raise AssertionError(f"cache probe did not report one backend device: {compile_trace}")
    if trace_int(compile_trace, "cache_hits_total") <= baseline["cache_hits_total"]:
        raise AssertionError(f"cache hit total did not advance: {compile_trace}")
    if trace_int(compile_trace, "cache_misses_total") < baseline["cache_misses_total"]:
        raise AssertionError(f"cache miss total regressed: {compile_trace}")
    if trace_int(compile_cache_trace, "backend_cache_compile_samples") != baseline["backend_cache_compile_samples"]:
        raise AssertionError(f"cache reuse should not add a compile latency sample: {compile_cache_trace}")
    if trace_int(compile_cache_trace, "backend_cache_compile_us_total") != baseline["backend_cache_compile_us_total"]:
        raise AssertionError(f"cache reuse should not add compile latency total: {compile_cache_trace}")
    if trace_int(compile_cache_trace, "backend_cache_compile_us_peak") != baseline["backend_cache_compile_us_peak"]:
        raise AssertionError(f"cache reuse should not change compile latency peak: {compile_cache_trace}")
    if trace_int(compile_cache_trace, "cache_pressure_failed") != 0:
        raise AssertionError(f"cache reuse compile pressure should be satisfied: {compile_cache_trace}")

    checked_fields = {
        "resident_constants": "resident_constants",
        "resident_constant_bytes": "resident_constant_bytes",
        "program_values": "program_value_count",
        "program_nodes": "program_node_count",
        "program_edges": "program_edge_count",
        "schedule_items": "program_schedule_item_count",
        "subprograms": "program_subprogram_count",
        "control_flows": "program_control_flow_count",
        "fusion_groups": "program_fusion_group_count",
        "materialization_boundaries": "program_materialization_boundary_count",
        "planned_releases": "program_planned_release_count",
        "planned_release_bytes": "program_planned_release_bytes",
        "peak_live_values": "program_peak_live_value_count",
        "peak_live_bytes": "program_peak_live_bytes",
    }
    for trace_key, summary_key in checked_fields.items():
        got = trace_int(compile_trace, trace_key)
        want = baseline[summary_key]
        if got != want:
            raise AssertionError(f"cache probe changed {trace_key}: got={got} want={want} trace={compile_trace}")

    if len(h2d_traces) != 1 or trace_int(h2d_traces[0], "bytes") != activation_bytes:
        raise AssertionError(f"cache probe should upload exactly one activation: {h2d_traces}")
    if len(execute_traces) != 1:
        raise AssertionError(f"cache probe should execute exactly once: {execute_traces}")
    if trace_int(execute_traces[-1], "backend_executes") != EXECUTE_COUNT + 1:
        raise AssertionError(f"cache probe should reuse backend execute counter: {execute_traces[-1]}")
    if trace_int(execute_traces[-1], "last_device_index") != 0 or trace_int(execute_traces[-1], "last_local_hardware_id") != 0:
        raise AssertionError(f"cache probe dispatched to an unexpected backend device slot: {execute_traces[-1]}")
    if trace_int(execute_traces[-1], "cache_pressure_failed") != 0:
        raise AssertionError(f"cache probe execute output allocation pressure should be satisfied: {execute_traces[-1]}")
    if trace_int(execute_traces[-1], "backend_completion_pending") != 0:
        raise AssertionError(f"cache probe should report completed backend dispatches: {execute_traces[-1]}")
    if trace_int(execute_traces[-1], "borrowed_constant_nodes") != baseline["resident_constants"] * (EXECUTE_COUNT + 1):
        raise AssertionError(f"cache probe should reuse resident constant borrow counter: {execute_traces[-1]}")
    if len(d2h_traces) != 1 or trace_int(d2h_traces[0], "bytes") != output_bytes:
        raise AssertionError(f"cache probe should read exactly one output: {d2h_traces}")

    return {
        "cache_probe_compile_elapsed_us": trace_int(compile_trace, "elapsed_us"),
        "cache_probe_hits_total": trace_int(compile_trace, "cache_hits_total"),
        "cache_probe_misses_total": trace_int(compile_trace, "cache_misses_total"),
        "cache_probe_execute_elapsed_us": trace_int(execute_traces[-1], "elapsed_us"),
    }


def np_softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def np_reference(
    x: np.ndarray,
    norm_w: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    wo: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
) -> np.ndarray:
    eps = np.float32(1.0e-5)
    scale = np.float32(0.5)
    denom = np.sum(x * x, axis=-1, keepdims=True) * np.float32(0.25) + eps
    h = x * (np.float32(1.0) / np.sqrt(denom)) * norm_w

    q = h @ wq
    k = h @ wk
    v = h @ wv
    attn = np_softmax((q @ k.T) * scale) @ v
    attn = attn @ wo

    gate = np.tanh(h @ w1)
    up = h @ w3
    mlp = (gate * up) @ w2
    return x + attn + mlp


def make_fixture(rng: np.random.Generator, seq_len: int, model_dim: int, ff_dim: int):
    x = rng.normal(size=(seq_len, model_dim)).astype(np.float32) * np.float32(0.2)
    norm_w = rng.normal(size=(model_dim,)).astype(np.float32) * np.float32(0.1) + np.float32(
        1.0
    )
    wq = rng.normal(size=(model_dim, model_dim)).astype(np.float32) * np.float32(0.1)
    wk = rng.normal(size=(model_dim, model_dim)).astype(np.float32) * np.float32(0.1)
    wv = rng.normal(size=(model_dim, model_dim)).astype(np.float32) * np.float32(0.1)
    wo = rng.normal(size=(model_dim, model_dim)).astype(np.float32) * np.float32(0.1)
    w1 = rng.normal(size=(model_dim, ff_dim)).astype(np.float32) * np.float32(0.1)
    w2 = rng.normal(size=(ff_dim, model_dim)).astype(np.float32) * np.float32(0.1)
    w3 = rng.normal(size=(model_dim, ff_dim)).astype(np.float32) * np.float32(0.1)
    return x, norm_w, wq, wk, wv, wo, w1, w2, w3


def fixture_weight_bytes(weights: tuple[np.ndarray, ...]) -> int:
    return sum(weight.nbytes for weight in weights)


def assert_fixture_budgets(name: str, summary: dict[str, int]) -> None:
    resident_overhead_budget = fixture_budget(
        name,
        "RESIDENT_CONSTANT_OVERHEAD_BUDGET_BYTES",
        DEFAULT_RESIDENT_CONSTANT_OVERHEAD_BUDGET,
    )
    resident_overhead = summary["resident_constant_bytes"] - summary["weight_bytes"]
    if resident_overhead < 0 or resident_overhead > resident_overhead_budget:
        raise AssertionError(
            f"{name}: resident constant overhead {resident_overhead} bytes exceeds "
            f"budget {resident_overhead_budget} bytes"
        )

    expected_activation_upload_bytes = summary["activation_bytes"] * EXECUTE_COUNT
    if summary["activation_upload_bytes"] != expected_activation_upload_bytes:
        raise AssertionError(
            f"{name}: activation upload bytes {summary['activation_upload_bytes']} != "
            f"{expected_activation_upload_bytes}"
        )

    expected_output_readback_bytes = summary["output_bytes"] * EXECUTE_COUNT
    if summary["output_readback_bytes"] != expected_output_readback_bytes:
        raise AssertionError(
            f"{name}: output readback bytes {summary['output_readback_bytes']} != "
            f"{expected_output_readback_bytes}"
        )

    if summary["steady_execute_count"] != EXECUTE_COUNT - 1:
        raise AssertionError(
            f"{name}: steady execute count {summary['steady_execute_count']} != {EXECUTE_COUNT - 1}"
        )

    steady_budget_us = fixture_budget(name, "STEADY_EXECUTE_BUDGET_US", DEFAULT_STEADY_EXECUTE_BUDGET_US)
    if summary["steady_execute_elapsed_us"] > steady_budget_us:
        raise AssertionError(
            f"{name}: steady execute time {summary['steady_execute_elapsed_us']}us "
            f"exceeds budget {steady_budget_us}us"
        )

    summary["resident_constant_overhead_bytes"] = resident_overhead
    summary["resident_constant_overhead_budget_bytes"] = resident_overhead_budget
    summary["steady_execute_budget_us"] = steady_budget_us


def run_fixture(
    jax: Any,
    jnp: Any,
    name: str,
    fixture: tuple[np.ndarray, ...],
) -> tuple[float, dict[str, int]]:
    x, norm_w, wq, wk, wv, wo, w1, w2, w3 = fixture
    weight_bytes = fixture_weight_bytes((norm_w, wq, wk, wv, wo, w1, w2, w3))

    def make_llama_like():
        @jax.jit
        def llama_like(input_x):
            eps = jnp.array(1.0e-5, dtype=jnp.float32)
            scale = jnp.array(0.5, dtype=jnp.float32)
            denom = (
                jnp.sum(input_x * input_x, axis=-1, keepdims=True)
                * jnp.array(0.25, dtype=jnp.float32)
                + eps
            )
            h = input_x * jax.lax.rsqrt(denom) * norm_w

            q = h @ wq
            k = h @ wk
            v = h @ wv
            scores = (q @ jnp.transpose(k)) * scale
            weights = jnp.exp(scores - jnp.max(scores, axis=-1, keepdims=True))
            weights = weights / jnp.sum(weights, axis=-1, keepdims=True)
            attn = (weights @ v) @ wo

            gate = jnp.tanh(h @ w1)
            up = h @ w3
            mlp = (gate * up) @ w2
            return input_x + attn + mlp

        return llama_like

    llama_like = make_llama_like()

    def run_repeated() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        first = np.asarray(jax.device_get(llama_like(x)))
        second = np.asarray(jax.device_get(llama_like(x)))
        third = np.asarray(jax.device_get(llama_like(x)))
        return first, second, third

    trace_text, outputs = capture_stderr(run_repeated)
    trace_summary = assert_resident_constant_trace(
        trace_text,
        activation_bytes=x.nbytes,
        min_resident_constant_bytes=weight_bytes,
        output_bytes=x.nbytes,
        execute_count=EXECUTE_COUNT,
    )
    got = outputs[0]
    want = np_reference(x, norm_w, wq, wk, wv, wo, w1, w2, w3)
    for index, output in enumerate(outputs):
        np.testing.assert_allclose(output, want, rtol=2.0e-4, atol=2.0e-4, err_msg=f"{name}/execute[{index}]")
    trace_summary["activation_bytes"] = x.nbytes
    trace_summary["output_bytes"] = x.nbytes
    trace_summary["weight_bytes"] = weight_bytes
    assert_fixture_budgets(name, trace_summary)

    cache_probe = make_llama_like()

    def run_cache_probe() -> np.ndarray:
        return np.asarray(jax.device_get(cache_probe(x)))

    cache_trace_text, cache_output = capture_stderr(run_cache_probe)
    np.testing.assert_allclose(cache_output, want, rtol=2.0e-4, atol=2.0e-4, err_msg=f"{name}/cache_probe")
    trace_summary.update(
        assert_cache_hit_trace(
            cache_trace_text,
            activation_bytes=x.nbytes,
            output_bytes=x.nbytes,
            baseline=trace_summary,
        )
    )
    print(f"{name} checksum: {float(np.sum(got)):.8f}")
    print(
        f"{name} resident constants: "
        f"{trace_summary['resident_constants']} "
        f"({trace_summary['resident_constant_bytes']} bytes), "
        f"closed weights: {weight_bytes} bytes, "
        f"overhead: {trace_summary['resident_constant_overhead_bytes']} bytes, "
        f"activation uploads: {trace_summary['activation_upload_bytes']} bytes, "
        f"output readbacks: {trace_summary['output_readback_bytes']} bytes"
    )
    print(
        f"{name} executable cache: "
        f"first_hit={trace_summary['cache_hit']} "
        f"first_backend_reuse={trace_summary['backend_cache_reuse']} "
        f"hits_total={trace_summary['cache_probe_hits_total']} "
        f"misses_total={trace_summary['cache_probe_misses_total']} "
        f"hit_compile_us={trace_summary['cache_probe_compile_elapsed_us']} "
        f"hit_execute_us={trace_summary['cache_probe_execute_elapsed_us']}"
    )
    print(
        f"{name} backend program: "
        f"values={trace_summary['program_value_count']} "
        f"nodes={trace_summary['program_node_count']} "
        f"edges={trace_summary['program_edge_count']} "
        f"schedule={trace_summary['program_schedule_item_count']} "
        f"subprograms={trace_summary['program_subprogram_count']} "
        f"control_flows={trace_summary['program_control_flow_count']} "
        f"fusion_groups={trace_summary['program_fusion_group_count']} "
        f"materializations={trace_summary['program_materialization_boundary_count']} "
        f"planned_releases={trace_summary['program_planned_release_count']} "
        f"planned_release_bytes={trace_summary['program_planned_release_bytes']} "
        f"peak_live={trace_summary['program_peak_live_value_count']} "
        f"peak_live_bytes={trace_summary['program_peak_live_bytes']} "
        f"fusion_executes={trace_summary['fusion_group_execute_count']} "
        f"evals={trace_summary['materialization_eval_count']} "
        f"eval_buffers={trace_summary['materialization_eval_buffer_count']} "
        f"released={trace_summary['released_intermediate_count']}"
    )
    print(
        f"{name} latency us: "
        f"compile={trace_summary['compile_elapsed_us']} "
        f"execute_total={trace_summary['execute_elapsed_us']} "
        f"execute_first={trace_summary['first_execute_elapsed_us']} "
        f"execute_last={trace_summary['last_execute_elapsed_us']} "
        f"steady_total={trace_summary['steady_execute_elapsed_us']} "
        f"steady_count={trace_summary['steady_execute_count']} "
        f"steady_budget={trace_summary['steady_execute_budget_us']}"
    )
    return float(np.sum(got)), trace_summary


def main() -> int:
    jax, jnp = import_jax()
    path = plugin_path()
    os.environ.setdefault("PJRTX_BACKEND", "metal_mlx")
    os.environ["PJRTX_TRACE"] = "1"
    register_plugin(path)
    jax.config.update("jax_platforms", "pjrtx")
    jax.config.update("jax_platform_name", "pjrtx")

    rng = np.random.default_rng(7)
    tiny_checksum, tiny_summary = run_fixture(jax, jnp, "tiny", make_fixture(rng, 2, 4, 8))
    medium_checksum, medium_summary = run_fixture(jax, jnp, "medium", make_fixture(rng, 4, 16, 32))

    print(f"JAX backend: {jax.default_backend()}")
    print(f"PjRTx plugin: {pathlib.Path(path)}")
    print(f"llama-like checksum: {tiny_checksum:.8f}")
    print(
        "resident constants: "
        f"{tiny_summary['resident_constants']} tiny / {medium_summary['resident_constants']} medium, "
        f"{tiny_summary['resident_constant_bytes']} tiny bytes / "
        f"{medium_summary['resident_constant_bytes']} medium bytes"
    )
    print(
        "warm execute us: "
        f"{tiny_summary['last_execute_elapsed_us']} tiny / "
        f"{medium_summary['last_execute_elapsed_us']} medium"
    )
    print(
        "steady execute us: "
        f"{tiny_summary['steady_execute_elapsed_us']} tiny / "
        f"{medium_summary['steady_execute_elapsed_us']} medium"
    )
    _ = medium_checksum
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
