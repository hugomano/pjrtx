#!/usr/bin/env python3
"""Exercise a PjRTx MLX/Metal backend custom call from JAX."""

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


TARGET = "pjrtx.mlx_metal.custom_binary_add_f32"
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


def assert_backend_custom_call_trace(trace_text: str) -> None:
    if os.environ.get("PJRTX_TRACE") != "1":
        return
    traces = parse_pjrtx_traces(trace_text)
    compile_traces = api_call_traces(traces, "PJRT_Client_Compile")
    execute_traces = api_call_traces(traces, "PJRT_LoadedExecutable_Execute")
    if not compile_traces:
        raise AssertionError("custom-call run did not emit PJRT_Client_Compile")
    if not execute_traces:
        raise AssertionError("custom-call run did not emit PJRT_LoadedExecutable_Execute")
    for trace in traces:
        if trace.get("event") != "pjrt_api_call":
            continue
        if trace_int(trace, "failed") != 0 or trace_int(trace, "error_code") != 0:
            raise AssertionError(f"custom-call PJRT API call failed: {trace}")


def main() -> int:
    jax, jnp = import_jax()

    os.environ.setdefault("PJRTX_BACKEND", "metal_mlx")
    path = plugin_path()
    register_plugin(path)

    jax.config.update("jax_platforms", "pjrtx,cpu")
    jax.config.update("jax_platform_name", "pjrtx")
    if not jax.devices("pjrtx"):
        raise RuntimeError("PjRTx backend did not initialize")

    def custom_add(x, y):
        result = jax.ffi.ffi_call(
            TARGET,
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            custom_call_api_version=0,
        )(x, y)
        return jnp.sqrt(result + jnp.asarray(1.0, dtype=x.dtype))

    lhs = np.array([1.5, 2.25, 4.0, 8.5], dtype=np.float32)
    rhs = np.array([4.0, -0.25, 12.0, 7.5], dtype=np.float32)
    want = np.sqrt(lhs + rhs + np.float32(1.0))

    lowered = jax.jit(custom_add, backend="pjrtx").lower(lhs, rhs)
    stablehlo = str(lowered.compiler_ir(dialect="stablehlo"))
    if "stablehlo.custom_call" not in stablehlo or TARGET not in stablehlo:
        raise AssertionError(f"JAX did not lower through stablehlo.custom_call:\n{stablehlo}")

    pjrtx_fn = jax.jit(custom_add, backend="pjrtx")
    trace_text, got = capture_stderr(lambda: jax.device_get(pjrtx_fn(lhs, rhs)))
    np.testing.assert_allclose(np.asarray(got), want, rtol=2.0e-4, atol=2.0e-4)
    assert_backend_custom_call_trace(trace_text)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"PjRTx plugin: {path}")
    print(f"custom call target: {TARGET}")
    print(f"result: {[float(v) for v in np.asarray(got)]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
