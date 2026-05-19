"""Shared Bazel/JAX harness utilities for the PjRTx plugin tests."""

from __future__ import annotations

import contextlib
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np

from jax_plugin_smoke import import_jax, plugin_path, register_plugin


TRACE_KV_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")
PLUGIN_NAME = "pjrtx"
_PLUGIN_REGISTERED = False


def configure_pjrtx(*, enable_x64: bool = False, trace: bool = False, allow_cpu: bool = True) -> tuple[Any, Any]:
    """Import JAX, register the PjRTx plugin, and select test platforms."""
    global _PLUGIN_REGISTERED
    if trace:
        os.environ.setdefault("PJRTX_TRACE", "1")
    os.environ.setdefault("PJRTX_BACKEND", "metal_mlx")

    jax, jnp = import_jax()
    if enable_x64:
        jax.config.update("jax_enable_x64", True)

    if not _PLUGIN_REGISTERED:
        register_plugin(plugin_path())
        _PLUGIN_REGISTERED = True
    platforms = f"{PLUGIN_NAME},cpu" if allow_cpu else PLUGIN_NAME
    jax.config.update("jax_platforms", platforms)
    jax.config.update("jax_platform_name", PLUGIN_NAME)
    return jax, jnp


@contextlib.contextmanager
def captured_stderr() -> Iterator[tempfile._TemporaryFileWrapper[bytes]]:
    """Capture process stderr, including C/C++ plugin writes."""
    sys.stderr.flush()
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile() as trace_file:
        try:
            os.dup2(trace_file.fileno(), 2)
            yield trace_file
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)


def capture_stderr(call: Callable[[], Any]) -> tuple[str, Any]:
    with captured_stderr() as trace_file:
        result = call()
        sys.stderr.flush()
        trace_file.seek(0)
        trace_text = trace_file.read().decode("utf-8", errors="replace")
    return trace_text, result


def parse_pjrtx_traces(trace_text: str) -> list[dict[str, str]]:
    traces: list[dict[str, str]] = []
    for line in trace_text.splitlines():
        if line.startswith("pjrtx_trace "):
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


def assert_successful_api_calls(name: str, traces: list[dict[str, str]]) -> None:
    for trace in traces:
        if trace.get("event") != "pjrt_api_call":
            continue
        if trace_int(trace, "failed") != 0 or trace_int(trace, "error_code") != 0:
            raise AssertionError(f"{name}: PJRT API call failed: {trace}")


def stablehlo_text(jax: Any, fn: Callable[..., Any], inputs: Sequence[Any]) -> str:
    lowered = jax.jit(fn, backend=PLUGIN_NAME).lower(*inputs)
    return str(lowered.compiler_ir(dialect="stablehlo"))


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


def assert_backend_trace(name: str, trace_text: str, *, expect_compile: bool) -> None:
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
    assert_successful_api_calls(name, traces)


def run_cpu_vs_pjrtx(
    jax: Any,
    name: str,
    fn: Callable[..., Any],
    *inputs: Any,
    check_trace: bool = True,
) -> None:
    cpu_fn = jax.jit(fn, backend="cpu")
    want = jax.device_get(cpu_fn(*inputs))

    pjrtx_fn = jax.jit(fn, backend=PLUGIN_NAME)
    try:
        first_trace, got_first = capture_stderr(lambda: jax.device_get(pjrtx_fn(*inputs)))
        second_trace, got_second = capture_stderr(lambda: jax.device_get(pjrtx_fn(*inputs)))
        assert_close(f"{name}/first", got_first, want)
        assert_close(f"{name}/second", got_second, want)
        if check_trace:
            assert_backend_trace(f"{name}/first", first_trace, expect_compile=True)
            assert_backend_trace(f"{name}/second", second_trace, expect_compile=False)
    except Exception as err:
        try:
            hlo = stablehlo_text(jax, fn, inputs)
        except Exception as lowering_err:
            hlo = f"<stablehlo capture failed: {lowering_err!r}>"
        raise AssertionError(f"{name}: failed with {err!r}\n\nStableHLO:\n{hlo}") from err
