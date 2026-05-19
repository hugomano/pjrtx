#!/usr/bin/env python3
"""Run a tiny JAX program through the registered PjRTx backend."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

import numpy as np

PLUGIN_RUNFILE = "src/plugin/libpjrtx_metal_plugin.dylib"
PLUGIN_NAME = "pjrtx"


def find_runfile(path: str) -> pathlib.Path:
    candidates: list[pathlib.Path] = []

    test_srcdir = os.environ.get("TEST_SRCDIR")
    test_workspace = os.environ.get("TEST_WORKSPACE")
    if test_srcdir and test_workspace:
        candidates.append(pathlib.Path(test_srcdir) / test_workspace / path)

    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if runfiles_dir:
        if test_workspace:
            candidates.append(pathlib.Path(runfiles_dir) / test_workspace / path)
        candidates.append(pathlib.Path(runfiles_dir) / path)

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repo_root / path,
            repo_root / "bazel-bin" / path,
            pathlib.Path.cwd() / path,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not find runfile {path}; tried:\n  {tried}")


def plugin_path() -> pathlib.Path:
    env_path = os.environ.get("PJRTX_PLUGIN_PATH")
    if env_path:
        return pathlib.Path(env_path)
    return find_runfile(PLUGIN_RUNFILE)


def import_jax() -> tuple[Any, Any]:
    try:
        import jax  # type: ignore[import-not-found]
        import jax.numpy as jnp  # type: ignore[import-not-found]
    except ModuleNotFoundError as err:
        print("JAX sandbox requires the pinned jax/jaxlib Bazel wheel deps.", file=sys.stderr)
        raise err
    return jax, jnp


def register_plugin(path: pathlib.Path) -> None:
    options: dict[str, str | int] = {
        "pjrtx_backend": os.environ.get("PJRTX_BACKEND", "metal_mlx"),
    }

    from jax._src import xla_bridge as xb  # type: ignore[import-not-found]

    if hasattr(xb, "register_plugin"):
        xb.register_plugin(
            PLUGIN_NAME,
            priority=500,
            library_path=str(path),
            options=options,
        )
        return

    from jaxlib import xla_client  # type: ignore[import-not-found]

    if not hasattr(xla_client, "load_pjrt_plugin_dynamically"):
        raise RuntimeError("jaxlib does not expose load_pjrt_plugin_dynamically")
    xla_client.load_pjrt_plugin_dynamically(PLUGIN_NAME, str(path))
    if hasattr(xla_client, "initialize_pjrt_plugin"):
        xla_client.initialize_pjrt_plugin(PLUGIN_NAME)


def main() -> int:
    path = plugin_path()
    jax, jnp = import_jax()
    register_plugin(path)

    jax.config.update("jax_platforms", PLUGIN_NAME)
    jax.config.update("jax_platform_name", PLUGIN_NAME)

    @jax.jit
    def add(x, y):
        return x + y

    lhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    rhs = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    got = [float(v) for v in jax.device_get(add(lhs, rhs))]
    want = [11.0, 22.0, 33.0, 44.0]
    if got != want:
        raise AssertionError(f"unexpected PjRTx result: got {got}, want {want}")

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"PjRTx plugin: {path}")
    print(f"result: {got}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
