#!/usr/bin/env python3
"""Load the PjRTx plugin dylib and verify that GetPjrtApi is exported."""

from __future__ import annotations

import ctypes
import os
import pathlib
import sys


PLUGIN_RUNFILE = "src/plugin/libpjrtx_metal_plugin.dylib"


def find_runfile(path: str) -> pathlib.Path:
    """Resolve a Bazel runfile while still allowing direct local execution."""

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


def main() -> int:
    env_path = os.environ.get("PJRTX_PLUGIN_PATH")
    plugin_path = pathlib.Path(env_path) if env_path else find_runfile(PLUGIN_RUNFILE)

    plugin = ctypes.CDLL(str(plugin_path))
    get_pjrt_api = plugin.GetPjrtApi
    get_pjrt_api.argtypes = []
    get_pjrt_api.restype = ctypes.c_void_p

    api_ptr = get_pjrt_api()
    if api_ptr == 0:
        print("GetPjrtApi returned null", file=sys.stderr)
        return 1

    print(f"PjRTx plugin loaded: {plugin_path}")
    print(f"GetPjrtApi: 0x{api_ptr:x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
