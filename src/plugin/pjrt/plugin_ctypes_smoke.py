#!/usr/bin/env python3
"""Load the PjRTx plugin dylib and verify that GetPjrtApi is exported."""

from __future__ import annotations

import ctypes
import os
import pathlib
import sys


PLUGIN_RUNFILE = "src/plugin/libpjrtx_metal_plugin.dylib"
PJRT_EXTENSION_TYPE_GPU_CUSTOM_CALL = 0


class PjrtExtensionBase(ctypes.Structure):
    pass


PjrtExtensionBasePtr = ctypes.POINTER(PjrtExtensionBase)
PjrtExtensionBase._fields_ = [
    ("struct_size", ctypes.c_size_t),
    ("type", ctypes.c_uint),
    ("next", PjrtExtensionBasePtr),
]


class PjrtApiVersion(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", PjrtExtensionBasePtr),
        ("major_version", ctypes.c_int),
        ("minor_version", ctypes.c_int),
    ]


class PjrtApiPrefix(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", PjrtExtensionBasePtr),
        ("pjrt_api_version", PjrtApiVersion),
    ]


class GpuRegisterCustomCallArgs(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("function_name", ctypes.c_char_p),
        ("function_name_size", ctypes.c_size_t),
        ("api_version", ctypes.c_int),
        ("handler_instantiate", ctypes.c_void_p),
        ("handler_prepare", ctypes.c_void_p),
        ("handler_initialize", ctypes.c_void_p),
        ("handler_execute", ctypes.c_void_p),
    ]


GpuRegisterCustomCallFn = ctypes.CFUNCTYPE(
    ctypes.c_void_p, ctypes.POINTER(GpuRegisterCustomCallArgs)
)


class GpuCustomCallExtension(ctypes.Structure):
    _fields_ = [
        ("base", PjrtExtensionBase),
        ("custom_call", GpuRegisterCustomCallFn),
    ]


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

    api = ctypes.cast(api_ptr, ctypes.POINTER(PjrtApiPrefix)).contents
    extension = api.extension_start
    gpu_custom_call: GpuCustomCallExtension | None = None
    while extension:
        base = extension.contents
        if base.type == PJRT_EXTENSION_TYPE_GPU_CUSTOM_CALL:
            gpu_custom_call = ctypes.cast(
                extension, ctypes.POINTER(GpuCustomCallExtension)
            ).contents
            break
        extension = base.next
    if gpu_custom_call is None:
        print("PJRT GPU custom-call extension not found", file=sys.stderr)
        return 1

    binary_add_marker = ctypes.cast(plugin.PjRTx_CustomCall_BinaryAdd, ctypes.c_void_p)
    extension_target = b"pjrtx.ctypes.extension_binary_add"
    extension_args = GpuRegisterCustomCallArgs(
        struct_size=ctypes.sizeof(GpuRegisterCustomCallArgs),
        function_name=extension_target,
        function_name_size=len(extension_target),
        api_version=0,
        handler_instantiate=None,
        handler_prepare=None,
        handler_initialize=None,
        handler_execute=binary_add_marker.value,
    )
    err_ptr = gpu_custom_call.custom_call(ctypes.byref(extension_args))
    if err_ptr:
        print("PJRT GPU custom-call extension registration failed", file=sys.stderr)
        return 1

    register_binary = plugin.PjRTx_RegisterCustomCallBinary
    register_binary.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    register_binary.restype = ctypes.c_void_p
    unregister = plugin.PjRTx_UnregisterCustomCall
    unregister.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    unregister.restype = None

    target = b"pjrtx.ctypes.binary_add"
    op = b"add"
    err_ptr = register_binary(target, len(target), op, len(op))
    if err_ptr:
        print("PjRTx_RegisterCustomCallBinary returned an error", file=sys.stderr)
        return 1
    unregister(target, len(target))

    print(f"PjRTx plugin loaded: {plugin_path}")
    print(f"GetPjrtApi: 0x{api_ptr:x}")
    print("PJRT GPU custom-call extension: ok")
    print("PjRTx custom call registration: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
