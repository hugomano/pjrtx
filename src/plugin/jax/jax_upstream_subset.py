#!/usr/bin/env python3
"""Run upstream JAX tests against the PjRTx plugin."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))


def bundled_jax_source() -> pathlib.Path | None:
    roots = []
    for env_name in ("RUNFILES_DIR", "TEST_SRCDIR"):
        if os.environ.get(env_name):
            roots.append(pathlib.Path(os.environ[env_name]).resolve())

    here = pathlib.Path(__file__).absolute()
    roots.extend([here.parent, *here.parents])

    seen: set[pathlib.Path] = set()
    for root in roots:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        manifest = root / "MANIFEST"
        if manifest.exists():
            try:
                for raw_line in manifest.read_text().splitlines():
                    if "+non_module_deps+jax_upstream/jax/__init__.py" not in raw_line:
                        continue
                    parts = raw_line.split(" ", 1)
                    if len(parts) == 2:
                        candidate = pathlib.Path(parts[1]).resolve().parents[1]
                        if (candidate / "tests").is_dir():
                            return candidate
            except OSError:
                pass

        direct_candidates = [
            root / "jax_upstream",
            root / "+non_module_deps+jax_upstream",
            root.parent / "+non_module_deps+jax_upstream",
            root / "external" / "+non_module_deps+jax_upstream",
            root / "external" / "jax_upstream",
        ]
        for candidate in direct_candidates:
            if (candidate / "jax" / "__init__.py").exists() and (candidate / "tests").is_dir():
                return candidate

        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for child in children:
            if not child.is_dir():
                continue
            if child.name.endswith("jax_upstream") and (child / "jax" / "__init__.py").exists():
                return child
    return None


def read_allowlist(path: pathlib.Path) -> list[str]:
    entries: list[str] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jax_source",
        default=os.environ.get("PJRTX_JAX_SOURCE", ""),
        help="Path to a local upstream JAX checkout. Defaults to the Bazel @jax_upstream runfile.",
    )
    parser.add_argument(
        "--allowlist",
        default=str(pathlib.Path(__file__).with_name("jax_upstream_allowlist.txt")),
        help="File containing pytest node ids relative to the JAX checkout.",
    )
    parser.add_argument(
        "--all-tests",
        action="store_true",
        help="Run the full upstream JAX tests/ tree instead of the curated allowlist.",
    )
    parser.add_argument("--trace", action="store_true", help="Enable PJRTX_TRACE=1.")
    parser.add_argument(
        "--jax-test-dut",
        default=os.environ.get("PJRTX_JAX_TEST_DUT", "pjrtx"),
        help="JAX test_util device_under_test() value for upstream tests.",
    )
    parser.add_argument(
        "--device-tags",
        default=os.environ.get("PJRTX_JAX_DEVICE_TAGS", "pjrtx,metal,gpu"),
        help="Comma-separated JAX test_util tags for skip/run_on_devices decorators.",
    )
    parser.add_argument(
        "--enable-x64",
        action="store_true",
        default=os.environ.get("PJRTX_JAX_ENABLE_X64", "") == "1",
        help="Enable jax_enable_x64 while running upstream tests.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        default=os.environ.get("PJRTX_JAX_ALLOW_CPU", "") == "1",
        help="Allow CPU as a secondary JAX platform. Defaults to plugin-only.",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Collect the selected upstream JAX tests without executing them.",
    )
    args, extra_pytest_args = parser.parse_known_args()
    if extra_pytest_args and extra_pytest_args[0] == "--":
        extra_pytest_args = extra_pytest_args[1:]

    jax_source = pathlib.Path(args.jax_source).resolve() if args.jax_source else bundled_jax_source()
    if jax_source is None:
        print("Could not locate Bazel @jax_upstream; set PJRTX_JAX_SOURCE=/path/to/jax or pass --jax_source.", file=sys.stderr)
        return 2

    if not jax_source.exists():
        print(f"JAX source checkout does not exist: {jax_source}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(jax_source))
    import pytest

    if args.all_tests:
        pytest_targets = [str(jax_source / "tests")]
    else:
        allowlist = pathlib.Path(args.allowlist)
        node_ids = read_allowlist(allowlist)
        if not node_ids:
            print(f"Upstream JAX allowlist is empty: {allowlist}", file=sys.stderr)
            return 2
        pytest_targets = [str(jax_source / node_id) for node_id in node_ids]

    pytest_args = [
        "-p",
        "pjrtx_pytest_plugin",
        "--import-mode=importlib",
        "-q",
    ]
    if args.trace:
        pytest_args.append("--pjrtx-trace")
    pytest_args.extend(
        [
            f"--pjrtx-jax-test-dut={args.jax_test_dut}",
            f"--pjrtx-device-tags={args.device_tags}",
        ]
    )
    if args.enable_x64:
        pytest_args.append("--pjrtx-enable-x64")
    if args.allow_cpu:
        pytest_args.append("--pjrtx-allow-cpu")
    if args.collect_only:
        pytest_args.append("--collect-only")
    pytest_args.extend(extra_pytest_args)
    pytest_args.extend(pytest_targets)
    return int(pytest.main(pytest_args))


if __name__ == "__main__":
    raise SystemExit(main())
