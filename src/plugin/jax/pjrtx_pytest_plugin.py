"""Pytest plugin that registers PjRTx before upstream JAX tests collect."""

from __future__ import annotations

import os
import pathlib
import sys
from collections.abc import Iterable

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from pjrtx_harness import configure_pjrtx

DEFAULT_DUT = "pjrtx"
DEFAULT_TAGS = "pjrtx,metal,gpu"


def _split_tags(raw_tags: str) -> set[str]:
    tags = {tag.strip() for tag in raw_tags.split(",")}
    tags.discard("")
    return tags


def _configure_jax_test_util(jax, *, dut: str, tags: Iterable[str]) -> None:  # type: ignore[no-untyped-def]
    """Mirror JAX's backend matrix flags for a dynamically registered plugin."""
    from jax._src import test_util as jtu  # type: ignore[import-not-found]

    jax.config.update("jax_test_dut", dut)
    device_tags = set(tags)
    device_tags.add(dut)

    def pjrtx_device_tags() -> set[str]:
        return set(device_tags)

    jtu._get_device_tags = pjrtx_device_tags  # type: ignore[attr-defined]


def pytest_configure(config) -> None:  # type: ignore[no-untyped-def]
    trace = bool(config.getoption("--pjrtx-trace", default=False))
    dut = str(config.getoption("--pjrtx-jax-test-dut", default=DEFAULT_DUT))
    raw_tags = str(config.getoption("--pjrtx-device-tags", default=DEFAULT_TAGS))
    enable_x64 = bool(config.getoption("--pjrtx-enable-x64", default=False))
    allow_cpu = bool(config.getoption("--pjrtx-allow-cpu", default=False))
    jax, _ = configure_pjrtx(enable_x64=enable_x64, trace=trace, allow_cpu=allow_cpu)
    _configure_jax_test_util(jax, dut=dut, tags=_split_tags(raw_tags))


def pytest_addoption(parser) -> None:  # type: ignore[no-untyped-def]
    parser.addoption(
        "--pjrtx-trace",
        action="store_true",
        default=False,
        help="Enable PJRTX_TRACE while running upstream JAX tests.",
    )
    parser.addoption(
        "--pjrtx-jax-test-dut",
        default=os.environ.get("PJRTX_JAX_TEST_DUT", DEFAULT_DUT),
        help="Value exposed by jax._src.test_util.device_under_test().",
    )
    parser.addoption(
        "--pjrtx-device-tags",
        default=os.environ.get("PJRTX_JAX_DEVICE_TAGS", DEFAULT_TAGS),
        help="Comma-separated JAX test-util tags matched by run_on_devices/skip_on_devices.",
    )
    parser.addoption(
        "--pjrtx-enable-x64",
        action="store_true",
        default=os.environ.get("PJRTX_JAX_ENABLE_X64", "") == "1",
        help="Enable jax_enable_x64 for upstream coverage runs.",
    )
    parser.addoption(
        "--pjrtx-allow-cpu",
        action="store_true",
        default=os.environ.get("PJRTX_JAX_ALLOW_CPU", "") == "1",
        help="Allow CPU as a secondary JAX platform. Off by default for upstream plugin coverage.",
    )
