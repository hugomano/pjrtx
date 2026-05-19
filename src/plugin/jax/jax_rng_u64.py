#!/usr/bin/env python3
"""CPU-vs-PjRTx check for x64 Threefry rng_bit_generator output."""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from jax_plugin_smoke import import_jax, plugin_path, register_plugin


def main() -> int:
    jax, jnp = import_jax()
    jax.config.update("jax_enable_x64", True)

    os.environ.setdefault("PJRTX_BACKEND", "metal_mlx")
    register_plugin(plugin_path())
    jax.config.update("jax_platforms", "pjrtx,cpu")
    jax.config.update("jax_platform_name", "pjrtx")

    def random_bits_u64(key_data: Any) -> Any:
        _, bits = jax.lax.rng_bit_generator(
            key_data,
            (2, 4),
            dtype=jnp.uint64,
            algorithm=jax.lax.RandomAlgorithm.RNG_THREE_FRY,
        )
        return bits

    key_data = np.array([0, 7, 0, 0], dtype=np.uint32)
    cpu_fn = jax.jit(random_bits_u64, backend="cpu")
    pjrtx_fn = jax.jit(random_bits_u64, backend="pjrtx")
    want = jax.device_get(cpu_fn(key_data))
    got_first = jax.device_get(pjrtx_fn(key_data))
    got_second = jax.device_get(pjrtx_fn(key_data))
    np.testing.assert_array_equal(np.asarray(got_first), np.asarray(want))
    np.testing.assert_array_equal(np.asarray(got_second), np.asarray(want))
    print("jax_rng_u64 ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
