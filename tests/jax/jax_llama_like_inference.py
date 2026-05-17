#!/usr/bin/env python3
"""Run a tiny Llama-like JAX forward pass through the PjRTx plugin."""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from jax_plugin_smoke import import_jax, plugin_path, register_plugin


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


def main() -> int:
    jax, jnp = import_jax()
    path = plugin_path()
    os.environ.setdefault("PJRTX_BACKEND", "metal_mlx")
    register_plugin(path)
    jax.config.update("jax_platforms", "pjrtx")
    jax.config.update("jax_platform_name", "pjrtx")

    @jax.jit
    def llama_like(x, norm_w, wq, wk, wv, wo, w1, w2, w3):
        eps = jnp.array(1.0e-5, dtype=jnp.float32)
        scale = jnp.array(0.5, dtype=jnp.float32)
        denom = jnp.sum(x * x, axis=-1, keepdims=True) * jnp.array(0.25, dtype=jnp.float32) + eps
        h = x * jax.lax.rsqrt(denom) * norm_w

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
        return x + attn + mlp

    rng = np.random.default_rng(7)
    x = rng.normal(size=(2, 4)).astype(np.float32) * np.float32(0.2)
    norm_w = rng.normal(size=(4,)).astype(np.float32) * np.float32(0.1) + np.float32(1.0)
    wq = rng.normal(size=(4, 4)).astype(np.float32) * np.float32(0.1)
    wk = rng.normal(size=(4, 4)).astype(np.float32) * np.float32(0.1)
    wv = rng.normal(size=(4, 4)).astype(np.float32) * np.float32(0.1)
    wo = rng.normal(size=(4, 4)).astype(np.float32) * np.float32(0.1)
    w1 = rng.normal(size=(4, 8)).astype(np.float32) * np.float32(0.1)
    w2 = rng.normal(size=(8, 4)).astype(np.float32) * np.float32(0.1)
    w3 = rng.normal(size=(4, 8)).astype(np.float32) * np.float32(0.1)

    got = np.asarray(
        jax.device_get(
            llama_like(
                x,
                norm_w,
                wq,
                wk,
                wv,
                wo,
                w1,
                w2,
                w3,
            )
        )
    )
    want = np_reference(x, norm_w, wq, wk, wv, wo, w1, w2, w3)
    np.testing.assert_allclose(got, want, rtol=2.0e-4, atol=2.0e-4)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"PjRTx plugin: {pathlib.Path(path)}")
    print(f"llama-like checksum: {float(np.sum(got)):.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
