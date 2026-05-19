#!/usr/bin/env bash
set -euo pipefail

repo="${TEST_SRCDIR:-}"
if [[ -n "${repo}" && -d "${repo}/_main" ]]; then
  cd "${repo}/_main"
fi

fail() {
  echo "architecture boundary violation: $*" >&2
  exit 1
}

if grep -Eq 'src/runtime|src/backend' src/compiler/compiler.zig; then
  fail "compiler must not import runtime or backend"
fi

if grep -Eq '@import\("c"\)|PjrtxMlx|pjrtx_mlx|mlx_metal' src/runtime/runtime.zig; then
  fail "runtime must not import or mention MLX/Metal C API symbols"
fi

if grep -Eq 'PjrtxMlx|pjrtx_mlx|mlx_metal_api' src/plugin/plugin.zig; then
  fail "plugin must not import or mention MLX C shim symbols"
fi

if grep -Eq 'executeInstruction|runtime_fallback|fallback_instruction_count|allow_runtime_fallback|test_only_runtime_fallback' src/runtime/runtime.zig; then
  fail "runtime must not contain an instruction-interpreter fallback"
fi

if grep -Eq 'evalReduce|evalDotGeneral|sortDenseBytes|seedFromBytes|nextRandomU32|readScalarAsF64|writeScalarFromF64|scalarIndexAt|dense host fallback|host shadow' src/runtime/runtime.zig; then
  fail "runtime must not contain CPU operation fallback helpers"
fi

if grep -Eq 'bufferFromHost\(.*\) orelse null|allocator\.dupe\(u8, src\)' src/runtime/runtime.zig; then
  fail "host imports must materialize backend device storage, not host-only buffers"
fi

if grep -Eq 'loadedExecutableExecuteLegacy|dense host fallback' src/plugin/plugin.zig; then
  fail "plugin execute path must stay graph/backend-only; no legacy or host fallback executor"
fi

if grep -Eq '//src/runtime|//src/backend' src/compiler/BUILD.bazel; then
  fail "compiler BUILD deps must stay independent of runtime/backend"
fi

if grep -Eq 'mlx_metal_api|mlx_metal_backend' src/runtime/BUILD.bazel; then
  fail "runtime BUILD deps must stay independent of MLX backend implementation"
fi

if find src/backend -maxdepth 1 -type f -name 'mlx_*' | grep -q .; then
  fail "backend implementations must live under backend-specific directories"
fi

echo "architecture boundaries OK"
