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

if find src/compiler -name '*.zig' -print0 | xargs -0 grep -Eq 'src/runtime|src/backend'; then
  fail "compiler must not import runtime or backend"
fi

if [[ -e src/core ]]; then
  fail "src/core must not exist; compiler owns IR and tensor vocabulary"
fi

if find src -name '*.zig' -print0 | xargs -0 grep -Eq 'src/core'; then
  fail "no Zig source may import src/core"
fi

if grep -R -Eq '//src/core' src MODULE.bazel; then
  fail "no Bazel target may depend on //src/core"
fi

if [[ -e src/backend/backend.zig || -e src/backend/registry.zig ]]; then
  fail "root backend vtable/registry files must not exist"
fi

if find src -name '*.zig' -print0 | xargs -0 grep -Eq 'Backend\.VTable|backend_registry|src/backend/registry'; then
  fail "dynamic backend vtable/registry usage must not return"
fi

if find src/runtime -name '*.zig' -print0 | xargs -0 grep -Eq '@import\("c"\)|PjrtxMlx|pjrtx_mlx|mlx_metal_api'; then
  fail "runtime must not import or mention MLX/Metal C API symbols"
fi

if grep -Eq 'PjrtxMlx|pjrtx_mlx|mlx_metal_api' src/plugin/plugin.zig; then
  fail "plugin must not import or mention MLX C shim symbols"
fi

if find src/runtime -name '*.zig' -print0 | xargs -0 grep -Eq 'executeInstruction|runtime_fallback|fallback_instruction_count|allow_runtime_fallback|test_only_runtime_fallback'; then
  fail "runtime must not contain an instruction-interpreter fallback"
fi

if find src/runtime -name '*.zig' -print0 | xargs -0 grep -Eq 'evalReduce|evalDotGeneral|sortDenseBytes|seedFromBytes|nextRandomU32|readScalarAsF64|writeScalarFromF64|scalarIndexAt|dense host fallback|host shadow'; then
  fail "runtime must not contain CPU operation fallback helpers"
fi

if find src/runtime -name '*.zig' -print0 | xargs -0 grep -Eq 'bufferFromHost\(.*\) orelse null|allocator\.dupe\(u8, src\)'; then
  fail "host imports must materialize backend device storage, not host-only buffers"
fi

if grep -Eq 'loadedExecutableExecuteLegacy|dense host fallback' src/plugin/plugin.zig; then
  fail "plugin execute path must stay graph/backend-only; no legacy or host fallback executor"
fi

if grep -Eq '//src/runtime|//src/backend' src/compiler/BUILD.bazel; then
  fail "compiler BUILD deps must stay independent of runtime/backend"
fi

if grep -Eq '//src/backend(:|")|//src/backend:registry|//src/backend:backend' src/runtime/BUILD.bazel; then
  fail "runtime must depend on the concrete MLX backend package, not a root backend facade"
fi

if grep -Eq 'mlx_metal_api|mlx_metal_backend' src/runtime/BUILD.bazel; then
  fail "runtime must not depend directly on MLX C shim targets"
fi

if find src/backend -maxdepth 1 -type f -name 'mlx_*' | grep -q .; then
  fail "backend implementations must live under backend-specific directories"
fi

if [[ -e src/backend/synthetic ]]; then
  fail "synthetic backend code must not exist in //src; MLX Metal is the only production backend"
fi

echo "architecture boundaries OK"
