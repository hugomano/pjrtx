#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${TEST_SRCDIR:-}" ]]; then
  if [[ -n "${TEST_WORKSPACE:-}" && -d "${TEST_SRCDIR}/${TEST_WORKSPACE}" ]]; then
    cd "${TEST_SRCDIR}/${TEST_WORKSPACE}"
  elif [[ -d "${TEST_SRCDIR}/_main" ]]; then
    cd "${TEST_SRCDIR}/_main"
  fi
fi

failures=0

fail() {
  printf 'architecture boundary violation: %s\n' "$1" >&2
  failures=1
}

source_files() {
  find pjrtx -type f \( -name '*.zig' -o -name BUILD.bazel \) \
    ! -path 'next/pjrtx/architecture/*' \
    | sort
}

forbid_pattern() {
  local message="$1"
  local file_prefix="$2"
  local pattern="$3"
  local matches

  matches="$(source_files | grep "^${file_prefix}" | xargs grep -En "${pattern}" 2>/dev/null || true)"
  if [[ -n "${matches}" ]]; then
    fail "${message}"
    printf '%s\n' "${matches}" >&2
  fi
}

expected_core_lines() {
  case "$1" in
    pjrtx/core/BUILD.bazel) echo 1 ;;
    pjrtx/compiler/BUILD.bazel) echo 1 ;;
    pjrtx/compiler/compiler.zig) echo 1 ;;
    pjrtx/compiler/mlir_state/BUILD.bazel) echo 1 ;;
    pjrtx/compiler/mlir_state/package.zig) echo 1 ;;
    pjrtx/compiler/mlir_state/types.zig) echo 1 ;;
    pjrtx/backend/BUILD.bazel) echo 2 ;;
    pjrtx/backend/backend.zig) echo 1 ;;
    pjrtx/backend/mlir_bridge.zig) echo 1 ;;
    pjrtx/runtime/BUILD.bazel) echo 2 ;;
    pjrtx/runtime/runtime.zig) echo 1 ;;
    pjrtx/runtime/mlir_bridge.zig) echo 1 ;;
    pjrtx/plugin/BUILD.bazel) echo 1 ;;
    pjrtx/plugin/plugin.zig) echo 1 ;;
    next/pjrtx/vertical_slice/BUILD.bazel) echo 5 ;;
    next/pjrtx/vertical_slice/backend_binding_test.zig) echo 1 ;;
    next/pjrtx/vertical_slice/execution_test.zig) echo 1 ;;
    next/pjrtx/vertical_slice/import_test.zig) echo 1 ;;
    next/pjrtx/vertical_slice/lowering_test.zig) echo 1 ;;
    next/pjrtx/vertical_slice/report_test.zig) echo 1 ;;
    *) echo 0 ;;
  esac
}

check_core_baseline() {
  local file
  local actual
  local expected

  while IFS= read -r file; do
    actual="$(grep -Ec 'pjrtx/core|//next/pjrtx/core' "$file" || true)"
    expected="$(expected_core_lines "$file")"
    if [[ "${actual}" != "${expected}" ]]; then
      fail "unexpected //next/pjrtx/core reference count in ${file}: expected ${expected}, got ${actual}"
      grep -En 'pjrtx/core|//next/pjrtx/core' "$file" >&2 || true
    fi
  done < <(source_files)
}

forbid_pattern \
  "compiler must not depend on backend, runtime, or plugin packages" \
  "pjrtx/compiler/" \
  'pjrtx/(backend|runtime|plugin)|//next/pjrtx/(backend|runtime|plugin)'

forbid_pattern \
  "compiler facts must not depend on deprecated core" \
  "pjrtx/compiler/facts/" \
  'pjrtx/core|//next/pjrtx/core'

forbid_pattern \
  "target must not depend on compiler, backend, runtime, plugin, or core packages" \
  "pjrtx/target/" \
  'pjrtx/(compiler|backend|runtime|plugin|core)|//next/pjrtx/(compiler|backend|runtime|plugin|core)'

forbid_pattern \
  "backend must not depend on runtime or plugin packages" \
  "pjrtx/backend/" \
  'pjrtx/(runtime|plugin)|//next/pjrtx/(runtime|plugin)'

forbid_pattern \
  "runtime must not depend on backend or plugin packages" \
  "pjrtx/runtime/" \
  'pjrtx/(backend|plugin)|//next/pjrtx/(backend|plugin)'

forbid_pattern \
  "plugin must not depend on backend implementation packages" \
  "pjrtx/plugin/" \
  'pjrtx/backend|//next/pjrtx/backend'

forbid_pattern \
  "deprecated core package must not depend on the migrated pjrtx packages" \
  "pjrtx/core/" \
  'pjrtx/(compiler|backend|runtime|plugin)|//next/pjrtx/(compiler|backend|runtime|plugin)'

core_sources_matches="$(source_files | xargs grep -En '//next/pjrtx/core:sources' 2>/dev/null || true)"
if [[ -n "${core_sources_matches}" ]]; then
  fail "do not reintroduce //next/pjrtx/core:sources"
  printf '%s\n' "${core_sources_matches}" >&2
fi

check_core_baseline

exit "${failures}"
