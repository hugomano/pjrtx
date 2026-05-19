// Stage 20: compiler executable contract. This is the handoff boundary before
// backend executable expansion and runtime plan extraction.

module @pjrtx_stage_20_executable_ready attributes {
  pjrtx.state = "executable_ready",
  pjrtx.stage = "executable_view_create"
} {
  "pjrtx.state_transition"() {from = "backend_bound", to = "executable_ready", pass = "executable_contract_verify"} : () -> ()
  "pjrtx.executable_contract"() {
    entry = @main,
    target = "npu_v0",
    schedule_commands = 4 : i32,
    backend_bindings = 2 : i32,
    kernels = 2 : i32,
    allocation_plan_required = true,
    runtime_stream_plan_required = true,
    profile_join_required = true,
    no_runtime_fallback = true
  } : () -> ()

  "pjrtx.executable.func"() ({
  ^entry(%arg0: !pjrtx.buffer, %arg1: !pjrtx.buffer, %arg2: !pjrtx.buffer):
    %result = "pjrtx.executable.call_schedule"(%arg0, %arg1, %arg2) {
      schedule = @main_schedule,
      backend = "npu_v0",
      no_runtime_fallback = true
    } : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.buffer
    "pjrtx.executable.return"(%result) : (!pjrtx.buffer) -> ()
  }) {sym_name = "main"} : () -> ()

  "pjrtx.executable.program"() ({
  ^entry(%lhs: !pjrtx.buffer, %rhs: !pjrtx.buffer, %bias: !pjrtx.buffer, %out: !pjrtx.buffer):
    %h2d = "pjrtx.exec.h2d"(%lhs, %rhs, %bias) {
      command = 0 : i64,
      transfer_edge = 0 : i32
    } : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %tmp, %matmul_done = "pjrtx.exec.kernel"(%h2d, %lhs, %rhs) {
      command = 1 : i64,
      kernel = @npu_matmul_f32_tile,
      lowering_region = 0 : i64
    } : (!pjrtx.event, !pjrtx.buffer, !pjrtx.buffer) -> (!pjrtx.buffer, !pjrtx.event)
    %epilogue_done = "pjrtx.exec.kernel"(%matmul_done, %tmp, %bias, %out) {
      command = 2 : i64,
      kernel = @npu_fused_broadcast_add_tanh_f32_tile,
      lowering_region = 1 : i64
    } : (!pjrtx.event, !pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %done = "pjrtx.exec.d2h"(%epilogue_done, %out) {
      command = 3 : i64,
      transfer_edge = 0 : i32
    } : (!pjrtx.event, !pjrtx.buffer) -> !pjrtx.event
    "pjrtx.exec.return"(%done, %out) : (!pjrtx.event, !pjrtx.buffer) -> ()
  }) {
    sym_name = "main_executable_program",
    no_runtime_fallback = true
  } : () -> ()
}
