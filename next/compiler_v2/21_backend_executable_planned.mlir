// Stage 21: backend executable call sequence. This is still explainable MLIR
// state, not an opaque submitted command buffer.

module @pjrtx_stage_21_backend_executable_planned attributes {
  pjrtx.state = "backend_executable_planned",
  pjrtx.stage = "backend_executable_plan"
} {
  "pjrtx.state_transition"() {from = "executable_ready", to = "backend_executable_planned", pass = "backend_executable_plan"} : () -> ()
  "pjrtx.backend_executable"() {
    backend = "npu_v0",
    entry = @main,
    calls = [0 : i64, 1 : i64],
    constants = [],
    command_count = 4 : i32
  } : () -> ()
  "pjrtx.backend_executable_call"() {index = 0 : i64, command = 1 : i64, binding = 0 : i64, call = "npu_matmul_f32_tile", inputs = [0 : i64, 1 : i64], outputs = [3 : i64]} : () -> ()
  "pjrtx.backend_executable_call"() {index = 1 : i64, command = 2 : i64, binding = 1 : i64, call = "npu_fused_broadcast_add_tanh_f32_tile", inputs = [3 : i64, 2 : i64], outputs = [4 : i64]} : () -> ()

  "pjrtx.backend.executable"() ({
  ^entry(%buf0: !pjrtx.buffer, %buf1: !pjrtx.buffer, %buf2: !pjrtx.buffer, %buf3: !pjrtx.buffer, %buf4: !pjrtx.buffer):
    "pjrtx.backend.call"(%buf0, %buf1, %buf3) {
      index = 0 : i64,
      command = 1 : i64,
      callee = @npu_matmul_f32_tile,
      lowering_region = 0 : i64
    } : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> ()
    "pjrtx.backend.call"(%buf3, %buf2, %buf4) {
      index = 1 : i64,
      command = 2 : i64,
      callee = @npu_fused_broadcast_add_tanh_f32_tile,
      lowering_region = 1 : i64
    } : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> ()
    "pjrtx.backend.return"(%buf4) : (!pjrtx.buffer) -> ()
  }) {sym_name = "main_backend_executable", backend = "npu_v0"} : () -> ()
}
