// Stage 19: backend binding. Lowering regions and schedule commands are bound
// to concrete backend calls without letting the backend make hidden compiler
// choices.

module @pjrtx_stage_19_backend_bound attributes {
  pjrtx.state = "backend_bound",
  pjrtx.stage = "backend_binding_select+backend_binding_verify"
} {
  "pjrtx.state_transition"() {from = "scheduled", to = "backend_bound", pass = "backend_binding_verify"} : () -> ()
  "pjrtx.backend_binding"() {
    command = 1 : i64,
    kernel = 0 : i64,
    lowering_region = 0 : i64,
    backend = "npu_v0",
    call = "library:npu_matmul_f32_tile",
    inputs = [0 : i64, 1 : i64],
    outputs = [3 : i64],
    verified = true
  } : () -> ()
  "pjrtx.backend_binding"() {
    command = 2 : i64,
    kernel = 1 : i64,
    lowering_region = 1 : i64,
    backend = "npu_v0",
    call = "generated:npu_fused_broadcast_add_tanh_f32_tile",
    inputs = [3 : i64, 2 : i64],
    outputs = [4 : i64],
    verified = true
  } : () -> ()

  "pjrtx.backend.func"() ({
  ^entry(%buf0: !pjrtx.buffer, %buf1: !pjrtx.buffer, %buf2: !pjrtx.buffer, %buf4: !pjrtx.buffer):
    %buf3 = "pjrtx.backend.bound_call"(%buf0, %buf1) {
      binding = 0 : i64,
      command = 1 : i64,
      callee = @npu_matmul_f32_tile
    } : (!pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.buffer
    "pjrtx.backend.bound_call"(%buf3, %buf2, %buf4) {
      binding = 1 : i64,
      command = 2 : i64,
      callee = @npu_fused_broadcast_add_tanh_f32_tile
    } : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> ()
    "pjrtx.backend.return"(%buf4) : (!pjrtx.buffer) -> ()
  }) {sym_name = "main_backend_bound"} : () -> ()
}
