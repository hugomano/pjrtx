// Stage 18A: async schedule IR.
// Commands are values with token dependencies. This models real overlap
// decisions without hiding the program in command metadata.

module @pjrtx_stage_18a_async_schedule_ir attributes {
  pjrtx.state = "scheduled",
  pjrtx.stage = "async_schedule_ir"
} {
  "pjrtx.async.func"() ({
  ^entry(%lhs: !pjrtx.buffer, %rhs: !pjrtx.buffer, %bias: !pjrtx.buffer, %out: !pjrtx.buffer):
    %h2d = "pjrtx.async.transfer"(%lhs, %rhs, %bias) {
      command = 0 : i64,
      stream = "dma0",
      direction = "host_to_device",
      bytes = 92 : i64
    } : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.async.token
    %matmul_out, %matmul_done = "pjrtx.async.launch"(%h2d, %lhs, %rhs) {
      command = 1 : i64,
      stream = "compute0",
      kernel = @npu_matmul_f32_tile,
      produces = 3 : i64
    } : (!pjrtx.async.token, !pjrtx.buffer, !pjrtx.buffer) -> (!pjrtx.buffer, !pjrtx.async.token)
    %epilogue_done = "pjrtx.async.launch"(%matmul_done, %matmul_out, %bias, %out) {
      command = 2 : i64,
      stream = "compute0",
      kernel = @npu_fused_broadcast_add_tanh_f32_tile,
      consumes = 3 : i64,
      produces = 6 : i64
    } : (!pjrtx.async.token, !pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.async.token
    %d2h = "pjrtx.async.transfer"(%epilogue_done, %out) {
      command = 3 : i64,
      stream = "dma0",
      direction = "device_to_host",
      bytes = 24 : i64
    } : (!pjrtx.async.token, !pjrtx.buffer) -> !pjrtx.async.token
    "pjrtx.async.await"(%d2h) : (!pjrtx.async.token) -> ()
    "pjrtx.async.return"(%out) : (!pjrtx.buffer) -> ()
  }) {
    sym_name = "main_async_schedule",
    rejected_overlaps = ["h2d_compute:input_dependency", "compute_d2h:result_dependency"]
  } : () -> ()
}
