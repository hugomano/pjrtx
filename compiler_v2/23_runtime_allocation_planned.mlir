// Stage 23: runtime allocation plan. Runtime gets physical buffers, lifetimes,
// transfer requirements, and peak pressure. It does not rediscover compiler IR.

module @pjrtx_stage_23_runtime_allocation_planned attributes {
  pjrtx.state = "runtime_allocation_planned",
  pjrtx.stage = "runtime_allocation_plan"
} {
  "pjrtx.state_transition"() {from = "backend_kernel_graph_planned", to = "runtime_allocation_planned", pass = "runtime_allocation_plan"} : () -> ()
  "pjrtx.runtime_allocation"() {id = 0 : i64, logical_buffer = 0 : i64, memory_space = "device_hbm", size_bytes = 32 : i64, alignment = 64 : i64, first_command = 0 : i64, last_command = 1 : i64} : () -> ()
  "pjrtx.runtime_allocation"() {id = 1 : i64, logical_buffer = 1 : i64, memory_space = "device_hbm", size_bytes = 48 : i64, alignment = 64 : i64, first_command = 0 : i64, last_command = 1 : i64} : () -> ()
  "pjrtx.runtime_allocation"() {id = 2 : i64, logical_buffer = 2 : i64, memory_space = "device_hbm", size_bytes = 12 : i64, alignment = 64 : i64, first_command = 0 : i64, last_command = 2 : i64} : () -> ()
  "pjrtx.runtime_allocation"() {id = 3 : i64, logical_buffer = 3 : i64, memory_space = "device_hbm", size_bytes = 24 : i64, alignment = 64 : i64, first_command = 1 : i64, last_command = 2 : i64} : () -> ()
  "pjrtx.runtime_allocation"() {id = 4 : i64, logical_buffer = 4 : i64, memory_space = "device_hbm", size_bytes = 24 : i64, alignment = 64 : i64, first_command = 2 : i64, last_command = 3 : i64} : () -> ()
  "pjrtx.runtime_transfer_requirement"() {command = 0 : i64, edge = 0 : i32, bytes = 92 : i64, direction = "host_to_device"} : () -> ()
  "pjrtx.runtime_transfer_requirement"() {command = 3 : i64, edge = 0 : i32, bytes = 24 : i64, direction = "device_to_host"} : () -> ()
  "pjrtx.runtime_peak_memory"() {memory_space = "device_hbm", peak_live_bytes = 116 : i64, status = "fits"} : () -> ()

  "pjrtx.runtime.alloc_func"() ({
  ^entry:
    %buf0 = "pjrtx.runtime.allocate"() {allocation = 0 : i64, memory_space = "device_hbm", size_bytes = 32 : i64} : () -> !pjrtx.buffer
    %buf1 = "pjrtx.runtime.allocate"() {allocation = 1 : i64, memory_space = "device_hbm", size_bytes = 48 : i64} : () -> !pjrtx.buffer
    %buf2 = "pjrtx.runtime.allocate"() {allocation = 2 : i64, memory_space = "device_hbm", size_bytes = 12 : i64} : () -> !pjrtx.buffer
    %buf3 = "pjrtx.runtime.allocate"() {allocation = 3 : i64, memory_space = "device_hbm", size_bytes = 24 : i64} : () -> !pjrtx.buffer
    %buf4 = "pjrtx.runtime.allocate"() {allocation = 4 : i64, memory_space = "device_hbm", size_bytes = 24 : i64} : () -> !pjrtx.buffer
    "pjrtx.runtime.alloc_return"(%buf0, %buf1, %buf2, %buf3, %buf4) : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> ()
  }) {sym_name = "main_runtime_allocations"} : () -> ()

  "pjrtx.runtime.allocated_program"() ({
  ^entry:
    %lhs = "pjrtx.runtime.allocate"() {allocation = 0 : i64} : () -> !pjrtx.buffer
    %rhs = "pjrtx.runtime.allocate"() {allocation = 1 : i64} : () -> !pjrtx.buffer
    %bias = "pjrtx.runtime.allocate"() {allocation = 2 : i64} : () -> !pjrtx.buffer
    %tmp = "pjrtx.runtime.allocate"() {allocation = 3 : i64} : () -> !pjrtx.buffer
    %out = "pjrtx.runtime.allocate"() {allocation = 4 : i64} : () -> !pjrtx.buffer
    %h2d = "pjrtx.runtime.h2d"(%lhs, %rhs, %bias) {
      command = 0 : i64,
      transfer_bytes = 92 : i64
    } : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %matmul_done = "pjrtx.runtime.launch"(%h2d, %lhs, %rhs, %tmp) {
      command = 1 : i64,
      kernel = @npu_matmul_f32_tile
    } : (!pjrtx.event, !pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %epilogue_done = "pjrtx.runtime.launch"(%matmul_done, %tmp, %bias, %out) {
      command = 2 : i64,
      kernel = @npu_fused_broadcast_add_tanh_f32_tile
    } : (!pjrtx.event, !pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %done = "pjrtx.runtime.d2h"(%epilogue_done, %out) {
      command = 3 : i64,
      transfer_bytes = 24 : i64
    } : (!pjrtx.event, !pjrtx.buffer) -> !pjrtx.event
    "pjrtx.runtime.return"(%done, %out) : (!pjrtx.event, !pjrtx.buffer) -> ()
  }) {sym_name = "main_runtime_allocated_program"} : () -> ()
}
