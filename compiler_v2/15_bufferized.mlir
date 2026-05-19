// Stage 15: buffer assignment. Logical values now have buffer lifetimes,
// alias/donation decisions, workspaces, and peak memory by space.

module @pjrtx_stage_15_bufferized attributes {
  pjrtx.state = "bufferized",
  pjrtx.stage = "buffer_assignment"
} {
  "pjrtx.state_transition"() {from = "performance_modeled", to = "bufferized", pass = "buffer_lifetime_plan+buffer_alias_plan"} : () -> ()
  "pjrtx.logical_buffer"() {id = 0 : i64, value = 0 : i64, size_bytes = 32 : i64, memory_space = "device_hbm", first_use = 0 : i64, last_use = 1 : i64} : () -> ()
  "pjrtx.logical_buffer"() {id = 1 : i64, value = 1 : i64, size_bytes = 48 : i64, memory_space = "device_hbm", first_use = 0 : i64, last_use = 1 : i64} : () -> ()
  "pjrtx.logical_buffer"() {id = 2 : i64, value = 2 : i64, size_bytes = 12 : i64, memory_space = "device_hbm", first_use = 0 : i64, last_use = 2 : i64} : () -> ()
  "pjrtx.logical_buffer"() {id = 3 : i64, value = 3 : i64, size_bytes = 24 : i64, memory_space = "device_hbm", first_use = 1 : i64, last_use = 2 : i64} : () -> ()
  "pjrtx.logical_buffer"() {id = 4 : i64, value = 6 : i64, size_bytes = 24 : i64, memory_space = "device_hbm", first_use = 2 : i64, last_use = 3 : i64} : () -> ()
  "pjrtx.physical_allocation"() {id = 0 : i64, memory_space = "device_hbm", size_bytes = 140 : i64, alignment = 64 : i64, buffers = [0 : i64, 1 : i64, 2 : i64, 3 : i64, 4 : i64]} : () -> ()
  "pjrtx.alias_plan"() {decision = "none", reason = "tiny fixture does not donate inputs or alias result"} : () -> ()
  "pjrtx.peak_memory"() {memory_space = "device_hbm", peak_live_bytes = 116 : i64, status = "fits"} : () -> ()

  func.func private @main_bufferized_memref(
      %lhs: memref<1x4xf32>,
      %rhs: memref<4x3xf32>,
      %bias: memref<3xf32>,
      %out: memref<1x3xf32>) attributes {
    pjrtx.lowering_level = "bufferized_memref",
    pjrtx.source_instructions = [0 : i64, 1 : i64, 2 : i64, 3 : i64]
  } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %f0 = arith.constant 0.0 : f32
    %tmp = memref.alloc() {pjrtx.logical_buffer = 3 : i64, memory_space = "device_hbm"} : memref<1x3xf32>
    scf.for %j = %c0 to %c3 step %c1 {
      memref.store %f0, %tmp[%c0, %j] : memref<1x3xf32>
      scf.for %k = %c0 to %c4 step %c1 {
        %acc = memref.load %tmp[%c0, %j] : memref<1x3xf32>
        %a = memref.load %lhs[%c0, %k] : memref<1x4xf32>
        %b = memref.load %rhs[%k, %j] : memref<4x3xf32>
        %prod = arith.mulf %a, %b : f32
        %next = arith.addf %acc, %prod : f32
        memref.store %next, %tmp[%c0, %j] : memref<1x3xf32>
      } {pjrtx.lowering_region = 0 : i64}
    }
    scf.for %j = %c0 to %c3 step %c1 {
      %m = memref.load %tmp[%c0, %j] : memref<1x3xf32>
      %b = memref.load %bias[%j] : memref<3xf32>
      %sum = arith.addf %m, %b : f32
      %tanh = math.tanh %sum : f32
      memref.store %tanh, %out[%c0, %j] : memref<1x3xf32>
    } {pjrtx.lowering_region = 1 : i64, pjrtx.fusion_candidate = 1 : i64}
    memref.dealloc %tmp : memref<1x3xf32>
    return
  }

  "pjrtx.buffer.func"() ({
  ^entry(%buf0: !pjrtx.buffer, %buf1: !pjrtx.buffer, %buf2: !pjrtx.buffer, %buf4: !pjrtx.buffer):
    %buf3 = "pjrtx.buffer.produce"(%buf0, %buf1) {
      logical_buffer = 3 : i64,
      lowering_region = 0 : i64,
      memory_space = "device_hbm",
      lifetime = "command[1..2]"
    } : (!pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.buffer
    "pjrtx.buffer.produce_into"(%buf3, %buf2, %buf4) {
      logical_buffer = 4 : i64,
      lowering_region = 1 : i64,
      memory_space = "device_hbm",
      lifetime = "command[2..3]"
    } : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> ()
    "pjrtx.buffer.return"(%buf4) : (!pjrtx.buffer) -> ()
  }) {sym_name = "main_bufferized"} : () -> ()
}
