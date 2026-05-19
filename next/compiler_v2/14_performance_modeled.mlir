// Stage 14: performance model. Cost, traffic, roofline, and bottleneck records
// are joined to target hardware facts, not backend logs.

module @pjrtx_stage_14_performance_modeled attributes {
  pjrtx.state = "performance_modeled",
  pjrtx.stage = "logical_cost_estimate+memory_traffic_refine+roofline"
} {
  "pjrtx.state_transition"() {from = "lowering_planned", to = "performance_modeled", pass = "performance_model"} : () -> ()
  "pjrtx.cost_ledger"() {
    id = 0 : i64,
    lowering_region = 0 : i64,
    source_instructions = [0 : i64],
    op_class = "matmul",
    dtype = "f32",
    accumulation_dtype = "f32",
    logical_ops = 24 : i64,
    hardware_ops = 24 : i64,
    expected_unit = 0 : i32
  } : () -> ()
  "pjrtx.cost_ledger"() {
    id = 1 : i64,
    lowering_region = 1 : i64,
    source_instructions = [1 : i64, 2 : i64, 3 : i64],
    op_class = "elementwise_transcendental",
    dtype = "f32",
    accumulation_dtype = "f32",
    logical_ops = 9 : i64,
    hardware_ops = 9 : i64,
    expected_unit = 1 : i32
  } : () -> ()
  "pjrtx.memory_traffic"() {lowering_region = 0 : i64, memory_space = "device_hbm", bytes_read = 80 : i64, bytes_written = 24 : i64} : () -> ()
  "pjrtx.memory_traffic"() {lowering_region = 0 : i64, memory_space = "local_sram", bytes_read = 80 : i64, bytes_written = 24 : i64} : () -> ()
  "pjrtx.memory_traffic"() {lowering_region = 1 : i64, memory_space = "device_hbm", bytes_read = 36 : i64, bytes_written = 24 : i64} : () -> ()
  "pjrtx.roofline_estimate"() {
    lowering_region = 0 : i64,
    ideal_compute_time_ns = 1 : i64,
    ideal_memory_time_ns = 1 : i64,
    ideal_interconnect_time_ns = 0 : i64,
    limiting_resource = "launch_overhead_for_tiny_fixture"
  } : () -> ()
  "pjrtx.roofline_estimate"() {
    lowering_region = 1 : i64,
    ideal_compute_time_ns = 1 : i64,
    ideal_memory_time_ns = 1 : i64,
    ideal_interconnect_time_ns = 0 : i64,
    limiting_resource = "transcendental_latency_model_unknown"
  } : () -> ()

  "pjrtx.perf.func"() ({
  ^entry(%arg0: memref<1x4xf32>, %arg1: memref<4x3xf32>, %arg2: memref<3xf32>, %out: memref<1x3xf32>):
    %matmul = "pjrtx.perf.call"(%arg0, %arg1) {
      lowering_region = 0 : i64,
      cost_ledger = 0 : i64,
      roofline = "launch_overhead_for_tiny_fixture",
      bytes_read = 80 : i64,
      bytes_written = 24 : i64
    } : (memref<1x4xf32>, memref<4x3xf32>) -> memref<1x3xf32>
    "pjrtx.perf.call"(%matmul, %arg2, %out) {
      lowering_region = 1 : i64,
      cost_ledger = 1 : i64,
      roofline = "transcendental_latency_model_unknown",
      bytes_read = 36 : i64,
      bytes_written = 24 : i64
    } : (memref<1x3xf32>, memref<3xf32>, memref<1x3xf32>) -> ()
    "pjrtx.perf.return"(%out) : (memref<1x3xf32>) -> ()
  }) {sym_name = "main_performance_modeled"} : () -> ()

  func.func private @main_perf_annotated_memref(
      %lhs: memref<1x4xf32>,
      %rhs: memref<4x3xf32>,
      %bias: memref<3xf32>,
      %out: memref<1x3xf32>) attributes {
    pjrtx.state = "performance_modeled"
  } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %f0 = arith.constant 0.0 : f32
    %tmp = memref.alloc() {pjrtx.memory_space = "device_hbm"} : memref<1x3xf32>
    scf.for %j = %c0 to %c3 step %c1 {
      memref.store %f0, %tmp[%c0, %j] : memref<1x3xf32>
      scf.for %k = %c0 to %c4 step %c1 {
        %acc = memref.load %tmp[%c0, %j] : memref<1x3xf32>
        %a = memref.load %lhs[%c0, %k] : memref<1x4xf32>
        %b = memref.load %rhs[%k, %j] : memref<4x3xf32>
        %prod = arith.mulf %a, %b : f32
        %next = arith.addf %acc, %prod : f32
        memref.store %next, %tmp[%c0, %j] : memref<1x3xf32>
      }
    } {pjrtx.cost_ledger = 0 : i64, pjrtx.roofline = "launch_overhead_for_tiny_fixture"}
    scf.for %j = %c0 to %c3 step %c1 {
      %m = memref.load %tmp[%c0, %j] : memref<1x3xf32>
      %b = memref.load %bias[%j] : memref<3xf32>
      %sum = arith.addf %m, %b : f32
      %tanh = math.tanh %sum : f32
      memref.store %tanh, %out[%c0, %j] : memref<1x3xf32>
    } {pjrtx.cost_ledger = 1 : i64, pjrtx.roofline = "transcendental_latency_model_unknown"}
    memref.dealloc %tmp : memref<1x3xf32>
    return
  }
}
