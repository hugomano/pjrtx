// Stage 05: target and topology contract attached. Hardware is a first-class
// MLIR fact before legality, cost, or lowering decisions.

module @pjrtx_stage_05_target_attached attributes {
  pjrtx.state = "target_attached",
  pjrtx.stage = "target_select",
  pjrtx.target.name = "npu_v0",
  pjrtx.target.kind = "npu_v0",
  pjrtx.target.fingerprint = 2001 : i64
} {
  "pjrtx.state_transition"() {
    from = "distributed_graph",
    to = "target_attached",
    pass = "target_select",
    reason = "compile request selected synthetic TRN2-like NPU target"
  } : () -> ()
  "pjrtx.target_spec"() {name = "npu_v0", kind = "npu_v0", fingerprint = 2001 : i64, replicas = 2 : i32, partitions = 1 : i32} : () -> ()
  "pjrtx.target_device"() {id = 0 : i32, local_hardware_id = 0 : i32, memory_spaces = [0 : i32, 1 : i32], execution_units = [0 : i32, 1 : i32, 2 : i32, 3 : i32]} : () -> ()
  "pjrtx.target_memory_space"() {id = 0 : i32, name = "device_hbm", kind = "device_hbm", capacity_bytes = 34359738368 : i64, bandwidth_bytes_per_second = 1.0e12 : f64} : () -> ()
  "pjrtx.target_memory_space"() {id = 1 : i32, name = "local_sram", kind = "local_sram", capacity_bytes = 67108864 : i64, bandwidth_bytes_per_second = 2.0e13 : f64} : () -> ()
  "pjrtx.target_memory_space"() {id = 2 : i32, name = "host_pinned", kind = "host_pinned"} : () -> ()
  "pjrtx.target_execution_unit"() {id = 0 : i32, name = "trn2_tensor_engine", kind = "matrix"} : () -> ()
  "pjrtx.target_execution_unit"() {id = 1 : i32, name = "trn2_vector_engine", kind = "vector"} : () -> ()
  "pjrtx.target_execution_unit"() {id = 2 : i32, name = "trn2_dma_engine", kind = "dma"} : () -> ()
  "pjrtx.target_execution_unit"() {id = 3 : i32, name = "trn2_collective_engine", kind = "collective"} : () -> ()
  "pjrtx.target_transfer_edge"() {id = 0 : i32, src = 2 : i32, dst = 0 : i32, bandwidth_bytes_per_second = 5.0e11 : f64, latency_ns = 1000 : i64, supports_async = true, engine_unit = 2 : i32} : () -> ()
  "pjrtx.target_transfer_edge"() {id = 1 : i32, src = 0 : i32, dst = 1 : i32, bandwidth_bytes_per_second = 2.0e12 : f64, latency_ns = 100 : i64, supports_async = true, engine_unit = 2 : i32} : () -> ()
  "pjrtx.target_dtype_rate"() {unit = 0 : i32, dtype = "f32", op_class = "matmul", ops_per_second = 5.0e13 : f64, source = "synthetic"} : () -> ()
  "pjrtx.target_dtype_rate"() {unit = 1 : i32, dtype = "f32", op_class = "elementwise", ops_per_second = 1.0e13 : f64, source = "synthetic"} : () -> ()

  "pjrtx.platform.func"() ({
  ^entry(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>):
    %dot = "pjrtx.platform.unknown_legal_matmul"(%arg0, %arg1) {
      graph_instruction = 0 : i64,
      target_fingerprint = 2001 : i64
    } : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %bias = "pjrtx.platform.unknown_legal_broadcast"(%arg2) {
      graph_instruction = 1 : i64,
      target_fingerprint = 2001 : i64
    } : (tensor<3xf32>) -> tensor<1x3xf32>
    %sum = "pjrtx.platform.unknown_legal_add"(%dot, %bias) {
      graph_instruction = 2 : i64,
      target_fingerprint = 2001 : i64
    } : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %result = "pjrtx.platform.unknown_legal_tanh"(%sum) {
      graph_instruction = 3 : i64,
      target_fingerprint = 2001 : i64
    } : (tensor<1x3xf32>) -> tensor<1x3xf32>
    "pjrtx.platform.return"(%result) : (tensor<1x3xf32>) -> ()
  }) {sym_name = "main_target_attached"} : () -> ()
}
