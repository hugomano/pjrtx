// Stage 22: backend kernel graph. Nodes and edges are concrete enough for
// graph generation, but source/lowering/profile provenance is preserved.

module @pjrtx_stage_22_backend_kernel_graph_planned attributes {
  pjrtx.state = "backend_kernel_graph_planned",
  pjrtx.stage = "kernel_graph_build"
} {
  "pjrtx.state_transition"() {from = "backend_executable_planned", to = "backend_kernel_graph_planned", pass = "kernel_graph_build"} : () -> ()
  "pjrtx.kernel_graph"() ({
    "pjrtx.kernel_graph.node"() {
      id = 0 : i64,
      call = 0 : i64,
      lowering_region = 0 : i64,
      kernel = @npu_matmul_f32_tile,
      source_instructions = [0 : i64],
      inputs = [0 : i64, 1 : i64],
      outputs = [3 : i64],
      hardware_unit = "trn2_tensor_engine",
      expected_dtype_rate = "f32.matmul.synthetic"
    } : () -> ()
    "pjrtx.kernel_graph.node"() {
      id = 1 : i64,
      call = 1 : i64,
      lowering_region = 1 : i64,
      kernel = @npu_fused_broadcast_add_tanh_f32_tile,
      source_instructions = [1 : i64, 2 : i64, 3 : i64],
      inputs = [3 : i64, 2 : i64],
      outputs = [4 : i64],
      hardware_unit = "trn2_vector_engine",
      expected_dtype_rate = "f32.elementwise.synthetic"
    } : () -> ()
    "pjrtx.kernel_graph.edge"() {
      id = 0 : i64,
      value = 3 : i64,
      buffer = 3 : i64,
      from_node = 0 : i64,
      to_node = 1 : i64
    } : () -> ()
  }) {symbol = @main_kernel_graph, backend = "npu_v0"} : () -> ()

  "pjrtx.kernel_graph.func"() ({
  ^entry(%buf0: !pjrtx.buffer, %buf1: !pjrtx.buffer, %buf2: !pjrtx.buffer):
    %buf3 = "pjrtx.kernel_graph.call_node"(%buf0, %buf1) {
      node = 0 : i64,
      kernel = @npu_matmul_f32_tile,
      produces_value = 3 : i64,
      hardware_unit = "trn2_tensor_engine"
    } : (!pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.buffer
    %buf4 = "pjrtx.kernel_graph.call_node"(%buf3, %buf2) {
      node = 1 : i64,
      kernel = @npu_fused_broadcast_add_tanh_f32_tile,
      consumes_value = 3 : i64,
      produces_value = 6 : i64,
      hardware_unit = "trn2_vector_engine"
    } : (!pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.buffer
    "pjrtx.kernel_graph.return"(%buf4) : (!pjrtx.buffer) -> ()
  }) {
    sym_name = "main_kernel_graph",
    backend = "npu_v0"
  } : () -> ()
}
