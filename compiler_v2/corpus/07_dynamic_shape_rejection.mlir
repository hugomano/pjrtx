// HLO coverage corpus: dynamic shape rejection.
// Covers an intentional no-fallback failure path. The program is still present,
// and the verifier records why it cannot become executable for V0.

module @pjrtx_hlo_dynamic_shape_rejection attributes {
  pjrtx.coverage = "dynamic_shape_rejection",
  pjrtx.final_state = "compile_failed",
  pjrtx.failure_stage = "pjrtx_dynamic_shape_gate"
} {
  func.func @stablehlo_input(%arg0: tensor<?x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<?x3xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<?x4xf32>, tensor<4x3xf32>) -> tensor<?x3xf32>
    return %0 : tensor<?x3xf32>
  }

  "pjrtx.graph.func"() ({
  ^entry(%arg0: tensor<?x4xf32>, %arg1: tensor<4x3xf32>):
    %dot = "pjrtx.graph.dot_general"(%arg0, %arg1) {
      source_instruction = 0 : i64,
      dynamic_dims = [0 : i64],
      shape_status = "dynamic"
    } : (tensor<?x4xf32>, tensor<4x3xf32>) -> tensor<?x3xf32>
    "pjrtx.graph.return"(%dot) : (tensor<?x3xf32>) -> ()
  }) {sym_name = "main_dynamic_graph"} : () -> ()

  "pjrtx.verifier_failure"() {
    pass = "pjrtx_dynamic_shape_gate",
    state = "graph_imported",
    feature = "dynamic_shape",
    source_instruction = 0 : i64,
    diagnostic = "V0 requires static tensor dimensions before target legality",
    executable_created = false,
    runtime_fallback = false
  } : () -> ()
}
