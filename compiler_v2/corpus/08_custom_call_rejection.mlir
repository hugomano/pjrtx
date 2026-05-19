// HLO coverage corpus: custom_call rejection.
// Covers backend-private escape hatches: custom calls are not allowed to skip
// target legality or explainability.

module @pjrtx_hlo_custom_call_rejection attributes {
  pjrtx.coverage = "custom_call_rejection",
  pjrtx.final_state = "compile_failed",
  pjrtx.failure_stage = "target_feature_legality"
} {
  func.func @stablehlo_input(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.custom_call @opaque_backend_magic(%arg0)
      {api_version = 2 : i32, has_side_effect = false}
      : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  "pjrtx.graph.func"() ({
  ^entry(%arg0: tensor<4xf32>):
    %0 = "pjrtx.graph.custom_call"(%arg0) {
      source_instruction = 0 : i64,
      call_target = "opaque_backend_magic",
      has_side_effect = false,
      explainability_status = "opaque"
    } : (tensor<4xf32>) -> tensor<4xf32>
    "pjrtx.graph.return"(%0) : (tensor<4xf32>) -> ()
  }) {sym_name = "main_custom_call_graph"} : () -> ()

  "pjrtx.verifier_failure"() {
    pass = "target_feature_legality",
    feature = "custom_call",
    source_instruction = 0 : i64,
    diagnostic = "custom_call lacks a PjRTx lowering contract and cannot become a backend-private fallback",
    executable_created = false,
    runtime_fallback = false
  } : () -> ()
}
