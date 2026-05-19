// Stage 08: fusion planning. Both accepted and rejected alternatives are kept
// because they explain the lowering surface seen by codegen.

module @pjrtx_stage_08_fusion_planned attributes {
  pjrtx.state = "fusion_planned",
  pjrtx.stage = "fusion_planning"
} {
  "pjrtx.state_transition"() {from = "layout_assigned", to = "fusion_planned", pass = "fusion_candidate_discovery+fusion_select"} : () -> ()
  "pjrtx.fusion_candidate"() {
    id = 0 : i64,
    kind = "matmul_epilogue",
    members = [0 : i64, 1 : i64, 2 : i64, 3 : i64],
    root = 3 : i64,
    bytes_saved = 72 : i64,
    launch_delta = -3 : i32,
    reason = "dot+broadcast/add/tanh could be one epilogue kernel"
  } : () -> ()
  "pjrtx.pressure_delta"() {
    candidate = 0 : i64,
    split_kernel_count = 2 : i32,
    fused_kernel_count = 1 : i32,
    split_peak_live_bytes = 108 : i64,
    fused_live_bytes = 188 : i64,
    additional_live_bytes = 80 : i64,
    global_bytes_saved = 72 : i64
  } : () -> ()
  "pjrtx.fusion_decision"() {
    candidate = 0 : i64,
    decision = "rejected",
    reason = "epilogue codegen contract is not explicit yet"
  } : () -> ()
  "pjrtx.fusion_candidate"() {
    id = 1 : i64,
    kind = "elementwise_chain",
    members = [1 : i64, 2 : i64, 3 : i64],
    root = 3 : i64,
    bytes_saved = 48 : i64,
    launch_delta = -2 : i32,
    reason = "pure elementwise chain with compatible layout and no side effects"
  } : () -> ()
  "pjrtx.fusion_decision"() {
    candidate = 1 : i64,
    decision = "accepted",
    reason = "fits vector unit and local memory pressure"
  } : () -> ()

  "pjrtx.platform.func"() ({
  ^entry(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>):
    %dot = "pjrtx.platform.matmul"(%arg0, %arg1) {
      graph_instruction = 0 : i64,
      fusion_boundary = "kept",
      rejected_fusion_candidate = 0 : i64
    } : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %result = "pjrtx.fusion.region"(%dot, %arg2) ({
    ^fused(%matmul_out: tensor<1x3xf32>, %bias_arg: tensor<3xf32>):
      %bias = "pjrtx.platform.broadcast"(%bias_arg) {
        graph_instruction = 1 : i64,
        dims = [1]
      } : (tensor<3xf32>) -> tensor<1x3xf32>
      %sum = "pjrtx.platform.add"(%matmul_out, %bias) {
        graph_instruction = 2 : i64
      } : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
      %tanh = "pjrtx.platform.tanh"(%sum) {
        graph_instruction = 3 : i64
      } : (tensor<1x3xf32>) -> tensor<1x3xf32>
      "pjrtx.yield"(%tanh) : (tensor<1x3xf32>) -> ()
    }) {
      fusion_candidate = 1 : i64,
      decision = "accepted",
      source_instructions = [1 : i64, 2 : i64, 3 : i64]
    } : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    "pjrtx.platform.return"(%result) : (tensor<1x3xf32>) -> ()
  }) {sym_name = "main_after_fusion"} : () -> ()
}
