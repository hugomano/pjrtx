// Stage 06: target legality. Unsupported features must fail here, not at
// backend submission or runtime.

module @pjrtx_stage_06_target_legal attributes {
  pjrtx.state = "target_legal",
  pjrtx.stage = "target_feature_legality",
  pjrtx.target.fingerprint = 2001 : i64
} {
  "pjrtx.state_transition"() {
    from = "target_attached",
    to = "target_legal",
    pass = "target_feature_legality",
    reason = "all graph operations have legal NPU V0 lowering candidates"
  } : () -> ()
  "pjrtx.legality"() {instruction = 0 : i64, op = "dot_general", decision = "legal", lowering_candidates = ["library_matmul", "generated_matmul"], expected_unit = 0 : i32} : () -> ()
  "pjrtx.legality"() {instruction = 1 : i64, op = "broadcast_in_dim", decision = "legal", lowering_candidates = ["generated_elementwise"], expected_unit = 1 : i32} : () -> ()
  "pjrtx.legality"() {instruction = 2 : i64, op = "add", decision = "legal", lowering_candidates = ["generated_elementwise"], expected_unit = 1 : i32} : () -> ()
  "pjrtx.legality"() {instruction = 3 : i64, op = "tanh", decision = "legal", lowering_candidates = ["generated_elementwise"], expected_unit = 1 : i32} : () -> ()
  "pjrtx.no_fallback_gate"() {
    status = "closed",
    runtime_interpreter = "forbidden",
    backend_private_repair = "forbidden"
  } : () -> ()

  "pjrtx.platform.func"() ({
  ^entry(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>):
    %dot = "pjrtx.platform.matmul"(%arg0, %arg1) {
      graph_instruction = 0 : i64,
      legal_lowerings = ["npu.matmul_f32", "generated.matmul_tile_f32"],
      expected_unit = "trn2_tensor_engine"
    } : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %bias = "pjrtx.platform.broadcast"(%arg2) {
      graph_instruction = 1 : i64,
      legal_lowerings = ["generated.elementwise_f32"],
      expected_unit = "trn2_vector_engine"
    } : (tensor<3xf32>) -> tensor<1x3xf32>
    %sum = "pjrtx.platform.add"(%dot, %bias) {
      graph_instruction = 2 : i64,
      legal_lowerings = ["generated.elementwise_f32"],
      expected_unit = "trn2_vector_engine"
    } : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %result = "pjrtx.platform.tanh"(%sum) {
      graph_instruction = 3 : i64,
      legal_lowerings = ["generated.elementwise_f32"],
      expected_unit = "trn2_vector_engine"
    } : (tensor<1x3xf32>) -> tensor<1x3xf32>
    "pjrtx.platform.return"(%result) : (tensor<1x3xf32>) -> ()
  }) {sym_name = "main_target_legal"} : () -> ()
}
