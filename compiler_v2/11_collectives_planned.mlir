// Stage 11: collective requirements are explicit. A no-collective program still
// records the checked fact so later scheduling does not infer silently.

module @pjrtx_stage_11_collectives_planned attributes {
  pjrtx.state = "collectives_planned",
  pjrtx.stage = "collective_group_channel_verify"
} {
  "pjrtx.state_transition"() {from = "memory_planned", to = "collectives_planned", pass = "collective_group_channel_verify"} : () -> ()
  "pjrtx.collective_group"() {
    id = 0 : i64,
    participants = [0 : i64, 1 : i64],
    replica_count = 2 : i32,
    partition_count = 1 : i32,
    decision = "verified",
    reason = "participants are inside replicas*partitions"
  } : () -> ()
  "pjrtx.collective_plan"() {
    id = 0 : i64,
    source_instruction = -1 : i64,
    kind = "none",
    algorithm_candidates = [],
    selected_algorithm = "none",
    reason = "program has no collective operation after sharding propagation"
  } : () -> ()

  "pjrtx.collective.func"() ({
  ^entry(%local_result: tensor<1x3xf32>):
    %checked = "pjrtx.collective.none"(%local_result) {
      collective_plan = 0 : i64,
      group = 0 : i64,
      reason = "no communication inserted"
    } : (tensor<1x3xf32>) -> tensor<1x3xf32>
    "pjrtx.collective.return"(%checked) : (tensor<1x3xf32>) -> ()
  }) {sym_name = "main_collectives_planned"} : () -> ()

  func.func private @main_after_collective_planning(
      %lhs: tensor<1x4xf32>,
      %rhs: tensor<4x3xf32>,
      %bias: tensor<3xf32>) -> tensor<1x3xf32> attributes {
    pjrtx.state = "collectives_planned",
    pjrtx.collective_plan = 0 : i64
  } {
    %c0 = arith.constant 0.0 : f32
    %matmul_init = tensor.empty() : tensor<1x3xf32>
    %matmul_zero = linalg.fill ins(%c0 : f32) outs(%matmul_init : tensor<1x3xf32>) -> tensor<1x3xf32>
    %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<1x4xf32>, tensor<4x3xf32>)
      outs(%matmul_zero : tensor<1x3xf32>) -> tensor<1x3xf32>
    %after_collectives = "pjrtx.collective.none"(%matmul) {
      collective_plan = 0 : i64,
      reason = "no collective required for this sharding"
    } : (tensor<1x3xf32>) -> tensor<1x3xf32>
    %epilogue_init = tensor.empty() : tensor<1x3xf32>
    %epilogue = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"],
      pjrtx.fusion_candidate = 1 : i64
    } ins(%after_collectives, %bias : tensor<1x3xf32>, tensor<3xf32>)
      outs(%epilogue_init : tensor<1x3xf32>) {
    ^bb0(%m: f32, %b: f32, %out: f32):
      %sum = arith.addf %m, %b : f32
      %tanh = math.tanh %sum : f32
      linalg.yield %tanh : f32
    } -> tensor<1x3xf32>
    return %epilogue : tensor<1x3xf32>
  }
}
