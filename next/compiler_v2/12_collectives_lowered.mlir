// Stage 12: collective lowering. For this example the lowering is an explicit
// empty lowering, which is still a verifier-visible decision.

module @pjrtx_stage_12_collectives_lowered attributes {
  pjrtx.state = "collectives_lowered",
  pjrtx.stage = "collective_algorithm_select"
} {
  "pjrtx.state_transition"() {from = "collectives_planned", to = "collectives_lowered", pass = "collective_algorithm_select"} : () -> ()
  "pjrtx.collective_lowering"() {
    id = 0 : i64,
    collective_plan = 0 : i64,
    lowering = "none",
    traffic_bytes = 0 : i64,
    commands = [],
    reason = "no collective command is required"
  } : () -> ()
  "pjrtx.collective_traffic"() {plan = 0 : i64, interconnect_bytes = 0 : i64, ideal_time_ns = 0 : i64} : () -> ()

  "pjrtx.collective_lowered.func"() ({
  ^entry(%local_result: tensor<1x3xf32>):
    %after_comm = "pjrtx.collective.lowered_none"(%local_result) {
      lowering = 0 : i64,
      commands = [],
      interconnect_bytes = 0 : i64
    } : (tensor<1x3xf32>) -> tensor<1x3xf32>
    "pjrtx.collective_lowered.return"(%after_comm) : (tensor<1x3xf32>) -> ()
  }) {sym_name = "main_collectives_lowered"} : () -> ()

  func.func private @main_after_collective_lowering(
      %lhs: tensor<1x4xf32>,
      %rhs: tensor<4x3xf32>,
      %bias: tensor<3xf32>) -> tensor<1x3xf32> attributes {
    pjrtx.state = "collectives_lowered",
    pjrtx.collective_lowering = 0 : i64
  } {
    %c0 = arith.constant 0.0 : f32
    %matmul_init = tensor.empty() : tensor<1x3xf32>
    %matmul_zero = linalg.fill ins(%c0 : f32) outs(%matmul_init : tensor<1x3xf32>) -> tensor<1x3xf32>
    %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<1x4xf32>, tensor<4x3xf32>)
      outs(%matmul_zero : tensor<1x3xf32>) -> tensor<1x3xf32>
    %comm = "pjrtx.collective.lowered_none"(%matmul) {
      lowering = 0 : i64,
      interconnect_bytes = 0 : i64
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
    } ins(%comm, %bias : tensor<1x3xf32>, tensor<3xf32>)
      outs(%epilogue_init : tensor<1x3xf32>) {
    ^bb0(%m: f32, %b: f32, %out: f32):
      %sum = arith.addf %m, %b : f32
      %tanh = math.tanh %sum : f32
      linalg.yield %tanh : f32
    } -> tensor<1x3xf32>
    return %epilogue : tensor<1x3xf32>
  }
}
