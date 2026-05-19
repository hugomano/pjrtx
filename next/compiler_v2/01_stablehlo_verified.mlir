// Stage 01: semantic verification.
// Shapes, dtypes, StableHLO semantics, source locations, tokens, side effects,
// and Shardy annotations have been checked before any optimization.

sdy.mesh @pjrtx_mesh = <["x"=2]>

module @pjrtx_stage_01_stablehlo_verified attributes {
  pjrtx.state = "stablehlo_verified",
  pjrtx.stage = "semantic_verification"
} {
  "pjrtx.state_transition"() {
    from = "imported",
    to = "stablehlo_verified",
    pass = "mlir_verify",
    verifier = "stablehlo+func+sdy",
    reason = "module verifier accepted source-level program"
  } : () -> ()
  "pjrtx.correctness_contract"() {
    math_mode = "strict",
    nan = "preserve",
    infinity = "preserve",
    signed_zero = "preserve",
    reduction_order = "stablehlo_default",
    sharding_semantics = "preserve",
    unsupported_tokens = "fail_before_lowering",
    unsupported_side_effects = "fail_before_lowering"
  } : () -> ()
  "pjrtx.shape_dtype_fact"() {
    values = [
      "v0:tensor<2x4xf32>",
      "v1:tensor<4x3xf32>",
      "v2:tensor<3xf32>",
      "v3:tensor<2x3xf32>",
      "v4:tensor<2x3xf32>",
      "v5:tensor<2x3xf32>",
      "v6:tensor<2x3xf32>"
    ]
  } : () -> ()

  func.func private @stablehlo_source(
      %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@pjrtx_mesh, [{"x"}, {}]>},
      %arg1: tensor<4x3xf32> {sdy.sharding = #sdy.sharding<@pjrtx_mesh, [{}, {}]>},
      %arg2: tensor<3xf32> {sdy.sharding = #sdy.sharding<@pjrtx_mesh, [{}]>})
      -> tensor<2x3xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@pjrtx_mesh, [{"x"}, {}]>]>}
      : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1]
      {sdy.sharding = #sdy.sharding_per_value<[<@pjrtx_mesh, [{"x"}, {}]>]>}
      : (tensor<3xf32>) -> tensor<2x3xf32>
    %2 = stablehlo.add %0, %1 : tensor<2x3xf32>
    %3 = stablehlo.tanh %2 : tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}
