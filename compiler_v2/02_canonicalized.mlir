// Stage 02: canonical graph-preserving MLIR cleanup.
// Canonicalization/CSE ran, but no PjRTx executable decision has been made.

sdy.mesh @pjrtx_mesh = <["x"=2]>

module @pjrtx_stage_02_canonicalized attributes {
  pjrtx.state = "canonicalized",
  pjrtx.stage = "canonicalization"
} {
  "pjrtx.state_transition"() {
    from = "stablehlo_verified",
    to = "canonicalized",
    pass = "mlir_canonicalize_cse",
    preserves_source = true,
    preserves_shardy = true
  } : () -> ()
  "pjrtx.rewrite_record"() {
    id = 0 : i64,
    pass = "broadcast_simplify",
    decision = "rejected",
    source_op = "stablehlo.broadcast_in_dim",
    reason = "broadcast from tensor<3xf32> to tensor<2x3xf32> is not identity"
  } : () -> ()
  "pjrtx.rewrite_record"() {
    id = 1 : i64,
    pass = "reshape_transpose_fold",
    decision = "skipped",
    reason = "program has no reshape or transpose"
  } : () -> ()

  func.func private @canonical_stablehlo(
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
