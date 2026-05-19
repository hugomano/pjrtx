// Stage 00: frontend import.
// Source-level StableHLO is present, and Shardy metadata is still attached to
// the semantic operations that came from the frontend.

sdy.mesh @pjrtx_mesh = <["x"=2]>

module @pjrtx_stage_00_frontend_import attributes {
  pjrtx.state = "imported",
  pjrtx.stage = "frontend_import",
  pjrtx.compile.mode = "jit",
  pjrtx.compile.options = "replicas=2;partitions=1;use_shardy=true"
} {
  "pjrtx.pass_record"() {
    index = 0 : i32,
    name = "stablehlo_parse",
    status = "ok",
    produced_state = "imported",
    preserves_source = true,
    preserves_shardy = true
  } : () -> ()

  func.func @main(
      %arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@pjrtx_mesh, [{"x"}, {}]>},
      %arg1: tensor<4x3xf32> {sdy.sharding = #sdy.sharding<@pjrtx_mesh, [{}, {}]>},
      %arg2: tensor<3xf32> {sdy.sharding = #sdy.sharding<@pjrtx_mesh, [{}]>})
      -> (tensor<2x3xf32> {sdy.sharding = #sdy.sharding<@pjrtx_mesh, [{"x"}, {}]>}) {
    %0 = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@pjrtx_mesh, [{"x"}, {}]>]>}
      : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1]
      {sdy.sharding = #sdy.sharding_per_value<[<@pjrtx_mesh, [{"x"}, {}]>]>}
      : (tensor<3xf32>) -> tensor<2x3xf32>
    %2 = stablehlo.add %0, %1
      {sdy.sharding = #sdy.sharding_per_value<[<@pjrtx_mesh, [{"x"}, {}]>]>}
      : tensor<2x3xf32>
    %3 = stablehlo.tanh %2
      {sdy.sharding = #sdy.sharding_per_value<[<@pjrtx_mesh, [{"x"}, {}]>]>}
      : tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}
