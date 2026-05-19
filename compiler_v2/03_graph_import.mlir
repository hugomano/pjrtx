// Stage 03: canonical StableHLO imported into stable PjRTx source, graph
// instruction, graph value, shape, dtype, and sharding facts.

module @pjrtx_stage_03_graph_import attributes {
  pjrtx.state = "graph_imported",
  pjrtx.stage = "canonical_graph_ir"
} {
  "pjrtx.state_transition"() {
    from = "canonicalized",
    to = "graph_imported",
    pass = "pjrtx_graph_import",
    reason = "all StableHLO ops map to typed V0 graph payloads"
  } : () -> ()

  "pjrtx.source"() {id = 0 : i64, symbol = "main.arg0", op = "func.argument"} : () -> ()
  "pjrtx.source"() {id = 1 : i64, symbol = "main.arg1", op = "func.argument"} : () -> ()
  "pjrtx.source"() {id = 2 : i64, symbol = "main.arg2", op = "func.argument"} : () -> ()
  "pjrtx.source"() {id = 3 : i64, symbol = "main.dot", op = "stablehlo.dot_general"} : () -> ()
  "pjrtx.source"() {id = 4 : i64, symbol = "main.bias_broadcast", op = "stablehlo.broadcast_in_dim"} : () -> ()
  "pjrtx.source"() {id = 5 : i64, symbol = "main.bias_add", op = "stablehlo.add"} : () -> ()
  "pjrtx.source"() {id = 6 : i64, symbol = "main.tanh", op = "stablehlo.tanh"} : () -> ()

  "pjrtx.graph_value"() {id = 0 : i64, source = 0 : i64, tensor = "tensor<2x4xf32>", sharding = "mesh=pjrtx_mesh axes=[x,unsharded]"} : () -> ()
  "pjrtx.graph_value"() {id = 1 : i64, source = 1 : i64, tensor = "tensor<4x3xf32>", sharding = "mesh=pjrtx_mesh axes=[unsharded,unsharded]"} : () -> ()
  "pjrtx.graph_value"() {id = 2 : i64, source = 2 : i64, tensor = "tensor<3xf32>", sharding = "mesh=pjrtx_mesh axes=[unsharded]"} : () -> ()
  "pjrtx.graph_value"() {id = 3 : i64, def = 0 : i64, tensor = "tensor<2x3xf32>", sharding = "mesh=pjrtx_mesh axes=[x,unsharded]"} : () -> ()
  "pjrtx.graph_value"() {id = 4 : i64, def = 1 : i64, tensor = "tensor<2x3xf32>", sharding = "mesh=pjrtx_mesh axes=[x,unsharded]"} : () -> ()
  "pjrtx.graph_value"() {id = 5 : i64, def = 2 : i64, tensor = "tensor<2x3xf32>", sharding = "mesh=pjrtx_mesh axes=[x,unsharded]"} : () -> ()
  "pjrtx.graph_value"() {id = 6 : i64, def = 3 : i64, tensor = "tensor<2x3xf32>", sharding = "mesh=pjrtx_mesh axes=[x,unsharded]"} : () -> ()

  "pjrtx.graph_instruction"() {
    id = 0 : i64,
    source = 3 : i64,
    op = "dot_general",
    inputs = [0 : i64, 1 : i64],
    outputs = [3 : i64],
    payload = "dot lhs_contracting=[1] rhs_contracting=[0] strict_f32"
  } : () -> ()
  "pjrtx.graph_instruction"() {id = 1 : i64, source = 4 : i64, op = "broadcast_in_dim", inputs = [2 : i64], outputs = [4 : i64], payload = "broadcast dims=[1]"} : () -> ()
  "pjrtx.graph_instruction"() {id = 2 : i64, source = 5 : i64, op = "add", inputs = [3 : i64, 4 : i64], outputs = [5 : i64], payload = "elementwise strict_f32"} : () -> ()
  "pjrtx.graph_instruction"() {id = 3 : i64, source = 6 : i64, op = "tanh", inputs = [5 : i64], outputs = [6 : i64], payload = "transcendental strict_f32"} : () -> ()

  "pjrtx.graph_verify"() {
    status = "ok",
    dynamic_shapes = "none",
    tokens = "none",
    side_effects = "none",
    unsupported_regions = "none"
  } : () -> ()

  "pjrtx.graph.func"() ({
  ^entry(%arg0: tensor<2x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>):
    %dot = "pjrtx.graph.dot_general"(%arg0, %arg1) {
      graph_instruction = 0 : i64,
      source = 3 : i64,
      lhs_contracting_dims = [1],
      rhs_contracting_dims = [0],
      precision = "DEFAULT",
      sharding = "mesh=pjrtx_mesh axes=[x,unsharded]",
      semantic = "strict_f32"
    } : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %bias = "pjrtx.graph.broadcast_in_dim"(%arg2) {
      graph_instruction = 1 : i64,
      source = 4 : i64,
      dims = [1],
      sharding = "mesh=pjrtx_mesh axes=[x,unsharded]"
    } : (tensor<3xf32>) -> tensor<2x3xf32>
    %sum = "pjrtx.graph.add"(%dot, %bias) {
      graph_instruction = 2 : i64,
      source = 5 : i64,
      semantic = "strict_f32"
    } : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %result = "pjrtx.graph.tanh"(%sum) {
      graph_instruction = 3 : i64,
      source = 6 : i64,
      semantic = "strict_f32_transcendental"
    } : (tensor<2x3xf32>) -> tensor<2x3xf32>
    "pjrtx.graph.return"(%result) : (tensor<2x3xf32>) -> ()
  }) {
    sym_name = "main",
    inputs = [0 : i64, 1 : i64, 2 : i64],
    outputs = [6 : i64]
  } : () -> ()
}
