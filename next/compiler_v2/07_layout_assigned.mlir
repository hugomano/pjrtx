// Stage 07: layout assignment. Layout is represented before fusion/tiling so
// later decisions can prove compatibility.

module @pjrtx_stage_07_layout_assigned attributes {
  pjrtx.state = "layout_assigned",
  pjrtx.stage = "layout_assignment"
} {
  "pjrtx.state_transition"() {
    from = "target_legal",
    to = "layout_assigned",
    pass = "layout_constraint_legality",
    reason = "NPU V0 accepts row-major sharded batch layout for this program"
  } : () -> ()
  "pjrtx.layout"() {value = 0 : i64, layout = "row_major", minor_to_major = [1 : i64, 0 : i64], reason = "input batch-major tensor"} : () -> ()
  "pjrtx.layout"() {value = 1 : i64, layout = "row_major", minor_to_major = [1 : i64, 0 : i64], reason = "matmul rhs library compatible"} : () -> ()
  "pjrtx.layout"() {value = 2 : i64, layout = "dense_1d", minor_to_major = [0 : i64], reason = "bias vector"} : () -> ()
  "pjrtx.layout"() {value = 3 : i64, layout = "row_major", minor_to_major = [1 : i64, 0 : i64], reason = "matmul output feeds epilogue"} : () -> ()
  "pjrtx.layout"() {value = 6 : i64, layout = "row_major", minor_to_major = [1 : i64, 0 : i64], reason = "result preserves batch sharding"} : () -> ()

  "pjrtx.platform.func"() ({
  ^entry(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>):
    %dot = "pjrtx.platform.matmul"(%arg0, %arg1) {
      graph_instruction = 0 : i64,
      input_layouts = ["row_major", "row_major"],
      output_layout = "row_major"
    } : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %bias = "pjrtx.platform.broadcast"(%arg2) {
      graph_instruction = 1 : i64,
      input_layouts = ["dense_1d"],
      output_layout = "row_major"
    } : (tensor<3xf32>) -> tensor<1x3xf32>
    %sum = "pjrtx.platform.add"(%dot, %bias) {graph_instruction = 2 : i64, layout = "row_major"} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %result = "pjrtx.platform.tanh"(%sum) {graph_instruction = 3 : i64, layout = "row_major"} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    "pjrtx.platform.return"(%result) : (tensor<1x3xf32>) -> ()
  }) {sym_name = "main_layout_assigned"} : () -> ()
}
