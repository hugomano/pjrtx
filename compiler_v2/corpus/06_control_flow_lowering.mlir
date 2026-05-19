// HLO coverage corpus: while/conditional lowering.
// Covers region preservation, loop-carried buffer state, conditional branches,
// and no-fallback target legality for control flow.

module @pjrtx_hlo_control_flow_lowering attributes {
  pjrtx.coverage = "control_flow_lowering",
  pjrtx.final_state = "lowering_planned"
} {
  func.func @stablehlo_while(%arg0: tensor<4xf32>, %limit: tensor<i32>) -> tensor<4xf32> {
    %i0 = stablehlo.constant dense<0> : tensor<i32>
    %tuple = "stablehlo.tuple"(%i0, %arg0) : (tensor<i32>, tensor<4xf32>) -> tuple<tensor<i32>, tensor<4xf32>>
    %while = "stablehlo.while"(%tuple) ({
    ^cond(%iter_tuple: tuple<tensor<i32>, tensor<4xf32>>):
      %i = "stablehlo.get_tuple_element"(%iter_tuple) {index = 0 : i32} : (tuple<tensor<i32>, tensor<4xf32>>) -> tensor<i32>
      %pred = stablehlo.compare LT, %i, %limit : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %pred : tensor<i1>
    }, {
    ^body(%body_tuple: tuple<tensor<i32>, tensor<4xf32>>):
      %i = "stablehlo.get_tuple_element"(%body_tuple) {index = 0 : i32} : (tuple<tensor<i32>, tensor<4xf32>>) -> tensor<i32>
      %v = "stablehlo.get_tuple_element"(%body_tuple) {index = 1 : i32} : (tuple<tensor<i32>, tensor<4xf32>>) -> tensor<4xf32>
      %one = stablehlo.constant dense<1> : tensor<i32>
      %next_i = stablehlo.add %i, %one : tensor<i32>
      %next_v = stablehlo.tanh %v : tensor<4xf32>
      %next = "stablehlo.tuple"(%next_i, %next_v) : (tensor<i32>, tensor<4xf32>) -> tuple<tensor<i32>, tensor<4xf32>>
      stablehlo.return %next : tuple<tensor<i32>, tensor<4xf32>>
    }) : (tuple<tensor<i32>, tensor<4xf32>>) -> tuple<tensor<i32>, tensor<4xf32>>
    %out = "stablehlo.get_tuple_element"(%while) {index = 1 : i32} : (tuple<tensor<i32>, tensor<4xf32>>) -> tensor<4xf32>
    return %out : tensor<4xf32>
  }

  "pjrtx.control_flow.func"() ({
  ^entry(%arg0: tensor<4xf32>, %limit: i32):
    %c0 = arith.constant 0 : i32
    %result:2 = scf.while (%i = %c0, %v = %arg0) : (i32, tensor<4xf32>) -> (i32, tensor<4xf32>) {
      %pred = arith.cmpi slt, %i, %limit : i32
      scf.condition(%pred) %i, %v : i32, tensor<4xf32>
    } do {
    ^bb0(%i: i32, %v: tensor<4xf32>):
      %one = arith.constant 1 : i32
      %next_i = arith.addi %i, %one : i32
      %next_v = "pjrtx.elementwise.tanh"(%v) {source_instruction = 0 : i64} : (tensor<4xf32>) -> tensor<4xf32>
      scf.yield %next_i, %next_v : i32, tensor<4xf32>
    }
    "pjrtx.control_flow.return"(%result#1) : (tensor<4xf32>) -> ()
  }) {
    sym_name = "main_while_lowered",
    legality = "target_legal_control_flow"
  } : () -> ()
}
