// Stage 26: backend profile joins. Backend-owned call/profile facts still join
// back to compiler lowering and source IDs.

module @pjrtx_stage_26_backend_profile_joined attributes {
  pjrtx.state = "backend_profile_joined",
  pjrtx.stage = "backend_profile_join"
} {
  "pjrtx.state_transition"() {from = "runtime_profile_joined", to = "backend_profile_joined", pass = "backend_profile_join"} : () -> ()
  "pjrtx.backend_profile_join"() {
    id = 0 : i64,
    backend_call = 0 : i64,
    kernel_graph_node = 0 : i64,
    lowering_region = 0 : i64,
    runtime_profile_event = 1 : i64,
    source_instructions = [0 : i64]
  } : () -> ()
  "pjrtx.backend_profile_join"() {
    id = 1 : i64,
    backend_call = 1 : i64,
    kernel_graph_node = 1 : i64,
    lowering_region = 1 : i64,
    runtime_profile_event = 2 : i64,
    source_instructions = [1 : i64, 2 : i64, 3 : i64]
  } : () -> ()
  "pjrtx.explain_trace"() {
    query = "which source operations became which kernels and profile events",
    rows = [
      "stablehlo.dot_general -> lowering_region.0 -> node.0 -> event.1",
      "stablehlo.broadcast_in_dim/add/tanh -> fusion.1 -> lowering_region.1 -> node.1 -> event.2"
    ]
  } : () -> ()

  "pjrtx.explain.func"() ({
  ^entry:
    %dot = "pjrtx.explain.trace_source_to_kernel"() {
      source = "stablehlo.dot_general",
      source_instruction = 0 : i64,
      lowering_region = 0 : i64,
      kernel_graph_node = 0 : i64,
      backend_call = 0 : i64,
      profile_event = 1 : i64
    } : () -> !pjrtx.trace_row
    %epilogue = "pjrtx.explain.trace_source_to_kernel"() {
      source = "stablehlo.broadcast_in_dim+stablehlo.add+stablehlo.tanh",
      source_instruction = 3 : i64,
      fusion_candidate = 1 : i64,
      lowering_region = 1 : i64,
      kernel_graph_node = 1 : i64,
      backend_call = 1 : i64,
      profile_event = 2 : i64
    } : () -> !pjrtx.trace_row
    "pjrtx.explain.return"(%dot, %epilogue) : (!pjrtx.trace_row, !pjrtx.trace_row) -> ()
  }) {sym_name = "main_explain_trace"} : () -> ()
}
