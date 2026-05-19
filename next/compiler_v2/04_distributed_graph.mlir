// Stage 04: Shardy and topology facts made explicit enough for collectives,
// partitioning, and later memory/traffic accounting.

module @pjrtx_stage_04_distributed_graph attributes {
  pjrtx.state = "distributed_graph",
  pjrtx.stage = "sharding_topology_analysis"
} {
  "pjrtx.state_transition"() {
    from = "graph_imported",
    to = "distributed_graph",
    pass = "shardy_metadata_propagation_report",
    preserves_source = true,
    preserves_shardy = true
  } : () -> ()
  "pjrtx.mesh"() {
    name = "pjrtx_mesh",
    axes = ["x"],
    axis_sizes = [2 : i64],
    device_assignment = [0 : i64, 1 : i64]
  } : () -> ()
  "pjrtx.partition"() {
    id = 0 : i64,
    mesh_axis = "x",
    participants = [0 : i64, 1 : i64],
    graph_values = [0 : i64, 3 : i64, 4 : i64, 5 : i64, 6 : i64],
    shard_shape = "tensor<1x?xf32>",
    reason = "batch dimension is partitioned across mesh axis x"
  } : () -> ()
  "pjrtx.placement_constraint"() {
    graph_values = [1 : i64, 2 : i64],
    placement = "replicated",
    reason = "weights and bias are unsharded in this tiny example"
  } : () -> ()
  "pjrtx.collective_requirement"() {
    id = 0 : i64,
    kind = "none",
    reason = "dot output sharding follows lhs batch sharding; no cross-partition reduction"
  } : () -> ()

  "pjrtx.dist.func"() ({
  ^entry(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>):
    %dot = "pjrtx.dist.dot_general"(%arg0, %arg1) {
      graph_instruction = 0 : i64,
      mesh = "pjrtx_mesh",
      shard = "x",
      local_shape = "tensor<1x3xf32>",
      collective = "none"
    } : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %bias = "pjrtx.dist.broadcast_in_dim"(%arg2) {
      graph_instruction = 1 : i64,
      dims = [1],
      broadcast_over_shard_axis = true
    } : (tensor<3xf32>) -> tensor<1x3xf32>
    %sum = "pjrtx.dist.add"(%dot, %bias) {
      graph_instruction = 2 : i64
    } : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %result = "pjrtx.dist.tanh"(%sum) {
      graph_instruction = 3 : i64
    } : (tensor<1x3xf32>) -> tensor<1x3xf32>
    "pjrtx.dist.return"(%result) : (tensor<1x3xf32>) -> ()
  }) {
    sym_name = "main_shard",
    mesh = "pjrtx_mesh",
    per_partition_program = true
  } : () -> ()
}
