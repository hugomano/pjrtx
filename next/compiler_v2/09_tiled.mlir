// Stage 09: tiling and partitioning. Logical, physical, memory, and hardware
// tile facts are explicit before memory planning and kernel IR.

module @pjrtx_stage_09_tiled attributes {
  pjrtx.state = "tiled",
  pjrtx.stage = "tiling_partitioning"
} {
  "pjrtx.state_transition"() {from = "fusion_planned", to = "tiled", pass = "tile_shape_select"} : () -> ()
  "pjrtx.tile_plan"() {
    id = 0 : i64,
    lowering_subject = "instruction:0",
    logical_tile_shape = [1 : i64, 3 : i64],
    physical_tile_shape = [1 : i64, 3 : i64],
    memory_tile_shape = [1 : i64, 3 : i64],
    core_mapping = "mesh_axis_x_to_device",
    matrix_unit_shape = "1x3x4",
    vector_width = 1 : i32,
    double_buffering = false,
    pipeline_stages = 1 : i32
  } : () -> ()
  "pjrtx.tile_plan"() {
    id = 1 : i64,
    lowering_subject = "fusion:1",
    logical_tile_shape = [1 : i64, 3 : i64],
    physical_tile_shape = [1 : i64, 3 : i64],
    memory_tile_shape = [1 : i64, 3 : i64],
    core_mapping = "mesh_axis_x_to_device",
    vector_width = 1 : i32,
    double_buffering = false,
    pipeline_stages = 1 : i32
  } : () -> ()

  "pjrtx.tile.func"() ({
  ^entry(%arg0: tensor<1x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %init = tensor.empty() : tensor<1x3xf32>
    %result = scf.for %tile_m = %c0 to %c1 step %c1 iter_args(%out = %init) -> (tensor<1x3xf32>) {
      %dot_tile = "pjrtx.tile.matmul"(%arg0, %arg1) {
        tile_plan = 0 : i64,
        logical_tile_shape = [1 : i64, 3 : i64],
        source_instructions = [0 : i64]
      } : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
      %epilogue_tile = "pjrtx.tile.fused_elementwise"(%dot_tile, %arg2) ({
      ^tile(%matmul_out: tensor<1x3xf32>, %bias_arg: tensor<3xf32>):
        %bias = "pjrtx.tile.broadcast"(%bias_arg) {dims = [1]} : (tensor<3xf32>) -> tensor<1x3xf32>
        %sum = "pjrtx.tile.add"(%matmul_out, %bias) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
        %tanh = "pjrtx.tile.tanh"(%sum) : (tensor<1x3xf32>) -> tensor<1x3xf32>
        "pjrtx.yield"(%tanh) : (tensor<1x3xf32>) -> ()
      }) {
        tile_plan = 1 : i64,
        fusion_candidate = 1 : i64,
        source_instructions = [1 : i64, 2 : i64, 3 : i64]
      } : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
      scf.yield %epilogue_tile : tensor<1x3xf32>
    }
    "pjrtx.tile.return"(%result) : (tensor<1x3xf32>) -> ()
  }) {sym_name = "main_tiled"} : () -> ()
}
