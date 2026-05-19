// Stage 13: executable lowering regions. This is the boundary consumed by
// performance modeling, bufferization, schedule, backend binding, and profile.

module @pjrtx_stage_13_lowering_planned attributes {
  pjrtx.state = "lowering_planned",
  pjrtx.stage = "lowering_region_form"
} {
  "pjrtx.state_transition"() {from = "collectives_lowered", to = "lowering_planned", pass = "lowering_region_form"} : () -> ()
  "pjrtx.lowering_region"() {
    id = 0 : i64,
    kind = "library_call",
    source_instructions = [0 : i64],
    tile_plan = 0 : i64,
    placement_records = [0 : i64, 1 : i64, 3 : i64],
    inputs = [0 : i64, 1 : i64],
    outputs = [3 : i64],
    backend_operation = "npu.matmul_f32",
    expected_unit = 0 : i32,
    math_policy = "strict"
  } : () -> ()
  "pjrtx.lowering_region"() {
    id = 1 : i64,
    kind = "generated_kernel",
    source_instructions = [1 : i64, 2 : i64, 3 : i64],
    fusion_candidate = 1 : i64,
    tile_plan = 1 : i64,
    placement_records = [2 : i64, 3 : i64, 4 : i64],
    inputs = [3 : i64, 2 : i64],
    outputs = [6 : i64],
    backend_operation = "npu.fused_broadcast_add_tanh_f32",
    expected_unit = 1 : i32,
    math_policy = "strict"
  } : () -> ()

  func.func private @main_linalg_tensor(
      %lhs: tensor<1x4xf32>,
      %rhs: tensor<4x3xf32>,
      %bias: tensor<3xf32>) -> tensor<1x3xf32> attributes {
    pjrtx.lowering_level = "tensor_linalg",
    pjrtx.source_instructions = [0 : i64, 1 : i64, 2 : i64, 3 : i64]
  } {
    %c0 = arith.constant 0.0 : f32
    %matmul_init = tensor.empty() : tensor<1x3xf32>
    %matmul_zero = linalg.fill ins(%c0 : f32) outs(%matmul_init : tensor<1x3xf32>) -> tensor<1x3xf32>
    %matmul = linalg.matmul
      ins(%lhs, %rhs : tensor<1x4xf32>, tensor<4x3xf32>)
      outs(%matmul_zero : tensor<1x3xf32>) -> tensor<1x3xf32>
    %epilogue_init = tensor.empty() : tensor<1x3xf32>
    %epilogue = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"],
      pjrtx.fusion_candidate = 1 : i64,
      pjrtx.source_instructions = [1 : i64, 2 : i64, 3 : i64]
    } ins(%matmul, %bias : tensor<1x3xf32>, tensor<3xf32>)
      outs(%epilogue_init : tensor<1x3xf32>) {
    ^bb0(%m: f32, %b: f32, %out: f32):
      %sum = arith.addf %m, %b : f32
      %tanh = math.tanh %sum : f32
      linalg.yield %tanh : f32
    } -> tensor<1x3xf32>
    return %epilogue : tensor<1x3xf32>
  }

  "pjrtx.lowering.func"() ({
  ^entry(%arg0: memref<1x4xf32>, %arg1: memref<4x3xf32>, %arg2: memref<3xf32>, %out: memref<1x3xf32>):
    %matmul = "pjrtx.lowering.library_call"(%arg0, %arg1) {
      lowering_region = 0 : i64,
      callee = @npu_matmul_f32_tile,
      source_instructions = [0 : i64],
      expected_unit = "trn2_tensor_engine",
      result_memory_space = "device_hbm"
    } : (memref<1x4xf32>, memref<4x3xf32>) -> memref<1x3xf32>
    "pjrtx.lowering.generated_region"(%matmul, %arg2, %out) ({
    ^kernel(%matmul_tile: memref<1x3xf32>, %bias: memref<3xf32>, %dst: memref<1x3xf32>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %j = %c0 to %c3 step %c1 {
        %m = memref.load %matmul_tile[%c0, %j] : memref<1x3xf32>
        %b = memref.load %bias[%j] : memref<3xf32>
        %sum = arith.addf %m, %b : f32
        %tanh = math.tanh %sum : f32
        memref.store %tanh, %dst[%c0, %j] : memref<1x3xf32>
      }
      "pjrtx.lowering.return"() : () -> ()
    }) {
      lowering_region = 1 : i64,
      fusion_candidate = 1 : i64,
      source_instructions = [1 : i64, 2 : i64, 3 : i64],
      expected_unit = "trn2_vector_engine"
    } : (memref<1x3xf32>, memref<3xf32>, memref<1x3xf32>) -> ()
    "pjrtx.lowering.return"(%out) : (memref<1x3xf32>) -> ()
  }) {sym_name = "main_lowered"} : () -> ()
}
