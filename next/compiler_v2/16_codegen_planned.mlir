// Stage 16: backend-neutral codegen records. Kernel IR contracts are explicit
// before tile legality, scheduling, or backend binding.

module @pjrtx_stage_16_codegen_planned attributes {
  pjrtx.state = "codegen_planned",
  pjrtx.stage = "kernel_codegen_plan"
} {
  "pjrtx.state_transition"() {from = "bufferized", to = "codegen_planned", pass = "kernel_codegen_plan"} : () -> ()
  "pjrtx.codegen_kernel"() {
    id = 0 : i64,
    lowering_region = 0 : i64,
    kind = "library_call",
    symbol = @npu_matmul_f32_tile,
    backend_operation = "npu.matmul_f32",
    abi = "buffers+tile+stream",
    inputs = [0 : i64, 1 : i64],
    outputs = [3 : i64],
    scratch_bytes = 0 : i64,
    expected_unit = 0 : i32
  } : () -> ()
  "pjrtx.codegen_kernel"() {
    id = 1 : i64,
    lowering_region = 1 : i64,
    kind = "generated_kernel",
    symbol = @npu_fused_broadcast_add_tanh_f32_tile,
    backend_operation = "npu.fused_broadcast_add_tanh_f32",
    abi = "buffers+tile+stream",
    inputs = [3 : i64, 2 : i64],
    outputs = [4 : i64],
    scratch_bytes = 0 : i64,
    expected_unit = 1 : i32
  } : () -> ()
  "pjrtx.kernel_ir"() {
    kernel = 1 : i64,
    block = "tile",
    ops = [
      "bias = broadcast(arg_bias)",
      "sum = add(matmul_out, bias)",
      "out = tanh(sum)"
    ],
    math_policy = "strict_f32"
  } : () -> ()

  "pjrtx.kernel.module"() ({
    "pjrtx.kernel"() ({
    ^entry(%lhs: memref<1x4xf32>, %rhs: memref<4x3xf32>, %dst: memref<1x3xf32>):
      "pjrtx.npu.matmul_tile"(%lhs, %rhs, %dst) {
        kernel = 0 : i64,
        tile_shape = [1 : i64, 3 : i64],
        source_instructions = [0 : i64],
        unit = "trn2_tensor_engine",
        dtype = "f32"
      } : (memref<1x4xf32>, memref<4x3xf32>, memref<1x3xf32>) -> ()
      "pjrtx.kernel.return"() : () -> ()
    }) {
      sym_name = "npu_matmul_f32_tile",
      lowering_region = 0 : i64,
      kind = "library_call_contract"
    } : () -> ()

    "pjrtx.kernel"() ({
    ^entry(%matmul: memref<1x3xf32>, %bias: memref<3xf32>, %dst: memref<1x3xf32>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %j = %c0 to %c3 step %c1 {
        %m = memref.load %matmul[%c0, %j] : memref<1x3xf32>
        %b = memref.load %bias[%j] : memref<3xf32>
        %sum = arith.addf %m, %b : f32
        %tanh = math.tanh %sum : f32
        memref.store %tanh, %dst[%c0, %j] : memref<1x3xf32>
      } {pjrtx.schedule_hint = "vector_lane_contiguous"}
      "pjrtx.kernel.return"() : () -> ()
    }) {
      sym_name = "npu_fused_broadcast_add_tanh_f32_tile",
      lowering_region = 1 : i64,
      kind = "generated_kernel",
      source_instructions = [1 : i64, 2 : i64, 3 : i64]
    } : () -> ()

    func.func private @npu_fused_broadcast_add_tanh_f32_vector(
        %matmul: memref<1x3xf32>,
        %bias: memref<3xf32>,
        %dst: memref<1x3xf32>) attributes {
      pjrtx.lowering_level = "vector_tile_kernel",
      pjrtx.kernel = 1 : i64,
      pjrtx.source_instructions = [1 : i64, 2 : i64, 3 : i64]
    } {
      %c0 = arith.constant 0 : index
      %cst = arith.constant 0.0 : f32
      %m = vector.transfer_read %matmul[%c0, %c0], %cst
        {in_bounds = [true, true]} : memref<1x3xf32>, vector<3xf32>
      %b = vector.transfer_read %bias[%c0], %cst
        {in_bounds = [true]} : memref<3xf32>, vector<3xf32>
      %sum = arith.addf %m, %b : vector<3xf32>
      %tanh = math.tanh %sum : vector<3xf32>
      vector.transfer_write %tanh, %dst[%c0, %c0]
        {in_bounds = [true, true]} : vector<3xf32>, memref<1x3xf32>
      return
    }
  }) {target = "npu_v0"} : () -> ()
}
