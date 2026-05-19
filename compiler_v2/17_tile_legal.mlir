// Stage 17: tile legality verification against memory, ABI, and execution-unit
// constraints.

module @pjrtx_stage_17_tile_legal attributes {
  pjrtx.state = "tile_legal",
  pjrtx.stage = "tile_legality_verify"
} {
  "pjrtx.state_transition"() {from = "codegen_planned", to = "tile_legal", pass = "tile_legality_verify"} : () -> ()
  "pjrtx.tile_legality"() {kernel = 0 : i64, decision = "legal", reason = "tile operands fit local SRAM and matmul ABI"} : () -> ()
  "pjrtx.tile_legality"() {kernel = 1 : i64, decision = "legal", reason = "fused elementwise live values fit local SRAM and vector ABI"} : () -> ()
  "pjrtx.codegen_contract"() {
    kernels = [0 : i64, 1 : i64],
    requires_alignment = 64 : i64,
    supports_async_launch = true,
    result_memory_space = "device_hbm",
    scratch_memory_space = "local_sram"
  } : () -> ()

  "pjrtx.kernel.module"() ({
    "pjrtx.kernel.ref"() {
      symbol = @npu_matmul_f32_tile,
      tile_legality = "legal",
      verified_memory_spaces = ["device_hbm", "local_sram"]
    } : () -> ()
    "pjrtx.kernel.ref"() {
      symbol = @npu_fused_broadcast_add_tanh_f32_tile,
      tile_legality = "legal",
      verified_memory_spaces = ["device_hbm", "local_sram"]
    } : () -> ()
  }) {target = "npu_v0", normal_form = "tile_legal"} : () -> ()

  "pjrtx.tile_checked.func"() ({
  ^entry(%lhs: memref<1x4xf32>, %rhs: memref<4x3xf32>, %bias: memref<3xf32>, %dst: memref<1x3xf32>):
    %tmp = "pjrtx.tile_checked.launch"(%lhs, %rhs) {
      kernel = @npu_matmul_f32_tile,
      tile_legality = "legal",
      produces_value = 3 : i64
    } : (memref<1x4xf32>, memref<4x3xf32>) -> memref<1x3xf32>
    "pjrtx.tile_checked.launch"(%tmp, %bias, %dst) {
      kernel = @npu_fused_broadcast_add_tanh_f32_tile,
      tile_legality = "legal",
      produces_value = 6 : i64
    } : (memref<1x3xf32>, memref<3xf32>, memref<1x3xf32>) -> ()
    "pjrtx.tile_checked.return"(%dst) : (memref<1x3xf32>) -> ()
  }) {sym_name = "main_tile_legal"} : () -> ()
}
