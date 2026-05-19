// Stage 16A: target instruction sketch.
// This file is intentionally Aster-like: the kernel is not a metadata object,
// it is the low-level program. Registers, DMA tokens, waits, vector lanes, and
// stores are SSA values and operations.

module @pjrtx_stage_16a_target_instruction_ir attributes {
  pjrtx.state = "codegen_planned",
  pjrtx.stage = "target_instruction_ir",
  pjrtx.target.name = "npu_v0"
} {
  "pjrtx.npu.module"() ({
    "pjrtx.npu.kernel"() ({
    ^entry(%lhs_hbm: !pjrtx.ptr<device_hbm>, %rhs_hbm: !pjrtx.ptr<device_hbm>, %dst_hbm: !pjrtx.ptr<device_hbm>):
      %lhs_sram, %lhs_tok = "pjrtx.npu.dma.load"(%lhs_hbm) {
        bytes = 16 : i64,
        src_space = "device_hbm",
        dst_space = "local_sram",
        engine = "trn2_dma_engine"
      } : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.sram_tile<1x4xf32>, !pjrtx.token<dma>)
      %rhs_sram, %rhs_tok = "pjrtx.npu.dma.load"(%rhs_hbm) {
        bytes = 48 : i64,
        src_space = "device_hbm",
        dst_space = "local_sram",
        engine = "trn2_dma_engine"
      } : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.sram_tile<4x3xf32>, !pjrtx.token<dma>)
      "pjrtx.npu.wait"(%lhs_tok, %rhs_tok) {kind = "dma"} : (!pjrtx.token<dma>, !pjrtx.token<dma>) -> ()
      %acc = "pjrtx.npu.mma"(%lhs_sram, %rhs_sram) {
        instruction = "matmul_f32_1x3x4",
        unit = "trn2_tensor_engine",
        source_instructions = [0 : i64]
      } : (!pjrtx.sram_tile<1x4xf32>, !pjrtx.sram_tile<4x3xf32>) -> !pjrtx.reg_tile<1x3xf32>
      %store_tok = "pjrtx.npu.dma.store"(%acc, %dst_hbm) {
        bytes = 12 : i64,
        src_space = "register_file",
        dst_space = "device_hbm"
      } : (!pjrtx.reg_tile<1x3xf32>, !pjrtx.ptr<device_hbm>) -> !pjrtx.token<dma>
      "pjrtx.npu.wait"(%store_tok) {kind = "dma"} : (!pjrtx.token<dma>) -> ()
      "pjrtx.npu.return"() : () -> ()
    }) {
      sym_name = "npu_matmul_f32_tile",
      lowering_region = 0 : i64,
      normal_form = "scheduled_instruction_ir"
    } : () -> ()

    "pjrtx.npu.kernel"() ({
    ^entry(%matmul_hbm: !pjrtx.ptr<device_hbm>, %bias_hbm: !pjrtx.ptr<device_hbm>, %dst_hbm: !pjrtx.ptr<device_hbm>):
      %matmul_vec, %matmul_tok = "pjrtx.npu.dma.load"(%matmul_hbm) {
        bytes = 12 : i64,
        src_space = "device_hbm",
        dst_space = "register_file"
      } : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.vreg<3xf32>, !pjrtx.token<dma>)
      %bias_vec, %bias_tok = "pjrtx.npu.dma.load"(%bias_hbm) {
        bytes = 12 : i64,
        src_space = "device_hbm",
        dst_space = "register_file"
      } : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.vreg<3xf32>, !pjrtx.token<dma>)
      "pjrtx.npu.wait"(%matmul_tok, %bias_tok) {kind = "dma"} : (!pjrtx.token<dma>, !pjrtx.token<dma>) -> ()
      %sum = "pjrtx.npu.vadd.f32"(%matmul_vec, %bias_vec) {
        unit = "trn2_vector_engine",
        source_instruction = 2 : i64
      } : (!pjrtx.vreg<3xf32>, !pjrtx.vreg<3xf32>) -> !pjrtx.vreg<3xf32>
      %out = "pjrtx.npu.vtanh.f32"(%sum) {
        unit = "trn2_vector_engine",
        source_instruction = 3 : i64,
        math_policy = "strict"
      } : (!pjrtx.vreg<3xf32>) -> !pjrtx.vreg<3xf32>
      %store_tok = "pjrtx.npu.dma.store"(%out, %dst_hbm) {
        bytes = 12 : i64,
        src_space = "register_file",
        dst_space = "device_hbm"
      } : (!pjrtx.vreg<3xf32>, !pjrtx.ptr<device_hbm>) -> !pjrtx.token<dma>
      "pjrtx.npu.wait"(%store_tok) {kind = "dma"} : (!pjrtx.token<dma>) -> ()
      "pjrtx.npu.return"() : () -> ()
    }) {
      sym_name = "npu_fused_broadcast_add_tanh_f32_tile",
      lowering_region = 1 : i64,
      fusion_candidate = 1 : i64,
      normal_form = "scheduled_instruction_ir"
    } : () -> ()
  }) {target = "npu_v0"} : () -> ()
}
