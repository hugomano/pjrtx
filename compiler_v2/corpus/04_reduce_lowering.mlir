// HLO coverage corpus: reduction lowering.
// Covers stablehlo.reduce, strict reduction semantics, linalg reduce, tree
// reduction sketch, and profile join back to the original reduction.

module @pjrtx_hlo_reduce_lowering attributes {
  pjrtx.coverage = "reduce_lowering",
  pjrtx.final_state = "codegen_planned"
} {
  func.func @stablehlo_input(%arg0: tensor<4x8xf32>, %init: tensor<f32>) -> tensor<4xf32> {
    %0 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.add across dimensions = [1]
      : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  "pjrtx.correctness_contract"() {
    source_instruction = 0 : i64,
    reduction_order = "stablehlo_default",
    reassociation = "forbidden_without_relaxed_math",
    nan = "preserve"
  } : () -> ()

  func.func @linalg_reduce(%arg0: tensor<4x8xf32>, %init_scalar: f32) -> tensor<4xf32> {
    %init = tensor.empty() : tensor<4xf32>
    %filled = linalg.fill ins(%init_scalar : f32) outs(%init : tensor<4xf32>) -> tensor<4xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d0)>
      ],
      iterator_types = ["parallel", "reduction"],
      pjrtx.source_instruction = 0 : i64
    } ins(%arg0 : tensor<4x8xf32>) outs(%filled : tensor<4xf32>) {
    ^bb0(%x: f32, %acc: f32):
      %next = arith.addf %acc, %x : f32
      linalg.yield %next : f32
    } -> tensor<4xf32>
    return %out : tensor<4xf32>
  }

  "pjrtx.npu.kernel"() ({
  ^entry(%input: !pjrtx.ptr<device_hbm>, %dst: !pjrtx.ptr<device_hbm>):
    %tile, %tok = "pjrtx.npu.dma.load"(%input) {bytes = 128 : i64, dst_space = "local_sram"} : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.sram_tile<4x8xf32>, !pjrtx.token<dma>)
    "pjrtx.npu.wait"(%tok) {kind = "dma"} : (!pjrtx.token<dma>) -> ()
    %partial = "pjrtx.npu.reduce_add.f32"(%tile) {
      axis = 1 : i32,
      order = "stablehlo_default",
      unit = "trn2_vector_engine"
    } : (!pjrtx.sram_tile<4x8xf32>) -> !pjrtx.vreg<4xf32>
    %store = "pjrtx.npu.dma.store"(%partial, %dst) {bytes = 16 : i64} : (!pjrtx.vreg<4xf32>, !pjrtx.ptr<device_hbm>) -> !pjrtx.token<dma>
    "pjrtx.npu.wait"(%store) {kind = "dma"} : (!pjrtx.token<dma>) -> ()
    "pjrtx.npu.return"() : () -> ()
  }) {sym_name = "npu_reduce_add_f32_axis1", source_instruction = 0 : i64} : () -> ()
}
