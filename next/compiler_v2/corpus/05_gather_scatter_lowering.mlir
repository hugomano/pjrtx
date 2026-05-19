// HLO coverage corpus: gather and scatter lowering.
// Covers bounds policy, index materialization, unsupported aliasing checks,
// and target kernels with explicit indexed memory operations.

module @pjrtx_hlo_gather_scatter_lowering attributes {
  pjrtx.coverage = "gather_scatter_lowering",
  pjrtx.final_state = "codegen_planned"
} {
  func.func @stablehlo_gather(%operand: tensor<8xf32>, %indices: tensor<4xi32>) -> tensor<4xf32> {
    %0 = "stablehlo.gather"(%operand, %indices) {
      dimension_numbers = #stablehlo.gather<
        offset_dims = [],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1>,
      slice_sizes = array<i64: 1>
    } : (tensor<8xf32>, tensor<4xi32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  func.func @stablehlo_scatter(%operand: tensor<8xf32>, %indices: tensor<4xi32>, %updates: tensor<4xf32>) -> tensor<8xf32> {
    %0 = "stablehlo.scatter"(%operand, %indices, %updates) ({
    ^bb0(%old: tensor<f32>, %new: tensor<f32>):
      %sum = stablehlo.add %old, %new : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {
      indices_are_sorted = false,
      unique_indices = false,
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    } : (tensor<8xf32>, tensor<4xi32>, tensor<4xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  "pjrtx.legality"() {
    op = "stablehlo.scatter",
    decision = "legal_with_atomic_add",
    reason = "non-unique indices require explicit atomic accumulation"
  } : () -> ()

  "pjrtx.npu.kernel"() ({
  ^entry(%operand: !pjrtx.ptr<device_hbm>, %indices: !pjrtx.ptr<device_hbm>, %dst: !pjrtx.ptr<device_hbm>):
    %idx, %idx_tok = "pjrtx.npu.dma.load"(%indices) {bytes = 16 : i64, dst_space = "register_file"} : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.ireg<4xi32>, !pjrtx.token<dma>)
    "pjrtx.npu.wait"(%idx_tok) {kind = "dma"} : (!pjrtx.token<dma>) -> ()
    %vals = "pjrtx.npu.gather.f32"(%operand, %idx) {
      bounds = "checked",
      source_instruction = 0 : i64
    } : (!pjrtx.ptr<device_hbm>, !pjrtx.ireg<4xi32>) -> !pjrtx.vreg<4xf32>
    %store = "pjrtx.npu.dma.store"(%vals, %dst) {bytes = 16 : i64} : (!pjrtx.vreg<4xf32>, !pjrtx.ptr<device_hbm>) -> !pjrtx.token<dma>
    "pjrtx.npu.wait"(%store) {kind = "dma"} : (!pjrtx.token<dma>) -> ()
    "pjrtx.npu.return"() : () -> ()
  }) {sym_name = "npu_gather_f32_checked"} : () -> ()

  "pjrtx.npu.kernel"() ({
  ^entry(%operand: !pjrtx.ptr<device_hbm>, %indices: !pjrtx.ptr<device_hbm>, %updates: !pjrtx.ptr<device_hbm>):
    %idx, %idx_tok = "pjrtx.npu.dma.load"(%indices) {bytes = 16 : i64} : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.ireg<4xi32>, !pjrtx.token<dma>)
    %upd, %upd_tok = "pjrtx.npu.dma.load"(%updates) {bytes = 16 : i64} : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.vreg<4xf32>, !pjrtx.token<dma>)
    "pjrtx.npu.wait"(%idx_tok, %upd_tok) {kind = "dma"} : (!pjrtx.token<dma>, !pjrtx.token<dma>) -> ()
    "pjrtx.npu.atomic_scatter_add.f32"(%operand, %idx, %upd) {
      source_instruction = 1 : i64,
      unique_indices = false
    } : (!pjrtx.ptr<device_hbm>, !pjrtx.ireg<4xi32>, !pjrtx.vreg<4xf32>) -> ()
    "pjrtx.npu.return"() : () -> ()
  }) {sym_name = "npu_scatter_add_f32_atomic"} : () -> ()
}
