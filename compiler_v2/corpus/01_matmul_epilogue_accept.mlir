// HLO coverage corpus: matmul + epilogue fusion accepted.
// Covers dot_general, broadcast_in_dim, add, tanh, fusion profitability,
// tensor/linalg lowering, vector epilogue codegen, and target instruction IR.

sdy.mesh @pjrtx_mesh = <["x"=2]>

module @pjrtx_hlo_matmul_epilogue_accept attributes {
  pjrtx.coverage = "matmul_epilogue_accept",
  pjrtx.final_state = "backend_kernel_graph_planned"
} {
  func.func @stablehlo_input(
      %lhs: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@pjrtx_mesh, [{"x"}, {}]>},
      %rhs: tensor<4x3xf32>,
      %bias: tensor<3xf32>) -> tensor<2x3xf32> {
    %dot = stablehlo.dot_general %lhs, %rhs,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %b = stablehlo.broadcast_in_dim %bias, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
    %sum = stablehlo.add %dot, %b : tensor<2x3xf32>
    %out = stablehlo.tanh %sum : tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }

  "pjrtx.fusion_decision"() {
    candidate = 0 : i64,
    kind = "matmul_epilogue",
    decision = "accepted",
    members = [0 : i64, 1 : i64, 2 : i64, 3 : i64],
    reason = "target epilogue ABI supports bias add and tanh without extra HBM round trip",
    bytes_saved = 96 : i64,
    additional_live_bytes = 32 : i64
  } : () -> ()

  func.func @linalg_tensor(
      %lhs: tensor<1x4xf32>,
      %rhs: tensor<4x3xf32>,
      %bias: tensor<3xf32>) -> tensor<1x3xf32> attributes {
    pjrtx.lowering_level = "tensor_linalg",
    pjrtx.fusion_candidate = 0 : i64
  } {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1x3xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<1x3xf32>) -> tensor<1x3xf32>
    %dot = linalg.matmul
      ins(%lhs, %rhs : tensor<1x4xf32>, tensor<4x3xf32>)
      outs(%filled : tensor<1x3xf32>) -> tensor<1x3xf32>
    %out_init = tensor.empty() : tensor<1x3xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"],
      pjrtx.epilogue = "bias_add_tanh"
    } ins(%dot, %bias : tensor<1x3xf32>, tensor<3xf32>)
      outs(%out_init : tensor<1x3xf32>) {
    ^bb0(%m: f32, %b: f32, %old: f32):
      %sum = arith.addf %m, %b : f32
      %tanh = math.tanh %sum : f32
      linalg.yield %tanh : f32
    } -> tensor<1x3xf32>
    return %out : tensor<1x3xf32>
  }

  "pjrtx.npu.kernel"() ({
  ^entry(%lhs_hbm: !pjrtx.ptr<device_hbm>, %rhs_hbm: !pjrtx.ptr<device_hbm>, %bias_hbm: !pjrtx.ptr<device_hbm>, %dst_hbm: !pjrtx.ptr<device_hbm>):
    %lhs, %lhs_tok = "pjrtx.npu.dma.load"(%lhs_hbm) {bytes = 16 : i64, dst_space = "local_sram"} : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.sram_tile<1x4xf32>, !pjrtx.token<dma>)
    %rhs, %rhs_tok = "pjrtx.npu.dma.load"(%rhs_hbm) {bytes = 48 : i64, dst_space = "local_sram"} : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.sram_tile<4x3xf32>, !pjrtx.token<dma>)
    %bias, %bias_tok = "pjrtx.npu.dma.load"(%bias_hbm) {bytes = 12 : i64, dst_space = "register_file"} : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.vreg<3xf32>, !pjrtx.token<dma>)
    "pjrtx.npu.wait"(%lhs_tok, %rhs_tok, %bias_tok) {kind = "dma"} : (!pjrtx.token<dma>, !pjrtx.token<dma>, !pjrtx.token<dma>) -> ()
    %acc = "pjrtx.npu.mma"(%lhs, %rhs) {source_instruction = 0 : i64, unit = "trn2_tensor_engine"} : (!pjrtx.sram_tile<1x4xf32>, !pjrtx.sram_tile<4x3xf32>) -> !pjrtx.vreg<3xf32>
    %sum = "pjrtx.npu.vadd.f32"(%acc, %bias) {source_instruction = 2 : i64, epilogue = true} : (!pjrtx.vreg<3xf32>, !pjrtx.vreg<3xf32>) -> !pjrtx.vreg<3xf32>
    %out = "pjrtx.npu.vtanh.f32"(%sum) {source_instruction = 3 : i64, math_policy = "strict"} : (!pjrtx.vreg<3xf32>) -> !pjrtx.vreg<3xf32>
    %store = "pjrtx.npu.dma.store"(%out, %dst_hbm) {bytes = 12 : i64, src_space = "register_file"} : (!pjrtx.vreg<3xf32>, !pjrtx.ptr<device_hbm>) -> !pjrtx.token<dma>
    "pjrtx.npu.wait"(%store) {kind = "dma"} : (!pjrtx.token<dma>) -> ()
    "pjrtx.npu.return"() : () -> ()
  }) {sym_name = "matmul_bias_tanh_tile", lowering_region = 0 : i64} : () -> ()
}
