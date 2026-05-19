// HLO coverage corpus: convolution lowering.
// Covers stablehlo.convolution, target layout choice, linalg conv lowering,
// tiling, and generated kernel loops.

module @pjrtx_hlo_convolution_lowering attributes {
  pjrtx.coverage = "convolution_lowering",
  pjrtx.final_state = "codegen_planned"
} {
  func.func @stablehlo_input(%input: tensor<1x5x5x1xf32>, %filter: tensor<3x3x1x2xf32>) -> tensor<1x3x3x2xf32> {
    %0 = stablehlo.convolution(%input, %filter)
      dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
      : (tensor<1x5x5x1xf32>, tensor<3x3x1x2xf32>) -> tensor<1x3x3x2xf32>
    return %0 : tensor<1x3x3x2xf32>
  }

  "pjrtx.layout"() {
    input = "nhwc",
    filter = "hwcf",
    output = "nhwc",
    reason = "NPU V0 direct-conv kernel expects NHWC/HWCF"
  } : () -> ()

  func.func @linalg_conv(%input: tensor<1x5x5x1xf32>, %filter: tensor<3x3x1x2xf32>) -> tensor<1x3x3x2xf32> {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1x3x3x2xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<1x3x3x2xf32>) -> tensor<1x3x3x2xf32>
    %out = linalg.conv_2d_nhwc_hwcf
      ins(%input, %filter : tensor<1x5x5x1xf32>, tensor<3x3x1x2xf32>)
      outs(%filled : tensor<1x3x3x2xf32>) -> tensor<1x3x3x2xf32>
    return %out : tensor<1x3x3x2xf32>
  }

  func.func @generated_conv_kernel(
      %input: memref<1x5x5x1xf32>,
      %filter: memref<3x3x1x2xf32>,
      %out: memref<1x3x3x2xf32>) attributes {
    pjrtx.lowering_level = "scf_direct_conv",
    pjrtx.tile_shape = [1 : i64, 1 : i64, 1 : i64, 2 : i64]
  } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %f0 = arith.constant 0.0 : f32
    scf.for %oh = %c0 to %c3 step %c1 {
      scf.for %ow = %c0 to %c3 step %c1 {
        scf.for %oc = %c0 to %c2 step %c1 {
          memref.store %f0, %out[%c0, %oh, %ow, %oc] : memref<1x3x3x2xf32>
          scf.for %kh = %c0 to %c3 step %c1 {
            scf.for %kw = %c0 to %c3 step %c1 {
              %acc = memref.load %out[%c0, %oh, %ow, %oc] : memref<1x3x3x2xf32>
              %ih = arith.addi %oh, %kh : index
              %iw = arith.addi %ow, %kw : index
              %x = memref.load %input[%c0, %ih, %iw, %c0] : memref<1x5x5x1xf32>
              %w = memref.load %filter[%kh, %kw, %c0, %oc] : memref<3x3x1x2xf32>
              %prod = arith.mulf %x, %w : f32
              %next = arith.addf %acc, %prod : f32
              memref.store %next, %out[%c0, %oh, %ow, %oc] : memref<1x3x3x2xf32>
            }
          }
        }
      }
    }
    return
  }
}
