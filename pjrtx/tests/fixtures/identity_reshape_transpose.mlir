module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<2x3xf32>) -> tensor<2x3xf32>
    %1 = stablehlo.transpose %0, dims = [0, 1] : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
}
