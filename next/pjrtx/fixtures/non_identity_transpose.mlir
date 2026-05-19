module {
  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
    return %0 : tensor<3x2xf32>
  }
}
