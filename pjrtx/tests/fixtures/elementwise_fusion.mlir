module {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<2x3xf32>
    %1 = stablehlo.tanh %0 : tensor<2x3xf32>
    return %1 : tensor<2x3xf32>
  }
}
