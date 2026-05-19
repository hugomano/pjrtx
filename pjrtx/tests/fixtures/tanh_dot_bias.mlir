module {
  func.func @main(%arg0: tensor<2x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>) -> tensor<2x3xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
    %2 = stablehlo.add %0, %1 : tensor<2x3xf32>
    %3 = stablehlo.tanh %2 : tensor<2x3xf32>
    return %3 : tensor<2x3xf32>
  }
}
