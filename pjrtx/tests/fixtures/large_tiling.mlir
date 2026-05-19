module {
  func.func @main(%arg0: tensor<4096x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      : (tensor<4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    return %0 : tensor<4096x4096xf32>
  }
}
