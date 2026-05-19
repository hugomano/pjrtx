module {
  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = "stablehlo.all_reduce"(%arg0) ({
    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
      stablehlo.return %sum : tensor<f32>
    }) {
      channel_handle = #stablehlo.channel_handle<handle = 7, type = 0>,
      replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
    } : (tensor<4xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
