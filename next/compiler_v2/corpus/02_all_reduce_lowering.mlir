// HLO coverage corpus: all_reduce lowering.
// Covers collective graph payload import, replica groups, channel handles,
// ring algorithm selection, async start/done, and profile provenance.

sdy.mesh @pjrtx_mesh = <["x"=2]>

module @pjrtx_hlo_all_reduce_lowering attributes {
  pjrtx.coverage = "all_reduce_lowering",
  pjrtx.final_state = "backend_kernel_graph_planned"
} {
  func.func @stablehlo_input(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.all_reduce %arg0
      {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
       channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>}
      ({
      ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        %sum = stablehlo.add %lhs, %rhs : tensor<f32>
        stablehlo.return %sum : tensor<f32>
      }) : (tensor<4xf32>) -> tensor<4xf32>
    %1 = stablehlo.tanh %0 : tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  "pjrtx.collective_spec"() {
    id = 0 : i64,
    source_instruction = 0 : i64,
    kind = "all_reduce",
    reduction = "add",
    dtype = "f32",
    participants = [0 : i64, 1 : i64],
    channel_id = 7 : i64,
    channel_type = 1 : i64
  } : () -> ()
  "pjrtx.collective_algorithm"() {
    collective = 0 : i64,
    selected = "ring",
    rejected = ["direct", "tree", "split"],
    reason = "two participants and additive f32 reduction use ring v0"
  } : () -> ()

  "pjrtx.collective.func"() ({
  ^entry(%local: tensor<2xf32>):
    %start = "pjrtx.collective.async_start"(%local) {
      collective = 0 : i64,
      algorithm = "ring",
      channel_id = 7 : i64,
      phase = "reduce_scatter"
    } : (tensor<2xf32>) -> !pjrtx.collective_token
    %independent = "pjrtx.compute.local_tanh"(%local) {
      overlap_candidate = true,
      reason = "independent local work may overlap with collective start"
    } : (tensor<2xf32>) -> tensor<2xf32>
    %reduced = "pjrtx.collective.async_done"(%start) {
      collective = 0 : i64,
      traffic_bytes = 16 : i64
    } : (!pjrtx.collective_token) -> tensor<2xf32>
    %out = "pjrtx.elementwise.tanh"(%reduced) {source_instruction = 1 : i64} : (tensor<2xf32>) -> tensor<2xf32>
    "pjrtx.collective.return"(%out) : (tensor<2xf32>) -> ()
  }) {sym_name = "main_all_reduce_lowered"} : () -> ()

  "pjrtx.async.func"() ({
  ^entry(%buf: !pjrtx.buffer):
    %c0 = "pjrtx.async.collective_start"(%buf) {command = 0 : i64, engine = "trn2_collective_engine", channel_id = 7 : i64} : (!pjrtx.buffer) -> !pjrtx.async.token
    %c1 = "pjrtx.async.collective_done"(%c0, %buf) {command = 1 : i64, traffic_bytes = 16 : i64} : (!pjrtx.async.token, !pjrtx.buffer) -> !pjrtx.async.token
    %c2 = "pjrtx.async.launch"(%c1, %buf) {command = 2 : i64, kernel = @npu_tanh_f32_tile} : (!pjrtx.async.token, !pjrtx.buffer) -> !pjrtx.async.token
    "pjrtx.async.return"(%c2) : (!pjrtx.async.token) -> ()
  }) {sym_name = "main_all_reduce_schedule"} : () -> ()
}
