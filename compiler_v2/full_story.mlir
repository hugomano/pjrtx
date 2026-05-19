// Full PjRTx compiler_v2 story.
//
// This file merges the staged lowering spine and the HLO coverage corpus into
// one narrative artifact. The rule is: every compiler fact must attach to a
// visible program representation. The program starts as StableHLO/Shardy, then
// becomes graph IR, distributed IR, platform IR, linalg/tensor IR, memref/scf
// IR, vector/kernel IR, target instruction IR, async schedule IR, backend
// graph IR, runtime IR, and finally profile/explain IR.

sdy.mesh @pjrtx_mesh = <["x"=2]>

module @pjrtx_compiler_v2_full_story attributes {
  pjrtx.story = "full_hlo_to_kernel_graph",
  pjrtx.target.name = "npu_v0",
  pjrtx.target.fingerprint = 2001 : i64,
  pjrtx.no_runtime_fallback = true
} {
  "pjrtx.story.index"() {
    stages = [
      "00 stablehlo_shardy_input",
      "01 semantic_verify",
      "02 canonicalize",
      "03 graph_import",
      "04 distributed_graph",
      "05 target_attach",
      "06 target_legal",
      "07 layout_assign",
      "08 fusion_plan",
      "09 tile_plan",
      "10 memory_plan",
      "11 collective_plan",
      "12 collective_lower",
      "13 linalg_lower",
      "14 performance_model",
      "15 bufferize",
      "16 codegen",
      "16a target_instruction_ir",
      "18 async_schedule",
      "21 backend_executable",
      "22 kernel_graph",
      "23 runtime_alloc",
      "24 runtime_stream",
      "25 profile_join",
      "26 backend_profile_join"
    ],
    coverage = [
      "matmul_epilogue_accept",
      "all_reduce_lowering",
      "convolution_lowering",
      "reduce_lowering",
      "gather_scatter_lowering",
      "control_flow_lowering",
      "dynamic_shape_rejection",
      "custom_call_rejection"
    ]
  } : () -> ()

  // 00-02: source program and semantics.
  func.func @stablehlo_shardy_input(
      %lhs: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@pjrtx_mesh, [{"x"}, {}]>},
      %rhs: tensor<4x3xf32> {sdy.sharding = #sdy.sharding<@pjrtx_mesh, [{}, {}]>},
      %bias: tensor<3xf32> {sdy.sharding = #sdy.sharding<@pjrtx_mesh, [{}]>})
      -> tensor<2x3xf32> attributes {
    pjrtx.state = "stablehlo_verified",
    pjrtx.math_mode = "strict",
    pjrtx.shardy = "preserved"
  } {
    %dot = stablehlo.dot_general %lhs, %rhs,
      contracting_dims = [1] x [0],
      precision = [DEFAULT, DEFAULT]
      {sdy.sharding = #sdy.sharding_per_value<[<@pjrtx_mesh, [{"x"}, {}]>]>}
      : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %b = stablehlo.broadcast_in_dim %bias, dims = [1]
      {sdy.sharding = #sdy.sharding_per_value<[<@pjrtx_mesh, [{"x"}, {}]>]>}
      : (tensor<3xf32>) -> tensor<2x3xf32>
    %sum = stablehlo.add %dot, %b : tensor<2x3xf32>
    %out = stablehlo.tanh %sum : tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }

  // 03: canonical graph IR. Stable IDs are now attached to program ops.
  "pjrtx.graph.func"() ({
  ^entry(%lhs: tensor<2x4xf32>, %rhs: tensor<4x3xf32>, %bias: tensor<3xf32>):
    %dot = "pjrtx.graph.dot_general"(%lhs, %rhs) {
      graph_instruction = 0 : i64,
      source = "stablehlo.dot_general",
      lhs_contracting_dims = [1],
      rhs_contracting_dims = [0],
      semantic = "strict_f32"
    } : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    %b = "pjrtx.graph.broadcast_in_dim"(%bias) {
      graph_instruction = 1 : i64,
      source = "stablehlo.broadcast_in_dim",
      dims = [1]
    } : (tensor<3xf32>) -> tensor<2x3xf32>
    %sum = "pjrtx.graph.add"(%dot, %b) {
      graph_instruction = 2 : i64,
      source = "stablehlo.add"
    } : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %out = "pjrtx.graph.tanh"(%sum) {
      graph_instruction = 3 : i64,
      source = "stablehlo.tanh"
    } : (tensor<2x3xf32>) -> tensor<2x3xf32>
    "pjrtx.graph.return"(%out) : (tensor<2x3xf32>) -> ()
  }) {sym_name = "main_graph"} : () -> ()

  // 04-07: distributed/target/platform IR. The program is now per-shard and
  // target-legal, with layout on the operations rather than in a side table.
  "pjrtx.platform.func"() ({
  ^entry(%lhs: tensor<1x4xf32>, %rhs: tensor<4x3xf32>, %bias: tensor<3xf32>):
    %dot = "pjrtx.platform.matmul"(%lhs, %rhs) {
      graph_instruction = 0 : i64,
      mesh = "pjrtx_mesh",
      shard = "x",
      layout = "row_major",
      legal_lowerings = ["npu.matmul_f32", "generated.matmul_tile_f32"],
      expected_unit = "trn2_tensor_engine"
    } : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %b = "pjrtx.platform.broadcast"(%bias) {
      graph_instruction = 1 : i64,
      layout = "row_major",
      expected_unit = "trn2_vector_engine"
    } : (tensor<3xf32>) -> tensor<1x3xf32>
    %sum = "pjrtx.platform.add"(%dot, %b) {
      graph_instruction = 2 : i64,
      layout = "row_major"
    } : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
    %out = "pjrtx.platform.tanh"(%sum) {
      graph_instruction = 3 : i64,
      layout = "row_major",
      math_policy = "strict"
    } : (tensor<1x3xf32>) -> tensor<1x3xf32>
    "pjrtx.platform.return"(%out) : (tensor<1x3xf32>) -> ()
  }) {
    sym_name = "main_target_legal",
    target = "npu_v0",
    state = "target_legal"
  } : () -> ()

  // 08: fusion is represented as a program region with a decision attached.
  "pjrtx.platform.func"() ({
  ^entry(%lhs: tensor<1x4xf32>, %rhs: tensor<4x3xf32>, %bias: tensor<3xf32>):
    %dot = "pjrtx.platform.matmul"(%lhs, %rhs) {
      graph_instruction = 0 : i64,
      rejected_fusion_candidate = 0 : i64,
      reason = "matmul epilogue kept separate in V0 spine"
    } : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
    %out = "pjrtx.fusion.region"(%dot, %bias) ({
    ^fused(%matmul_out: tensor<1x3xf32>, %bias_arg: tensor<3xf32>):
      %b = "pjrtx.platform.broadcast"(%bias_arg) {graph_instruction = 1 : i64, dims = [1]} : (tensor<3xf32>) -> tensor<1x3xf32>
      %sum = "pjrtx.platform.add"(%matmul_out, %b) {graph_instruction = 2 : i64} : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
      %tanh = "pjrtx.platform.tanh"(%sum) {graph_instruction = 3 : i64} : (tensor<1x3xf32>) -> tensor<1x3xf32>
      "pjrtx.yield"(%tanh) : (tensor<1x3xf32>) -> ()
    }) {
      fusion_candidate = 1 : i64,
      decision = "accepted",
      source_instructions = [1 : i64, 2 : i64, 3 : i64]
    } : (tensor<1x3xf32>, tensor<3xf32>) -> tensor<1x3xf32>
    "pjrtx.platform.return"(%out) : (tensor<1x3xf32>) -> ()
  }) {sym_name = "main_fusion_planned"} : () -> ()

  // 09-10: tile and memory movement are program IR. Prefetches and local-memory
  // tiles are explicit values.
  "pjrtx.memory.func"() ({
  ^entry(%lhs_hbm: memref<1x4xf32>, %rhs_hbm: memref<4x3xf32>, %bias_hbm: memref<3xf32>, %out_hbm: memref<1x3xf32>):
    %lhs_sram = "pjrtx.mem.prefetch"(%lhs_hbm) {edge = 1 : i32, from = "device_hbm", to = "local_sram"} : (memref<1x4xf32>) -> memref<1x4xf32, 1>
    %rhs_sram = "pjrtx.mem.prefetch"(%rhs_hbm) {edge = 1 : i32, from = "device_hbm", to = "local_sram"} : (memref<4x3xf32>) -> memref<4x3xf32, 1>
    %matmul_hbm = "pjrtx.mem.compute_tile"(%lhs_sram, %rhs_sram) {
      tile_plan = 0 : i64,
      lowering_subject = "instruction:0"
    } : (memref<1x4xf32, 1>, memref<4x3xf32, 1>) -> memref<1x3xf32>
    %matmul_sram = "pjrtx.mem.prefetch"(%matmul_hbm) {edge = 1 : i32, from = "device_hbm", to = "local_sram"} : (memref<1x3xf32>) -> memref<1x3xf32, 1>
    %bias_sram = "pjrtx.mem.prefetch"(%bias_hbm) {edge = 1 : i32, from = "device_hbm", to = "local_sram"} : (memref<3xf32>) -> memref<3xf32, 1>
    "pjrtx.mem.compute_tile"(%matmul_sram, %bias_sram, %out_hbm) {
      tile_plan = 1 : i64,
      lowering_subject = "fusion:1"
    } : (memref<1x3xf32, 1>, memref<3xf32, 1>, memref<1x3xf32>) -> ()
    "pjrtx.mem.return"(%out_hbm) : (memref<1x3xf32>) -> ()
  }) {
    sym_name = "main_memory_planned",
    state = "memory_planned",
    peak_local_sram_bytes = 92 : i64
  } : () -> ()

  // 11-12: collective lowering is still program IR. This happy path has no
  // collective command, but the no-op is explicit and verifiable.
  func.func @main_after_collective_lowering(
      %lhs: tensor<1x4xf32>,
      %rhs: tensor<4x3xf32>,
      %bias: tensor<3xf32>) -> tensor<1x3xf32> attributes {
    pjrtx.state = "collectives_lowered",
    pjrtx.collective_lowering = 0 : i64
  } {
    %zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<1x3xf32>
    %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<1x3xf32>) -> tensor<1x3xf32>
    %dot = linalg.matmul
      ins(%lhs, %rhs : tensor<1x4xf32>, tensor<4x3xf32>)
      outs(%filled : tensor<1x3xf32>) -> tensor<1x3xf32>
    %comm = "pjrtx.collective.lowered_none"(%dot) {interconnect_bytes = 0 : i64} : (tensor<1x3xf32>) -> tensor<1x3xf32>
    %out_init = tensor.empty() : tensor<1x3xf32>
    %out = linalg.generic {
      indexing_maps = [
        affine_map<(d0, d1) -> (d0, d1)>,
        affine_map<(d0, d1) -> (d1)>,
        affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"],
      pjrtx.fusion_candidate = 1 : i64
    } ins(%comm, %bias : tensor<1x3xf32>, tensor<3xf32>)
      outs(%out_init : tensor<1x3xf32>) {
    ^bb0(%m: f32, %b: f32, %old: f32):
      %sum = arith.addf %m, %b : f32
      %tanh = math.tanh %sum : f32
      linalg.yield %tanh : f32
    } -> tensor<1x3xf32>
    return %out : tensor<1x3xf32>
  }

  // 13-15: lower to linalg and then explicit buffer/scf code.
  func.func @main_linalg_tensor(
      %lhs: tensor<1x4xf32>,
      %rhs: tensor<4x3xf32>,
      %bias: tensor<3xf32>) -> tensor<1x3xf32> attributes {
    pjrtx.state = "lowering_planned"
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
      pjrtx.fusion_candidate = 1 : i64
    } ins(%dot, %bias : tensor<1x3xf32>, tensor<3xf32>)
      outs(%out_init : tensor<1x3xf32>) {
    ^bb0(%m: f32, %b: f32, %old: f32):
      %sum = arith.addf %m, %b : f32
      %tanh = math.tanh %sum : f32
      linalg.yield %tanh : f32
    } -> tensor<1x3xf32>
    return %out : tensor<1x3xf32>
  }

  func.func @main_bufferized_memref(
      %lhs: memref<1x4xf32>,
      %rhs: memref<4x3xf32>,
      %bias: memref<3xf32>,
      %out: memref<1x3xf32>) attributes {
    pjrtx.state = "bufferized",
    pjrtx.performance_model = "attached_to_loops"
  } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %f0 = arith.constant 0.0 : f32
    %tmp = memref.alloc() {pjrtx.logical_buffer = 3 : i64} : memref<1x3xf32>
    scf.for %j = %c0 to %c3 step %c1 {
      memref.store %f0, %tmp[%c0, %j] : memref<1x3xf32>
      scf.for %k = %c0 to %c4 step %c1 {
        %acc = memref.load %tmp[%c0, %j] : memref<1x3xf32>
        %a = memref.load %lhs[%c0, %k] : memref<1x4xf32>
        %b = memref.load %rhs[%k, %j] : memref<4x3xf32>
        %prod = arith.mulf %a, %b : f32
        %next = arith.addf %acc, %prod : f32
        memref.store %next, %tmp[%c0, %j] : memref<1x3xf32>
      }
    } {pjrtx.lowering_region = 0 : i64, pjrtx.cost_ledger = 0 : i64}
    scf.for %j = %c0 to %c3 step %c1 {
      %m = memref.load %tmp[%c0, %j] : memref<1x3xf32>
      %b = memref.load %bias[%j] : memref<3xf32>
      %sum = arith.addf %m, %b : f32
      %tanh = math.tanh %sum : f32
      memref.store %tanh, %out[%c0, %j] : memref<1x3xf32>
    } {pjrtx.lowering_region = 1 : i64, pjrtx.cost_ledger = 1 : i64}
    memref.dealloc %tmp : memref<1x3xf32>
    return
  }

  // 16: vector kernel and Aster-like instruction kernel are both visible.
  func.func @npu_fused_broadcast_add_tanh_f32_vector(
      %matmul: memref<1x3xf32>,
      %bias: memref<3xf32>,
      %dst: memref<1x3xf32>) attributes {
    pjrtx.state = "codegen_planned",
    pjrtx.lowering_level = "vector_tile_kernel"
  } {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.0 : f32
    %m = vector.transfer_read %matmul[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x3xf32>, vector<3xf32>
    %b = vector.transfer_read %bias[%c0], %cst {in_bounds = [true]} : memref<3xf32>, vector<3xf32>
    %sum = arith.addf %m, %b : vector<3xf32>
    %tanh = math.tanh %sum : vector<3xf32>
    vector.transfer_write %tanh, %dst[%c0, %c0] {in_bounds = [true, true]} : vector<3xf32>, memref<1x3xf32>
    return
  }

  "pjrtx.npu.kernel"() ({
  ^entry(%matmul_hbm: !pjrtx.ptr<device_hbm>, %bias_hbm: !pjrtx.ptr<device_hbm>, %dst_hbm: !pjrtx.ptr<device_hbm>):
    %matmul_vec, %matmul_tok = "pjrtx.npu.dma.load"(%matmul_hbm) {bytes = 12 : i64, dst_space = "register_file"} : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.vreg<3xf32>, !pjrtx.token<dma>)
    %bias_vec, %bias_tok = "pjrtx.npu.dma.load"(%bias_hbm) {bytes = 12 : i64, dst_space = "register_file"} : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.vreg<3xf32>, !pjrtx.token<dma>)
    "pjrtx.npu.wait"(%matmul_tok, %bias_tok) {kind = "dma"} : (!pjrtx.token<dma>, !pjrtx.token<dma>) -> ()
    %sum = "pjrtx.npu.vadd.f32"(%matmul_vec, %bias_vec) {source_instruction = 2 : i64} : (!pjrtx.vreg<3xf32>, !pjrtx.vreg<3xf32>) -> !pjrtx.vreg<3xf32>
    %out = "pjrtx.npu.vtanh.f32"(%sum) {source_instruction = 3 : i64, math_policy = "strict"} : (!pjrtx.vreg<3xf32>) -> !pjrtx.vreg<3xf32>
    %store_tok = "pjrtx.npu.dma.store"(%out, %dst_hbm) {bytes = 12 : i64} : (!pjrtx.vreg<3xf32>, !pjrtx.ptr<device_hbm>) -> !pjrtx.token<dma>
    "pjrtx.npu.wait"(%store_tok) {kind = "dma"} : (!pjrtx.token<dma>) -> ()
    "pjrtx.npu.return"() : () -> ()
  }) {
    sym_name = "npu_fused_broadcast_add_tanh_f32_tile",
    normal_form = "scheduled_instruction_ir"
  } : () -> ()

  // 18-26: executable dataflow, backend graph, runtime, and profile joins.
  "pjrtx.async.func"() ({
  ^entry(%lhs: !pjrtx.buffer, %rhs: !pjrtx.buffer, %bias: !pjrtx.buffer, %out: !pjrtx.buffer):
    %h2d = "pjrtx.async.transfer"(%lhs, %rhs, %bias) {command = 0 : i64, stream = "dma0", bytes = 92 : i64} : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.async.token
    %tmp, %matmul_done = "pjrtx.async.launch"(%h2d, %lhs, %rhs) {command = 1 : i64, kernel = @npu_matmul_f32_tile} : (!pjrtx.async.token, !pjrtx.buffer, !pjrtx.buffer) -> (!pjrtx.buffer, !pjrtx.async.token)
    %epilogue_done = "pjrtx.async.launch"(%matmul_done, %tmp, %bias, %out) {command = 2 : i64, kernel = @npu_fused_broadcast_add_tanh_f32_tile} : (!pjrtx.async.token, !pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.async.token
    %d2h = "pjrtx.async.transfer"(%epilogue_done, %out) {command = 3 : i64, stream = "dma0", bytes = 24 : i64} : (!pjrtx.async.token, !pjrtx.buffer) -> !pjrtx.async.token
    "pjrtx.async.await"(%d2h) : (!pjrtx.async.token) -> ()
    "pjrtx.async.return"(%out) : (!pjrtx.buffer) -> ()
  }) {sym_name = "main_async_schedule"} : () -> ()

  "pjrtx.kernel_graph.func"() ({
  ^entry(%lhs: !pjrtx.buffer, %rhs: !pjrtx.buffer, %bias: !pjrtx.buffer):
    %tmp = "pjrtx.kernel_graph.call_node"(%lhs, %rhs) {
      node = 0 : i64,
      kernel = @npu_matmul_f32_tile,
      source_instructions = [0 : i64]
    } : (!pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.buffer
    %out = "pjrtx.kernel_graph.call_node"(%tmp, %bias) {
      node = 1 : i64,
      kernel = @npu_fused_broadcast_add_tanh_f32_tile,
      source_instructions = [1 : i64, 2 : i64, 3 : i64]
    } : (!pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.buffer
    "pjrtx.kernel_graph.return"(%out) : (!pjrtx.buffer) -> ()
  }) {sym_name = "main_kernel_graph", state = "backend_kernel_graph_planned"} : () -> ()

  "pjrtx.profiled.program"() ({
  ^entry(%lhs: !pjrtx.buffer, %rhs: !pjrtx.buffer, %bias: !pjrtx.buffer, %out: !pjrtx.buffer):
    %h2d_event = "pjrtx.profiled.h2d"(%lhs, %rhs, %bias) {command = 0 : i64, profile_event = 0 : i64} : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %tmp, %matmul_event = "pjrtx.profiled.kernel"(%h2d_event, %lhs, %rhs) {command = 1 : i64, profile_event = 1 : i64, lowering_region = 0 : i64} : (!pjrtx.event, !pjrtx.buffer, !pjrtx.buffer) -> (!pjrtx.buffer, !pjrtx.event)
    %epilogue_event = "pjrtx.profiled.kernel"(%matmul_event, %tmp, %bias, %out) {command = 2 : i64, profile_event = 2 : i64, lowering_region = 1 : i64} : (!pjrtx.event, !pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %done = "pjrtx.profiled.d2h"(%epilogue_event, %out) {command = 3 : i64, profile_event = 3 : i64} : (!pjrtx.event, !pjrtx.buffer) -> !pjrtx.event
    "pjrtx.profiled.return"(%done, %out) : (!pjrtx.event, !pjrtx.buffer) -> ()
  }) {sym_name = "main_profiled_program"} : () -> ()

  "pjrtx.explain.func"() ({
  ^entry:
    %dot = "pjrtx.explain.trace_source_to_kernel"() {
      source = "stablehlo.dot_general",
      lowering_region = 0 : i64,
      kernel_graph_node = 0 : i64,
      profile_event = 1 : i64
    } : () -> !pjrtx.trace_row
    %epilogue = "pjrtx.explain.trace_source_to_kernel"() {
      source = "stablehlo.broadcast_in_dim+stablehlo.add+stablehlo.tanh",
      fusion_candidate = 1 : i64,
      lowering_region = 1 : i64,
      kernel_graph_node = 1 : i64,
      profile_event = 2 : i64
    } : () -> !pjrtx.trace_row
    "pjrtx.explain.return"(%dot, %epilogue) : (!pjrtx.trace_row, !pjrtx.trace_row) -> ()
  }) {sym_name = "main_explain_trace", state = "backend_profile_joined"} : () -> ()

  // Coverage branches merged from the corpus. These are compact, program-
  // bearing slices that stress HLO features beyond the tiny matmul story.
  module @coverage_all_reduce {
    func.func @stablehlo_all_reduce(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = stablehlo.all_reduce %arg0
        {replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
         channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>}
        ({
        ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
          %sum = stablehlo.add %lhs, %rhs : tensor<f32>
          stablehlo.return %sum : tensor<f32>
        }) : (tensor<4xf32>) -> tensor<4xf32>
      return %0 : tensor<4xf32>
    }
    "pjrtx.collective.func"() ({
    ^entry(%local: tensor<2xf32>):
      %start = "pjrtx.collective.async_start"(%local) {algorithm = "ring", channel_id = 7 : i64} : (tensor<2xf32>) -> !pjrtx.collective_token
      %reduced = "pjrtx.collective.async_done"(%start) {traffic_bytes = 16 : i64} : (!pjrtx.collective_token) -> tensor<2xf32>
      "pjrtx.collective.return"(%reduced) : (tensor<2xf32>) -> ()
    }) {sym_name = "all_reduce_ring"} : () -> ()
  }

  module @coverage_convolution {
    func.func @stablehlo_conv(%input: tensor<1x5x5x1xf32>, %filter: tensor<3x3x1x2xf32>) -> tensor<1x3x3x2xf32> {
      %0 = stablehlo.convolution(%input, %filter)
        dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
        window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [false, false]}
        {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
        : (tensor<1x5x5x1xf32>, tensor<3x3x1x2xf32>) -> tensor<1x3x3x2xf32>
      return %0 : tensor<1x3x3x2xf32>
    }
    func.func @linalg_conv(%input: tensor<1x5x5x1xf32>, %filter: tensor<3x3x1x2xf32>) -> tensor<1x3x3x2xf32> {
      %zero = arith.constant 0.0 : f32
      %init = tensor.empty() : tensor<1x3x3x2xf32>
      %filled = linalg.fill ins(%zero : f32) outs(%init : tensor<1x3x3x2xf32>) -> tensor<1x3x3x2xf32>
      %out = linalg.conv_2d_nhwc_hwcf
        ins(%input, %filter : tensor<1x5x5x1xf32>, tensor<3x3x1x2xf32>)
        outs(%filled : tensor<1x3x3x2xf32>) -> tensor<1x3x3x2xf32>
      return %out : tensor<1x3x3x2xf32>
    }
  }

  module @coverage_reduce_gather_scatter_control {
    func.func @stablehlo_reduce(%arg0: tensor<4x8xf32>, %init: tensor<f32>) -> tensor<4xf32> {
      %0 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.add across dimensions = [1]
        : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
      return %0 : tensor<4xf32>
    }
    "pjrtx.npu.kernel"() ({
    ^entry(%operand: !pjrtx.ptr<device_hbm>, %indices: !pjrtx.ptr<device_hbm>, %dst: !pjrtx.ptr<device_hbm>):
      %idx, %tok = "pjrtx.npu.dma.load"(%indices) {bytes = 16 : i64} : (!pjrtx.ptr<device_hbm>) -> (!pjrtx.ireg<4xi32>, !pjrtx.token<dma>)
      "pjrtx.npu.wait"(%tok) {kind = "dma"} : (!pjrtx.token<dma>) -> ()
      %vals = "pjrtx.npu.gather.f32"(%operand, %idx) {bounds = "checked"} : (!pjrtx.ptr<device_hbm>, !pjrtx.ireg<4xi32>) -> !pjrtx.vreg<4xf32>
      %store = "pjrtx.npu.dma.store"(%vals, %dst) {bytes = 16 : i64} : (!pjrtx.vreg<4xf32>, !pjrtx.ptr<device_hbm>) -> !pjrtx.token<dma>
      "pjrtx.npu.wait"(%store) {kind = "dma"} : (!pjrtx.token<dma>) -> ()
      "pjrtx.npu.return"() : () -> ()
    }) {sym_name = "npu_gather_f32_checked"} : () -> ()
    "pjrtx.control_flow.func"() ({
    ^entry(%arg0: tensor<4xf32>, %limit: i32):
      %c0 = arith.constant 0 : i32
      %result:2 = scf.while (%i = %c0, %v = %arg0) : (i32, tensor<4xf32>) -> (i32, tensor<4xf32>) {
        %pred = arith.cmpi slt, %i, %limit : i32
        scf.condition(%pred) %i, %v : i32, tensor<4xf32>
      } do {
      ^bb0(%i: i32, %v: tensor<4xf32>):
        %one = arith.constant 1 : i32
        %next_i = arith.addi %i, %one : i32
        %next_v = "pjrtx.elementwise.tanh"(%v) : (tensor<4xf32>) -> tensor<4xf32>
        scf.yield %next_i, %next_v : i32, tensor<4xf32>
      }
      "pjrtx.control_flow.return"(%result#1) : (tensor<4xf32>) -> ()
    }) {sym_name = "while_lowered"} : () -> ()
  }

  module @coverage_failures {
    func.func @dynamic_shape_rejected(%arg0: tensor<?x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<?x3xf32> {
      %0 = stablehlo.dot_general %arg0, %arg1,
        contracting_dims = [1] x [0],
        precision = [DEFAULT, DEFAULT]
        : (tensor<?x4xf32>, tensor<4x3xf32>) -> tensor<?x3xf32>
      return %0 : tensor<?x3xf32>
    }
    "pjrtx.verifier_failure"() {
      pass = "pjrtx_dynamic_shape_gate",
      feature = "dynamic_shape",
      diagnostic = "V0 requires static tensor dimensions before target legality",
      executable_created = false,
      runtime_fallback = false
    } : () -> ()
    func.func @custom_call_rejected(%arg0: tensor<4xf32>) -> tensor<4xf32> {
      %0 = stablehlo.custom_call @opaque_backend_magic(%arg0)
        {api_version = 2 : i32, has_side_effect = false}
        : (tensor<4xf32>) -> tensor<4xf32>
      return %0 : tensor<4xf32>
    }
    "pjrtx.verifier_failure"() {
      pass = "target_feature_legality",
      feature = "custom_call",
      diagnostic = "custom_call lacks a PjRTx lowering contract and cannot become a backend-private fallback",
      executable_created = false,
      runtime_fallback = false
    } : () -> ()
  }
}
