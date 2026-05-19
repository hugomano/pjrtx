// Stage 10: memory-space assignment and transfer intent. This is not runtime
// allocation yet; it is the compiler's placement and movement plan.

module @pjrtx_stage_10_memory_planned attributes {
  pjrtx.state = "memory_planned",
  pjrtx.stage = "memory_space_assignment"
} {
  "pjrtx.state_transition"() {from = "tiled", to = "memory_planned", pass = "memory_space_assign"} : () -> ()
  "pjrtx.memory_space_assignment"() {value = 0 : i64, memory_space = 0 : i32, placement = "device_hbm", size_bytes = 32 : i64, reason = "sharded input lives in HBM"} : () -> ()
  "pjrtx.memory_space_assignment"() {value = 1 : i64, memory_space = 0 : i32, placement = "device_hbm", size_bytes = 48 : i64, reason = "replicated weight lives in HBM"} : () -> ()
  "pjrtx.memory_space_assignment"() {value = 2 : i64, memory_space = 0 : i32, placement = "device_hbm", size_bytes = 12 : i64, reason = "bias lives in HBM"} : () -> ()
  "pjrtx.memory_space_assignment"() {value = 3 : i64, memory_space = 0 : i32, placement = "device_hbm", size_bytes = 12 : i64, reason = "matmul output feeds next kernel"} : () -> ()
  "pjrtx.memory_space_assignment"() {value = 6 : i64, memory_space = 0 : i32, placement = "device_hbm", size_bytes = 12 : i64, reason = "program result"} : () -> ()
  "pjrtx.prefetch"() {id = 0 : i64, from = "device_hbm", to = "local_sram", values = [0 : i64, 1 : i64], edge = 1 : i32, reason = "matmul tile staging"} : () -> ()
  "pjrtx.prefetch"() {id = 1 : i64, from = "device_hbm", to = "local_sram", values = [3 : i64, 2 : i64], edge = 1 : i32, reason = "fused elementwise tile staging"} : () -> ()
  "pjrtx.memory_pressure"() {space = "local_sram", peak_live_bytes = 92 : i64, capacity_bytes = 67108864 : i64, status = "fits"} : () -> ()
  "pjrtx.memory_pressure"() {space = "device_hbm", peak_live_bytes = 116 : i64, capacity_bytes = 34359738368 : i64, status = "fits"} : () -> ()

  "pjrtx.memory.func"() ({
  ^entry(%arg0_hbm: memref<1x4xf32>, %arg1_hbm: memref<4x3xf32>, %arg2_hbm: memref<3xf32>, %out_hbm: memref<1x3xf32>):
    %lhs_tile = "pjrtx.mem.prefetch"(%arg0_hbm) {
      prefetch = 0 : i64,
      from = "device_hbm",
      to = "local_sram"
    } : (memref<1x4xf32>) -> memref<1x4xf32, 1>
    %rhs_tile = "pjrtx.mem.prefetch"(%arg1_hbm) {
      prefetch = 0 : i64,
      from = "device_hbm",
      to = "local_sram"
    } : (memref<4x3xf32>) -> memref<4x3xf32, 1>
    %matmul_hbm = "pjrtx.mem.compute_tile"(%lhs_tile, %rhs_tile) {
      lowering_subject = "instruction:0",
      result_memory_space = "device_hbm"
    } : (memref<1x4xf32, 1>, memref<4x3xf32, 1>) -> memref<1x3xf32>
    %matmul_tile = "pjrtx.mem.prefetch"(%matmul_hbm) {prefetch = 1 : i64, from = "device_hbm", to = "local_sram"} : (memref<1x3xf32>) -> memref<1x3xf32, 1>
    %bias_tile = "pjrtx.mem.prefetch"(%arg2_hbm) {prefetch = 1 : i64, from = "device_hbm", to = "local_sram"} : (memref<3xf32>) -> memref<3xf32, 1>
    "pjrtx.mem.compute_tile"(%matmul_tile, %bias_tile, %out_hbm) {
      lowering_subject = "fusion:1",
      result_memory_space = "device_hbm"
    } : (memref<1x3xf32, 1>, memref<3xf32, 1>, memref<1x3xf32>) -> ()
    "pjrtx.mem.return"(%out_hbm) : (memref<1x3xf32>) -> ()
  }) {sym_name = "main_memory_planned"} : () -> ()
}
