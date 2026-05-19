// Stage 18: executable schedule with streams, dependencies, overlap candidates,
// barriers, transfers, and kernel commands.

module @pjrtx_stage_18_scheduled attributes {
  pjrtx.state = "scheduled",
  pjrtx.stage = "schedule_build+schedule_overlap_plan+schedule_verify"
} {
  "pjrtx.state_transition"() {from = "tile_legal", to = "scheduled", pass = "schedule_build"} : () -> ()
  "pjrtx.stream"() {id = 0 : i32, kind = "dma", capacity = 1 : i32} : () -> ()
  "pjrtx.stream"() {id = 1 : i32, kind = "compute", capacity = 1 : i32} : () -> ()
  "pjrtx.schedule_command"() {id = 0 : i64, kind = "host_to_device", stream = 0 : i32, buffers = [0 : i64, 1 : i64, 2 : i64], waits = [], signals = [10 : i64]} : () -> ()
  "pjrtx.schedule_command"() {id = 1 : i64, kind = "kernel", stream = 1 : i32, kernel = 0 : i64, waits = [10 : i64], signals = [11 : i64]} : () -> ()
  "pjrtx.schedule_command"() {id = 2 : i64, kind = "kernel", stream = 1 : i32, kernel = 1 : i64, waits = [11 : i64], signals = [12 : i64]} : () -> ()
  "pjrtx.schedule_command"() {id = 3 : i64, kind = "device_to_host", stream = 0 : i32, buffers = [4 : i64], waits = [12 : i64], signals = [13 : i64]} : () -> ()
  "pjrtx.schedule_overlap"() {id = 0 : i64, decision = "rejected", commands = [0 : i64, 1 : i64], reason = "kernel waits on all fixture inputs"} : () -> ()
  "pjrtx.schedule_overlap"() {id = 1 : i64, decision = "rejected", commands = [2 : i64, 3 : i64], reason = "D2H waits on final result"} : () -> ()

  "pjrtx.schedule.func"() ({
  ^entry:
    %h2d_done = "pjrtx.cmd.h2d"() {
      command = 0 : i64,
      stream = 0 : i32,
      buffers = [0 : i64, 1 : i64, 2 : i64]
    } : () -> !pjrtx.event
    %matmul_done = "pjrtx.cmd.launch"(%h2d_done) {
      command = 1 : i64,
      stream = 1 : i32,
      kernel = @npu_matmul_f32_tile,
      inputs = [0 : i64, 1 : i64],
      outputs = [3 : i64]
    } : (!pjrtx.event) -> !pjrtx.event
    %epilogue_done = "pjrtx.cmd.launch"(%matmul_done) {
      command = 2 : i64,
      stream = 1 : i32,
      kernel = @npu_fused_broadcast_add_tanh_f32_tile,
      inputs = [3 : i64, 2 : i64],
      outputs = [4 : i64]
    } : (!pjrtx.event) -> !pjrtx.event
    %d2h_done = "pjrtx.cmd.d2h"(%epilogue_done) {
      command = 3 : i64,
      stream = 0 : i32,
      buffers = [4 : i64]
    } : (!pjrtx.event) -> !pjrtx.event
    "pjrtx.schedule.return"(%d2h_done) : (!pjrtx.event) -> ()
  }) {sym_name = "main_schedule"} : () -> ()
}
