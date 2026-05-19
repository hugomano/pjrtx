// Stage 24: runtime stream and event plan. The schedule is now in runtime
// terms while still pointing back to compiler commands.

module @pjrtx_stage_24_runtime_stream_planned attributes {
  pjrtx.state = "runtime_stream_planned",
  pjrtx.stage = "runtime_stream_plan"
} {
  "pjrtx.state_transition"() {from = "runtime_allocation_planned", to = "runtime_stream_planned", pass = "runtime_stream_plan"} : () -> ()
  "pjrtx.runtime_stream_step"() {command = 0 : i64, stream = 0 : i32, wait_events = [], start_event = 100 : i64, done_event = 101 : i64, kind = "h2d"} : () -> ()
  "pjrtx.runtime_stream_step"() {command = 1 : i64, stream = 1 : i32, wait_events = [101 : i64], start_event = 102 : i64, done_event = 103 : i64, kind = "kernel"} : () -> ()
  "pjrtx.runtime_stream_step"() {command = 2 : i64, stream = 1 : i32, wait_events = [103 : i64], start_event = 104 : i64, done_event = 105 : i64, kind = "kernel"} : () -> ()
  "pjrtx.runtime_stream_step"() {command = 3 : i64, stream = 0 : i32, wait_events = [105 : i64], start_event = 106 : i64, done_event = 107 : i64, kind = "d2h"} : () -> ()
  "pjrtx.runtime_event_contract"() {events = [100 : i64, 101 : i64, 102 : i64, 103 : i64, 104 : i64, 105 : i64, 106 : i64, 107 : i64], semantics = "happens_before"} : () -> ()

  "pjrtx.runtime.stream_func"() ({
  ^entry(%buf0: !pjrtx.buffer, %buf1: !pjrtx.buffer, %buf2: !pjrtx.buffer, %buf3: !pjrtx.buffer, %buf4: !pjrtx.buffer):
    %e101 = "pjrtx.runtime.h2d"(%buf0, %buf1, %buf2) {command = 0 : i64, stream = 0 : i32} : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %e103 = "pjrtx.runtime.launch"(%e101, %buf0, %buf1, %buf3) {command = 1 : i64, stream = 1 : i32, kernel = @npu_matmul_f32_tile} : (!pjrtx.event, !pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %e105 = "pjrtx.runtime.launch"(%e103, %buf3, %buf2, %buf4) {command = 2 : i64, stream = 1 : i32, kernel = @npu_fused_broadcast_add_tanh_f32_tile} : (!pjrtx.event, !pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %e107 = "pjrtx.runtime.d2h"(%e105, %buf4) {command = 3 : i64, stream = 0 : i32} : (!pjrtx.event, !pjrtx.buffer) -> !pjrtx.event
    "pjrtx.runtime.stream_return"(%e107) : (!pjrtx.event) -> ()
  }) {sym_name = "main_runtime_stream"} : () -> ()
}
