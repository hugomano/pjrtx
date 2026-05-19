// Stage 25: synthetic/real profile events joined back to compiler facts. This
// is the final explainability loop: source -> lowering -> kernel -> hardware
// estimate -> observed event.

module @pjrtx_stage_25_runtime_profile_joined attributes {
  pjrtx.state = "runtime_profile_joined",
  pjrtx.stage = "profile_event_join"
} {
  "pjrtx.state_transition"() {from = "runtime_stream_planned", to = "runtime_profiled", pass = "runtime_profile_collect"} : () -> ()
  "pjrtx.runtime_profile_event"() {id = 0 : i64, command = 0 : i64, kind = "h2d", bytes = 92 : i64, logical_ops = 0 : i64, status = "synthetic"} : () -> ()
  "pjrtx.runtime_profile_event"() {id = 1 : i64, command = 1 : i64, kind = "kernel", bytes = 104 : i64, logical_ops = 24 : i64, status = "synthetic"} : () -> ()
  "pjrtx.runtime_profile_event"() {id = 2 : i64, command = 2 : i64, kind = "kernel", bytes = 60 : i64, logical_ops = 9 : i64, status = "synthetic"} : () -> ()
  "pjrtx.runtime_profile_event"() {id = 3 : i64, command = 3 : i64, kind = "d2h", bytes = 24 : i64, logical_ops = 0 : i64, status = "synthetic"} : () -> ()
  "pjrtx.state_transition"() {from = "runtime_profiled", to = "runtime_profile_joined", pass = "profile_event_join"} : () -> ()
  "pjrtx.profile_join"() {
    id = 0 : i64,
    subject = "lowering_region",
    subject_id = 0 : i64,
    profile_events = [1 : i64],
    source_instructions = [0 : i64],
    hardware_unit = "trn2_tensor_engine",
    limiting_resource = "launch_overhead_for_tiny_fixture"
  } : () -> ()
  "pjrtx.profile_join"() {
    id = 1 : i64,
    subject = "lowering_region",
    subject_id = 1 : i64,
    profile_events = [2 : i64],
    source_instructions = [1 : i64, 2 : i64, 3 : i64],
    hardware_unit = "trn2_vector_engine",
    limiting_resource = "transcendental_latency_model_unknown"
  } : () -> ()

  "pjrtx.profile.func"() ({
  ^entry:
    %matmul_event = "pjrtx.profile.event"() {
      event = 1 : i64,
      command = 1 : i64,
      bytes = 104 : i64,
      logical_ops = 24 : i64
    } : () -> !pjrtx.profile_event
    %epilogue_event = "pjrtx.profile.event"() {
      event = 2 : i64,
      command = 2 : i64,
      bytes = 60 : i64,
      logical_ops = 9 : i64
    } : () -> !pjrtx.profile_event
    "pjrtx.profile.join"(%matmul_event) {
      join = 0 : i64,
      lowering_region = 0 : i64,
      source_instructions = [0 : i64]
    } : (!pjrtx.profile_event) -> ()
    "pjrtx.profile.join"(%epilogue_event) {
      join = 1 : i64,
      lowering_region = 1 : i64,
      source_instructions = [1 : i64, 2 : i64, 3 : i64]
    } : (!pjrtx.profile_event) -> ()
    "pjrtx.profile.return"() : () -> ()
  }) {sym_name = "main_runtime_profile"} : () -> ()

  "pjrtx.profiled.program"() ({
  ^entry(%lhs: !pjrtx.buffer, %rhs: !pjrtx.buffer, %bias: !pjrtx.buffer, %out: !pjrtx.buffer):
    %h2d_event = "pjrtx.profiled.h2d"(%lhs, %rhs, %bias) {
      command = 0 : i64,
      profile_event = 0 : i64
    } : (!pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %tmp, %matmul_event = "pjrtx.profiled.kernel"(%h2d_event, %lhs, %rhs) {
      command = 1 : i64,
      kernel = @npu_matmul_f32_tile,
      profile_event = 1 : i64,
      lowering_region = 0 : i64
    } : (!pjrtx.event, !pjrtx.buffer, !pjrtx.buffer) -> (!pjrtx.buffer, !pjrtx.event)
    %epilogue_event = "pjrtx.profiled.kernel"(%matmul_event, %tmp, %bias, %out) {
      command = 2 : i64,
      kernel = @npu_fused_broadcast_add_tanh_f32_tile,
      profile_event = 2 : i64,
      lowering_region = 1 : i64
    } : (!pjrtx.event, !pjrtx.buffer, !pjrtx.buffer, !pjrtx.buffer) -> !pjrtx.event
    %done = "pjrtx.profiled.d2h"(%epilogue_event, %out) {
      command = 3 : i64,
      profile_event = 3 : i64
    } : (!pjrtx.event, !pjrtx.buffer) -> !pjrtx.event
    "pjrtx.profiled.return"(%done, %out) : (!pjrtx.event, !pjrtx.buffer) -> ()
  }) {sym_name = "main_profiled_program"} : () -> ()
}
