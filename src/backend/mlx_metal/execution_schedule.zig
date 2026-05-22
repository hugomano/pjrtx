const std = @import("std");
const ir = @import("src/compiler/ir");

const executable_mod = @import("executable.zig");
const liveness_mod = @import("execution_liveness.zig");
const materialization_mod = @import("execution_materialization.zig");
const metalcpp_fusion_runner = @import("metalcpp_fusion_runner.zig");
const node_mod = @import("execution_node.zig");
const profiling_mod = @import("profiling.zig");
const program_mod = @import("program.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const Error = types.Error;
const CompiledExecutable = executable_mod.Executable;
const ExecuteProfile = profiling_mod.Execute;
const ValueBindings = values_mod.ValueBindings;

fn traceScheduleFailure(executable: *const CompiledExecutable, item: program_mod.ScheduleItem, err: Error) void {
    var snapshot = profiling_mod.ScheduleFailureSnapshot{
        .schedule_kind = @tagName(item.kind),
        .schedule_index = item.index,
        .schedule_count = item.count,
        .err = @errorName(err),
    };
    switch (item.kind) {
        .node => {
            if (item.index < executable.program.nodes.len) {
                const node = executable.program.nodes[item.index];
                snapshot.node_kind = @tagName(node.kind);
                snapshot.instruction_index = node.instruction_index;
                if (node.instruction_index < executable.plan.instructions.len) {
                    const instruction = executable.plan.instructions[node.instruction_index];
                    snapshot.op = @tagName(instruction.kind);
                    if (instruction.outputs.len > 0 and instruction.outputs[0].index < executable.plan.values.len) {
                        const descriptor = executable.plan.values[instruction.outputs[0].index].descriptor;
                        snapshot.output_value = instruction.outputs[0].index;
                        snapshot.dtype = @tagName(descriptor.element_type);
                        snapshot.rank = descriptor.dims.len;
                    }
                }
            }
        },
        .fusion_group => {
            if (item.index < executable.program.fusion_groups.len) {
                const group = executable.program.fusion_groups[item.index];
                snapshot.group_first_node = group.first_node;
                snapshot.group_last_node = group.last_node;
                snapshot.group_node_count = group.node_count;
            }
        },
        .materialization_boundary => {},
    }
    profiling_mod.writeScheduleFailure(snapshot);
}

/// Walks a backend program schedule and delegates each item to its owner.
pub const ScheduleDispatch = struct {
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    profile: ?*ExecuteProfile = null,
    profile_enabled: bool = false,
    profile_verbose: bool = false,

    /// Executes the full schedule, including materialization boundaries.
    pub fn run(self: ScheduleDispatch, bindings: *ValueBindings) Error!bool {
        for (self.executable.program.schedule, 0..) |schedule_item, schedule_index| {
            const schedule_start_ns = profiling_mod.start(self.profile_enabled);
            if (!try self.dispatch(schedule_item, bindings, true)) return false;
            const schedule_us = profiling_mod.elapsedUs(schedule_start_ns);
            self.recordProfile(schedule_index, schedule_item, schedule_us);
        }
        return true;
    }

    /// Replays the schedule while building an MLX compiled-program trace.
    pub fn runForCompiledTrace(self: ScheduleDispatch, bindings: *ValueBindings) Error!void {
        for (self.executable.program.schedule) |schedule_item| {
            if (!try self.dispatch(schedule_item, bindings, false)) return error.CommandSubmissionFailed;
        }
    }

    fn dispatch(self: ScheduleDispatch, schedule_item: program_mod.ScheduleItem, bindings: *ValueBindings, materialize_boundaries: bool) Error!bool {
        switch (schedule_item.kind) {
            .node => {
                return ((node_mod.ProgramNodeDispatch{
                    .allocator = self.allocator,
                    .executable = self.executable,
                    .device_index = self.device_index,
                    .values = bindings,
                    .release_inputs = true,
                }).run(schedule_item.index) catch |err| {
                    traceScheduleFailure(self.executable, schedule_item, err);
                    return err;
                }) != null;
            },
            .fusion_group => {
                return ((FusionGroupDispatch{
                    .allocator = self.allocator,
                    .executable = self.executable,
                    .device_index = self.device_index,
                    .values = bindings,
                }).run(schedule_item.index, schedule_item.count) catch |err| {
                    traceScheduleFailure(self.executable, schedule_item, err);
                    return err;
                }) != null;
            },
            .materialization_boundary => {
                if (!materialize_boundaries) return true;
                (materialization_mod.MaterializationBoundaryEval{
                    .program = &self.executable.program,
                    .values = bindings,
                }).run(schedule_item.index, schedule_item.count) catch |err| {
                    traceScheduleFailure(self.executable, schedule_item, err);
                    return err;
                };
                self.executable.recordMaterializationEval(schedule_item.count);
                return true;
            },
        }
    }

    fn recordProfile(self: ScheduleDispatch, schedule_index: usize, schedule_item: program_mod.ScheduleItem, schedule_us: u64) void {
        if (!self.profile_enabled) return;
        const profile = self.profile orelse return;
        profile.schedule_us +|= schedule_us;
        profile.schedule_peak_us = @max(profile.schedule_peak_us, schedule_us);
        switch (schedule_item.kind) {
            .node => {
                profile.node_us +|= schedule_us;
                profile.node_peak_us = @max(profile.node_peak_us, schedule_us);
            },
            .fusion_group => {
                profile.fusion_group_us +|= schedule_us;
                profile.fusion_group_peak_us = @max(profile.fusion_group_peak_us, schedule_us);
            },
            .materialization_boundary => {
                profile.materialization_eval_us +|= schedule_us;
                profile.materialization_eval_peak_us = @max(profile.materialization_eval_peak_us, schedule_us);
            },
        }
        if (self.profile_verbose) {
            writeScheduleProfile(self.executable, schedule_index, schedule_item, schedule_us);
        }
    }
};

const FusionGroupDispatch = struct {
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    values: *ValueBindings,

    pub fn run(self: FusionGroupDispatch, group_index: usize, scheduled_node_count: usize) Error!?void {
        if (group_index >= self.executable.program.fusion_groups.len) return error.CommandSubmissionFailed;
        const group = self.executable.program.fusion_groups[group_index];
        if (scheduled_node_count != group.node_indices.len) return error.CommandSubmissionFailed;
        switch (group.kind) {
            .view_elementwise => {},
        }
        self.executable.recordFusionGroupExecute();
        if ((try (metalcpp_fusion_runner.FusionRunner{
            .allocator = self.allocator,
            .plan = self.executable.plan,
            .program = &self.executable.program,
            .device_local_hardware_id = self.executable.device_local_hardware_ids[self.device_index],
            .resident_kernel = self.executable.metalCppFusionKernel(self.device_index, group.id),
            .values = self.values,
        }).run(group)) != null) {
            const released = (liveness_mod.LivenessRelease{
                .program = &self.executable.program,
                .values = self.values,
            }).fusionGroup(group);
            if (released != 0) self.executable.recordReleasedIntermediateValues(released);
            return {};
        }
        for (group.node_indices) |group_node_index| {
            if (group_node_index >= self.executable.program.nodes.len) return error.CommandSubmissionFailed;
            const node = self.executable.program.nodes[group_node_index];
            if (node.fusion_group != group_index) return error.CommandSubmissionFailed;
            (try (node_mod.ProgramNodeDispatch{
                .allocator = self.allocator,
                .executable = self.executable,
                .device_index = self.device_index,
                .values = self.values,
                .release_inputs = false,
            }).run(group_node_index)) orelse return null;
        }
        const released = (liveness_mod.LivenessRelease{
            .program = &self.executable.program,
            .values = self.values,
        }).fusionGroup(group);
        if (released != 0) self.executable.recordReleasedIntermediateValues(released);
        return {};
    }
};

/// Emits an executable-level execution profile snapshot.
pub fn writeExecuteProfile(executable: *const CompiledExecutable, device_index: usize, argument_count: usize, output_count: usize, profile: ExecuteProfile) void {
    profiling_mod.writeExecute(.{
        .address = @intFromPtr(executable),
        .schedule_count = executable.program.schedule.len,
        .node_count = executable.program.nodes.len,
        .fusion_group_count = executable.program.fusion_groups.len,
        .materialization_boundary_count = executable.program.materialization_boundaries.len,
    }, device_index, argument_count, output_count, profile);
}

fn writeScheduleProfile(executable: *const CompiledExecutable, schedule_index: usize, item: program_mod.ScheduleItem, elapsed_us: u64) void {
    var snapshot = profiling_mod.ScheduleItemSnapshot{
        .executable_address = @intFromPtr(executable),
        .schedule_index = schedule_index,
        .kind = switch (item.kind) {
            .node => .node,
            .fusion_group => .fusion_group,
            .materialization_boundary => .materialization_boundary,
        },
        .index = item.index,
        .count = item.count,
    };
    switch (item.kind) {
        .node => {
            if (item.index < executable.program.nodes.len) {
                const node = executable.program.nodes[item.index];
                snapshot.node_kind = @tagName(node.kind);
                snapshot.instruction_index = node.instruction_index;
                if (node.instruction_index < executable.plan.instructions.len) {
                    const instruction = executable.plan.instructions[node.instruction_index];
                    snapshot.op = @tagName(instruction.kind);
                }
            }
        },
        .fusion_group => {
            var group_ops_buffer: [128][]const u8 = undefined;
            if (item.index < executable.program.fusion_groups.len) {
                const group = executable.program.fusion_groups[item.index];
                snapshot.group_first_node = group.first_node;
                snapshot.group_last_node = group.last_node;
                snapshot.group_node_count = group.node_count;
                var op_count: usize = 0;
                for (group.node_indices) |group_node_index| {
                    if (op_count >= group_ops_buffer.len) break;
                    if (group_node_index >= executable.program.nodes.len) {
                        group_ops_buffer[op_count] = "?";
                    } else {
                        const group_node = executable.program.nodes[group_node_index];
                        group_ops_buffer[op_count] = if (group_node.instruction_index < executable.plan.instructions.len)
                            @tagName(executable.plan.instructions[group_node.instruction_index].kind)
                        else
                            "?";
                    }
                    op_count += 1;
                }
                snapshot.group_ops = group_ops_buffer[0..op_count];
            }
        },
        .materialization_boundary => {},
    }
    profiling_mod.writeScheduleItem(snapshot, elapsed_us);
}
