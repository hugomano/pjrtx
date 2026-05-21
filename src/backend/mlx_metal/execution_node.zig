const std = @import("std");
const ir = @import("src/compiler/ir");

const control_flow_mod = @import("execution_control_flow.zig");
const custom_call_dispatch_mod = @import("execution_custom_call.zig");
const elementwise_mod = @import("execution_node_elementwise.zig");
const executable_mod = @import("executable.zig");
const generation_mod = @import("execution_node_generation.zig");
const indexing_mod = @import("execution_node_indexing.zig");
const linalg_mod = @import("execution_node_linalg.zig");
const liveness_mod = @import("execution_liveness.zig");
const reduction_mod = @import("execution_node_reduction.zig");
const structural_mod = @import("execution_node_structural.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const CompiledExecutable = executable_mod.Executable;
const ValueBindings = values_mod.ValueBindings;

/// Executes one backend program node and records produced device handles.
pub const ProgramNodeDispatch = struct {
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    values: *ValueBindings,
    release_inputs: bool,

    /// Dispatches the node at the given program index using device-resident operands.
    pub fn run(self: ProgramNodeDispatch, node_index: usize) Error!?void {
        const executable = self.executable;
        const plan = executable.plan;
        if (node_index >= executable.program.nodes.len) return error.CommandSubmissionFailed;
        const node = executable.program.nodes[node_index];
        const instruction_index = node.instruction_index;
        const instruction = plan.instructions[instruction_index];

        if (node.kind == .control_flow) {
            (try (control_flow_mod.ControlFlowDispatch{
                .executable = executable,
                .device_index = self.device_index,
                .values = self.values,
                .release_inputs = self.release_inputs,
            }).run(node, instruction)) orelse return null;
            return {};
        }

        if (try self.runStored(instruction_index, instruction)) |_| {
            self.releaseNodeInputsIfNeeded(node.inputs, instruction_index);
            return {};
        }

        if (instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        if (output_id.index >= self.values.handles.len or output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
        const output_descriptor = plan.values[output_id.index].descriptor;
        const output_dims = instruction.dims orelse output_descriptor.dims;
        const next = (try self.runSingle(instruction, output_descriptor, output_dims)) orelse return null;
        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, output_id, next);
        self.releaseNodeInputsIfNeeded(node.inputs, instruction_index);
        return {};
    }

    fn runStored(self: ProgramNodeDispatch, instruction_index: usize, instruction: ir.PlanInstruction) Error!?void {
        if (try self.structural().runStored(instruction_index, instruction)) |_| return {};
        if (try self.indexing().runStored(instruction)) |_| return {};
        if (try self.reduction().runStored(instruction)) |_| return {};
        if (try self.generation().runStored(instruction)) |_| return {};
        return null;
    }

    fn runSingle(self: ProgramNodeDispatch, instruction: ir.PlanInstruction, output_descriptor: ir.BufferDescriptor, output_dims: []const i64) Error!?BufferHandle {
        if (try self.structural().run(instruction)) |handle| return handle;
        if (instruction.kind == .custom_call) return (custom_call_dispatch_mod.CustomCallDispatch{ .values = self.values }).run(instruction);
        if (try (elementwise_mod.Context{ .values = self.values }).run(instruction, output_descriptor, output_dims)) |handle| return handle;
        if (try self.indexing().run(instruction, output_dims)) |handle| return handle;
        if (try (linalg_mod.Context{ .values = self.values }).run(instruction, output_dims)) |handle| return handle;
        if (try self.reduction().run(instruction, output_dims)) |handle| return handle;
        if (try self.generation().run(instruction, output_descriptor, output_dims)) |handle| return handle;
        return null;
    }

    fn structural(self: ProgramNodeDispatch) structural_mod.Context {
        return .{
            .executable = self.executable,
            .device_index = self.device_index,
            .values = self.values,
        };
    }

    fn indexing(self: ProgramNodeDispatch) indexing_mod.Context {
        return .{
            .allocator = self.allocator,
            .plan = self.executable.plan,
            .values = self.values,
        };
    }

    fn reduction(self: ProgramNodeDispatch) reduction_mod.Context {
        return .{
            .plan = self.executable.plan,
            .values = self.values,
        };
    }

    fn generation(self: ProgramNodeDispatch) generation_mod.Context {
        return .{
            .executable = self.executable,
            .device_index = self.device_index,
            .values = self.values,
        };
    }

    fn releaseNodeInputsIfNeeded(self: ProgramNodeDispatch, inputs: []const ir.ValueId, instruction_index: usize) void {
        if (!self.release_inputs) return;
        const released = (liveness_mod.LivenessRelease{
            .program = &self.executable.program,
            .values = self.values,
        }).afterNode(inputs, instruction_index);
        if (released != 0) self.executable.recordReleasedIntermediateValues(released);
    }
};
