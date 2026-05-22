const std = @import("std");
const ir = @import("src/compiler/ir");

const executable_mod = @import("executable.zig");
const program_mod = @import("program.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");
const while_mod = @import("execution_control_flow_while.zig");

const Error = types.Error;
const CompiledExecutable = executable_mod.Executable;
const ValueBindings = values_mod.ValueBindings;

/// Executes backend control-flow nodes using device-side MLX primitives only.
pub const ControlFlowDispatch = struct {
    executable: *CompiledExecutable,
    device_index: usize,
    values: *ValueBindings,
    release_inputs: bool,

    /// Executes one scheduled control-flow node and stores its output handles.
    pub fn run(self: ControlFlowDispatch, node: program_mod.Node, instruction: ir.PlanInstruction) Error!?void {
        const control_flow_index = node.control_flow orelse return error.CommandSubmissionFailed;
        if (control_flow_index >= self.executable.program.control_flows.len) return error.CommandSubmissionFailed;
        const control_flow = self.executable.program.control_flows[control_flow_index];
        switch (control_flow.kind) {
            .while_loop => return (while_mod.WhileLoopDispatch{
                .executable = self.executable,
                .device_index = self.device_index,
                .values = self.values,
                .release_inputs = self.release_inputs,
            }).run(node, instruction, control_flow_index, control_flow),
        }
    }
};
