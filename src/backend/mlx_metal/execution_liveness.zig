const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const program_mod = @import("program.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = @import("execution_types.zig").BufferHandle;
const ValueBindings = values_mod.ValueBindings;

/// Releases owned intermediate value handles after their final scheduled use.
pub const LivenessRelease = struct {
    program: *const program_mod.Program,
    values: *ValueBindings,

    /// Releases node inputs whose liveness ends at the current instruction.
    pub fn afterNode(self: LivenessRelease, input_ids: []const ir.ValueId, instruction_index: usize) usize {
        var released: usize = 0;
        for (input_ids) |input_id| {
            if (input_id.index >= self.values.handles.len or input_id.index >= self.program.values.len) continue;
            const value = self.program.values[input_id.index];
            if (value.is_output) continue;
            if (value.last_use_node != @as(?usize, instruction_index)) continue;
            if (!self.values.owned[input_id.index]) continue;
            if (self.values.handles[input_id.index]) |old| buffer_mod.Opaque.destroy(old);
            self.values.handles[input_id.index] = null;
            self.values.owned[input_id.index] = false;
            released += 1;
        }
        return released;
    }

    /// Releases fusion-group inputs whose final consumer is inside the group.
    pub fn fusionGroup(self: LivenessRelease, group: program_mod.FusionGroup) usize {
        var released: usize = 0;
        for (group.node_indices) |node_index| {
            if (node_index >= self.program.nodes.len) continue;
            const node = self.program.nodes[node_index];
            for (node.inputs) |input_id| {
                if (input_id.index >= self.values.handles.len or input_id.index >= self.program.values.len) continue;
                const value = self.program.values[input_id.index];
                if (value.is_output) continue;
                const last_use = value.last_use_node orelse continue;
                if (last_use > group.last_node) continue;
                if (!self.values.owned[input_id.index]) continue;
                if (self.values.handles[input_id.index]) |old| buffer_mod.Opaque.destroy(old);
                self.values.handles[input_id.index] = null;
                self.values.owned[input_id.index] = false;
                released += 1;
            }
        }
        return released;
    }
};
