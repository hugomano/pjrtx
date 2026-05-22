const constants_mod = @import("executable_constants.zig");
const types = @import("execution_types.zig");

/// Identifies resident constant buffers for one executable/device pair.
pub const Source = struct {
    instruction_count: usize,
    device_index: usize,
    constant_handles: []const ?types.BufferHandle,

    /// Returns the resident buffer for one constant instruction on this device.
    pub fn instructionConstant(self: Source, instruction_index: usize) types.Error!types.BufferHandle {
        const slot = constants_mod.constantIndex(self.instruction_count, self.device_index, instruction_index);
        if (slot >= self.constant_handles.len) return error.CommandSubmissionFailed;
        return self.constant_handles[slot] orelse error.CommandSubmissionFailed;
    }
};
