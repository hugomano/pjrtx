const std = @import("std");

const resident_inputs = @import("executable_metal_graph_resident_inputs.zig");
const storage = @import("executable_metal_graph_graph_request_storage.zig");
const types = @import("execution_types.zig");

/// Collects call-time buffers for one single-kernel graph request.
pub const GraphRequestInputs = struct {
    /// Returns argument handles followed by resident constant handles.
    pub fn collect(
        allocator: std.mem.Allocator,
        request: storage.Request,
        source: resident_inputs.Source,
        arguments: []const types.BufferHandle,
    ) types.Error![]types.BufferHandle {
        if (arguments.len + request.constant_instruction_indices.len != request.input_specs.len) {
            return error.CommandSubmissionFailed;
        }
        const handles = try allocator.alloc(types.BufferHandle, request.input_specs.len);
        errdefer allocator.free(handles);
        @memcpy(handles[0..arguments.len], arguments);
        for (request.constant_instruction_indices, 0..) |instruction_index, constant_index| {
            handles[arguments.len + constant_index] = try source.instructionConstant(instruction_index);
        }
        return handles;
    }
};
