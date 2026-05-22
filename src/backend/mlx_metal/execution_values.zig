const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const types = @import("execution_types.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ExecutableOutput = types.ExecutableOutput;

/// Owns the per-execute value table and destroys intermediate handles it owns.
pub const ValueBindings = struct {
    allocator: std.mem.Allocator,
    handles: []?BufferHandle,
    owned: []bool,

    /// Creates a value table seeded with borrowed PJRT argument buffers.
    pub fn init(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, arguments: []const BufferHandle) Error!ValueBindings {
        const handles = try allocator.alloc(?BufferHandle, plan.values.len);
        errdefer allocator.free(handles);
        @memset(handles, null);

        const owned = try allocator.alloc(bool, plan.values.len);
        errdefer allocator.free(owned);
        @memset(owned, false);

        var parameter_index: usize = 0;
        for (plan.values) |value| {
            if (value.role != .parameter) continue;
            if (parameter_index >= arguments.len or value.id.index >= handles.len) return error.CommandSubmissionFailed;
            handles[value.id.index] = arguments[parameter_index];
            parameter_index += 1;
        }

        return .{
            .allocator = allocator,
            .handles = handles,
            .owned = owned,
        };
    }

    /// Releases all device handles owned by this execution value table.
    pub fn deinit(self: *ValueBindings) void {
        destroyOwnedValueHandles(self.handles, self.owned);
        self.allocator.free(self.owned);
        self.allocator.free(self.handles);
        self.* = undefined;
    }
};

/// Clones or transfers executable output handles from a completed value table.
pub const OutputBindings = struct {
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    values: *ValueBindings,

    /// Returns cloned outputs, or null when an output handle cannot be materialized.
    pub fn cloneOrNull(self: OutputBindings) Error!?[]ExecutableOutput {
        return self.clone(.return_null);
    }

    /// Returns cloned outputs and converts missing output handles into submission failure.
    pub fn cloneOrFail(self: OutputBindings) Error![]ExecutableOutput {
        return (try self.clone(.fail)) orelse error.CommandSubmissionFailed;
    }

    const MissingOutput = enum {
        return_null,
        fail,
    };

    fn clone(self: OutputBindings, missing_output: MissingOutput) Error!?[]ExecutableOutput {
        const plan = self.plan;
        const outputs = try self.allocator.alloc(ExecutableOutput, plan.output_ids.len);
        errdefer self.allocator.free(outputs);
        var initialized: usize = 0;
        errdefer {
            for (outputs[0..initialized]) |output| buffer_mod.Opaque.destroy(output.handle);
        }

        for (plan.output_ids, 0..) |output_id, output_index| {
            if (output_id.index >= self.values.handles.len) return error.CommandSubmissionFailed;
            const value = self.values.handles[output_id.index] orelse return error.CommandSubmissionFailed;
            const handle = if (self.values.owned[output_id.index]) blk: {
                self.values.owned[output_id.index] = false;
                break :blk value;
            } else (try buffer_mod.Opaque.clone(value)) orelse switch (missing_output) {
                .return_null => return null,
                .fail => return error.CommandSubmissionFailed,
            };
            const descriptor = plan.values[output_id.index].descriptor;
            outputs[output_index] = .{
                .handle = handle,
                .element_type = descriptor.element_type,
                .dims = descriptor.dims,
                .byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims),
            };
            initialized += 1;
        }
        return outputs;
    }
};

fn destroyOwnedValueHandles(value_handles: []?BufferHandle, value_owned: []const bool) void {
    for (value_handles, value_owned) |maybe_handle, owned| {
        if (owned) {
            if (maybe_handle) |handle| buffer_mod.Opaque.destroy(handle);
        }
    }
}

/// Resolves start-index value ids into a temporary borrowed buffer-handle slice.
pub fn startHandles(allocator: std.mem.Allocator, value_handles: []const ?BufferHandle, ids: []const ir.ValueId) ![]BufferHandle {
    const handles = try allocator.alloc(BufferHandle, ids.len);
    errdefer allocator.free(handles);
    for (ids, 0..) |id, index| {
        if (id.index >= value_handles.len) return error.CommandSubmissionFailed;
        handles[index] = value_handles[id.index] orelse return error.CommandSubmissionFailed;
    }
    return handles;
}

/// Stores a newly owned handle, replacing and destroying any previous owned handle.
pub fn storeOwnedValueHandle(
    value_handles: []?BufferHandle,
    value_owned: []bool,
    value_id: ir.ValueId,
    handle: BufferHandle,
) Error!void {
    if (value_id.index >= value_handles.len) return error.CommandSubmissionFailed;
    if (value_owned[value_id.index]) {
        if (value_handles[value_id.index]) |old| buffer_mod.Opaque.destroy(old);
    }
    value_handles[value_id.index] = handle;
    value_owned[value_id.index] = true;
}

/// Stores a borrowed handle, replacing and destroying any previous owned handle.
pub fn storeBorrowedValueHandle(
    value_handles: []?BufferHandle,
    value_owned: []bool,
    value_id: ir.ValueId,
    handle: BufferHandle,
) Error!void {
    if (value_id.index >= value_handles.len) return error.CommandSubmissionFailed;
    if (value_owned[value_id.index]) {
        if (value_handles[value_id.index]) |old| buffer_mod.Opaque.destroy(old);
    }
    value_handles[value_id.index] = handle;
    value_owned[value_id.index] = false;
}

/// Applies a reverse view for descending sort directions while preserving handle ownership.
pub fn reverseIfDescending(
    handle: BufferHandle,
    dimension: i64,
    output_dims: []const i64,
    direction: ir.CompareOp,
) Error!?BufferHandle {
    return switch (direction) {
        .lt, .le => handle,
        .gt, .ge => blk: {
            const dimensions = [_]i64{dimension};
            const reversed = buffer_mod.Opaque.reverse(handle, &dimensions, output_dims) catch |err| {
                buffer_mod.Opaque.destroy(handle);
                return err;
            };
            buffer_mod.Opaque.destroy(handle);
            break :blk reversed;
        },
        else => blk: {
            buffer_mod.Opaque.destroy(handle);
            break :blk null;
        },
    };
}
