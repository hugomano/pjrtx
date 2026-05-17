const std = @import("std");
const c = @import("c");
const backend = @import("src/backend");
const core = @import("src/core");

pub fn create() backend.Backend {
    return .{ .vtable = &vtable };
}

fn kind(_: backend.Backend) core.BackendKind {
    return .metal_mlx;
}

fn capabilities(_: backend.Backend) backend.Capabilities {
    return .{
        .kind = .metal_mlx,
        .name = "metal_mlx",
        .supports_device_buffers = true,
        .supports_unified_memory = true,
    };
}

fn enumerateDevices(_: backend.Backend, allocator: std.mem.Allocator, _: usize) backend.Error![]core.DeviceDescriptor {
    var metal_devices: [core.MAX_DEVICES]c.PjrtxMlxMetalDeviceInfo = undefined;
    const copied = c.pjrtx_mlx_metal_copy_devices(&metal_devices, core.MAX_DEVICES);
    const count: usize = if (copied <= 0) 1 else @intCast(copied);
    const devices = try allocator.alloc(core.DeviceDescriptor, count);
    errdefer allocator.free(devices);

    for (devices, 0..) |*device, i| {
        const metal_device = if (copied > 0) metal_devices[i] else std.mem.zeroes(c.PjrtxMlxMetalDeviceInfo);
        const name_bytes = if (copied > 0) cNameBytes(&metal_device.name) else "Metal/MLX device";
        const name = try allocator.dupe(u8, name_bytes);
        errdefer allocator.free(name);

        var debug_buffer: [256]u8 = undefined;
        var debug_writer = std.Io.Writer.fixed(&debug_buffer);
        debug_writer.print("PjRTx Metal/MLX device {d}: {s}", .{ i, name }) catch return error.BufferAllocationFailed;
        const debug_string = try allocator.dupe(u8, debug_writer.buffered());
        errdefer allocator.free(debug_string);

        const id: i32 = @intCast(i);
        device.* = .{
            .id = id,
            .local_hardware_id = if (copied > 0) metal_device.ordinal else id,
            .registry_id = if (copied > 0) metal_device.registry_id else 0,
            .name = name,
            .debug_string = debug_string,
            .memory_bytes = if (copied > 0) metal_device.recommended_max_working_set_size else 0,
            .has_unified_memory = copied <= 0 or metal_device.has_unified_memory != 0,
            .default_memory_id = id,
        };
    }
    return devices;
}

fn releaseDeviceDescriptors(_: backend.Backend, allocator: std.mem.Allocator, descriptors: []core.DeviceDescriptor) void {
    for (descriptors) |descriptor| {
        allocator.free(descriptor.name);
        allocator.free(descriptor.debug_string);
    }
    allocator.free(descriptors);
}

fn bufferFromHost(_: backend.Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, src: []const u8) backend.Error!?backend.BufferHandle {
    if (src.len == 0) return null;
    const dtype = mlxDtype(element_type) orelse return error.UnsupportedElementType;
    const handle = c.pjrtx_mlx_metal_buffer_from_host_typed(device_local_hardware_id, src.ptr, src.len, dtype, dims.ptr, dims.len) orelse return error.BufferAllocationFailed;
    return @ptrCast(handle);
}

fn cloneBuffer(_: backend.Backend, src: backend.BufferHandle) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_clone(@ptrCast(@alignCast(src))) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn convert(_: backend.Backend, src: backend.BufferHandle, output_type: core.BufferType) backend.Error!?backend.BufferHandle {
    const dtype = mlxDtype(output_type) orelse return error.UnsupportedElementType;
    const handle = c.pjrtx_mlx_metal_buffer_astype(@ptrCast(@alignCast(src)), dtype) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn binary(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, op: core.ElementwiseBinaryOp) backend.Error!?backend.BufferHandle {
    const op_code = mlxBinaryOpCode(op) orelse return error.CommandSubmissionFailed;
    const handle = c.pjrtx_mlx_metal_buffer_binary(@ptrCast(@alignCast(lhs)), @ptrCast(@alignCast(rhs)), op_code) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn unary(_: backend.Backend, src: backend.BufferHandle, op: core.ElementwiseUnaryOp) backend.Error!?backend.BufferHandle {
    const op_code = mlxUnaryOpCode(op) orelse return error.CommandSubmissionFailed;
    const handle = c.pjrtx_mlx_metal_buffer_unary(@ptrCast(@alignCast(src)), op_code) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn reshape(_: backend.Backend, src: backend.BufferHandle, dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_reshape(@ptrCast(@alignCast(src)), dims.ptr, dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn transpose(_: backend.Backend, src: backend.BufferHandle, permutation: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_transpose(@ptrCast(@alignCast(src)), permutation.ptr, permutation.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn broadcastInDim(_: backend.Backend, src: backend.BufferHandle, broadcast_dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_broadcast_in_dim(@ptrCast(@alignCast(src)), broadcast_dimensions.ptr, broadcast_dimensions.len, output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn slice(_: backend.Backend, src: backend.BufferHandle, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_slice(@ptrCast(@alignCast(src)), start_indices.ptr, limit_indices.ptr, strides.ptr, start_indices.len, output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn dynamicSlice(_: backend.Backend, src: backend.BufferHandle, start_buffers: []const backend.BufferHandle, slice_sizes: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    if (start_buffers.len != slice_sizes.len) return error.ShapeMismatch;
    const handle = c.pjrtx_mlx_metal_buffer_dynamic_slice(
        @ptrCast(@alignCast(src)),
        @ptrCast(start_buffers.ptr),
        start_buffers.len,
        slice_sizes.ptr,
        slice_sizes.len,
        output_dims.ptr,
        output_dims.len,
    ) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn dynamicUpdateSlice(_: backend.Backend, src: backend.BufferHandle, update: backend.BufferHandle, start_buffers: []const backend.BufferHandle, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_dynamic_update_slice(
        @ptrCast(@alignCast(src)),
        @ptrCast(@alignCast(update)),
        @ptrCast(start_buffers.ptr),
        start_buffers.len,
        output_dims.ptr,
        output_dims.len,
    ) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn pad(_: backend.Backend, src: backend.BufferHandle, padding_value: backend.BufferHandle, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_pad(
        @ptrCast(@alignCast(src)),
        @ptrCast(@alignCast(padding_value)),
        edge_padding_low.ptr,
        edge_padding_high.ptr,
        interior_padding.ptr,
        edge_padding_low.len,
        output_dims.ptr,
        output_dims.len,
    ) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn reverse(_: backend.Backend, src: backend.BufferHandle, dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_reverse(
        @ptrCast(@alignCast(src)),
        dimensions.ptr,
        dimensions.len,
        output_dims.ptr,
        output_dims.len,
    ) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn concatenate(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, dimension: i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_concatenate(@ptrCast(@alignCast(lhs)), @ptrCast(@alignCast(rhs)), dimension, output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn gatherAxis(_: backend.Backend, operand: backend.BufferHandle, indices: backend.BufferHandle, axis: i64, index_vector_dim: i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_gather_axis(
        @ptrCast(@alignCast(operand)),
        @ptrCast(@alignCast(indices)),
        axis,
        index_vector_dim,
        output_dims.ptr,
        output_dims.len,
    ) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn sort(_: backend.Backend, src: backend.BufferHandle, dimension: i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_sort(
        @ptrCast(@alignCast(src)),
        dimension,
        output_dims.ptr,
        output_dims.len,
    ) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn dotGeneral(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_dot_general(
        @ptrCast(@alignCast(lhs)),
        @ptrCast(@alignCast(rhs)),
        lhs_batch_dimensions.ptr,
        lhs_batch_dimensions.len,
        rhs_batch_dimensions.ptr,
        rhs_batch_dimensions.len,
        lhs_contracting_dimensions.ptr,
        lhs_contracting_dimensions.len,
        rhs_contracting_dimensions.ptr,
        rhs_contracting_dimensions.len,
        output_dims.ptr,
        output_dims.len,
    ) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn reduce(_: backend.Backend, src: backend.BufferHandle, op: core.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const code: c_int = switch (op) {
        .reduce_sum => c.PJRTX_MLX_METAL_REDUCE_SUM,
        .reduce_max => c.PJRTX_MLX_METAL_REDUCE_MAX,
        else => return error.CommandSubmissionFailed,
    };
    const handle = c.pjrtx_mlx_metal_buffer_reduce(@ptrCast(@alignCast(src)), code, dimensions.ptr, dimensions.len, output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn compare(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, direction: core.CompareOp, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_compare(@ptrCast(@alignCast(lhs)), @ptrCast(@alignCast(rhs)), mlxCompareOpCode(direction), output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn select(_: backend.Backend, pred: backend.BufferHandle, on_true: backend.BufferHandle, on_false: backend.BufferHandle, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_select(@ptrCast(@alignCast(pred)), @ptrCast(@alignCast(on_true)), @ptrCast(@alignCast(on_false)), output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn copyToHost(_: backend.Backend, src: backend.BufferHandle, dst: []u8) backend.Error!void {
    const ok = c.pjrtx_mlx_metal_buffer_copy_to_host(@ptrCast(@alignCast(src)), dst.ptr, dst.len);
    if (ok == 0) return error.BufferCopyFailed;
}

fn destroyBuffer(_: backend.Backend, buffer: backend.BufferHandle) void {
    c.pjrtx_mlx_metal_buffer_destroy(@ptrCast(@alignCast(buffer)));
}

const CompiledExecutable = struct {
    allocator: std.mem.Allocator,
    plan: *const core.ExecutablePlan,
    device_local_hardware_ids: []i32,
    constant_handles: []?backend.BufferHandle,
};

fn compileExecutable(backend_impl: backend.Backend, allocator: std.mem.Allocator, plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32) backend.Error!?backend.ExecutableHandle {
    if (device_local_hardware_ids.len == 0) return null;
    for (plan.instructions) |instruction| {
        if (!executableSupportsInstruction(instruction.kind)) return null;
        if (instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        if (output_id.index >= plan.values.len) return null;
        if (instruction.kind == .constant and instruction.literal == null) return null;
    }

    const executable = try allocator.create(CompiledExecutable);
    errdefer allocator.destroy(executable);
    const ids = try allocator.dupe(i32, device_local_hardware_ids);
    errdefer allocator.free(ids);
    const constant_handles = try allocator.alloc(?backend.BufferHandle, plan.instructions.len * device_local_hardware_ids.len);
    errdefer allocator.free(constant_handles);
    @memset(constant_handles, null);
    errdefer destroyConstantHandles(backend_impl, constant_handles);

    for (device_local_hardware_ids, 0..) |local_hardware_id, device_index| {
        for (plan.instructions, 0..) |instruction, instruction_index| {
            if (instruction.kind != .constant) continue;
            const output_id = instruction.outputs[0];
            const descriptor = plan.values[output_id.index].descriptor;
            const literal = instruction.literal.?;
            constant_handles[constantIndex(plan.instructions.len, device_index, instruction_index)] = (try bufferFromHost(
                backend_impl,
                local_hardware_id,
                descriptor.element_type,
                descriptor.dims,
                literal,
            )) orelse return error.CommandSubmissionFailed;
        }
    }

    executable.* = .{
        .allocator = allocator,
        .plan = plan,
        .device_local_hardware_ids = ids,
        .constant_handles = constant_handles,
    };
    return @ptrCast(executable);
}

fn executeExecutable(backend_impl: backend.Backend, allocator: std.mem.Allocator, executable_handle: backend.ExecutableHandle, device_index: usize, arguments: []const backend.BufferHandle) backend.Error!?[]backend.ExecutableOutput {
    const executable: *CompiledExecutable = @ptrCast(@alignCast(executable_handle));
    const plan = executable.plan;
    if (device_index >= executable.device_local_hardware_ids.len) return error.CommandSubmissionFailed;
    if (arguments.len < plan.parameter_shardings.len) return error.CommandSubmissionFailed;

    var value_handles = try allocator.alloc(?backend.BufferHandle, plan.values.len);
    defer allocator.free(value_handles);
    @memset(value_handles, null);

    var value_owned = try allocator.alloc(bool, plan.values.len);
    defer allocator.free(value_owned);
    @memset(value_owned, false);
    defer destroyOwnedValueHandles(backend_impl, value_handles, value_owned);

    var parameter_index: usize = 0;
    for (plan.values) |value| {
        if (value.role != .parameter) continue;
        if (parameter_index >= arguments.len or value.id.index >= value_handles.len) return error.CommandSubmissionFailed;
        value_handles[value.id.index] = arguments[parameter_index];
        parameter_index += 1;
    }

    for (plan.instructions, 0..) |instruction, instruction_index| {
        if (instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        if (output_id.index >= value_handles.len) return error.CommandSubmissionFailed;
        const output_descriptor = plan.values[output_id.index].descriptor;
        const output_dims = instruction.dims orelse output_descriptor.dims;
        const next = switch (instruction.kind) {
            .constant => blk: {
                const cached = executable.constant_handles[constantIndex(plan.instructions.len, device_index, instruction_index)] orelse return null;
                break :blk (try cloneBuffer(backend_impl, cached)) orelse return null;
            },
            .copy_arg0, .reduce_precision => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try cloneBuffer(backend_impl, input)) orelse return null;
            },
            .convert => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try convert(backend_impl, input, output_descriptor.element_type)) orelse return null;
            },
            .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .remainder => blk: {
                const op = executableBinaryOp(instruction.kind) orelse return null;
                const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try binary(backend_impl, lhs, rhs, op)) orelse return null;
            },
            .negate, .exp, .tanh, .sqrt, .rsqrt, .abs, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .sign => blk: {
                const op = executableUnaryOp(instruction.kind) orelse return null;
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try unary(backend_impl, input, op)) orelse return null;
            },
            .reshape => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try reshape(backend_impl, input, output_dims)) orelse return null;
            },
            .transpose => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try transpose(backend_impl, input, instruction.permutation orelse return null)) orelse return null;
            },
            .broadcast_in_dim => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try broadcastInDim(backend_impl, input, instruction.broadcast_dimensions orelse return null, output_dims)) orelse return null;
            },
            .slice => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try slice(
                    backend_impl,
                    input,
                    instruction.start_indices orelse return null,
                    instruction.limit_indices orelse return null,
                    instruction.strides orelse return null,
                    output_dims,
                )) orelse return null;
            },
            .dynamic_slice => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const starts = try startHandles(allocator, value_handles, instruction.inputs[1..]);
                defer allocator.free(starts);
                break :blk (try dynamicSlice(backend_impl, input, starts, instruction.slice_sizes orelse return null, output_dims)) orelse return null;
            },
            .dynamic_update_slice => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const update = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                const starts = try startHandles(allocator, value_handles, instruction.inputs[2..]);
                defer allocator.free(starts);
                break :blk (try dynamicUpdateSlice(backend_impl, input, update, starts, output_dims)) orelse return null;
            },
            .pad => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const padding_value = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try pad(
                    backend_impl,
                    input,
                    padding_value,
                    instruction.edge_padding_low orelse return null,
                    instruction.edge_padding_high orelse return null,
                    instruction.interior_padding orelse return null,
                    output_dims,
                )) orelse return null;
            },
            .reverse => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try reverse(backend_impl, input, instruction.dimensions orelse &.{}, output_dims)) orelse return null;
            },
            .concatenate => blk: {
                const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try concatenate(backend_impl, lhs, rhs, instruction.dimension orelse return null, output_dims)) orelse return null;
            },
            .gather => blk: {
                const gather_axis = supportedGatherAxis(instruction) orelse return null;
                const operand = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const indices = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try gatherAxis(backend_impl, operand, indices, gather_axis, instruction.index_vector_dim orelse 0, output_dims)) orelse return null;
            },
            .sort => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const sorted = (try sort(backend_impl, input, instruction.dimension orelse return null, output_dims)) orelse return null;
                switch (instruction.compare_direction orelse .lt) {
                    .lt, .le => break :blk sorted,
                    .gt, .ge => {
                        const dimensions = [_]i64{instruction.dimension.?};
                        const reversed = reverse(backend_impl, sorted, &dimensions, output_dims) catch |err| {
                            destroyBuffer(backend_impl, sorted);
                            return err;
                        };
                        destroyBuffer(backend_impl, sorted);
                        break :blk reversed orelse return null;
                    },
                    else => {
                        destroyBuffer(backend_impl, sorted);
                        return null;
                    },
                }
            },
            .dot_general => blk: {
                const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try dotGeneral(
                    backend_impl,
                    lhs,
                    rhs,
                    instruction.lhs_batch_dimensions orelse &.{},
                    instruction.rhs_batch_dimensions orelse &.{},
                    instruction.lhs_contracting_dimensions orelse &.{},
                    instruction.rhs_contracting_dimensions orelse &.{},
                    output_dims,
                )) orelse return null;
            },
            .reduce_sum, .reduce_max => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try reduce(backend_impl, input, instruction.kind, instruction.reduce_dimensions orelse &.{}, output_dims)) orelse return null;
            },
            .compare => blk: {
                const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try compare(backend_impl, lhs, rhs, instruction.compare_direction orelse .eq, output_dims)) orelse return null;
            },
            .select => blk: {
                const pred = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const on_true = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                const on_false = value_handles[instruction.inputs[2].index] orelse return error.CommandSubmissionFailed;
                break :blk (try select(backend_impl, pred, on_true, on_false, output_dims)) orelse return null;
            },
            else => return null,
        };

        if (value_owned[output_id.index]) {
            if (value_handles[output_id.index]) |old| destroyBuffer(backend_impl, old);
        }
        value_handles[output_id.index] = next;
        value_owned[output_id.index] = true;
    }

    const outputs = try allocator.alloc(backend.ExecutableOutput, plan.output_ids.len);
    errdefer allocator.free(outputs);
    var initialized: usize = 0;
    errdefer {
        for (outputs[0..initialized]) |output| destroyBuffer(backend_impl, output.handle);
    }

    for (plan.output_ids, 0..) |output_id, output_index| {
        if (output_id.index >= value_handles.len) return error.CommandSubmissionFailed;
        const value = value_handles[output_id.index] orelse return error.CommandSubmissionFailed;
        const handle = if (value_owned[output_id.index]) blk: {
            value_owned[output_id.index] = false;
            break :blk value;
        } else (try cloneBuffer(backend_impl, value)) orelse return null;
        const descriptor = plan.values[output_id.index].descriptor;
        outputs[output_index] = .{
            .handle = handle,
            .element_type = descriptor.element_type,
            .dims = descriptor.dims,
            .byte_size = core.denseByteSize(descriptor.element_type, descriptor.dims),
        };
        initialized += 1;
    }

    return outputs;
}

fn destroyExecutable(backend_impl: backend.Backend, executable_handle: backend.ExecutableHandle) void {
    const executable: *CompiledExecutable = @ptrCast(@alignCast(executable_handle));
    destroyConstantHandles(backend_impl, executable.constant_handles);
    executable.allocator.free(executable.constant_handles);
    executable.allocator.free(executable.device_local_hardware_ids);
    executable.allocator.destroy(executable);
}

fn destroyConstantHandles(backend_impl: backend.Backend, constant_handles: []?backend.BufferHandle) void {
    for (constant_handles) |maybe_handle| {
        if (maybe_handle) |handle| destroyBuffer(backend_impl, handle);
    }
}

fn constantIndex(instruction_count: usize, device_index: usize, instruction_index: usize) usize {
    return device_index * instruction_count + instruction_index;
}

fn executableSupportsInstruction(kind_: core.PlanInstructionKind) bool {
    return switch (kind_) {
        .constant,
        .copy_arg0,
        .reduce_precision,
        .convert,
        .add,
        .subtract,
        .multiply,
        .divide,
        .maximum,
        .minimum,
        .power,
        .remainder,
        .negate,
        .exp,
        .tanh,
        .sqrt,
        .rsqrt,
        .abs,
        .ceil,
        .floor,
        .log,
        .log1p,
        .logistic,
        .sine,
        .cosine,
        .sign,
        .reshape,
        .transpose,
        .broadcast_in_dim,
        .slice,
        .dynamic_slice,
        .dynamic_update_slice,
        .pad,
        .reverse,
        .concatenate,
        .gather,
        .sort,
        .dot_general,
        .reduce_sum,
        .reduce_max,
        .compare,
        .select,
        => true,
        else => false,
    };
}

fn destroyOwnedValueHandles(backend_impl: backend.Backend, value_handles: []?backend.BufferHandle, value_owned: []const bool) void {
    for (value_handles, value_owned) |maybe_handle, owned| {
        if (owned) {
            if (maybe_handle) |handle| destroyBuffer(backend_impl, handle);
        }
    }
}

fn startHandles(allocator: std.mem.Allocator, value_handles: []const ?backend.BufferHandle, ids: []const core.ValueId) ![]backend.BufferHandle {
    const handles = try allocator.alloc(backend.BufferHandle, ids.len);
    errdefer allocator.free(handles);
    for (ids, 0..) |id, index| {
        if (id.index >= value_handles.len) return error.CommandSubmissionFailed;
        handles[index] = value_handles[id.index] orelse return error.CommandSubmissionFailed;
    }
    return handles;
}

fn supportedGatherAxis(instruction: core.PlanInstruction) ?i64 {
    const start_index_map = instruction.start_index_map orelse return null;
    const slice_sizes = instruction.slice_sizes orelse return null;
    const collapsed_slice_dims = instruction.collapsed_slice_dims orelse return null;
    if (start_index_map.len != 1 or collapsed_slice_dims.len != 1) return null;
    const axis = start_index_map[0];
    if (axis != 0 or collapsed_slice_dims[0] != axis) return null;
    if (slice_sizes.len == 0 or slice_sizes[@intCast(axis)] != 1) return null;
    return axis;
}

fn executableBinaryOp(instruction_kind: core.PlanInstructionKind) ?core.ElementwiseBinaryOp {
    return switch (instruction_kind) {
        .add => .add,
        .subtract => .subtract,
        .multiply => .multiply,
        .divide => .divide,
        .maximum => .maximum,
        .minimum => .minimum,
        .power => .power,
        .remainder => .remainder,
        else => null,
    };
}

fn executableUnaryOp(instruction_kind: core.PlanInstructionKind) ?core.ElementwiseUnaryOp {
    return switch (instruction_kind) {
        .negate => .negate,
        .exp => .exp,
        .tanh => .tanh,
        .sqrt => .sqrt,
        .rsqrt => .rsqrt,
        .abs => .abs,
        .ceil => .ceil,
        .floor => .floor,
        .log => .log,
        .log1p => .log1p,
        .logistic => .logistic,
        .sine => .sine,
        .cosine => .cosine,
        .sign => .sign,
        else => null,
    };
}

fn cNameBytes(name: *const [128]u8) []const u8 {
    const end = std.mem.indexOfScalar(u8, name, 0) orelse name.len;
    return name[0..end];
}

fn mlxDtype(element_type: core.BufferType) ?c_int {
    return switch (element_type) {
        .pred => c.PJRTX_MLX_METAL_DTYPE_PRED,
        .s8 => c.PJRTX_MLX_METAL_DTYPE_S8,
        .s32 => c.PJRTX_MLX_METAL_DTYPE_S32,
        .u8 => c.PJRTX_MLX_METAL_DTYPE_U8,
        .u32 => c.PJRTX_MLX_METAL_DTYPE_U32,
        .f16 => c.PJRTX_MLX_METAL_DTYPE_F16,
        .f32 => c.PJRTX_MLX_METAL_DTYPE_F32,
        .bf16 => c.PJRTX_MLX_METAL_DTYPE_BF16,
        else => null,
    };
}

fn mlxBinaryOpCode(op: core.ElementwiseBinaryOp) ?c_int {
    return switch (op) {
        .add => c.PJRTX_MLX_METAL_U8_BINARY_ADD,
        .subtract => c.PJRTX_MLX_METAL_U8_BINARY_SUBTRACT,
        .multiply => c.PJRTX_MLX_METAL_U8_BINARY_MULTIPLY,
        .divide => c.PJRTX_MLX_METAL_U8_BINARY_DIVIDE,
        .maximum => c.PJRTX_MLX_METAL_BINARY_MAXIMUM,
        .minimum => c.PJRTX_MLX_METAL_BINARY_MINIMUM,
        .power => c.PJRTX_MLX_METAL_BINARY_POWER,
        .remainder => c.PJRTX_MLX_METAL_BINARY_REMAINDER,
        else => null,
    };
}

fn mlxUnaryOpCode(op: core.ElementwiseUnaryOp) ?c_int {
    return switch (op) {
        .negate => c.PJRTX_MLX_METAL_U8_UNARY_NEGATE,
        .exp => c.PJRTX_MLX_METAL_UNARY_EXP,
        .tanh => c.PJRTX_MLX_METAL_UNARY_TANH,
        .sqrt => c.PJRTX_MLX_METAL_UNARY_SQRT,
        .rsqrt => c.PJRTX_MLX_METAL_UNARY_RSQRT,
        .abs => c.PJRTX_MLX_METAL_UNARY_ABS,
        .ceil => c.PJRTX_MLX_METAL_UNARY_CEIL,
        .floor => c.PJRTX_MLX_METAL_UNARY_FLOOR,
        .log => c.PJRTX_MLX_METAL_UNARY_LOG,
        .log1p => c.PJRTX_MLX_METAL_UNARY_LOG1P,
        .logistic => c.PJRTX_MLX_METAL_UNARY_LOGISTIC,
        .sine => c.PJRTX_MLX_METAL_UNARY_SIN,
        .cosine => c.PJRTX_MLX_METAL_UNARY_COS,
        .sign => c.PJRTX_MLX_METAL_UNARY_SIGN,
        else => null,
    };
}

fn mlxCompareOpCode(op: core.CompareOp) c_int {
    return switch (op) {
        .eq => c.PJRTX_MLX_METAL_COMPARE_EQ,
        .ne => c.PJRTX_MLX_METAL_COMPARE_NE,
        .ge => c.PJRTX_MLX_METAL_COMPARE_GE,
        .gt => c.PJRTX_MLX_METAL_COMPARE_GT,
        .le => c.PJRTX_MLX_METAL_COMPARE_LE,
        .lt => c.PJRTX_MLX_METAL_COMPARE_LT,
    };
}

const vtable: backend.Backend.VTable = .{
    .kind = kind,
    .capabilities = capabilities,
    .enumerateDevices = enumerateDevices,
    .releaseDeviceDescriptors = releaseDeviceDescriptors,
    .bufferFromHost = bufferFromHost,
    .cloneBuffer = cloneBuffer,
    .convert = convert,
    .binary = binary,
    .unary = unary,
    .reshape = reshape,
    .transpose = transpose,
    .broadcastInDim = broadcastInDim,
    .slice = slice,
    .dynamicSlice = dynamicSlice,
    .dynamicUpdateSlice = dynamicUpdateSlice,
    .pad = pad,
    .reverse = reverse,
    .concatenate = concatenate,
    .gatherAxis = gatherAxis,
    .sort = sort,
    .dotGeneral = dotGeneral,
    .reduce = reduce,
    .compare = compare,
    .select = select,
    .compileExecutable = compileExecutable,
    .executeExecutable = executeExecutable,
    .destroyExecutable = destroyExecutable,
    .copyToHost = copyToHost,
    .destroyBuffer = destroyBuffer,
};

test "mlx metal backend exposes opaque backend interface" {
    const b = create();
    try std.testing.expectEqual(core.BackendKind.metal_mlx, b.kind());
    const devices = try b.enumerateDevices(std.testing.allocator, 1);
    defer b.releaseDeviceDescriptors(std.testing.allocator, devices);
    try std.testing.expect(devices.len >= 1);
}

test "mlx metal backend executable runs resident device buffers" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{4};
    const assignment = [_]i32{0};

    const values = try allocator.alloc(core.Value, 3);
    errdefer allocator.free(values);
    for (values, 0..) |*value, i| {
        value.* = .{
            .id = .{ .index = @intCast(i) },
            .role = if (i == 0) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    var parameter_shardings = try allocator.alloc(core.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(core.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = core.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(core.PlanInstruction, &.{
            .{
                .kind = .constant,
                .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 1 }}),
                .literal = try allocator.dupe(u8, &.{ 10, 20, 30, 40 }),
            },
            .{
                .kind = .add,
                .inputs = try allocator.dupe(core.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
                .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }}),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);

    const lhs = (try b.bufferFromHost(local_hardware_id, .u8, &dims, &.{ 1, 2, 3, 4 })) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(lhs);

    const outputs = (try b.executeExecutable(allocator, executable, 0, &.{lhs})) orelse return error.TestUnexpectedResult;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| b.destroyBuffer(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(core.BufferType.u8, outputs[0].element_type);
    try std.testing.expectEqualSlices(i64, &dims, outputs[0].dims);
    var actual: [4]u8 = undefined;
    try b.copyToHost(outputs[0].handle, &actual);
    try std.testing.expectEqualSlices(u8, &.{ 11, 22, 33, 44 }, &actual);

    const second_lhs = (try b.bufferFromHost(local_hardware_id, .u8, &dims, &.{ 2, 4, 6, 8 })) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(second_lhs);
    const second_outputs = (try b.executeExecutable(allocator, executable, 0, &.{second_lhs})) orelse return error.TestUnexpectedResult;
    defer allocator.free(second_outputs);
    defer {
        for (second_outputs) |output| b.destroyBuffer(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), second_outputs.len);
    try b.copyToHost(second_outputs[0].handle, &actual);
    try std.testing.expectEqualSlices(u8, &.{ 12, 24, 36, 48 }, &actual);
}
