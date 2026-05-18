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

fn iota(_: backend.Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, iota_dimension: i64) backend.Error!?backend.BufferHandle {
    const dtype = mlxDtype(element_type) orelse return error.UnsupportedElementType;
    const handle = c.pjrtx_mlx_metal_buffer_iota(device_local_hardware_id, dtype, dims.ptr, dims.len, iota_dimension) orelse return error.CommandSubmissionFailed;
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

fn gather(_: backend.Backend, operand: backend.BufferHandle, indices: backend.BufferHandle, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_gather(
        @ptrCast(@alignCast(operand)),
        @ptrCast(@alignCast(indices)),
        start_index_map.ptr,
        start_index_map.len,
        collapsed_slice_dims.ptr,
        collapsed_slice_dims.len,
        operand_batching_dims.ptr,
        operand_batching_dims.len,
        start_indices_batching_dims.ptr,
        start_indices_batching_dims.len,
        index_vector_dim,
        slice_sizes.ptr,
        slice_sizes.len,
        offset_dims.ptr,
        offset_dims.len,
        output_dims.ptr,
        output_dims.len,
    ) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn scatterAxis(_: backend.Backend, operand: backend.BufferHandle, indices: backend.BufferHandle, updates: backend.BufferHandle, axis: i64, index_vector_dim: i64, update_kind: core.ScatterUpdateKind, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_scatter_axis(
        @ptrCast(@alignCast(operand)),
        @ptrCast(@alignCast(indices)),
        @ptrCast(@alignCast(updates)),
        axis,
        index_vector_dim,
        mlxScatterUpdateCode(update_kind),
        output_dims.ptr,
        output_dims.len,
    ) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn scatter(_: backend.Backend, operand: backend.BufferHandle, indices: backend.BufferHandle, updates: backend.BufferHandle, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: core.ScatterUpdateKind, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_scatter(
        @ptrCast(@alignCast(operand)),
        @ptrCast(@alignCast(indices)),
        @ptrCast(@alignCast(updates)),
        scatter_dims_to_operand_dims.ptr,
        scatter_dims_to_operand_dims.len,
        inserted_window_dims.ptr,
        inserted_window_dims.len,
        update_window_dims.ptr,
        update_window_dims.len,
        input_batching_dims.ptr,
        input_batching_dims.len,
        scatter_indices_batching_dims.ptr,
        scatter_indices_batching_dims.len,
        index_vector_dim,
        mlxScatterUpdateCode(update_kind),
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

fn argsort(_: backend.Backend, src: backend.BufferHandle, dimension: i64, output_type: core.BufferType, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const dtype = mlxDtype(output_type) orelse return error.UnsupportedElementType;
    const handle = c.pjrtx_mlx_metal_buffer_argsort(
        @ptrCast(@alignCast(src)),
        dimension,
        dtype,
        output_dims.ptr,
        output_dims.len,
    ) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn takeAlongAxis(_: backend.Backend, src: backend.BufferHandle, indices: backend.BufferHandle, dimension: i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_take_along_axis(
        @ptrCast(@alignCast(src)),
        @ptrCast(@alignCast(indices)),
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
        .reduce_and => c.PJRTX_MLX_METAL_REDUCE_AND,
        .reduce_or => c.PJRTX_MLX_METAL_REDUCE_OR,
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

fn clamp(_: backend.Backend, min: backend.BufferHandle, value: backend.BufferHandle, max: backend.BufferHandle, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_clamp(@ptrCast(@alignCast(min)), @ptrCast(@alignCast(value)), @ptrCast(@alignCast(max)), output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
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
    program: backend.Program,
};

const LoweringIssue = struct {
    instruction_index: ?usize = null,
    value_id: ?core.ValueId = null,
    op: ?core.PlanInstructionKind = null,
    detail: []const u8,
    feature: []const u8 = "mlx-backend-executable",
};

fn buildBackendProgram(allocator: std.mem.Allocator, plan: *const core.ExecutablePlan) !backend.Program {
    var nodes = try allocator.alloc(backend.ProgramNode, plan.instructions.len);
    errdefer allocator.free(nodes);
    var initialized_nodes: usize = 0;
    errdefer {
        for (nodes[0..initialized_nodes]) |node| {
            allocator.free(node.inputs);
            allocator.free(node.outputs);
        }
    }

    const last_uses = try allocator.alloc(usize, plan.values.len);
    errdefer allocator.free(last_uses);
    @memset(last_uses, 0);

    const output_values = try allocator.alloc(bool, plan.values.len);
    errdefer allocator.free(output_values);
    @memset(output_values, false);
    for (plan.output_ids) |output_id| {
        if (output_id.index < output_values.len) output_values[output_id.index] = true;
    }

    for (plan.instructions, 0..) |instruction, instruction_index| {
        for (instruction.inputs) |input_id| {
            if (input_id.index < last_uses.len) last_uses[input_id.index] = instruction_index;
        }
        nodes[instruction_index] = .{
            .instruction_index = instruction_index,
            .kind = programNodeKind(instruction.kind),
            .inputs = try allocator.dupe(core.ValueId, instruction.inputs),
            .outputs = try allocator.dupe(core.ValueId, instruction.outputs),
            .materializes = instructionMaterializes(instruction.kind),
        };
        initialized_nodes += 1;
    }

    return .{
        .allocator = allocator,
        .nodes = nodes,
        .last_uses = last_uses,
        .output_values = output_values,
    };
}

fn programNodeKind(instruction_kind: core.PlanInstructionKind) backend.ProgramNodeKind {
    return switch (instruction_kind) {
        .constant => .constant,
        .copy_arg0 => .parameter,
        .reshape, .transpose, .broadcast_in_dim, .slice => .view,
        .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .atan2, .remainder, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical, .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .abs, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .is_finite, .round_nearest_even, .compare, .select, .clamp => .elementwise,
        .reduce_sum, .reduce_max, .reduce_and, .reduce_or => .reduction,
        .dot_general => .matmul,
        .sort, .top_k, .gather, .scatter, .dynamic_slice, .dynamic_update_slice, .pad, .reverse, .concatenate, .iota, .convert, .reduce_precision => .materialize,
        else => .library_call,
    };
}

fn instructionMaterializes(instruction_kind: core.PlanInstructionKind) bool {
    return switch (instruction_kind) {
        .reshape, .transpose, .broadcast_in_dim, .slice => false,
        else => true,
    };
}

fn compileExecutable(backend_impl: backend.Backend, allocator: std.mem.Allocator, plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32) backend.Error!?backend.ExecutableHandle {
    if (executableLoweringIssue(plan, device_local_hardware_ids)) |_| return null;

    const executable = try allocator.create(CompiledExecutable);
    errdefer allocator.destroy(executable);
    const ids = try allocator.dupe(i32, device_local_hardware_ids);
    errdefer allocator.free(ids);
    const constant_handles = try allocator.alloc(?backend.BufferHandle, plan.instructions.len * device_local_hardware_ids.len);
    errdefer allocator.free(constant_handles);
    @memset(constant_handles, null);
    errdefer destroyConstantHandles(backend_impl, constant_handles);
    var program = try buildBackendProgram(allocator, plan);
    errdefer program.deinit();

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
        .program = program,
    };
    return @ptrCast(executable);
}

fn writeExecutableLoweringDiagnostic(_: backend.Backend, plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    if (executableLoweringIssue(plan, device_local_hardware_ids)) |issue| {
        try writeLoweringIssue(plan, issue, writer);
        return;
    }
    try writer.writeAll("invalid executable plan: pass=mlx-backend-legalization detail=\"MLX backend rejected executable plan without a specific issue\" feature=mlx-backend-executable");
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

    for (executable.program.nodes) |node| {
        const instruction_index = node.instruction_index;
        const instruction = plan.instructions[instruction_index];
        if (instruction.kind == .sort and instruction.inputs.len == 2 and instruction.outputs.len == 2) {
            const dimension = instruction.dimension orelse return null;
            const direction = instruction.compare_direction orelse .lt;
            const keys = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const values = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            const key_output_id = instruction.outputs[0];
            const value_output_id = instruction.outputs[1];
            if (key_output_id.index >= plan.values.len or value_output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
            const key_descriptor = plan.values[key_output_id.index].descriptor;
            const value_descriptor = plan.values[value_output_id.index].descriptor;
            const key_dims = instruction.dims orelse key_descriptor.dims;
            const value_dims = value_descriptor.dims;
            const sorted_keys = (try sort(backend_impl, keys, dimension, key_dims)) orelse return null;
            const directed_keys = (try reverseIfDescending(backend_impl, sorted_keys, dimension, key_dims, direction)) orelse return null;
            errdefer destroyBuffer(backend_impl, directed_keys);
            const order = (try argsort(backend_impl, keys, dimension, value_descriptor.element_type, value_dims)) orelse return null;
            const directed_order = (try reverseIfDescending(backend_impl, order, dimension, value_dims, direction)) orelse return null;
            errdefer destroyBuffer(backend_impl, directed_order);
            const sorted_values = (try takeAlongAxis(backend_impl, values, directed_order, dimension, value_dims)) orelse return null;

            try storeOwnedValueHandle(backend_impl, value_handles, value_owned, key_output_id, directed_keys);
            errdefer value_owned[key_output_id.index] = false;
            try storeOwnedValueHandle(backend_impl, value_handles, value_owned, value_output_id, sorted_values);
            destroyBuffer(backend_impl, directed_order);
            releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, instruction.inputs, instruction_index);
            continue;
        }
        if (instruction.kind == .top_k and instruction.inputs.len == 1 and instruction.outputs.len == 2) {
            const input_id = instruction.inputs[0];
            const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
            const input_descriptor = plan.values[input_id.index].descriptor;
            if (input_descriptor.dims.len == 0) return null;
            const axis: i64 = @intCast(input_descriptor.dims.len - 1);
            const k = instruction.top_k_k orelse return null;
            const values_id = instruction.outputs[0];
            const indices_id = instruction.outputs[1];
            if (values_id.index >= plan.values.len or indices_id.index >= plan.values.len) return error.CommandSubmissionFailed;
            const values_descriptor = plan.values[values_id.index].descriptor;
            const indices_descriptor = plan.values[indices_id.index].descriptor;

            const starts = try allocator.alloc(i64, input_descriptor.dims.len);
            defer allocator.free(starts);
            const limits = try allocator.dupe(i64, input_descriptor.dims);
            defer allocator.free(limits);
            const strides = try allocator.alloc(i64, input_descriptor.dims.len);
            defer allocator.free(strides);
            @memset(starts, 0);
            @memset(strides, 1);
            limits[limits.len - 1] = k;

            const sorted_values = (try sort(backend_impl, input, axis, input_descriptor.dims)) orelse return null;
            const descending_values = (try reverseIfDescending(backend_impl, sorted_values, axis, input_descriptor.dims, .gt)) orelse return null;
            errdefer destroyBuffer(backend_impl, descending_values);
            const top_values = (try slice(backend_impl, descending_values, starts, limits, strides, values_descriptor.dims)) orelse return null;
            destroyBuffer(backend_impl, descending_values);

            const sorted_indices = (try argsort(backend_impl, input, axis, indices_descriptor.element_type, input_descriptor.dims)) orelse return null;
            const descending_indices = (try reverseIfDescending(backend_impl, sorted_indices, axis, input_descriptor.dims, .gt)) orelse return null;
            errdefer destroyBuffer(backend_impl, descending_indices);
            const top_indices = (try slice(backend_impl, descending_indices, starts, limits, strides, indices_descriptor.dims)) orelse return null;
            destroyBuffer(backend_impl, descending_indices);

            try storeOwnedValueHandle(backend_impl, value_handles, value_owned, values_id, top_values);
            errdefer value_owned[values_id.index] = false;
            try storeOwnedValueHandle(backend_impl, value_handles, value_owned, indices_id, top_indices);
            releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, instruction.inputs, instruction_index);
            continue;
        }
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
            .iota => blk: {
                break :blk (try iota(
                    backend_impl,
                    executable.device_local_hardware_ids[device_index],
                    output_descriptor.element_type,
                    output_dims,
                    instruction.iota_dimension orelse return null,
                )) orelse return null;
            },
            .convert => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try convert(backend_impl, input, output_descriptor.element_type)) orelse return null;
            },
            .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .atan2, .remainder, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => blk: {
                const op = executableBinaryOp(instruction.kind) orelse return null;
                const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try binary(backend_impl, lhs, rhs, op)) orelse return null;
            },
            .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .abs, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .is_finite, .round_nearest_even => blk: {
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
                const operand = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const indices = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try gather(
                    backend_impl,
                    operand,
                    indices,
                    instruction.start_index_map orelse return null,
                    instruction.collapsed_slice_dims orelse &.{},
                    instruction.operand_batching_dims orelse &.{},
                    instruction.start_indices_batching_dims orelse &.{},
                    instruction.index_vector_dim orelse 0,
                    instruction.slice_sizes orelse return null,
                    instruction.offset_dims orelse &.{},
                    output_dims,
                )) orelse return null;
            },
            .scatter => blk: {
                const operand = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const indices = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                const updates = value_handles[instruction.inputs[2].index] orelse return error.CommandSubmissionFailed;
                if (supportedScatterAxis(instruction)) |scatter_axis| {
                    break :blk (try scatterAxis(
                        backend_impl,
                        operand,
                        indices,
                        updates,
                        scatter_axis,
                        instruction.index_vector_dim orelse 0,
                        instruction.scatter_update_kind orelse .set,
                        output_dims,
                    )) orelse return null;
                }
                break :blk (try scatter(
                    backend_impl,
                    operand,
                    indices,
                    updates,
                    instruction.scatter_dims_to_operand_dims orelse return null,
                    instruction.inserted_window_dims orelse return null,
                    instruction.update_window_dims orelse &.{},
                    instruction.input_batching_dims orelse &.{},
                    instruction.scatter_indices_batching_dims orelse &.{},
                    instruction.index_vector_dim orelse 0,
                    instruction.scatter_update_kind orelse .set,
                    output_dims,
                )) orelse return null;
            },
            .sort => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const sorted = (try sort(backend_impl, input, instruction.dimension orelse return null, output_dims)) orelse return null;
                break :blk (try reverseIfDescending(backend_impl, sorted, instruction.dimension.?, output_dims, instruction.compare_direction orelse .lt)) orelse return null;
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
            .reduce_sum, .reduce_max, .reduce_and, .reduce_or => blk: {
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
            .clamp => blk: {
                const min = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const value = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                const max = value_handles[instruction.inputs[2].index] orelse return error.CommandSubmissionFailed;
                break :blk (try clamp(backend_impl, min, value, max, output_dims)) orelse return null;
            },
            else => return null,
        };

        try storeOwnedValueHandle(backend_impl, value_handles, value_owned, output_id, next);
        releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, instruction.inputs, instruction_index);
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
    executable.program.deinit();
    destroyConstantHandles(backend_impl, executable.constant_handles);
    executable.allocator.free(executable.constant_handles);
    executable.allocator.free(executable.device_local_hardware_ids);
    executable.allocator.destroy(executable);
}

fn releaseDeadInputs(
    backend_impl: backend.Backend,
    program: *const backend.Program,
    value_handles: []?backend.BufferHandle,
    value_owned: []bool,
    input_ids: []const core.ValueId,
    instruction_index: usize,
) void {
    for (input_ids) |input_id| {
        if (input_id.index >= value_handles.len or input_id.index >= program.last_uses.len) continue;
        if (program.output_values[input_id.index]) continue;
        if (program.last_uses[input_id.index] != instruction_index) continue;
        if (!value_owned[input_id.index]) continue;
        if (value_handles[input_id.index]) |old| destroyBuffer(backend_impl, old);
        value_handles[input_id.index] = null;
        value_owned[input_id.index] = false;
    }
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
        .iota,
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
        .atan2,
        .remainder,
        .and_,
        .or_,
        .xor,
        .shift_left,
        .shift_right_arithmetic,
        .shift_right_logical,
        .negate,
        .exp,
        .expm1,
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
        .not_,
        .sign,
        .is_finite,
        .round_nearest_even,
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
        .scatter,
        .sort,
        .top_k,
        .dot_general,
        .reduce_sum,
        .reduce_max,
        .reduce_and,
        .reduce_or,
        .compare,
        .select,
        .clamp,
        => true,
        else => false,
    };
}

fn executableLoweringIssue(plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32) ?LoweringIssue {
    if (device_local_hardware_ids.len == 0) return .{
        .detail = "backend executable requires at least one device",
        .feature = "mlx-device-assignment",
    };
    for (plan.instructions, 0..) |instruction, instruction_index| {
        if (!executableSupportsInstruction(instruction.kind)) return .{
            .instruction_index = instruction_index,
            .op = instruction.kind,
            .detail = "operation is not supported by the MLX backend executable",
        };
        const valid_output_count = instruction.outputs.len == 1 or ((instruction.kind == .sort or instruction.kind == .top_k) and instruction.outputs.len == 2);
        if (!valid_output_count) return .{
            .instruction_index = instruction_index,
            .op = instruction.kind,
            .detail = "MLX executable lowering requires one output per instruction except two-output sort/top_k",
            .feature = "mlx-executable-values",
        };
        for (instruction.outputs) |output_id| {
            if (output_id.index >= plan.values.len) return .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "instruction output value is outside the executable value table",
                .feature = "mlx-executable-values",
            };
        }
        if (instructionLoweringIssue(plan, instruction, instruction_index, instruction.outputs[0])) |issue| return issue;
    }
    return null;
}

fn instructionLoweringIssue(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    const output_descriptor = plan.values[output_id.index].descriptor;
    return switch (instruction.kind) {
        .constant => if (instruction.literal == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "constant lowering requires an embedded literal",
            .feature = "mlx-constant-literal",
        } else null,
        .iota => blk: {
            const dim = instruction.iota_dimension orelse break :blk .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "iota lowering requires an iota dimension",
                .feature = "mlx-iota",
            };
            if (dim < 0 or dim >= @as(i64, @intCast(output_descriptor.dims.len))) break :blk .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "iota dimension is outside the output rank",
                .feature = "mlx-iota",
            };
            break :blk null;
        },
        .atan2, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => validateBinaryElementwiseLowering(plan, instruction, instruction_index, output_id),
        .expm1, .not_, .is_finite, .round_nearest_even => validateUnaryElementwiseLowering(plan, instruction, instruction_index, output_id),
        .transpose => if (instruction.permutation == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "transpose lowering requires a permutation",
            .feature = "mlx-layout",
        } else null,
        .broadcast_in_dim => if (instruction.broadcast_dimensions == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "broadcast_in_dim lowering requires broadcast dimensions",
            .feature = "mlx-shape",
        } else null,
        .slice => if (instruction.start_indices == null or instruction.limit_indices == null or instruction.strides == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "slice lowering requires static starts, limits, and strides",
            .feature = "mlx-slice",
        } else null,
        .dynamic_slice => if (instruction.slice_sizes == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "dynamic_slice lowering requires static slice sizes",
            .feature = "mlx-dynamic-slice",
        } else null,
        .dynamic_update_slice => if (instruction.inputs.len < 2) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "dynamic_update_slice lowering requires operand and update inputs",
            .feature = "mlx-dynamic-update-slice",
        } else null,
        .pad => validatePadLowering(plan, instruction, instruction_index, output_id),
        .concatenate => if (instruction.dimension == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "concatenate lowering requires a dimension",
            .feature = "mlx-concatenate",
        } else null,
        .gather => validateGatherLowering(plan, instruction, instruction_index, output_id),
        .scatter => validateScatterLowering(plan, instruction, instruction_index, output_id),
        .sort => validateSortLowering(instruction, instruction_index, output_id),
        .top_k => validateTopKLowering(plan, instruction, instruction_index, output_id),
        .dot_general => validateDotGeneralLowering(plan, instruction, instruction_index, output_id),
        .reduce_sum, .reduce_max, .reduce_and, .reduce_or => validateReduceLowering(plan, instruction, instruction_index, output_id),
        .compare => validateCompareLowering(plan, instruction, instruction_index, output_id),
        .select => validateSelectLowering(plan, instruction, instruction_index, output_id),
        .clamp => validateClampLowering(plan, instruction, instruction_index, output_id),
        else => null,
    };
}

fn validateBinaryElementwiseLowering(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "binary operand lhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "binary operand rhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or lhs.element_type != output.element_type or !dimsEqual(lhs.dims, rhs.dims) or !dimsEqual(lhs.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "binary lowering requires matching input/output dtypes and shapes",
        .feature = "mlx-elementwise-binary",
    };
    switch (instruction.kind) {
        .atan2 => if (!isSupportedFloat(lhs.element_type)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "atan2 lowering requires an MLX-supported floating dtype",
            .feature = "mlx-atan2",
        },
        .and_, .or_, .xor => if (!isSupportedInteger(lhs.element_type) and lhs.element_type != .pred) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "logical/bitwise lowering requires pred or MLX-supported integer dtype",
            .feature = "mlx-bitwise",
        },
        .shift_left, .shift_right_arithmetic, .shift_right_logical => if (!isSupportedInteger(lhs.element_type)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "shift lowering requires an MLX-supported integer dtype",
            .feature = "mlx-shift",
        },
        else => {},
    }
    return null;
}

fn validateUnaryElementwiseLowering(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "unary operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (!dimsEqual(input.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "unary lowering requires matching input/output shapes",
        .feature = "mlx-elementwise-unary",
    };
    switch (instruction.kind) {
        .expm1, .round_nearest_even => if (!isSupportedFloat(input.element_type) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "floating unary lowering requires matching MLX-supported floating dtype",
            .feature = "mlx-unary-float",
        },
        .is_finite => if (!isSupportedFloat(input.element_type) or output.element_type != .pred) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "is_finite lowering requires floating input and pred output",
            .feature = "mlx-is-finite",
        },
        .not_ => if ((!isSupportedInteger(input.element_type) and input.element_type != .pred) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "not lowering requires pred or MLX-supported integer dtype",
            .feature = "mlx-not",
        },
        else => {},
    }
    return null;
}

fn validatePadLowering(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    const low = instruction.edge_padding_low orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires low edge padding",
        .feature = "mlx-pad",
    };
    const high = instruction.edge_padding_high orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires high edge padding",
        .feature = "mlx-pad",
    };
    const interior = instruction.interior_padding orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires interior padding metadata",
        .feature = "mlx-pad",
    };
    if (instruction.inputs.len < 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires operand and scalar padding value inputs",
        .feature = "mlx-pad",
    };
    const input_descriptor = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "pad operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const padding_descriptor = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "pad scalar value is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output_descriptor = plan.values[output_id.index].descriptor;
    if (padding_descriptor.element_type != input_descriptor.element_type or padding_descriptor.dims.len != 0) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "pad lowering requires a scalar padding value with the operand dtype",
        .feature = "mlx-pad-scalar",
    };
    if (low.len != input_descriptor.dims.len or high.len != input_descriptor.dims.len or interior.len != input_descriptor.dims.len or output_descriptor.dims.len != input_descriptor.dims.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad metadata rank does not match operand/output rank",
        .feature = "mlx-pad",
    };
    for (input_descriptor.dims, 0..) |dim, axis| {
        if (low[axis] < 0 or high[axis] < 0 or interior[axis] < 0) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "pad lowering requires non-negative edge and interior padding",
            .feature = "mlx-pad",
        };
        const interior_slots = if (dim > 0) (dim - 1) * interior[axis] else 0;
        if (output_descriptor.dims[axis] != dim + low[axis] + high[axis] + interior_slots) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "pad output shape must equal StableHLO edge plus interior padding formula",
            .feature = "mlx-pad",
        };
    }
    return null;
}

fn validateGatherLowering(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    const operand = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "gather operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const indices = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "gather indices are outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (operand.element_type != output.element_type or !isSupportedInteger(indices.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather lowering requires matching operand/output dtypes and integer indices",
        .feature = "mlx-gather-types",
    };
    const slice_sizes = instruction.slice_sizes orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather lowering requires static slice sizes",
        .feature = "mlx-gather-slice-sizes",
    };
    if (slice_sizes.len != operand.dims.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather slice size rank must match operand rank",
        .feature = "mlx-gather-slice-sizes",
    };
    if (!validGatherShape(
        operand.dims,
        indices.dims,
        instruction.start_index_map orelse &.{},
        instruction.collapsed_slice_dims orelse &.{},
        instruction.operand_batching_dims orelse &.{},
        instruction.start_indices_batching_dims orelse &.{},
        instruction.index_vector_dim orelse 0,
        slice_sizes,
        instruction.offset_dims orelse &.{},
        output.dims,
    )) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather metadata and output shape must match MLX general gather semantics",
        .feature = "mlx-gather-general-shape",
    };
    return null;
}

fn validateScatterLowering(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    const operand = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "scatter operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const indices = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "scatter indices are outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const updates = inputDescriptor(plan, instruction, 2) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 2) instruction.inputs[2] else output_id,
        .op = instruction.kind,
        .detail = "scatter updates are outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (operand.element_type != updates.element_type or operand.element_type != output.element_type or !dimsEqual(operand.dims, output.dims) or !isSupportedInteger(indices.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "scatter lowering requires matching operand/update/output dtypes and integer indices",
        .feature = "mlx-scatter-types",
    };
    if (instruction.scatter_update_kind == null) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "scatter combiner must be set or add",
        .feature = "mlx-scatter-combiner",
    };
    if (supportedScatterAxis(instruction)) |axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(operand.dims.len))) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "scatter axis is outside the operand rank",
            .feature = "mlx-scatter-axis",
        };
        if (!scatterUpdateShapeMatchesAxis(operand.dims, indices.dims, updates.dims, instruction.index_vector_dim orelse 0, axis)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "scatter update shape must match MLX scatter semantics for the scattered axis",
            .feature = "mlx-scatter-update-shape",
        };
        return null;
    }
    if (!validScatterShape(
        operand.dims,
        indices.dims,
        updates.dims,
        instruction.scatter_dims_to_operand_dims orelse &.{},
        instruction.inserted_window_dims orelse &.{},
        instruction.update_window_dims orelse &.{},
        instruction.input_batching_dims orelse &.{},
        instruction.scatter_indices_batching_dims orelse &.{},
        instruction.index_vector_dim orelse 0,
        output.dims,
    )) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "scatter metadata and update shape must match MLX scatter semantics",
        .feature = "mlx-scatter-shape",
    };
    return null;
}

fn validateSortLowering(instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    if (instruction.dimension == null) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "sort lowering requires a sort dimension",
        .feature = "mlx-sort",
    };
    if (!(instruction.inputs.len == 1 and instruction.outputs.len == 1) and !(instruction.inputs.len == 2 and instruction.outputs.len == 2)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "sort lowering supports single-value sort or key/value sort with two operands and two outputs",
        .feature = "mlx-sort-arity",
    };
    switch (instruction.compare_direction orelse .lt) {
        .lt, .le, .gt, .ge => return null,
        else => return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "sort lowering supports lt/le/gt/ge comparator directions only",
            .feature = "mlx-sort-comparator",
        },
    }
}

fn validateTopKLowering(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "top_k lowering requires one input and values/indices outputs",
        .feature = "mlx-top-k-arity",
    };
    const k = instruction.top_k_k orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "top_k lowering requires static k",
        .feature = "mlx-top-k",
    };
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "top_k input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    if (input.dims.len == 0 or k < 0 or k > input.dims[input.dims.len - 1]) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "top_k supports static k along the last non-scalar dimension",
        .feature = "mlx-top-k-shape",
    };
    return null;
}

fn validateDotGeneralLowering(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "dot_general lhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "dot_general rhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != .f32 or rhs.element_type != .f32 or output.element_type != .f32) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "dot_general lowering currently supports f32 matmul-like tensors only",
        .feature = "mlx-dot-general-f32",
    };
    if (!dotGeneralIsMatmulLike(
        lhs.dims,
        rhs.dims,
        instruction.lhs_batch_dimensions orelse &.{},
        instruction.rhs_batch_dimensions orelse &.{},
        instruction.lhs_contracting_dimensions orelse &.{},
        instruction.rhs_contracting_dimensions orelse &.{},
        output.dims,
    )) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "dot_general lowering currently supports matmul-like contracting dimensions only",
        .feature = "mlx-dot-general-matmul",
    };
    return null;
}

fn validateReduceLowering(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "reduce operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    const supported_reduce_type = switch (instruction.kind) {
        .reduce_sum, .reduce_max => input.element_type == .f32 and output.element_type == .f32,
        .reduce_and, .reduce_or => input.element_type == .pred and output.element_type == .pred,
        else => false,
    };
    if (!supported_reduce_type) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce lowering supports f32 sum/max and pred and/or only",
        .feature = "mlx-reduce-types",
    };
    if (!validReduceShape(input.dims, instruction.reduce_dimensions orelse &.{}, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce output shape must equal input shape with reduced axes removed",
        .feature = "mlx-reduce-shape",
    };
    return null;
}

fn validateCompareLowering(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "compare lhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "compare rhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or !isSupportedComparable(lhs.element_type) or output.element_type != .pred or !dimsEqual(lhs.dims, rhs.dims) or !dimsEqual(lhs.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "compare lowering requires same-shape MLX-supported numeric inputs and pred outputs",
        .feature = "mlx-compare",
    };
    return null;
}

fn validateSelectLowering(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    const pred = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "select predicate is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const on_true = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "select true operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const on_false = inputDescriptor(plan, instruction, 2) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 2) instruction.inputs[2] else output_id,
        .op = instruction.kind,
        .detail = "select false operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (pred.element_type != .pred or on_true.element_type != on_false.element_type or !dimsEqual(pred.dims, on_true.dims) or !dimsEqual(on_true.dims, on_false.dims) or !dimsEqual(on_true.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "select lowering requires same-shape pred/true/false tensors",
        .feature = "mlx-select",
    };
    return null;
}

fn validateClampLowering(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, instruction_index: usize, output_id: core.ValueId) ?LoweringIssue {
    const min = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "clamp min operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const value = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "clamp value operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const max = inputDescriptor(plan, instruction, 2) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 2) instruction.inputs[2] else output_id,
        .op = instruction.kind,
        .detail = "clamp max operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (min.element_type != value.element_type or max.element_type != value.element_type or output.element_type != value.element_type or !dimsEqual(value.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "clamp lowering requires matching min/value/max/output dtypes and value/output shapes",
        .feature = "mlx-clamp",
    };
    if (min.dims.len != 0 and !dimsEqual(min.dims, value.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "clamp min must be scalar or match the value shape",
        .feature = "mlx-clamp-bounds",
    };
    if (max.dims.len != 0 and !dimsEqual(max.dims, value.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[2],
        .op = instruction.kind,
        .detail = "clamp max must be scalar or match the value shape",
        .feature = "mlx-clamp-bounds",
    };
    return null;
}

fn inputDescriptor(plan: *const core.ExecutablePlan, instruction: core.PlanInstruction, input_index: usize) ?core.BufferDescriptor {
    if (input_index >= instruction.inputs.len) return null;
    const id = instruction.inputs[input_index];
    if (id.index >= plan.values.len) return null;
    return plan.values[id.index].descriptor;
}

fn dimsEqual(lhs: []const i64, rhs: []const i64) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |a, b| if (a != b) return false;
    return true;
}

fn validReduceShape(input_dims: []const i64, dimensions: []const i64, output_dims: []const i64) bool {
    var reduced = [_]bool{false} ** 64;
    if (input_dims.len > reduced.len) return false;
    for (dimensions) |dim_i64| {
        if (dim_i64 < 0 or dim_i64 >= @as(i64, @intCast(input_dims.len))) return false;
        const dim: usize = @intCast(dim_i64);
        if (reduced[dim]) return false;
        reduced[dim] = true;
    }
    var expected_rank: usize = 0;
    for (0..input_dims.len) |axis| {
        if (!reduced[axis]) expected_rank += 1;
    }
    if (output_dims.len != expected_rank) return false;
    var out_axis: usize = 0;
    for (input_dims, 0..) |dim, axis| {
        if (reduced[axis]) continue;
        if (output_dims[out_axis] != dim) return false;
        out_axis += 1;
    }
    return true;
}

fn isSupportedFloat(element_type: core.BufferType) bool {
    return switch (element_type) {
        .f16, .f32, .bf16 => true,
        else => false,
    };
}

fn isSupportedInteger(element_type: core.BufferType) bool {
    return switch (element_type) {
        .s8, .s32, .u8, .u32 => true,
        else => false,
    };
}

fn isSupportedComparable(element_type: core.BufferType) bool {
    return isSupportedFloat(element_type) or isSupportedInteger(element_type);
}

fn dotGeneralIsMatmulLike(lhs_dims: []const i64, rhs_dims: []const i64, lhs_batch: []const i64, rhs_batch: []const i64, lhs_contract: []const i64, rhs_contract: []const i64, output_dims: []const i64) bool {
    if (lhs_contract.len != 1 or rhs_contract.len != 1 or lhs_batch.len != rhs_batch.len or lhs_dims.len == 0 or rhs_dims.len < 2 or output_dims.len == 0) return false;
    const lhs_k = lhs_contract[0];
    const rhs_k = rhs_contract[0];
    if (lhs_k < 0 or rhs_k < 0) return false;
    if (lhs_k != @as(i64, @intCast(lhs_dims.len - 1)) or rhs_k != @as(i64, @intCast(rhs_dims.len - 2))) return false;
    if (lhs_dims[@intCast(lhs_k)] != rhs_dims[@intCast(rhs_k)]) return false;
    for (lhs_batch, rhs_batch) |lhs_axis, rhs_axis| {
        if (lhs_axis < 0 or rhs_axis < 0) return false;
        if (@as(usize, @intCast(lhs_axis)) >= lhs_dims.len or @as(usize, @intCast(rhs_axis)) >= rhs_dims.len) return false;
        if (lhs_dims[@intCast(lhs_axis)] != rhs_dims[@intCast(rhs_axis)]) return false;
    }
    return true;
}

fn writeLoweringIssue(plan: *const core.ExecutablePlan, issue: LoweringIssue, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll("invalid executable plan: pass=mlx-backend-legalization");
    if (issue.instruction_index) |index| try writer.print(" instruction={d}", .{index});
    if (issue.op) |op| try writer.print(" op={s}", .{@tagName(op)});
    if (issue.value_id) |value_id| {
        try writer.print(" value={d}", .{value_id.index});
        if (value_id.index < plan.values.len) {
            const descriptor = plan.values[value_id.index].descriptor;
            try writer.print(" dtype={s} rank={d} shape=", .{ @tagName(descriptor.element_type), descriptor.dims.len });
            try writeDims(writer, descriptor.dims);
            try writer.print(" sharding={s}", .{shardingLabel(plan, value_id)});
        }
    }
    try writer.print(" detail=\"{s}\" feature={s}", .{ issue.detail, issue.feature });
}

fn writeDims(writer: *std.Io.Writer, dims: []const i64) std.Io.Writer.Error!void {
    try writer.writeAll("[");
    for (dims, 0..) |dim, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{dim});
    }
    try writer.writeAll("]");
}

fn shardingLabel(plan: *const core.ExecutablePlan, value_id: core.ValueId) []const u8 {
    for (plan.output_ids, 0..) |output_id, index| {
        if (output_id.index == value_id.index and index < plan.output_shardings.len) return @tagName(plan.output_shardings[index].kind);
    }
    if (value_id.index < plan.values.len and plan.values[value_id.index].role == .parameter) {
        var parameter_index: usize = 0;
        for (plan.values) |value| {
            if (value.role != .parameter) continue;
            if (value.id.index == value_id.index and parameter_index < plan.parameter_shardings.len) return @tagName(plan.parameter_shardings[parameter_index].kind);
            parameter_index += 1;
        }
    }
    return "internal";
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

fn storeOwnedValueHandle(
    backend_impl: backend.Backend,
    value_handles: []?backend.BufferHandle,
    value_owned: []bool,
    value_id: core.ValueId,
    handle: backend.BufferHandle,
) backend.Error!void {
    if (value_id.index >= value_handles.len) return error.CommandSubmissionFailed;
    if (value_owned[value_id.index]) {
        if (value_handles[value_id.index]) |old| destroyBuffer(backend_impl, old);
    }
    value_handles[value_id.index] = handle;
    value_owned[value_id.index] = true;
}

fn reverseIfDescending(
    backend_impl: backend.Backend,
    handle: backend.BufferHandle,
    dimension: i64,
    output_dims: []const i64,
    direction: core.CompareOp,
) backend.Error!?backend.BufferHandle {
    return switch (direction) {
        .lt, .le => handle,
        .gt, .ge => blk: {
            const dimensions = [_]i64{dimension};
            const reversed = reverse(backend_impl, handle, &dimensions, output_dims) catch |err| {
                destroyBuffer(backend_impl, handle);
                return err;
            };
            destroyBuffer(backend_impl, handle);
            break :blk reversed;
        },
        else => blk: {
            destroyBuffer(backend_impl, handle);
            break :blk null;
        },
    };
}

fn supportedGatherAxis(instruction: core.PlanInstruction) ?i64 {
    const start_index_map = instruction.start_index_map orelse return null;
    const slice_sizes = instruction.slice_sizes orelse return null;
    const collapsed_slice_dims = instruction.collapsed_slice_dims orelse return null;
    if (start_index_map.len != 1 or collapsed_slice_dims.len != 1) return null;
    const axis = start_index_map[0];
    if (axis < 0 or collapsed_slice_dims[0] != axis) return null;
    if (slice_sizes.len == 0 or axis >= @as(i64, @intCast(slice_sizes.len)) or slice_sizes[@intCast(axis)] != 1) return null;
    return axis;
}

fn gatherHasExplicitIndexVector(indices_dims: []const i64, start_axis_count: usize, index_vector_dim: i64) bool {
    if (index_vector_dim < 0 or index_vector_dim >= @as(i64, @intCast(indices_dims.len))) return false;
    return indices_dims[@intCast(index_vector_dim)] == @as(i64, @intCast(start_axis_count));
}

fn markUniqueAxis(seen: []bool, axis: i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(seen.len))) return false;
    const index: usize = @intCast(axis);
    if (seen[index]) return false;
    seen[index] = true;
    return true;
}

fn validGatherShape(operand_dims: []const i64, indices_dims: []const i64, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) bool {
    if (operand_dims.len > 64 or output_dims.len > 64 or start_index_map.len == 0 or slice_sizes.len != operand_dims.len) return false;
    if (start_index_map.len > 1 and !gatherHasExplicitIndexVector(indices_dims, start_index_map.len, index_vector_dim)) return false;
    if (operand_batching_dims.len != start_indices_batching_dims.len) return false;

    var gathered = [_]bool{false} ** 64;
    for (start_index_map) |axis| {
        if (!markUniqueAxis(gathered[0..operand_dims.len], axis)) return false;
    }
    for (operand_batching_dims, start_indices_batching_dims) |operand_axis, indices_axis| {
        if (!markUniqueAxis(gathered[0..operand_dims.len], operand_axis)) return false;
        if (indices_axis < 0 or indices_axis >= @as(i64, @intCast(indices_dims.len)) or indices_axis == index_vector_dim) return false;
        if (operand_dims[@intCast(operand_axis)] != indices_dims[@intCast(indices_axis)]) return false;
        if (slice_sizes[@intCast(operand_axis)] != 1) return false;
    }

    var collapsed = [_]bool{false} ** 64;
    for (collapsed_slice_dims) |axis| {
        if (!markUniqueAxis(collapsed[0..operand_dims.len], axis)) return false;
        if (slice_sizes[@intCast(axis)] != 1) return false;
    }
    for (operand_batching_dims) |axis| {
        if (!markUniqueAxis(collapsed[0..operand_dims.len], axis)) return false;
    }

    var non_collapsed_slice_rank: usize = 0;
    for (operand_dims, 0..) |dim, axis| {
        if (dim < 0 or slice_sizes[axis] < 0 or slice_sizes[axis] > dim) return false;
        if (!collapsed[axis]) non_collapsed_slice_rank += 1;
    }
    if (offset_dims.len != non_collapsed_slice_rank) return false;

    const explicit_vector = gatherHasExplicitIndexVector(indices_dims, start_index_map.len, index_vector_dim);
    const index_prefix_rank = indices_dims.len - @as(usize, if (explicit_vector) 1 else 0);
    if (output_dims.len != index_prefix_rank + non_collapsed_slice_rank) return false;

    var output_is_offset = [_]bool{false} ** 64;
    for (offset_dims) |axis| {
        if (!markUniqueAxis(output_is_offset[0..output_dims.len], axis)) return false;
    }

    var index_axis: usize = 0;
    var slice_axis: usize = 0;
    for (output_dims, 0..) |output_dim, output_axis| {
        if (output_is_offset[output_axis]) {
            while (slice_axis < operand_dims.len and collapsed[slice_axis]) slice_axis += 1;
            if (slice_axis >= slice_sizes.len or output_dim != slice_sizes[slice_axis]) return false;
            slice_axis += 1;
        } else {
            while (explicit_vector and index_axis == @as(usize, @intCast(index_vector_dim))) index_axis += 1;
            if (index_axis >= indices_dims.len or output_dim != indices_dims[index_axis]) return false;
            index_axis += 1;
        }
    }
    return true;
}

fn supportedScatterAxis(instruction: core.PlanInstruction) ?i64 {
    const scatter_dims_to_operand_dims = instruction.scatter_dims_to_operand_dims orelse return null;
    const inserted_window_dims = instruction.inserted_window_dims orelse return null;
    const input_batching_dims = instruction.input_batching_dims orelse &.{};
    const scatter_indices_batching_dims = instruction.scatter_indices_batching_dims orelse &.{};
    if (scatter_dims_to_operand_dims.len != 1 or inserted_window_dims.len != 1) return null;
    if (input_batching_dims.len != 0 or scatter_indices_batching_dims.len != 0) return null;
    const axis = scatter_dims_to_operand_dims[0];
    if (axis < 0 or inserted_window_dims[0] != axis) return null;
    return axis;
}

fn validScatterShape(operand_dims: []const i64, indices_dims: []const i64, update_dims: []const i64, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, output_dims: []const i64) bool {
    if (operand_dims.len > 64 or !dimsEqual(operand_dims, output_dims)) return false;
    if (scatter_dims_to_operand_dims.len == 0 or inserted_window_dims.len + update_window_dims.len + input_batching_dims.len != operand_dims.len or input_batching_dims.len != scatter_indices_batching_dims.len) return false;
    const explicit_vector = gatherHasExplicitIndexVector(indices_dims, scatter_dims_to_operand_dims.len, index_vector_dim);
    if (scatter_dims_to_operand_dims.len > 1 and !explicit_vector) return false;

    var scatter_axes = [_]bool{false} ** 64;
    for (scatter_dims_to_operand_dims) |axis| {
        if (!markUniqueAxis(scatter_axes[0..operand_dims.len], axis)) return false;
    }
    for (input_batching_dims, scatter_indices_batching_dims) |operand_axis, indices_axis| {
        if (!markUniqueAxis(scatter_axes[0..operand_dims.len], operand_axis)) return false;
        if (indices_axis < 0 or indices_axis >= @as(i64, @intCast(indices_dims.len)) or indices_axis == index_vector_dim) return false;
        if (operand_dims[@intCast(operand_axis)] != indices_dims[@intCast(indices_axis)]) return false;
    }

    var window_axes = [_]bool{false} ** 64;
    for (inserted_window_dims) |axis| {
        if (!markUniqueAxis(window_axes[0..operand_dims.len], axis)) return false;
    }
    for (update_window_dims) |axis| {
        if (!markUniqueAxis(window_axes[0..operand_dims.len], axis)) return false;
    }
    for (input_batching_dims) |axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len)) or window_axes[@intCast(axis)]) return false;
    }

    const index_prefix_rank = indices_dims.len - @as(usize, if (explicit_vector) 1 else 0);
    if (update_dims.len != index_prefix_rank + update_window_dims.len) return false;
    var update_axis: usize = 0;
    for (indices_dims, 0..) |dim, axis| {
        if (explicit_vector and axis == @as(usize, @intCast(index_vector_dim))) continue;
        if (update_axis >= update_dims.len or update_dims[update_axis] != dim) return false;
        update_axis += 1;
    }
    if (update_axis != index_prefix_rank) return false;
    for (update_window_dims, 0..) |operand_axis, window_axis| {
        if (operand_axis < 0 or operand_axis >= @as(i64, @intCast(operand_dims.len))) return false;
        const dim = update_dims[index_prefix_rank + window_axis];
        if (dim < 0 or dim > operand_dims[@intCast(operand_axis)]) return false;
    }
    return true;
}

fn scatterUpdateShapeMatchesAxis(operand_dims: []const i64, indices_dims: []const i64, update_dims: []const i64, index_vector_dim: i64, axis: i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len))) return false;
    var expected_rank: usize = operand_dims.len - 1;
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        expected_rank += 1;
    }
    if (update_dims.len != expected_rank) return false;

    var update_index: usize = 0;
    for (operand_dims[0..@intCast(axis)]) |dim| {
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    const axis_index: usize = @intCast(axis);
    for (operand_dims[axis_index + 1 ..]) |dim| {
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    return update_index == update_dims.len;
}

fn gatherOutputShapeMatchesTake(operand_dims: []const i64, indices_dims: []const i64, index_vector_dim: i64, axis: i64, output_dims: []const i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len))) return false;
    var expected_rank: usize = operand_dims.len - 1;
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        expected_rank += 1;
    }
    if (output_dims.len != expected_rank) return false;

    var out_index: usize = 0;
    for (operand_dims[0..@intCast(axis)]) |dim| {
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    const axis_index: usize = @intCast(axis);
    for (operand_dims[axis_index + 1 ..]) |dim| {
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    return out_index == output_dims.len;
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
        .atan2 => .atan2,
        .remainder => .remainder,
        .and_ => .and_,
        .or_ => .or_,
        .xor => .xor,
        .shift_left => .shift_left,
        .shift_right_arithmetic, .shift_right_logical => .shift_right_logical,
        else => null,
    };
}

fn executableUnaryOp(instruction_kind: core.PlanInstructionKind) ?core.ElementwiseUnaryOp {
    return switch (instruction_kind) {
        .negate => .negate,
        .exp => .exp,
        .expm1 => .expm1,
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
        .not_ => .not_,
        .sign => .sign,
        .is_finite => .is_finite,
        .round_nearest_even => .round_nearest_even,
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
        .atan2 => c.PJRTX_MLX_METAL_BINARY_ATAN2,
        .remainder => c.PJRTX_MLX_METAL_BINARY_REMAINDER,
        .and_ => c.PJRTX_MLX_METAL_BINARY_AND,
        .or_ => c.PJRTX_MLX_METAL_BINARY_OR,
        .xor => c.PJRTX_MLX_METAL_BINARY_XOR,
        .shift_left => c.PJRTX_MLX_METAL_BINARY_SHIFT_LEFT,
        .shift_right_arithmetic, .shift_right_logical => c.PJRTX_MLX_METAL_BINARY_SHIFT_RIGHT,
    };
}

fn mlxUnaryOpCode(op: core.ElementwiseUnaryOp) ?c_int {
    return switch (op) {
        .negate => c.PJRTX_MLX_METAL_U8_UNARY_NEGATE,
        .exp => c.PJRTX_MLX_METAL_UNARY_EXP,
        .expm1 => c.PJRTX_MLX_METAL_UNARY_EXPM1,
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
        .not_ => c.PJRTX_MLX_METAL_UNARY_NOT,
        .sign => c.PJRTX_MLX_METAL_UNARY_SIGN,
        .is_finite => c.PJRTX_MLX_METAL_UNARY_ISFINITE,
        .round_nearest_even => c.PJRTX_MLX_METAL_UNARY_ROUND,
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

fn mlxScatterUpdateCode(update_kind: core.ScatterUpdateKind) c_int {
    return switch (update_kind) {
        .set => c.PJRTX_MLX_METAL_SCATTER_SET,
        .add => c.PJRTX_MLX_METAL_SCATTER_ADD,
    };
}

const vtable: backend.Backend.VTable = .{
    .kind = kind,
    .capabilities = capabilities,
    .enumerateDevices = enumerateDevices,
    .releaseDeviceDescriptors = releaseDeviceDescriptors,
    .bufferFromHost = bufferFromHost,
    .iota = iota,
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
    .gather = gather,
    .gatherAxis = gatherAxis,
    .scatter = scatter,
    .scatterAxis = scatterAxis,
    .sort = sort,
    .argsort = argsort,
    .takeAlongAxis = takeAlongAxis,
    .dotGeneral = dotGeneral,
    .reduce = reduce,
    .compare = compare,
    .select = select,
    .clamp = clamp,
    .compileExecutable = compileExecutable,
    .writeExecutableLoweringDiagnostic = writeExecutableLoweringDiagnostic,
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
    const compiled: *CompiledExecutable = @ptrCast(@alignCast(executable));
    try std.testing.expectEqual(@as(usize, 2), compiled.program.nodes.len);
    try std.testing.expectEqual(backend.ProgramNodeKind.constant, compiled.program.nodes[0].kind);
    try std.testing.expectEqual(backend.ProgramNodeKind.elementwise, compiled.program.nodes[1].kind);
    try std.testing.expect(compiled.program.nodes[0].materializes);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.last_uses[1]);
    try std.testing.expect(compiled.program.output_values[2]);

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

test "mlx metal backend executable materializes iota on device" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{ 2, 3 };
    const assignment = [_]i32{0};

    const values = try allocator.alloc(core.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    const parameter_shardings = try allocator.alloc(core.ShardingPlan, 0);
    const output_shardings = try allocator.alloc(core.ShardingPlan, 1);
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
        .output_ids = try allocator.dupe(core.ValueId, &.{.{ .index = 0 }}),
        .instructions = try allocator.dupe(core.PlanInstruction, &.{.{
            .kind = .iota,
            .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 0 }}),
            .dims = try allocator.dupe(i64, &dims),
            .iota_dimension = 1,
        }}),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);

    const outputs = (try b.executeExecutable(allocator, executable, 0, &.{})) orelse return error.TestUnexpectedResult;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| b.destroyBuffer(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(core.BufferType.f32, outputs[0].element_type);
    try std.testing.expectEqualSlices(i64, &dims, outputs[0].dims);
    var actual: [6]f32 = undefined;
    try b.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 2.0, 0.0, 1.0, 2.0 }, &actual);
}

test "mlx metal backend executable lowers clamp with scalar bounds" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{3};
    const assignment = [_]i32{0};
    const min_literal_value: f32 = -1.0;
    const max_literal_value: f32 = 2.0;

    const values = try allocator.alloc(core.Value, 4);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .constant,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &.{}),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .constant,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &.{}),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[3] = .{
        .id = .{ .index = 3 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

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
        .output_ids = try allocator.dupe(core.ValueId, &.{.{ .index = 3 }}),
        .instructions = try allocator.dupe(core.PlanInstruction, &.{
            .{
                .kind = .constant,
                .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 0 }}),
                .literal = try allocator.dupe(u8, std.mem.asBytes(&min_literal_value)),
            },
            .{
                .kind = .constant,
                .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }}),
                .literal = try allocator.dupe(u8, std.mem.asBytes(&max_literal_value)),
            },
            .{
                .kind = .clamp,
                .inputs = try allocator.dupe(core.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 }, .{ .index = 2 } }),
                .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 3 }}),
                .dims = try allocator.dupe(i64, &dims),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);

    const input = [_]f32{ -2.0, 0.5, 3.0 };
    const input_buffer = (try b.bufferFromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&input))) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(input_buffer);

    const outputs = (try b.executeExecutable(allocator, executable, 0, &.{input_buffer})) orelse return error.TestUnexpectedResult;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| b.destroyBuffer(output.handle);
    }

    var actual: [3]f32 = undefined;
    try b.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 0.5, 2.0 }, &actual);
}

test "mlx metal backend executable rejects unsupported gather form during lowering" {
    const b = create();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const operand_dims = [_]i64{ 4, 2 };
    const index_dims = [_]i64{2};
    const output_dims = [_]i64{ 2, 2 };

    const values = try allocator.alloc(core.Value, 3);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &operand_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .s32,
            .dims = try allocator.dupe(i64, &index_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &output_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var parameter_shardings = try allocator.alloc(core.ShardingPlan, 2);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    parameter_shardings[1] = .{
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
        .instructions = try allocator.dupe(core.PlanInstruction, &.{.{
            .kind = .gather,
            .inputs = try allocator.dupe(core.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }}),
            .dims = try allocator.dupe(i64, &output_dims),
            .start_index_map = try allocator.dupe(i64, &.{ 0, 1 }),
            .collapsed_slice_dims = try allocator.dupe(i64, &.{1}),
            .slice_sizes = try allocator.dupe(i64, &.{ 4, 1 }),
            .index_vector_dim = 1,
        }}),
    };
    defer plan.deinit();

    const maybe_executable = try b.compileExecutable(allocator, &plan, &assignment);
    if (maybe_executable) |executable| b.destroyExecutable(executable);
    try std.testing.expect(maybe_executable == null);
    var diagnostics = std.Io.Writer.Allocating.init(allocator);
    defer diagnostics.deinit();
    try b.writeExecutableLoweringDiagnostic(&plan, &assignment, &diagnostics.writer);
    const message = diagnostics.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, message, "op=gather") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "feature=mlx-gather-general-shape") != null);
}
