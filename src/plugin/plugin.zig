const std = @import("std");

const backend_api = @import("src/backend");
const backend_registry = @import("src/backend/registry");
const compiler = @import("src/compiler");
const core = @import("src/core");
const runtime = @import("src/runtime");
const c = @import("c");

const allocator = std.heap.c_allocator;

const platform_name = "pjrtx";
const platform_version = "PjRTx Metal/MLX";
const device_kind = "Metal";
const plugin_name = "PjRTx";
const stablehlo_version = [_]i64{ 1, 0, 0 };
const backend_option = "pjrtx_backend";
const default_memory_kind = "device";

const MachTimebaseInfo = extern struct {
    numer: u32,
    denom: u32,
};

extern "c" fn mach_absolute_time() u64;
extern "c" fn mach_timebase_info(info: *MachTimebaseInfo) c_int;

const PjrtxError = struct {
    base: c.PJRT_Error,
    code: c.PJRT_Error_Code,
    message: []u8,
};

const DeviceAttributes = struct {
    attrs: [5]c.PJRT_NamedValue,
};

const SerializedTopology = struct {
    bytes: []u8,
};

const PJRT_Gpu_Register_Custom_Call_Args = extern struct {
    struct_size: usize,
    function_name: [*c]const u8,
    function_name_size: usize,
    api_version: c_int,
    handler_instantiate: ?*anyopaque,
    handler_prepare: ?*anyopaque,
    handler_initialize: ?*anyopaque,
    handler_execute: ?*anyopaque,
};

const PJRT_Gpu_Custom_Call = extern struct {
    base: c.PJRT_Extension_Base,
    custom_call: *const fn (args: [*c]PJRT_Gpu_Register_Custom_Call_Args) callconv(.c) ?*c.PJRT_Error,
};

const Executable = struct {
    client: *runtime.Client,
    plan: *compiler.ExecutablePlan,
    graph: runtime.ExecutableGraph,
    logical_ids: []c.PJRT_LogicalDeviceIds,
    optimized_program: []u8,
    parameter_memory_kinds: [][*c]const u8,
    parameter_memory_kind_sizes: []usize,
    output_memory_kinds: [][*c]const u8,
    output_memory_kind_sizes: []usize,
    fingerprint: []u8,
    name: []const u8 = "pjrtx_executable",
    deleted: bool = false,
    graph_released: bool = false,

    fn deinit(self: *Executable) void {
        self.releaseGraph();
        allocator.free(self.fingerprint);
        allocator.free(self.output_memory_kind_sizes);
        allocator.free(self.output_memory_kinds);
        allocator.free(self.parameter_memory_kind_sizes);
        allocator.free(self.parameter_memory_kinds);
        allocator.free(self.optimized_program);
        allocator.free(self.logical_ids);
        self.plan.deinit();
        allocator.destroy(self.plan);
        allocator.destroy(self);
    }

    fn releaseGraph(self: *Executable) void {
        if (self.graph_released) return;
        self.graph.deinit();
        self.graph_released = true;
    }
};

const AsyncHostToDeviceTransferManager = struct {
    allocator: std.mem.Allocator,
    client: *runtime.Client,
    device: *runtime.Device,
    memory: *runtime.Memory,
    buffers: []*runtime.Buffer,
    staging: [][]u8,
    backend_transfers: []?backend_api.AsyncHostToDeviceTransferHandle,
    written: []usize,
    retrieved: []bool,
    completed: []bool,

    fn create(client: *runtime.Client, memory: *runtime.Memory, shape_specs: []const c.PJRT_ShapeSpec) !*AsyncHostToDeviceTransferManager {
        if (memory.addressable_devices.len == 0) return error.InvalidArgument;
        const device = memory.addressable_devices[0];
        const shard_index = deviceIndex(client, device) orelse return error.InvalidArgument;

        const manager = try allocator.create(AsyncHostToDeviceTransferManager);
        errdefer allocator.destroy(manager);

        const buffers = try allocator.alloc(*runtime.Buffer, shape_specs.len);
        errdefer allocator.free(buffers);
        const staging = try allocator.alloc([]u8, shape_specs.len);
        errdefer allocator.free(staging);
        const backend_transfers = try allocator.alloc(?backend_api.AsyncHostToDeviceTransferHandle, shape_specs.len);
        errdefer allocator.free(backend_transfers);
        const written = try allocator.alloc(usize, shape_specs.len);
        errdefer allocator.free(written);
        const retrieved = try allocator.alloc(bool, shape_specs.len);
        errdefer allocator.free(retrieved);
        const completed = try allocator.alloc(bool, shape_specs.len);
        errdefer allocator.free(completed);

        @memset(written, 0);
        @memset(retrieved, false);
        @memset(completed, false);
        @memset(backend_transfers, null);

        var initialized: usize = 0;
        errdefer {
            for (staging[0..initialized]) |bytes| if (bytes.len != 0) allocator.free(bytes);
            for (backend_transfers[0..initialized]) |maybe_transfer| {
                if (maybe_transfer) |transfer| client.backend.destroyAsyncHostToDeviceTransfer(transfer);
            }
            for (buffers[0..initialized]) |buffer| buffer.deinit();
        }

        for (shape_specs, 0..) |shape_spec, i| {
            const dims = shape_spec.dims[0..shape_spec.num_dims];
            const byte_size = denseByteSize(shape_spec.element_type, dims);
            const element_type = runtimeTypeFromPjrt(shape_spec.element_type);
            backend_transfers[i] = client.backend.beginAsyncHostToDeviceTransfer(device.local_hardware_id, element_type, dims, byte_size) catch null;
            if (backend_transfers[i] != null) {
                staging[i] = &.{};
                buffers[i] = try runtime.Buffer.initPendingBackendTransfer(
                    allocator,
                    client.backend,
                    element_type,
                    dims,
                    device,
                    memory,
                    shard_index,
                );
            } else {
                staging[i] = try allocator.alloc(u8, byte_size);
                @memset(staging[i], 0);
                buffers[i] = try runtime.Buffer.initDeviceAllocationForBackend(
                    allocator,
                    client.backend,
                    element_type,
                    dims,
                    device,
                    memory,
                    shard_index,
                );
                buffers[i].ready_event = runtime.Event.pending();
            }
            initialized += 1;
        }

        manager.* = .{
            .allocator = allocator,
            .client = client,
            .device = device,
            .memory = memory,
            .buffers = buffers,
            .staging = staging,
            .backend_transfers = backend_transfers,
            .written = written,
            .retrieved = retrieved,
            .completed = completed,
        };
        return manager;
    }

    fn deinit(self: *AsyncHostToDeviceTransferManager) void {
        for (self.staging) |bytes| if (bytes.len != 0) self.allocator.free(bytes);
        for (self.backend_transfers) |maybe_transfer| {
            if (maybe_transfer) |transfer| self.client.backend.destroyAsyncHostToDeviceTransfer(transfer);
        }
        for (self.buffers, self.retrieved) |buffer, was_retrieved| {
            if (!was_retrieved) buffer.deinit();
        }
        self.allocator.free(self.completed);
        self.allocator.free(self.retrieved);
        self.allocator.free(self.written);
        self.allocator.free(self.backend_transfers);
        self.allocator.free(self.staging);
        self.allocator.free(self.buffers);
        self.allocator.destroy(self);
    }

    fn index(self: *AsyncHostToDeviceTransferManager, buffer_index: c_int) !usize {
        if (buffer_index < 0) return error.InvalidArgument;
        const i: usize = @intCast(buffer_index);
        if (i >= self.buffers.len) return error.InvalidArgument;
        return i;
    }

    fn bufferByteSize(self: *const AsyncHostToDeviceTransferManager, i: usize) usize {
        return self.buffers[i].byte_size;
    }

    fn finishBuffer(self: *AsyncHostToDeviceTransferManager, i: usize) !void {
        if (self.completed[i]) return error.InvalidArgument;
        const buffer = self.buffers[i];
        _ = self.client.trimExecutableCacheForAllocation(self.memory, buffer.byte_size);
        if (self.backend_transfers[i]) |transfer| {
            const backend_buffer = try self.client.backend.finishAsyncHostToDeviceTransfer(transfer) orelse return error.UnsupportedRuntimeFeature;
            self.backend_transfers[i] = null;
            errdefer self.client.backend.destroyBuffer(backend_buffer);
            try buffer.replaceBackendStorage(backend_buffer);
            buffer.memory.stats.host_to_device_bytes += @intCast(buffer.byte_size);
        } else {
            try buffer.replaceBackendStorageFromHost(self.staging[i]);
        }
        buffer.ready_event.setReady();
        self.completed[i] = true;
    }
};

fn copyBytesWithIo(dst: []u8, src: []const u8) !void {
    if (dst.len != src.len) return error.InvalidArgument;
    var reader: std.Io.Reader = .fixed(src);
    var writer = std.Io.Writer.fixed(dst);
    const copied = try reader.streamRemaining(&writer);
    if (copied != src.len) return error.InvalidArgument;
}

fn fillMemoryKindArrays(kinds: [][*c]const u8, sizes: []usize) void {
    for (kinds, sizes) |*kind, *size| {
        kind.* = default_memory_kind.ptr;
        size.* = default_memory_kind.len;
    }
}

fn updateOptionalI64Slice(hasher: *std.hash.Wyhash, values: ?[]const i64) void {
    const present = values != null;
    hasher.update(std.mem.asBytes(&present));
    if (values) |slice| hasher.update(std.mem.sliceAsBytes(slice));
}

fn updateOptionalBytes(hasher: *std.hash.Wyhash, bytes: ?[]const u8) void {
    const present = bytes != null;
    hasher.update(std.mem.asBytes(&present));
    if (bytes) |slice| hasher.update(slice);
}

fn updateShardingFingerprint(hasher: *std.hash.Wyhash, shardings: []const compiler.ShardingPlan) void {
    hasher.update(std.mem.asBytes(&shardings.len));
    for (shardings) |sharding| {
        hasher.update(std.mem.asBytes(&sharding.kind));
        hasher.update(sharding.mesh_name);
        hasher.update(std.mem.sliceAsBytes(sharding.device_assignment));
    }
}

fn updateInstructionFingerprint(hasher: *std.hash.Wyhash, instruction: compiler.PlanInstruction) void {
    hasher.update(std.mem.asBytes(&instruction.kind));
    hasher.update(std.mem.sliceAsBytes(instruction.inputs));
    hasher.update(std.mem.sliceAsBytes(instruction.outputs));
    hasher.update(std.mem.sliceAsBytes(instruction.region_ids));
    updateOptionalI64Slice(hasher, instruction.dims);
    updateOptionalI64Slice(hasher, instruction.permutation);
    updateOptionalI64Slice(hasher, instruction.broadcast_dimensions);
    updateOptionalI64Slice(hasher, instruction.start_indices);
    updateOptionalI64Slice(hasher, instruction.limit_indices);
    updateOptionalI64Slice(hasher, instruction.strides);
    updateOptionalI64Slice(hasher, instruction.slice_sizes);
    updateOptionalI64Slice(hasher, instruction.edge_padding_low);
    updateOptionalI64Slice(hasher, instruction.edge_padding_high);
    updateOptionalI64Slice(hasher, instruction.interior_padding);
    updateOptionalI64Slice(hasher, instruction.offset_dims);
    updateOptionalI64Slice(hasher, instruction.collapsed_slice_dims);
    updateOptionalI64Slice(hasher, instruction.operand_batching_dims);
    updateOptionalI64Slice(hasher, instruction.start_indices_batching_dims);
    updateOptionalI64Slice(hasher, instruction.start_index_map);
    updateOptionalI64Slice(hasher, instruction.update_window_dims);
    updateOptionalI64Slice(hasher, instruction.inserted_window_dims);
    updateOptionalI64Slice(hasher, instruction.input_batching_dims);
    updateOptionalI64Slice(hasher, instruction.scatter_indices_batching_dims);
    updateOptionalI64Slice(hasher, instruction.scatter_dims_to_operand_dims);
    hasher.update(std.mem.asBytes(&instruction.index_vector_dim));
    hasher.update(std.mem.asBytes(&instruction.scatter_update_kind));
    hasher.update(std.mem.asBytes(&instruction.dimension));
    hasher.update(std.mem.asBytes(&instruction.top_k_k));
    hasher.update(std.mem.asBytes(&instruction.iota_dimension));
    hasher.update(std.mem.asBytes(&instruction.fft_kind));
    updateOptionalI64Slice(hasher, instruction.dimensions);
    hasher.update(std.mem.asBytes(&instruction.tuple_index));
    hasher.update(std.mem.asBytes(&instruction.lower));
    hasher.update(std.mem.asBytes(&instruction.triangular_left_side));
    hasher.update(std.mem.asBytes(&instruction.triangular_lower));
    hasher.update(std.mem.asBytes(&instruction.triangular_unit_diagonal));
    hasher.update(std.mem.asBytes(&instruction.triangular_transpose));
    updateOptionalBytes(hasher, instruction.custom_call_target);
    updateOptionalI64Slice(hasher, instruction.reduce_dimensions);
    updateOptionalI64Slice(hasher, instruction.lhs_batch_dimensions);
    updateOptionalI64Slice(hasher, instruction.rhs_batch_dimensions);
    updateOptionalI64Slice(hasher, instruction.lhs_contracting_dimensions);
    updateOptionalI64Slice(hasher, instruction.rhs_contracting_dimensions);
    hasher.update(std.mem.asBytes(&instruction.compare_direction));
    updateOptionalBytes(hasher, instruction.literal);
}

fn bytesFromC(ptr: [*c]const u8, len: usize) ?[]const u8 {
    if (ptr == null and len != 0) return null;
    if (len == 0) return &.{};
    return ptr[0..len];
}

fn parseUnaryCustomCallOp(name: []const u8) ?backend_api.CustomCallRegistration {
    const op: core.ElementwiseUnaryOp = if (std.mem.eql(u8, name, "negate"))
        .negate
    else if (std.mem.eql(u8, name, "exp"))
        .exp
    else if (std.mem.eql(u8, name, "expm1"))
        .expm1
    else if (std.mem.eql(u8, name, "tanh"))
        .tanh
    else if (std.mem.eql(u8, name, "sqrt"))
        .sqrt
    else if (std.mem.eql(u8, name, "rsqrt"))
        .rsqrt
    else if (std.mem.eql(u8, name, "abs"))
        .abs
    else if (std.mem.eql(u8, name, "cbrt"))
        .cbrt
    else if (std.mem.eql(u8, name, "ceil"))
        .ceil
    else if (std.mem.eql(u8, name, "floor"))
        .floor
    else if (std.mem.eql(u8, name, "log"))
        .log
    else if (std.mem.eql(u8, name, "log1p"))
        .log1p
    else if (std.mem.eql(u8, name, "logistic"))
        .logistic
    else if (std.mem.eql(u8, name, "sine"))
        .sine
    else if (std.mem.eql(u8, name, "cosine"))
        .cosine
    else if (std.mem.eql(u8, name, "not"))
        .not_
    else if (std.mem.eql(u8, name, "sign"))
        .sign
    else if (std.mem.eql(u8, name, "is_finite"))
        .is_finite
    else if (std.mem.eql(u8, name, "round_nearest_afz"))
        .round_nearest_afz
    else if (std.mem.eql(u8, name, "round_nearest_even"))
        .round_nearest_even
    else if (std.mem.eql(u8, name, "popcnt"))
        .popcnt
    else if (std.mem.eql(u8, name, "count_leading_zeros"))
        .count_leading_zeros
    else
        return null;
    return .{ .target = "", .kind = .unary, .unary_op = op };
}

fn parseBinaryCustomCallOp(name: []const u8) ?backend_api.CustomCallRegistration {
    const op: core.ElementwiseBinaryOp = if (std.mem.eql(u8, name, "add"))
        .add
    else if (std.mem.eql(u8, name, "subtract"))
        .subtract
    else if (std.mem.eql(u8, name, "multiply"))
        .multiply
    else if (std.mem.eql(u8, name, "divide"))
        .divide
    else if (std.mem.eql(u8, name, "maximum"))
        .maximum
    else if (std.mem.eql(u8, name, "minimum"))
        .minimum
    else if (std.mem.eql(u8, name, "power"))
        .power
    else if (std.mem.eql(u8, name, "atan2"))
        .atan2
    else if (std.mem.eql(u8, name, "remainder"))
        .remainder
    else if (std.mem.eql(u8, name, "and"))
        .and_
    else if (std.mem.eql(u8, name, "or"))
        .or_
    else if (std.mem.eql(u8, name, "xor"))
        .xor
    else if (std.mem.eql(u8, name, "shift_left"))
        .shift_left
    else if (std.mem.eql(u8, name, "shift_right_logical"))
        .shift_right_logical
    else if (std.mem.eql(u8, name, "shift_right_arithmetic"))
        .shift_right_arithmetic
    else
        return null;
    return .{ .target = "", .kind = .binary, .binary_op = op };
}

fn registerPjrtxCustomCall(registration: backend_api.CustomCallRegistration) ?*c.PJRT_Error {
    var backend_impl = backend_registry.create(.metal_mlx);
    backend_impl.registerCustomCall(registration) catch |err| switch (err) {
        error.InvalidCustomCall => return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "invalid PjRTx custom call registration"),
        error.OutOfMemory => return makeError(c.PJRT_Error_Code_RESOURCE_EXHAUSTED, "failed to allocate PjRTx custom call registration"),
        else => return makeError(c.PJRT_Error_Code_INTERNAL, "failed to register PjRTx custom call"),
    };
    return null;
}

fn markerAddress(comptime marker: anytype) usize {
    return @intFromPtr(&marker);
}

fn registerPjrtxCustomCallHandler(target: []const u8, api_version: c_int, handler_execute: ?*anyopaque) ?*c.PJRT_Error {
    if (api_version != 0) return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "PjRTx custom calls currently support PJRT GPU untyped API version 0");
    const handler = handler_execute orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call handler_execute is null");
    const handler_address = @intFromPtr(handler);

    if (handler_address == markerAddress(PjRTx_CustomCall_Identity)) {
        return registerPjrtxCustomCall(.{ .target = target, .kind = .identity });
    }
    if (handler_address == markerAddress(PjRTx_CustomCall_UnarySqrt)) {
        return registerPjrtxCustomCall(.{ .target = target, .kind = .unary, .unary_op = .sqrt });
    }
    if (handler_address == markerAddress(PjRTx_CustomCall_BinaryAdd)) {
        return registerPjrtxCustomCall(.{ .target = target, .kind = .binary, .binary_op = .add });
    }

    return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "custom call handler is not a PjRTx MLX executable handler");
}

fn pjrtGpuRegisterCustomCall(args: [*c]PJRT_Gpu_Register_Custom_Call_Args) callconv(.c) ?*c.PJRT_Error {
    if (args == null) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call registration args are null");
    if (args[0].struct_size < @sizeOf(PJRT_Gpu_Register_Custom_Call_Args)) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call registration args struct is too small");
    }
    const target = bytesFromC(args[0].function_name, args[0].function_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call target is null");
    return registerPjrtxCustomCallHandler(target, args[0].api_version, args[0].handler_execute);
}

fn updateTargetDeviceFingerprint(hasher: *std.hash.Wyhash, client: *const runtime.Client, plan: *const compiler.ExecutablePlan) void {
    const device_count = plan.options.numDevices();
    hasher.update(std.mem.asBytes(&device_count));
    for (0..device_count) |i| {
        const device_id = if (plan.options.device_assignment.len != 0) plan.options.device_assignment[i] else client.devices[i].id;
        hasher.update(std.mem.asBytes(&device_id));
        const device = client.lookupDevice(device_id) orelse {
            const missing_device: i32 = -1;
            hasher.update(std.mem.asBytes(&missing_device));
            continue;
        };
        hasher.update(std.mem.asBytes(&device.local_hardware_id));
        hasher.update(std.mem.asBytes(&device.registry_id));
        hasher.update(std.mem.asBytes(&device.process_index));
        hasher.update(std.mem.asBytes(&device.addressable));
        hasher.update(std.mem.asBytes(&device.memory_bytes));
        hasher.update(std.mem.asBytes(&device.has_unified_memory));
        hasher.update(std.mem.asBytes(&device.default_memory_id));
        hasher.update(device.name);
        hasher.update(device.debug_string);
        const memory = client.lookupMemory(device.default_memory_id) orelse continue;
        hasher.update(std.mem.asBytes(&memory.id));
        hasher.update(std.mem.asBytes(&memory.kind));
        hasher.update(std.mem.sliceAsBytes(memory.addressable_device_ids));
    }
}

fn allocExecutableFingerprint(
    alloc: std.mem.Allocator,
    client: *const runtime.Client,
    optimized_program: []const u8,
    plan: *const compiler.ExecutablePlan,
) ![]u8 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update("pjrtx-executable-cache-v4");
    hasher.update(optimized_program);
    const caps = client.backend.capabilities();
    hasher.update(@tagName(client.backend_kind));
    hasher.update(caps.name);
    hasher.update(std.mem.asBytes(&caps.kind));
    hasher.update(std.mem.asBytes(&caps.supports_device_buffers));
    hasher.update(std.mem.asBytes(&caps.supports_unified_memory));
    hasher.update(std.mem.asBytes(&caps.supports_async_execution));
    const custom_call_registry_version = client.backend.customCallRegistryVersion();
    hasher.update(std.mem.asBytes(&custom_call_registry_version));
    updateTargetDeviceFingerprint(&hasher, client, plan);
    hasher.update(std.mem.asBytes(&plan.options.num_replicas));
    hasher.update(std.mem.asBytes(&plan.options.num_partitions));
    hasher.update(std.mem.asBytes(&plan.options.use_shardy_partitioner));
    hasher.update(std.mem.sliceAsBytes(plan.options.device_assignment));
    updateShardingFingerprint(&hasher, plan.parameter_shardings);
    updateShardingFingerprint(&hasher, plan.output_shardings);
    hasher.update(std.mem.sliceAsBytes(plan.output_ids));
    hasher.update(std.mem.sliceAsBytes(plan.donated_parameter_indices));
    for (plan.output_aliases) |alias| {
        hasher.update(std.mem.asBytes(&alias.output_index));
        hasher.update(std.mem.asBytes(&alias.parameter_index));
        hasher.update(std.mem.asBytes(&alias.kind));
    }
    for (plan.values) |value| {
        hasher.update(std.mem.asBytes(&value.role));
        hasher.update(std.mem.asBytes(&value.descriptor.element_type));
        hasher.update(std.mem.sliceAsBytes(value.descriptor.dims));
        hasher.update(std.mem.asBytes(&value.descriptor.layout));
        hasher.update(std.mem.asBytes(&value.descriptor.device_id));
        hasher.update(std.mem.asBytes(&value.descriptor.memory_id));
        hasher.update(std.mem.asBytes(&value.descriptor.shard_index));
        hasher.update(std.mem.asBytes(&value.storage));
        hasher.update(std.mem.sliceAsBytes(value.elements));
    }
    for (plan.regions) |region| {
        hasher.update(std.mem.asBytes(&region.id));
        hasher.update(std.mem.asBytes(&region.parent_instruction_index));
        hasher.update(std.mem.asBytes(&region.kind));
        for (region.argument_descriptors) |descriptor| {
            hasher.update(std.mem.asBytes(&descriptor.element_type));
            hasher.update(std.mem.sliceAsBytes(descriptor.dims));
            hasher.update(std.mem.asBytes(&descriptor.layout));
        }
        for (region.return_descriptors) |descriptor| {
            hasher.update(std.mem.asBytes(&descriptor.element_type));
            hasher.update(std.mem.sliceAsBytes(descriptor.dims));
            hasher.update(std.mem.asBytes(&descriptor.layout));
        }
    }
    for (plan.instructions) |instruction| updateInstructionFingerprint(&hasher, instruction);
    return std.fmt.allocPrint(alloc, "pjrtx-{x}", .{hasher.final()});
}

var api_storage: c.PJRT_Api = undefined;
var gpu_custom_call_extension: PJRT_Gpu_Custom_Call = .{
    .base = .{
        .struct_size = @sizeOf(PJRT_Gpu_Custom_Call),
        .type = c.PJRT_Extension_Type_Gpu_Custom_Call,
        .next = null,
    },
    .custom_call = pjrtGpuRegisterCustomCall,
};
var api_ready = false;
var attrs_ready = false;
var attrs: [5]c.PJRT_NamedValue = undefined;

fn cstrLen(ptr: [*c]const u8) usize {
    return std.mem.len(ptr);
}

fn makeError(code: c.PJRT_Error_Code, message: []const u8) ?*c.PJRT_Error {
    const err = allocator.create(PjrtxError) catch return null;
    err.* = .{
        .base = .{ .vtable = null },
        .code = code,
        .message = allocator.dupe(u8, message) catch {
            allocator.destroy(err);
            return null;
        },
    };
    return @ptrCast(err);
}

fn unimplemented(message: []const u8) ?*c.PJRT_Error {
    return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, message);
}

fn elementSize(t: c.PJRT_Buffer_Type) usize {
    return switch (t) {
        c.PJRT_Buffer_Type_PRED, c.PJRT_Buffer_Type_S8, c.PJRT_Buffer_Type_U8 => 1,
        c.PJRT_Buffer_Type_S16, c.PJRT_Buffer_Type_U16, c.PJRT_Buffer_Type_F16, c.PJRT_Buffer_Type_BF16 => 2,
        c.PJRT_Buffer_Type_S32, c.PJRT_Buffer_Type_U32, c.PJRT_Buffer_Type_F32 => 4,
        c.PJRT_Buffer_Type_S64, c.PJRT_Buffer_Type_U64, c.PJRT_Buffer_Type_F64, c.PJRT_Buffer_Type_C64 => 8,
        c.PJRT_Buffer_Type_C128 => 16,
        else => 0,
    };
}

fn runtimeTypeFromPjrt(t: c.PJRT_Buffer_Type) runtime.BufferType {
    return switch (t) {
        c.PJRT_Buffer_Type_PRED => .pred,
        c.PJRT_Buffer_Type_S8 => .s8,
        c.PJRT_Buffer_Type_S16 => .s16,
        c.PJRT_Buffer_Type_S32 => .s32,
        c.PJRT_Buffer_Type_S64 => .s64,
        c.PJRT_Buffer_Type_U8 => .u8,
        c.PJRT_Buffer_Type_U16 => .u16,
        c.PJRT_Buffer_Type_U32 => .u32,
        c.PJRT_Buffer_Type_U64 => .u64,
        c.PJRT_Buffer_Type_F16 => .f16,
        c.PJRT_Buffer_Type_F32 => .f32,
        c.PJRT_Buffer_Type_F64 => .f64,
        c.PJRT_Buffer_Type_BF16 => .bf16,
        c.PJRT_Buffer_Type_C64 => .c64,
        c.PJRT_Buffer_Type_C128 => .c128,
        else => .invalid,
    };
}

fn pjrtTypeFromRuntime(t: runtime.BufferType) c.PJRT_Buffer_Type {
    return switch (t) {
        .invalid => c.PJRT_Buffer_Type_INVALID,
        .pred => c.PJRT_Buffer_Type_PRED,
        .s8 => c.PJRT_Buffer_Type_S8,
        .s16 => c.PJRT_Buffer_Type_S16,
        .s32 => c.PJRT_Buffer_Type_S32,
        .s64 => c.PJRT_Buffer_Type_S64,
        .u8 => c.PJRT_Buffer_Type_U8,
        .u16 => c.PJRT_Buffer_Type_U16,
        .u32 => c.PJRT_Buffer_Type_U32,
        .u64 => c.PJRT_Buffer_Type_U64,
        .f16 => c.PJRT_Buffer_Type_F16,
        .f32 => c.PJRT_Buffer_Type_F32,
        .f64 => c.PJRT_Buffer_Type_F64,
        .bf16 => c.PJRT_Buffer_Type_BF16,
        .c64 => c.PJRT_Buffer_Type_C64,
        .c128 => c.PJRT_Buffer_Type_C128,
    };
}

fn denseByteSize(t: c.PJRT_Buffer_Type, dims: []const i64) usize {
    var elems: usize = 1;
    for (dims) |dim| elems *= @intCast(dim);
    return elems * elementSize(t);
}

fn clientFromC(client: ?*c.PJRT_Client) *runtime.Client {
    return @ptrCast(@alignCast(client.?));
}

fn deviceFromC(device: ?*c.PJRT_Device) *runtime.Device {
    return @ptrCast(@alignCast(device.?));
}

fn memoryFromC(memory: ?*c.PJRT_Memory) *runtime.Memory {
    return @ptrCast(@alignCast(memory.?));
}

fn deviceIndex(client: *const runtime.Client, device: *const runtime.Device) ?usize {
    for (client.devices, 0..) |*candidate, i| {
        if (candidate == device or candidate.id == device.id) return i;
    }
    return null;
}

fn bufferFromC(buffer: ?*c.PJRT_Buffer) *runtime.Buffer {
    return @ptrCast(@alignCast(buffer.?));
}

fn asyncH2DManagerFromC(manager: ?*c.PJRT_AsyncHostToDeviceTransferManager) *AsyncHostToDeviceTransferManager {
    return @ptrCast(@alignCast(manager.?));
}

fn executableFromC(executable: ?*c.PJRT_LoadedExecutable) *Executable {
    return @ptrCast(@alignCast(executable.?));
}

fn topologyFromC(topology: ?*const c.PJRT_TopologyDescription) *runtime.Client {
    return @ptrCast(@alignCast(@constCast(topology.?)));
}

fn initAttrs() void {
    if (attrs_ready) return;
    attrs = std.mem.zeroes(@TypeOf(attrs));

    attrs[0].struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attrs[0].name = "plugin_name";
    attrs[0].name_size = "plugin_name".len;
    attrs[0].type = c.PJRT_NamedValue_kString;
    attrs[0].unnamed_0.string_value = plugin_name;
    attrs[0].value_size = plugin_name.len;

    attrs[1].struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attrs[1].name = "xla_version";
    attrs[1].name_size = "xla_version".len;
    attrs[1].type = c.PJRT_NamedValue_kString;
    attrs[1].unnamed_0.string_value = "local";
    attrs[1].value_size = "local".len;

    attrs[2].struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attrs[2].name = "stablehlo_current_version";
    attrs[2].name_size = "stablehlo_current_version".len;
    attrs[2].type = c.PJRT_NamedValue_kInt64List;
    attrs[2].unnamed_0.int64_array_value = &stablehlo_version;
    attrs[2].value_size = stablehlo_version.len;

    attrs[3].struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attrs[3].name = "stablehlo_minimum_version";
    attrs[3].name_size = "stablehlo_minimum_version".len;
    attrs[3].type = c.PJRT_NamedValue_kInt64List;
    attrs[3].unnamed_0.int64_array_value = &stablehlo_version;
    attrs[3].value_size = stablehlo_version.len;

    attrs[4].struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attrs[4].name = "pjrtx_default_backend";
    attrs[4].name_size = "pjrtx_default_backend".len;
    attrs[4].type = c.PJRT_NamedValue_kString;
    attrs[4].unnamed_0.string_value = "metal_mlx";
    attrs[4].value_size = "metal_mlx".len;

    attrs_ready = true;
}

const ClientCreateConfig = struct {
    backend_kind: runtime.BackendKind = .metal_mlx,
};

fn traceEnabled() bool {
    if (envFlag("PJRTX_PROFILE")) return true;
    return envFlag("PJRTX_TRACE");
}

fn envFlag(comptime name: [:0]const u8) bool {
    const value = std.c.getenv(name) orelse return false;
    const text = std.mem.span(value);
    return text.len != 0 and !std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false");
}

fn executableCacheMaxBytesFromEnv() ?u64 {
    const value = std.c.getenv("PJRTX_EXECUTABLE_CACHE_MAX_BYTES") orelse return null;
    const text = std.mem.span(value);
    if (text.len == 0) return null;
    return std.fmt.parseUnsigned(u64, text, 10) catch null;
}

fn nowNs() u64 {
    var info: MachTimebaseInfo = undefined;
    if (mach_timebase_info(&info) != 0 or info.denom == 0) return mach_absolute_time();
    const ticks: u128 = mach_absolute_time();
    return @intCast((ticks * info.numer) / info.denom);
}

fn elapsedUs(start_ns: u64) u64 {
    return @intCast((nowNs() -| start_ns) / 1000);
}

fn trace(comptime fmt: []const u8, args: anytype) void {
    if (!traceEnabled()) return;
    std.debug.print("pjrtx_trace " ++ fmt ++ "\n", args);
}

fn clientCreateConfigFromArgs(args: c.PJRT_Client_Create_Args) !ClientCreateConfig {
    var config: ClientCreateConfig = .{};
    if (args.create_options != null) {
        for (0..args.num_options) |i| {
            const option = args.create_options[i];
            const name = option.name[0..option.name_size];
            if (std.mem.eql(u8, name, backend_option)) {
                if (option.type != c.PJRT_NamedValue_kString) return error.InvalidBackend;
                const value = option.unnamed_0.string_value[0..option.value_size];
                if (std.mem.eql(u8, value, "metal_mlx")) {
                    config.backend_kind = .metal_mlx;
                } else {
                    return error.InvalidBackend;
                }
            } else {
                return error.InvalidBackend;
            }
        }
    }
    return config;
}

fn isPjrtxTextCompileOptions(text: []const u8) bool {
    if (text.len == 0) return false;
    for (text) |byte| {
        if (!(std.ascii.isPrint(byte) or std.ascii.isWhitespace(byte))) return false;
    }
    return std.mem.indexOf(u8, text, "replicas=") != null or
        std.mem.indexOf(u8, text, "partitions=") != null or
        std.mem.indexOf(u8, text, "assignment=") != null or
        std.mem.indexOf(u8, text, "use_shardy=") != null;
}

fn pjrtErrorDestroy(args: [*c]c.PJRT_Error_Destroy_Args) callconv(.c) void {
    if (args[0].@"error") |base| {
        const err: *PjrtxError = @ptrCast(@alignCast(base));
        allocator.free(err.message);
        allocator.destroy(err);
    }
}

fn pjrtErrorMessage(args: [*c]c.PJRT_Error_Message_Args) callconv(.c) void {
    const err: *const PjrtxError = @ptrCast(@alignCast(args[0].@"error".?));
    args[0].message = err.message.ptr;
    args[0].message_size = err.message.len;
}

fn pjrtErrorGetCode(args: [*c]c.PJRT_Error_GetCode_Args) callconv(.c) ?*c.PJRT_Error {
    const err: *const PjrtxError = @ptrCast(@alignCast(args[0].@"error".?));
    args[0].code = err.code;
    return null;
}

fn pjrtErrorForEachPayload(_: [*c]c.PJRT_Error_ForEachPayload_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn pjrtPluginInitialize(_: [*c]c.PJRT_Plugin_Initialize_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn pjrtPluginAttributes(args: [*c]c.PJRT_Plugin_Attributes_Args) callconv(.c) ?*c.PJRT_Error {
    initAttrs();
    args[0].attributes = &attrs;
    args[0].num_attributes = attrs.len;
    return null;
}

fn eventCreateReady() ?*c.PJRT_Event {
    const event = allocator.create(runtime.Event) catch return null;
    event.* = runtime.Event.ready();
    return @ptrCast(event);
}

fn eventCreatePending() ?*c.PJRT_Event {
    const event = allocator.create(runtime.Event) catch return null;
    event.* = runtime.Event.pending();
    return @ptrCast(event);
}

fn eventCreateFailed(message: []const u8) ?*c.PJRT_Event {
    const event = allocator.create(runtime.Event) catch return null;
    event.* = runtime.Event.failed(message);
    return @ptrCast(event);
}

fn eventCreateFromRuntime(source: runtime.Event) ?*c.PJRT_Event {
    return switch (source.state) {
        .pending => eventCreatePending(),
        .ready => eventCreateReady(),
        .failed => eventCreateFailed(source.message),
    };
}

fn eventSetReady(event: ?*c.PJRT_Event) void {
    if (event) |opaque_event| {
        eventFromC(opaque_event).setReady();
    }
}

fn eventSetFailed(event: ?*c.PJRT_Event, message: []const u8) void {
    if (event) |opaque_event| {
        eventFromC(opaque_event).setFailed(message);
    }
}

fn eventFromC(event: *c.PJRT_Event) *runtime.Event {
    return @ptrCast(@alignCast(event));
}

const EventOnReadyBridge = struct {
    callback: c.PJRT_Event_OnReadyCallback,
    user_arg: ?*anyopaque,
};

fn eventOnReadyBridge(message: ?[]const u8, user_arg: ?*anyopaque) void {
    const bridge: *EventOnReadyBridge = @ptrCast(@alignCast(user_arg.?));
    defer allocator.destroy(bridge);
    const maybe_error = if (message) |msg| makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, msg) else null;
    bridge.callback.?(maybe_error, bridge.user_arg);
}

fn pjrtEventDestroy(args: [*c]c.PJRT_Event_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args[0].event) |event| {
        const runtime_event = eventFromC(event);
        runtime_event.deinit();
        allocator.destroy(runtime_event);
    }
    return null;
}

fn pjrtEventIsReady(args: [*c]c.PJRT_Event_IsReady_Args) callconv(.c) ?*c.PJRT_Error {
    const event = eventFromC(args[0].event.?);
    args[0].is_ready = event.isReady();
    return null;
}

fn pjrtEventError(args: [*c]c.PJRT_Event_Error_Args) callconv(.c) ?*c.PJRT_Error {
    const event = eventFromC(args[0].event.?);
    if (event.state == .failed) return makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, event.message);
    return null;
}

fn pjrtEventAwait(args: [*c]c.PJRT_Event_Await_Args) callconv(.c) ?*c.PJRT_Error {
    const event = eventFromC(args[0].event.?);
    event.awaitReady() catch |err| return switch (err) {
        error.EventFailed => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, event.message),
        error.EventPending => makeError(c.PJRT_Error_Code_INTERNAL, "pending event has no scheduler completion source"),
    };
    return null;
}

fn pjrtEventOnReady(args: [*c]c.PJRT_Event_OnReady_Args) callconv(.c) ?*c.PJRT_Error {
    const event = eventFromC(args[0].event.?);
    const callback = args[0].callback orelse return null;
    const bridge = allocator.create(EventOnReadyBridge) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate event callback");
    bridge.* = .{
        .callback = callback,
        .user_arg = args[0].user_arg,
    };
    event.onReady(eventOnReadyBridge, bridge) catch |err| {
        allocator.destroy(bridge);
        return switch (err) {
            error.TooManyEventCallbacks => makeError(c.PJRT_Error_Code_RESOURCE_EXHAUSTED, "too many callbacks registered on event"),
        };
    };
    return null;
}

fn pjrtClientCreate(args: [*c]c.PJRT_Client_Create_Args) callconv(.c) ?*c.PJRT_Error {
    const config = clientCreateConfigFromArgs(args[0]) catch {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "invalid PjRTx client create option");
    };
    const client = switch (config.backend_kind) {
        .metal_mlx => runtime.Client.init(allocator, backend_registry.create(.metal_mlx), 1),
    } catch {
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to create PjRTx client");
    };
    if (executableCacheMaxBytesFromEnv()) |max_resident_bytes| {
        client.setExecutableCacheMaxResidentBytes(max_resident_bytes);
    }
    args[0].client = @ptrCast(client);
    return null;
}

fn pjrtClientDestroy(args: [*c]c.PJRT_Client_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args[0].client) |client| clientFromC(client).deinit();
    return null;
}

fn pjrtClientPlatformName(args: [*c]c.PJRT_Client_PlatformName_Args) callconv(.c) ?*c.PJRT_Error {
    _ = clientFromC(args[0].client);
    args[0].platform_name = platform_name;
    args[0].platform_name_size = platform_name.len;
    return null;
}

fn pjrtClientProcessIndex(args: [*c]c.PJRT_Client_ProcessIndex_Args) callconv(.c) ?*c.PJRT_Error {
    _ = clientFromC(args[0].client);
    args[0].process_index = 0;
    return null;
}

fn pjrtClientPlatformVersion(args: [*c]c.PJRT_Client_PlatformVersion_Args) callconv(.c) ?*c.PJRT_Error {
    _ = clientFromC(args[0].client);
    args[0].platform_version = platform_version;
    args[0].platform_version_size = platform_version.len;
    return null;
}

fn pjrtClientTopologyDescription(args: [*c]c.PJRT_Client_TopologyDescription_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].topology = @ptrCast(clientFromC(args[0].client));
    return null;
}

fn pjrtClientDevices(args: [*c]c.PJRT_Client_Devices_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    args[0].devices = @ptrCast(client.device_handles.ptr);
    args[0].num_devices = client.device_handles.len;
    return null;
}

fn pjrtClientAddressableDevices(args: [*c]c.PJRT_Client_AddressableDevices_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    args[0].addressable_devices = @ptrCast(client.device_handles.ptr);
    args[0].num_addressable_devices = client.device_handles.len;
    return null;
}

fn pjrtClientLookupDevice(args: [*c]c.PJRT_Client_LookupDevice_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    const device = client.lookupDevice(args[0].id) orelse return makeError(c.PJRT_Error_Code_NOT_FOUND, "device id not found");
    args[0].device = @ptrCast(@constCast(device));
    return null;
}

fn pjrtClientLookupAddressableDevice(args: [*c]c.PJRT_Client_LookupAddressableDevice_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    const device = client.lookupDevice(args[0].local_hardware_id) orelse return makeError(c.PJRT_Error_Code_NOT_FOUND, "local hardware id not found");
    args[0].addressable_device = @ptrCast(@constCast(device));
    return null;
}

fn pjrtClientAddressableMemories(args: [*c]c.PJRT_Client_AddressableMemories_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    args[0].addressable_memories = @ptrCast(client.memory_handles.ptr);
    args[0].num_addressable_memories = client.memory_handles.len;
    return null;
}

fn pjrtClientCompile(args: [*c]c.PJRT_Client_Compile_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const client = clientFromC(args[0].client);
    var options: compiler.CompileOptions = .{
        .num_partitions = @intCast(client.devices.len),
    };
    var parsed_options = false;
    if (args[0].compile_options != null and args[0].compile_options_size != 0) {
        const text = args[0].compile_options[0..args[0].compile_options_size];
        if (isPjrtxTextCompileOptions(text)) {
            var options_reader: std.Io.Reader = .fixed(text);
            options = compiler.parseTextCompileOptionsFromReader(allocator, &options_reader) catch {
                return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "invalid PjRTx text compile options");
            };
            parsed_options = true;
        }
    }
    defer if (parsed_options) allocator.free(options.device_assignment);

    if (options.numDevices() > client.devices.len) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "compile options require more devices than the client exposes");
    }
    for (options.device_assignment) |device_id| {
        if (client.lookupDevice(device_id) == null) {
            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "compile options reference an unknown device id");
        }
    }

    var analysis: ?compiler.ModuleAnalysis = null;
    defer if (analysis) |*owned_analysis| owned_analysis.deinit();
    if (args[0].program != null and args[0].program[0].code != null and args[0].program[0].code_size != 0) {
        const program = args[0].program[0];
        const module_bytes = program.code[0..program.code_size];
        const format_text = if (program.format != null) program.format[0..program.format_size] else "";
        trace("event=compile_start format={s} program_bytes={d} compile_options_bytes={d}", .{
            format_text,
            module_bytes.len,
            args[0].compile_options_size,
        });
        var module_reader: std.Io.Reader = .fixed(module_bytes);
        var diagnostics = std.Io.Writer.Allocating.init(allocator);
        defer diagnostics.deinit();
        analysis = compiler.analyzeProgramFromReader(allocator, format_text, &module_reader, &diagnostics.writer) catch |err| {
            const message = diagnostics.writer.buffered();
            trace("event=compile_ingest_error err={s} diagnostic_bytes={d}", .{ @errorName(err), message.len });
            const fallback = "failed to ingest StableHLO/MLIR program";
            const text = if (message.len == 0) fallback else message;
            const code: c.PJRT_Error_Code = @intCast(switch (err) {
                error.UnsupportedOp, error.UnsupportedSharding, error.UnsupportedProgramEncoding, error.GspmdNotEnabled, error.InvalidManualComputation => c.PJRT_Error_Code_UNIMPLEMENTED,
                error.UnsupportedProgramFormat, error.InvalidStablehloModule => c.PJRT_Error_Code_INVALID_ARGUMENT,
                else => c.PJRT_Error_Code_INTERNAL,
            });
            return makeError(code, text);
        };
    }

    var plan = if (analysis) |owned_analysis|
        compiler.makeExecutablePlan(allocator, options, owned_analysis) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable plan")
    else
        compiler.makeReplicatedPlan(allocator, options, 1, 1) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable plan");
    var plan_moved = false;
    errdefer if (!plan_moved) plan.deinit();

    var plan_diagnostics = std.Io.Writer.Allocating.init(allocator);
    defer plan_diagnostics.deinit();
    compiler.verifyExecutablePlan(allocator, plan, &plan_diagnostics.writer) catch |err| {
        const message = plan_diagnostics.writer.buffered();
        const fallback = "invalid executable plan";
        const text = if (message.len == 0) fallback else message;
        const code: c.PJRT_Error_Code = @intCast(switch (err) {
            error.InvalidExecutablePlan => c.PJRT_Error_Code_INVALID_ARGUMENT,
            else => c.PJRT_Error_Code_INTERNAL,
        });
        return makeError(code, text);
    };

    const logical_ids = allocator.alloc(c.PJRT_LogicalDeviceIds, plan.options.numDevices()) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable logical device ids");
    errdefer allocator.free(logical_ids);
    for (logical_ids, 0..) |*id, index| {
        id.* = .{
            .replica = @intCast(index / @as(usize, @intCast(plan.options.num_partitions))),
            .partition = @intCast(index % @as(usize, @intCast(plan.options.num_partitions))),
        };
    }

    const optimized_program = if (analysis) |owned_analysis|
        allocator.dupe(u8, owned_analysis.source) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate optimized program")
    else
        allocator.dupe(u8, "module {}\n") catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate optimized program");
    errdefer allocator.free(optimized_program);

    const parameter_memory_kinds = allocator.alloc([*c]const u8, plan.parameter_shardings.len) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate parameter memory kind table");
    errdefer allocator.free(parameter_memory_kinds);
    const parameter_memory_kind_sizes = allocator.alloc(usize, plan.parameter_shardings.len) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate parameter memory kind sizes");
    errdefer allocator.free(parameter_memory_kind_sizes);
    fillMemoryKindArrays(parameter_memory_kinds, parameter_memory_kind_sizes);

    const output_memory_kinds = allocator.alloc([*c]const u8, plan.output_shardings.len) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate output memory kind table");
    errdefer allocator.free(output_memory_kinds);
    const output_memory_kind_sizes = allocator.alloc(usize, plan.output_shardings.len) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate output memory kind sizes");
    errdefer allocator.free(output_memory_kind_sizes);
    fillMemoryKindArrays(output_memory_kinds, output_memory_kind_sizes);

    const fingerprint = allocExecutableFingerprint(allocator, client, optimized_program, &plan) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable fingerprint");
    errdefer allocator.free(fingerprint);
    const cache_hit = client.recordExecutableCompile(fingerprint) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to update executable cache");

    const plan_ptr = allocator.create(compiler.ExecutablePlan) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable plan storage");
    plan_ptr.* = plan;
    plan_moved = true;
    errdefer {
        plan_ptr.deinit();
        allocator.destroy(plan_ptr);
    }

    var lowering_diagnostics = std.Io.Writer.Allocating.init(allocator);
    defer lowering_diagnostics.deinit();
    var graph = runtime.ExecutableGraph.initWithOptions(allocator, client, plan_ptr, .{
        .diagnostic_writer = &lowering_diagnostics.writer,
        .cache_fingerprint = fingerprint,
    }) catch |err| switch (err) {
        error.UnsupportedRuntimeFeature => {
            const message = lowering_diagnostics.writer.buffered();
            const fallback = "program is not fully lowered to the MLX backend executable; runtime execution is device-only";
            return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, if (message.len == 0) fallback else message);
        },
        else => return makeError(c.PJRT_Error_Code_INTERNAL, "failed to build executable graph"),
    };
    errdefer graph.deinit();

    const executable = allocator.create(Executable) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable");
    executable.* = .{
        .client = client,
        .plan = plan_ptr,
        .graph = graph,
        .logical_ids = logical_ids,
        .optimized_program = optimized_program,
        .parameter_memory_kinds = parameter_memory_kinds,
        .parameter_memory_kind_sizes = parameter_memory_kind_sizes,
        .output_memory_kinds = output_memory_kinds,
        .output_memory_kind_sizes = output_memory_kind_sizes,
        .fingerprint = fingerprint,
    };
    args[0].executable = @ptrCast(executable);
    const backend_stats = graph.backendExecutableStats() orelse backend_api.ExecutableStats{};
    trace(
        "event=compile backend={s} backend_devices={d} values={d} instructions={d} cache_hit={d} cache_hits_total={d} cache_misses_total={d} backend_cache_reuse={d} backend_cache_entries={d} backend_cache_bytes={d} backend_cache_peak_bytes={d} backend_cache_evictions={d} backend_cache_evicted_bytes={d} backend_executable={d} lowered={d} unlowered={d} resident_constants={d} resident_constant_bytes={d} program_values={d} program_nodes={d} program_edges={d} schedule_items={d} subprograms={d} control_flows={d} fusion_groups={d} materialization_boundaries={d} planned_releases={d} planned_release_bytes={d} peak_live_values={d} peak_live_bytes={d} elapsed_us={d}",
        .{
            @tagName(client.backend_kind),
            backend_stats.program_device_count,
            plan_ptr.values.len,
            plan_ptr.instructions.len,
            @intFromBool(cache_hit),
            client.executable_cache.stats.hits,
            client.executable_cache.stats.misses,
            @intFromBool(graph.lowering.backend_executable_cache_reused),
            client.executable_cache.stats.resident_entries,
            client.executable_cache.stats.resident_bytes,
            client.executable_cache.stats.peak_resident_bytes,
            client.executable_cache.stats.evictions,
            client.executable_cache.stats.evicted_resident_bytes,
            @intFromBool(graph.backend_executable != null),
            graph.lowering.lowered_instruction_count,
            0,
            backend_stats.resident_constant_count,
            backend_stats.resident_constant_bytes,
            backend_stats.program_value_count,
            backend_stats.program_node_count,
            backend_stats.program_edge_count,
            backend_stats.program_schedule_item_count,
            backend_stats.program_subprogram_count,
            backend_stats.program_control_flow_count,
            backend_stats.program_fusion_group_count,
            backend_stats.program_materialization_boundary_count,
            backend_stats.program_planned_release_count,
            backend_stats.program_planned_release_bytes,
            backend_stats.program_peak_live_value_count,
            backend_stats.program_peak_live_bytes,
            elapsedUs(trace_start_ns),
        },
    );
    trace("event=compile_aliases output_aliases={d} donated_parameters={d}", .{ plan_ptr.output_aliases.len, plan_ptr.donated_parameter_indices.len });
    trace(
        "event=compile_cache cache_hit={d} backend_cache_reuse={d} backend_cache_entries={d} backend_cache_bytes={d} backend_cache_compile_samples={d} backend_cache_compile_us_total={d} backend_cache_compile_us_peak={d} cache_trim_bytes={d} cache_trim_evictions={d} cache_pressure_remaining_bytes={d} cache_pressure_failed={d} elapsed_us={d}",
        .{
            @intFromBool(cache_hit),
            @intFromBool(graph.lowering.backend_executable_cache_reused),
            client.executable_cache.stats.resident_entries,
            client.executable_cache.stats.resident_bytes,
            client.executable_cache.stats.compile_latency_samples,
            client.executable_cache.stats.compile_latency_us_total,
            client.executable_cache.stats.compile_latency_us_peak,
            graph.last_compile_cache_trim.freed_bytes,
            graph.last_compile_cache_trim.evicted_entries,
            graph.last_compile_cache_trim.remaining_resident_bytes,
            @intFromBool(graph.last_compile_cache_trim.still_over_capacity),
            elapsedUs(trace_start_ns),
        },
    );
    return null;
}

fn pjrtClientDefaultDeviceAssignment(args: [*c]c.PJRT_Client_DefaultDeviceAssignment_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    const needed: usize = @intCast(args[0].num_replicas * args[0].num_partitions);
    if (needed > client.devices.len or needed > args[0].default_assignment_size) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "invalid default device assignment request");
    }
    for (0..needed) |i| args[0].default_assignment[i] = @intCast(i);
    return null;
}

fn pjrtClientBufferFromHostBuffer(args: [*c]c.PJRT_Client_BufferFromHostBuffer_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const client = clientFromC(args[0].client);
    const device = if (args[0].device) |dev| deviceFromC(dev) else &client.devices[0];
    const memory = if (args[0].memory) |mem| memoryFromC(mem) else device.default_memory;
    const shard_index = deviceIndex(client, device) orelse {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "buffer device does not belong to client");
    };
    const dims = args[0].dims[0..args[0].num_dims];
    const byte_size = denseByteSize(args[0].type, dims);
    const data = @as([*]const u8, @ptrCast(args[0].data))[0..byte_size];
    const trim = client.trimExecutableCacheForAllocation(memory, byte_size);
    const buffer = runtime.Buffer.initHostCopyForBackend(allocator, client.backend, runtimeTypeFromPjrt(args[0].type), dims, device, memory, shard_index, data) catch |err| {
        return switch (err) {
            error.InvalidArgument => makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "memory is not addressable by requested device"),
            else => makeError(c.PJRT_Error_Code_INTERNAL, "failed to create host buffer copy"),
        };
    };
    args[0].buffer = @ptrCast(buffer);
    args[0].done_with_host_buffer = eventCreatePending();
    eventSetReady(args[0].done_with_host_buffer);
    trace(
        "event=h2d bytes={d} dtype={s} rank={d} device={d} backend_storage={d} cache_trim_bytes={d} cache_trim_evictions={d} cache_pressure_remaining_bytes={d} cache_pressure_failed={d} elapsed_us={d}",
        .{
            byte_size,
            @tagName(buffer.element_type),
            dims.len,
            device.id,
            @intFromBool(buffer.hasBackendStorage()),
            trim.freed_bytes,
            trim.evicted_entries,
            trim.remaining_resident_bytes,
            @intFromBool(trim.still_over_capacity),
            elapsedUs(trace_start_ns),
        },
    );
    return null;
}

fn pjrtClientCreateUninitializedBuffer(args: [*c]c.PJRT_Client_CreateUninitializedBuffer_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const client = clientFromC(args[0].client);
    const memory = if (args[0].memory) |mem| memoryFromC(mem) else blk: {
        const device = if (args[0].device) |dev| deviceFromC(dev) else &client.devices[0];
        break :blk device.default_memory;
    };
    const device = if (args[0].device) |dev| deviceFromC(dev) else blk: {
        if (memory.addressable_devices.len == 0) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "memory is not addressable by any device");
        break :blk memory.addressable_devices[0];
    };
    const shard_index = deviceIndex(client, device) orelse {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "buffer device does not belong to client");
    };
    if (!memory.isAddressableBy(device)) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "memory is not addressable by requested device");
    }
    const dims = args[0].shape_dims[0..args[0].shape_num_dims];
    const byte_size = denseByteSize(args[0].shape_element_type, dims);
    const trim = client.trimExecutableCacheForAllocation(memory, byte_size);
    const buffer = runtime.Buffer.initDeviceAllocationForBackend(allocator, client.backend, runtimeTypeFromPjrt(args[0].shape_element_type), dims, device, memory, shard_index) catch |err| {
        return switch (err) {
            error.InvalidArgument => makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "memory is not addressable by requested device"),
            error.UnsupportedElementType, error.UnsupportedRuntimeFeature => makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "backend cannot allocate requested buffer type"),
            else => makeError(c.PJRT_Error_Code_INTERNAL, "failed to create device buffer"),
        };
    };
    args[0].buffer = @ptrCast(buffer);
    trace(
        "event=device_alloc bytes={d} dtype={s} rank={d} device={d} cache_trim_bytes={d} cache_trim_evictions={d} cache_pressure_remaining_bytes={d} cache_pressure_failed={d} elapsed_us={d}",
        .{
            byte_size,
            @tagName(buffer.element_type),
            dims.len,
            device.id,
            trim.freed_bytes,
            trim.evicted_entries,
            trim.remaining_resident_bytes,
            @intFromBool(trim.still_over_capacity),
            elapsedUs(trace_start_ns),
        },
    );
    return null;
}

fn pjrtClientCreateBuffersForAsyncHostToDevice(args: [*c]c.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const client = clientFromC(args[0].client);
    const memory = if (args[0].memory) |mem| memoryFromC(mem) else client.devices[0].default_memory;
    if (args[0].shape_specs == null and args[0].num_shape_specs != 0) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "shape specs are null");
    }
    const shape_specs = args[0].shape_specs[0..args[0].num_shape_specs];
    const manager = AsyncHostToDeviceTransferManager.create(client, memory, shape_specs) catch |err| {
        return switch (err) {
            error.InvalidArgument => makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "memory is not addressable by any device"),
            error.UnsupportedRuntimeFeature, error.UnsupportedElementType => makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "backend cannot allocate async transfer buffer"),
            else => makeError(c.PJRT_Error_Code_INTERNAL, "failed to create async host-to-device transfer manager"),
        };
    };
    args[0].transfer_manager = @ptrCast(manager);
    var total_bytes: usize = 0;
    for (manager.buffers) |buffer| total_bytes += buffer.byte_size;
    trace(
        "event=async_h2d_create buffers={d} bytes={d} device={d} memory={d} elapsed_us={d}",
        .{ manager.buffers.len, total_bytes, manager.device.id, manager.memory.id, elapsedUs(trace_start_ns) },
    );
    return null;
}

fn asyncH2DDestroy(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args[0].transfer_manager) |manager| asyncH2DManagerFromC(manager).deinit();
    return null;
}

fn asyncH2DTransferData(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const manager = asyncH2DManagerFromC(args[0].transfer_manager);
    const i = manager.index(args[0].buffer_index) catch {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async transfer buffer index is out of range");
    };
    if (args[0].offset < 0 or args[0].transfer_size < 0) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async transfer offset and size must be non-negative");
    }
    const offset: usize = @intCast(args[0].offset);
    const transfer_size: usize = @intCast(args[0].transfer_size);
    if (args[0].data == null and transfer_size != 0) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async transfer data is null");
    }
    const buffer_size = manager.bufferByteSize(i);
    if (offset > buffer_size or transfer_size > buffer_size - offset) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async transfer range exceeds buffer size");
    }

    if (transfer_size != 0) {
        const src = @as([*]const u8, @ptrCast(args[0].data))[0..transfer_size];
        if (manager.backend_transfers[i]) |transfer| {
            manager.client.backend.writeAsyncHostToDeviceTransfer(transfer, offset, src) catch |err| {
                return switch (err) {
                    error.BufferCopyFailed => makeError(c.PJRT_Error_Code_INTERNAL, "backend async transfer write failed"),
                    else => makeError(c.PJRT_Error_Code_INTERNAL, "failed to write backend async transfer"),
                };
            };
        } else {
            copyBytesWithIo(manager.staging[i][offset .. offset + transfer_size], src) catch {
                return makeError(c.PJRT_Error_Code_INTERNAL, "failed to stage async transfer data");
            };
        }
        manager.written[i] = @max(manager.written[i], offset + transfer_size);
    }

    args[0].done_with_h2d_transfer = eventCreatePending();
    if (args[0].done_with_h2d_transfer == null) {
        return makeError(c.PJRT_Error_Code_RESOURCE_EXHAUSTED, "failed to allocate async transfer event");
    }

    if (args[0].is_last_transfer) {
        if (manager.written[i] < buffer_size) {
            eventSetFailed(args[0].done_with_h2d_transfer, "async transfer completed before full buffer was written");
            manager.buffers[i].ready_event.setFailed("async transfer completed before full buffer was written");
            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async transfer completed before full buffer was written");
        }
        manager.finishBuffer(i) catch |err| {
            eventSetFailed(args[0].done_with_h2d_transfer, "failed to install async transfer buffer");
            manager.buffers[i].ready_event.setFailed("failed to install async transfer buffer");
            return switch (err) {
                error.ShapeMismatch => makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async transfer data does not match buffer shape"),
                error.BufferDeleted, error.BufferDonated => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "async transfer buffer is deleted or donated"),
                error.UnsupportedRuntimeFeature, error.UnsupportedElementType => makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "backend cannot import async transfer buffer"),
                else => makeError(c.PJRT_Error_Code_INTERNAL, "failed to install async transfer buffer"),
            };
        };
    }

    eventSetReady(args[0].done_with_h2d_transfer);
    trace(
        "event=async_h2d_transfer buffer={d} offset={d} bytes={d} last={d} elapsed_us={d}",
        .{ i, offset, transfer_size, @intFromBool(args[0].is_last_transfer), elapsedUs(trace_start_ns) },
    );
    return null;
}

fn asyncH2DTransferLiteral(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args) callconv(.c) ?*c.PJRT_Error {
    const manager = asyncH2DManagerFromC(args[0].transfer_manager);
    const i = manager.index(args[0].buffer_index) catch {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async transfer buffer index is out of range");
    };
    const dims = args[0].shape_dims[0..args[0].shape_num_dims];
    const byte_size = denseByteSize(args[0].shape_element_type, dims);
    if (byte_size != manager.bufferByteSize(i)) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async literal size does not match target buffer");
    }
    if (args[0].data == null and byte_size != 0) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async literal data is null");
    }
    if (byte_size != 0) {
        const src = @as([*]const u8, @ptrCast(args[0].data))[0..byte_size];
        if (manager.backend_transfers[i]) |transfer| {
            manager.client.backend.writeAsyncHostToDeviceTransfer(transfer, 0, src) catch {
                return makeError(c.PJRT_Error_Code_INTERNAL, "failed to write backend async transfer literal");
            };
        } else {
            copyBytesWithIo(manager.staging[i], src) catch {
                return makeError(c.PJRT_Error_Code_INTERNAL, "failed to stage async transfer literal");
            };
        }
        manager.written[i] = byte_size;
    }
    args[0].done_with_h2d_transfer = eventCreatePending();
    if (args[0].done_with_h2d_transfer == null) {
        return makeError(c.PJRT_Error_Code_RESOURCE_EXHAUSTED, "failed to allocate async transfer event");
    }
    manager.finishBuffer(i) catch |err| {
        eventSetFailed(args[0].done_with_h2d_transfer, "failed to install async transfer literal");
        manager.buffers[i].ready_event.setFailed("failed to install async transfer literal");
        return switch (err) {
            error.ShapeMismatch => makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async literal data does not match buffer shape"),
            error.BufferDeleted, error.BufferDonated => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "async literal buffer is deleted or donated"),
            error.UnsupportedRuntimeFeature, error.UnsupportedElementType => makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "backend cannot import async transfer literal"),
            else => makeError(c.PJRT_Error_Code_INTERNAL, "failed to install async transfer literal"),
        };
    };
    eventSetReady(args[0].done_with_h2d_transfer);
    return null;
}

fn asyncH2DRetrieveBuffer(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args) callconv(.c) ?*c.PJRT_Error {
    const manager = asyncH2DManagerFromC(args[0].transfer_manager);
    const i = manager.index(args[0].buffer_index) catch {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async transfer buffer index is out of range");
    };
    manager.retrieved[i] = true;
    args[0].buffer_out = @ptrCast(manager.buffers[i]);
    return null;
}

fn asyncH2DDevice(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_Device_Args) callconv(.c) ?*c.PJRT_Error {
    const manager = asyncH2DManagerFromC(args[0].transfer_manager);
    args[0].device_out = @ptrCast(manager.device);
    return null;
}

fn asyncH2DBufferCount(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args) callconv(.c) ?*c.PJRT_Error {
    const manager = asyncH2DManagerFromC(args[0].transfer_manager);
    args[0].buffer_count = manager.buffers.len;
    return null;
}

fn asyncH2DBufferSize(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args) callconv(.c) ?*c.PJRT_Error {
    const manager = asyncH2DManagerFromC(args[0].transfer_manager);
    const i = manager.index(args[0].buffer_index) catch {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async transfer buffer index is out of range");
    };
    args[0].buffer_size = manager.bufferByteSize(i);
    return null;
}

fn asyncH2DSetBufferError(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args) callconv(.c) ?*c.PJRT_Error {
    const manager = asyncH2DManagerFromC(args[0].transfer_manager);
    const i = manager.index(args[0].buffer_index) catch {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "async transfer buffer index is out of range");
    };
    const message = bytesFromC(args[0].error_message, args[0].error_message_size) orelse "async transfer buffer failed";
    manager.buffers[i].ready_event.setFailed(message);
    return null;
}

fn asyncH2DAddMetadata(_: [*c]c.PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn pjrtClientDmaMap(_: [*c]c.PJRT_Client_DmaMap_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn pjrtClientDmaUnmap(_: [*c]c.PJRT_Client_DmaUnmap_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn topologyDescriptionCreate(_: [*c]c.PJRT_TopologyDescription_Create_Args) callconv(.c) ?*c.PJRT_Error {
    return unimplemented("standalone topology creation is not implemented yet");
}

fn topologyDescriptionDestroy(_: [*c]c.PJRT_TopologyDescription_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn topologyDescriptionPlatformName(args: [*c]c.PJRT_TopologyDescription_PlatformName_Args) callconv(.c) ?*c.PJRT_Error {
    _ = topologyFromC(args[0].topology);
    args[0].platform_name = platform_name;
    args[0].platform_name_size = platform_name.len;
    return null;
}

fn topologyDescriptionPlatformVersion(args: [*c]c.PJRT_TopologyDescription_PlatformVersion_Args) callconv(.c) ?*c.PJRT_Error {
    _ = topologyFromC(args[0].topology);
    args[0].platform_version = platform_version;
    args[0].platform_version_size = platform_version.len;
    return null;
}

fn topologyDescriptionGetDeviceDescriptions(args: [*c]c.PJRT_TopologyDescription_GetDeviceDescriptions_Args) callconv(.c) ?*c.PJRT_Error {
    const client = topologyFromC(args[0].topology);
    args[0].descriptions = @ptrCast(client.device_handles.ptr);
    args[0].num_descriptions = client.device_handles.len;
    return null;
}

fn topologySerializedDelete(serialized_topology: ?*c.PJRT_SerializedTopology) callconv(.c) void {
    if (serialized_topology) |opaque_topology| {
        const topology: *SerializedTopology = @ptrCast(@alignCast(opaque_topology));
        allocator.free(topology.bytes);
        allocator.destroy(topology);
    }
}

fn topologyDescriptionSerialize(args: [*c]c.PJRT_TopologyDescription_Serialize_Args) callconv(.c) ?*c.PJRT_Error {
    const client = topologyFromC(args[0].topology);
    var writer = std.Io.Writer.Allocating.init(allocator);
    defer writer.deinit();
    writer.writer.print("platform={s};version={s};devices={d}", .{ platform_name, platform_version, client.devices.len }) catch {
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to serialize topology");
    };
    for (client.devices) |device| {
        writer.writer.print(";device={d}:{d}:{d}:{s}", .{ device.id, device.local_hardware_id, device.process_index, device.name }) catch {
            return makeError(c.PJRT_Error_Code_INTERNAL, "failed to serialize topology");
        };
    }

    const topology = allocator.create(SerializedTopology) catch {
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate serialized topology");
    };
    topology.* = .{ .bytes = writer.toOwnedSlice() catch {
        allocator.destroy(topology);
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate serialized topology bytes");
    } };

    args[0].serialized_bytes = topology.bytes.ptr;
    args[0].serialized_bytes_size = topology.bytes.len;
    args[0].serialized_topology = @ptrCast(topology);
    args[0].serialized_topology_deleter = topologySerializedDelete;
    return null;
}

fn topologyDescriptionDeserialize(_: [*c]c.PJRT_TopologyDescription_Deserialize_Args) callconv(.c) ?*c.PJRT_Error {
    return unimplemented("topology deserialization is not implemented yet");
}

fn topologyDescriptionAttributes(args: [*c]c.PJRT_TopologyDescription_Attributes_Args) callconv(.c) ?*c.PJRT_Error {
    _ = topologyFromC(args[0].topology);
    args[0].attributes = null;
    args[0].num_attributes = 0;
    return null;
}

fn topologyDescriptionFingerprint(args: [*c]c.PJRT_TopologyDescription_Fingerprint_Args) callconv(.c) ?*c.PJRT_Error {
    const client = topologyFromC(args[0].topology);
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(platform_name);
    hasher.update(platform_version);
    for (client.devices) |device| {
        hasher.update(std.mem.asBytes(&device.id));
        hasher.update(std.mem.asBytes(&device.local_hardware_id));
        hasher.update(std.mem.asBytes(&device.process_index));
        hasher.update(device.name);
    }
    args[0].fingerprint = hasher.final();
    return null;
}

fn deviceDescriptionId(args: [*c]c.PJRT_DeviceDescription_Id_Args) callconv(.c) ?*c.PJRT_Error {
    const device: *runtime.Device = @ptrCast(@alignCast(args[0].device_description.?));
    args[0].id = device.id;
    return null;
}

fn deviceDescriptionProcessIndex(args: [*c]c.PJRT_DeviceDescription_ProcessIndex_Args) callconv(.c) ?*c.PJRT_Error {
    const device: *runtime.Device = @ptrCast(@alignCast(args[0].device_description.?));
    args[0].process_index = device.process_index;
    return null;
}

fn deviceDescriptionAttributes(args: [*c]c.PJRT_DeviceDescription_Attributes_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].attributes = null;
    args[0].num_attributes = 0;
    return null;
}

fn deviceDescriptionKind(args: [*c]c.PJRT_DeviceDescription_Kind_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].device_kind = device_kind;
    args[0].device_kind_size = device_kind.len;
    return null;
}

fn deviceDescriptionDebugString(args: [*c]c.PJRT_DeviceDescription_DebugString_Args) callconv(.c) ?*c.PJRT_Error {
    const device: *runtime.Device = @ptrCast(@alignCast(args[0].device_description.?));
    args[0].debug_string = device.debug_string.ptr;
    args[0].debug_string_size = device.debug_string.len;
    return null;
}

fn deviceDescriptionToString(args: [*c]c.PJRT_DeviceDescription_ToString_Args) callconv(.c) ?*c.PJRT_Error {
    return deviceDescriptionDebugString(@ptrCast(args));
}

fn deviceGetDescription(args: [*c]c.PJRT_Device_GetDescription_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].device_description = @ptrCast(args[0].device);
    return null;
}

fn deviceIsAddressable(args: [*c]c.PJRT_Device_IsAddressable_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].is_addressable = deviceFromC(args[0].device).addressable;
    return null;
}

fn deviceLocalHardwareId(args: [*c]c.PJRT_Device_LocalHardwareId_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].local_hardware_id = deviceFromC(args[0].device).local_hardware_id;
    return null;
}

fn deviceAddressableMemories(args: [*c]c.PJRT_Device_AddressableMemories_Args) callconv(.c) ?*c.PJRT_Error {
    const device = deviceFromC(args[0].device);
    args[0].memories = @ptrCast(device.addressable_memories.ptr);
    args[0].num_memories = device.addressable_memories.len;
    return null;
}

fn deviceDefaultMemory(args: [*c]c.PJRT_Device_DefaultMemory_Args) callconv(.c) ?*c.PJRT_Error {
    const device = deviceFromC(args[0].device);
    args[0].memory = @ptrCast(device.default_memory);
    return null;
}

fn deviceMemoryStats(args: [*c]c.PJRT_Device_MemoryStats_Args) callconv(.c) ?*c.PJRT_Error {
    const device = deviceFromC(args[0].device);
    const stats = device.default_memory.stats;
    args[0].bytes_in_use = clampI64(stats.totalBytesInUse());
    args[0].peak_bytes_in_use = clampI64(stats.peakTotalBytesInUse());
    args[0].peak_bytes_in_use_is_set = true;
    args[0].num_allocs = clampI64(stats.live_allocs +| stats.executable_cache_resident_entries);
    args[0].num_allocs_is_set = true;
    args[0].largest_alloc_size = clampI64(@max(stats.largest_alloc_size, stats.executable_cache_largest_resident_bytes));
    args[0].largest_alloc_size_is_set = true;
    if (stats.capacity_bytes != 0) {
        args[0].bytes_limit = clampI64(stats.capacity_bytes);
        args[0].bytes_limit_is_set = true;
        args[0].largest_free_block_bytes = if (stats.totalBytesInUse() >= stats.capacity_bytes)
            0
        else
            clampI64(stats.capacity_bytes - stats.totalBytesInUse());
        args[0].largest_free_block_bytes_is_set = true;
        args[0].bytes_reservable_limit = args[0].largest_free_block_bytes;
        args[0].bytes_reservable_limit_is_set = true;
    }
    return null;
}

fn clampI64(value: u64) i64 {
    return @intCast(@min(value, @as(u64, @intCast(std.math.maxInt(i64)))));
}

fn initDeviceAttrString(attr: *c.PJRT_NamedValue, name: []const u8, value: []const u8) void {
    attr.* = std.mem.zeroes(c.PJRT_NamedValue);
    attr.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attr.name = name.ptr;
    attr.name_size = name.len;
    attr.type = c.PJRT_NamedValue_kString;
    attr.unnamed_0.string_value = value.ptr;
    attr.value_size = value.len;
}

fn initDeviceAttrInt64(attr: *c.PJRT_NamedValue, name: []const u8, value: i64) void {
    attr.* = std.mem.zeroes(c.PJRT_NamedValue);
    attr.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attr.name = name.ptr;
    attr.name_size = name.len;
    attr.type = c.PJRT_NamedValue_kInt64;
    attr.unnamed_0.int64_value = value;
    attr.value_size = 1;
}

fn initDeviceAttrBool(attr: *c.PJRT_NamedValue, name: []const u8, value: bool) void {
    attr.* = std.mem.zeroes(c.PJRT_NamedValue);
    attr.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attr.name = name.ptr;
    attr.name_size = name.len;
    attr.type = c.PJRT_NamedValue_kBool;
    attr.unnamed_0.bool_value = value;
    attr.value_size = 1;
}

fn deviceAttributesDelete(device_attributes: ?*c.PJRT_Device_Attributes) callconv(.c) void {
    if (device_attributes) |opaque_attrs| {
        const owned: *DeviceAttributes = @ptrCast(@alignCast(opaque_attrs));
        allocator.destroy(owned);
    }
}

fn deviceGetAttributes(args: [*c]c.PJRT_Device_GetAttributes_Args) callconv(.c) ?*c.PJRT_Error {
    const device = deviceFromC(args[0].device);
    const owned = allocator.create(DeviceAttributes) catch {
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate device attributes");
    };
    owned.* = .{ .attrs = std.mem.zeroes([5]c.PJRT_NamedValue) };

    initDeviceAttrString(&owned.attrs[0], "device_name", device.name);
    initDeviceAttrInt64(&owned.attrs[1], "pjrtx_registry_id", clampI64(device.registry_id));
    initDeviceAttrInt64(&owned.attrs[2], "pjrtx_recommended_working_set_size", clampI64(device.memory_bytes));
    initDeviceAttrInt64(&owned.attrs[3], "pjrtx_has_unified_memory", if (device.has_unified_memory) 1 else 0);
    initDeviceAttrInt64(&owned.attrs[4], "pjrtx_default_memory_id", device.default_memory_id);

    args[0].attributes = &owned.attrs;
    args[0].num_attributes = owned.attrs.len;
    args[0].device_attributes = @ptrCast(owned);
    args[0].attributes_deleter = deviceAttributesDelete;
    return null;
}

fn memoryId(args: [*c]c.PJRT_Memory_Id_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].id = memoryFromC(args[0].memory).id;
    return null;
}

fn memoryKind(args: [*c]c.PJRT_Memory_Kind_Args) callconv(.c) ?*c.PJRT_Error {
    const memory = memoryFromC(args[0].memory);
    const text = @tagName(memory.kind);
    args[0].kind = text.ptr;
    args[0].kind_size = text.len;
    return null;
}

fn memoryDebugString(args: [*c]c.PJRT_Memory_DebugString_Args) callconv(.c) ?*c.PJRT_Error {
    const memory = memoryFromC(args[0].memory);
    args[0].debug_string = memory.debug_string.ptr;
    args[0].debug_string_size = memory.debug_string.len;
    return null;
}

fn memoryToString(args: [*c]c.PJRT_Memory_ToString_Args) callconv(.c) ?*c.PJRT_Error {
    const memory = memoryFromC(args[0].memory);
    args[0].to_string = memory.debug_string.ptr;
    args[0].to_string_size = memory.debug_string.len;
    return null;
}

fn memoryAddressableByDevices(args: [*c]c.PJRT_Memory_AddressableByDevices_Args) callconv(.c) ?*c.PJRT_Error {
    const memory = memoryFromC(args[0].memory);
    args[0].devices = @ptrCast(memory.addressable_devices.ptr);
    args[0].num_devices = memory.addressable_devices.len;
    return null;
}

fn loadedExecutableDestroy(args: [*c]c.PJRT_LoadedExecutable_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    executableFromC(args[0].executable).deinit();
    return null;
}

fn loadedExecutableGetExecutable(args: [*c]c.PJRT_LoadedExecutable_GetExecutable_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].executable = @ptrCast(args[0].loaded_executable);
    return null;
}

fn loadedExecutableAddressableDevices(args: [*c]c.PJRT_LoadedExecutable_AddressableDevices_Args) callconv(.c) ?*c.PJRT_Error {
    const executable = executableFromC(args[0].executable);
    const num_devices = @min(executable.plan.options.numDevices(), executable.client.device_handles.len);
    args[0].addressable_devices = @ptrCast(executable.client.device_handles[0..num_devices].ptr);
    args[0].num_addressable_devices = num_devices;
    return null;
}

fn loadedExecutableAddressableDeviceLogicalIds(args: [*c]c.PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args) callconv(.c) ?*c.PJRT_Error {
    const executable = executableFromC(args[0].executable);
    args[0].addressable_device_logical_ids = executable.logical_ids.ptr;
    args[0].num_addressable_device_logical_ids = executable.logical_ids.len;
    return null;
}

fn deviceAssignmentSerializedDeleter(_: ?*c.PJRT_DeviceAssignmentSerialized) callconv(.c) void {}

fn loadedExecutableGetDeviceAssignment(args: [*c]c.PJRT_LoadedExecutable_GetDeviceAssignment_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].serialized_bytes = null;
    args[0].serialized_bytes_size = 0;
    args[0].serialized_device_assignment = null;
    args[0].serialized_device_assignment_deleter = deviceAssignmentSerializedDeleter;
    return null;
}

fn loadedExecutableDelete(args: [*c]c.PJRT_LoadedExecutable_Delete_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const executable = executableFromC(args[0].executable);
    executable.deleted = true;
    executable.releaseGraph();
    trace(
        "event=loaded_executable_delete cache_hits_total={d} cache_misses_total={d} backend_cache_entries={d} backend_cache_bytes={d} backend_cache_peak_bytes={d} backend_cache_evictions={d} backend_cache_evicted_bytes={d} backend_cache_compile_samples={d} backend_cache_compile_us_total={d} backend_cache_compile_us_peak={d} elapsed_us={d}",
        .{
            executable.client.executable_cache.stats.hits,
            executable.client.executable_cache.stats.misses,
            executable.client.executable_cache.stats.resident_entries,
            executable.client.executable_cache.stats.resident_bytes,
            executable.client.executable_cache.stats.peak_resident_bytes,
            executable.client.executable_cache.stats.evictions,
            executable.client.executable_cache.stats.evicted_resident_bytes,
            executable.client.executable_cache.stats.compile_latency_samples,
            executable.client.executable_cache.stats.compile_latency_us_total,
            executable.client.executable_cache.stats.compile_latency_us_peak,
            elapsedUs(trace_start_ns),
        },
    );
    return null;
}

fn loadedExecutableIsDeleted(args: [*c]c.PJRT_LoadedExecutable_IsDeleted_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].is_deleted = executableFromC(args[0].executable).deleted;
    return null;
}

fn graphExecuteError(err: runtime.GraphExecuteError) ?*c.PJRT_Error {
    return switch (err) {
        error.OutOfMemory => makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable graph execution state"),
        error.InvalidArgument => makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "invalid executable graph arguments or device assignment"),
        error.UnsupportedElementType => makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "executable graph contains an operation unsupported for this element type"),
        error.ShapeMismatch => makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "executable graph shape validation failed during execution"),
        error.UnsupportedRuntimeFeature => makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "executable graph is not fully lowered to the MLX backend executable"),
        error.BufferDeleted => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "execute attempted to use a deleted buffer"),
        error.BufferDonated => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "execute attempted to use a donated buffer"),
        error.BufferNotReady => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "execute attempted to use a buffer that is not ready"),
        error.BufferReadinessFailed => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "execute attempted to use a buffer with failed readiness"),
        error.Internal => makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute executable graph"),
    };
}

fn planDonatesParameter(plan: *const compiler.ExecutablePlan, parameter_index: usize) bool {
    for (plan.donated_parameter_indices) |candidate| {
        if (candidate == parameter_index) return true;
    }
    return false;
}

fn executeOptionsKeepParameter(options: ?*c.PJRT_ExecuteOptions, parameter_index: usize) bool {
    const execute_options = options orelse return false;
    if (execute_options.non_donatable_input_indices == null) return false;
    for (0..execute_options.num_non_donatable_input_indices) |index| {
        const non_donatable = execute_options.non_donatable_input_indices[index];
        if (non_donatable >= 0 and @as(usize, @intCast(non_donatable)) == parameter_index) return true;
    }
    return false;
}

fn validateExecuteDonationOptions(options: ?*c.PJRT_ExecuteOptions, num_args: usize) ?*c.PJRT_Error {
    const execute_options = options orelse return null;
    if (execute_options.num_non_donatable_input_indices != 0 and execute_options.non_donatable_input_indices == null) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "non_donatable_input_indices count requires a non-null index list");
    }
    for (0..execute_options.num_non_donatable_input_indices) |index| {
        const non_donatable = execute_options.non_donatable_input_indices[index];
        if (non_donatable < 0 or @as(usize, @intCast(non_donatable)) >= num_args) {
            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "non_donatable_input_indices contains an out-of-range argument index");
        }
    }
    return null;
}

fn validateExecuteDonationAliasHazards(
    executable: *const Executable,
    options: ?*c.PJRT_ExecuteOptions,
    execute_args: c.PJRT_LoadedExecutable_Execute_Args,
) ?*c.PJRT_Error {
    for (0..execute_args.num_devices) |donor_device_index| {
        for (0..execute_args.num_args) |donor_argument_index| {
            if (!planDonatesParameter(executable.plan, donor_argument_index)) continue;
            if (executeOptionsKeepParameter(options, donor_argument_index)) continue;
            const donor_buffer = execute_args.argument_lists[donor_device_index][donor_argument_index].?;
            for (0..execute_args.num_devices) |other_device_index| {
                for (0..execute_args.num_args) |other_argument_index| {
                    if (donor_device_index == other_device_index and donor_argument_index == other_argument_index) continue;
                    if (execute_args.argument_lists[other_device_index][other_argument_index] == donor_buffer) {
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "donated execute argument aliases another argument");
                    }
                }
            }
        }
    }
    return null;
}

fn appendUniqueDonatedArgument(list: *std.ArrayList(*runtime.Buffer), buffer: *runtime.Buffer) !void {
    for (list.items) |existing| {
        if (existing == buffer) return;
    }
    try list.append(allocator, buffer);
}

fn validateExecuteLists(executable: *const Executable, execute_args: c.PJRT_LoadedExecutable_Execute_Args) ?*c.PJRT_Error {
    const expected_args = executable.plan.parameter_shardings.len;
    const expected_outputs = executable.plan.output_ids.len;
    if (execute_args.num_devices == 0) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx execute requires at least one device");
    if (execute_args.num_devices > executable.graph.device_ids.len) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx execute requested more devices than the executable graph contains");
    if (execute_args.num_args != expected_args) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx execute argument count does not match executable parameters");
    if (expected_args != 0 and execute_args.argument_lists == null) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx execute requires non-null argument_lists for executable parameters");
    if (expected_outputs != 0 and execute_args.output_lists == null) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx execute requires non-null output_lists for executable outputs");
    for (0..execute_args.num_devices) |device_index| {
        if (expected_args != 0 and execute_args.argument_lists[device_index] == null) {
            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx execute requires a non-null argument list for every device");
        }
        if (expected_outputs != 0 and execute_args.output_lists[device_index] == null) {
            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx execute requires a non-null output list for every device");
        }
        for (0..expected_args) |argument_index| {
            if (execute_args.argument_lists[device_index][argument_index] == null) {
                return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx execute argument list contains a null buffer");
            }
        }
    }
    return null;
}

fn clearExecuteResults(execute_args: c.PJRT_LoadedExecutable_Execute_Args, output_count: usize) void {
    for (0..execute_args.num_devices) |device_index| {
        if (execute_args.output_lists != null) {
            const outputs = execute_args.output_lists[device_index];
            if (outputs != null) {
                for (0..output_count) |output_index| outputs[output_index] = null;
            }
        }
        if (execute_args.device_complete_events) |events| events[device_index] = null;
    }
}

fn cleanupExecuteResults(execute_args: c.PJRT_LoadedExecutable_Execute_Args, output_count: usize) void {
    for (0..execute_args.num_devices) |device_index| {
        if (execute_args.output_lists != null) {
            const outputs = execute_args.output_lists[device_index];
            if (outputs != null) {
                for (0..output_count) |output_index| {
                    if (outputs[output_index]) |output| {
                        bufferFromC(output).deinit();
                        outputs[output_index] = null;
                    }
                }
            }
        }
        if (execute_args.device_complete_events) |events| {
            if (events[device_index]) |event| {
                var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
                event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
                event_destroy_args.event = event;
                _ = pjrtEventDestroy(&event_destroy_args);
                events[device_index] = null;
            }
        }
    }
}

fn cleanupUnassignedExecuteOutputs(outputs: []const *runtime.Buffer, assigned_count: usize) void {
    if (assigned_count >= outputs.len) return;
    for (outputs[assigned_count..]) |output| output.deinit();
}

test "execute cleanup destroys only unassigned runtime outputs" {
    const test_allocator = std.testing.allocator;
    const client = try runtime.Client.init(test_allocator, backend_registry.create(.metal_mlx), 1);
    defer client.deinit();

    const dims = [_]i64{4};
    const first_data = [_]u8{ 1, 2, 3, 4 };
    const second_data = [_]u8{ 5, 6, 7, 8 };
    const first = try runtime.Buffer.initHostCopyForBackend(test_allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &first_data);
    defer first.deinit();
    const second = try runtime.Buffer.initHostCopyForBackend(test_allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &second_data);

    const bytes_before_cleanup = client.memories[0].stats.bytes_in_use;
    const second_bytes: u64 = @intCast(second.byte_size);
    var outputs = [_]*runtime.Buffer{ first, second };
    cleanupUnassignedExecuteOutputs(&outputs, 1);

    try first.ensureUsable();
    try std.testing.expectEqual(bytes_before_cleanup - second_bytes, client.memories[0].stats.bytes_in_use);
}

fn loadedExecutableExecute(args: [*c]c.PJRT_LoadedExecutable_Execute_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const executable = executableFromC(args[0].executable);
    if (executable.deleted) return makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "loaded executable has been deleted");
    if (validateExecuteLists(executable, args[0])) |err| return err;
    if (validateExecuteDonationOptions(args[0].options, args[0].num_args)) |err| return err;
    if (validateExecuteDonationAliasHazards(executable, args[0].options, args[0])) |err| return err;
    const output_count = executable.plan.output_ids.len;
    clearExecuteResults(args[0], output_count);
    var donated_arguments = std.ArrayList(*runtime.Buffer).empty;
    defer donated_arguments.deinit(allocator);
    var total_outputs: usize = 0;
    var backend_candidate = executable.graph.backend_executable != null;
    var execute_cache_trim_bytes: u64 = 0;
    var execute_cache_trim_evictions: u64 = 0;
    var execute_cache_pressure_remaining_bytes: u64 = 0;
    var execute_cache_pressure_failed = false;
    var backend_completion_pending_count: u64 = 0;
    for (0..args[0].num_devices) |device_index| {
        const arguments = allocator.alloc(*runtime.Buffer, args[0].num_args) catch {
            cleanupExecuteResults(args[0], output_count);
            return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable graph argument list");
        };
        defer allocator.free(arguments);
        for (arguments, 0..) |*argument, argument_index| {
            argument.* = bufferFromC(args[0].argument_lists[device_index][argument_index]);
            backend_candidate = backend_candidate and argument.*.hasBackendStorage();
            if (planDonatesParameter(executable.plan, argument_index) and !executeOptionsKeepParameter(args[0].options, argument_index)) {
                appendUniqueDonatedArgument(&donated_arguments, argument.*) catch {
                    cleanupExecuteResults(args[0], output_count);
                    return makeError(c.PJRT_Error_Code_INTERNAL, "failed to record donated executable argument");
                };
            }
        }

        const execute_result = executable.graph.executeDevice(allocator, executable.client, executable.plan, device_index, arguments) catch |err| {
            cleanupExecuteResults(args[0], output_count);
            return graphExecuteError(err);
        };
        execute_cache_trim_bytes +|= executable.graph.last_execute_cache_trim.freed_bytes;
        execute_cache_trim_evictions +|= executable.graph.last_execute_cache_trim.evicted_entries;
        execute_cache_pressure_remaining_bytes +|= executable.graph.last_execute_cache_trim.remaining_resident_bytes;
        execute_cache_pressure_failed = execute_cache_pressure_failed or executable.graph.last_execute_cache_trim.still_over_capacity;
        if (executable.graph.last_backend_completion.kind == .pending) backend_completion_pending_count += 1;
        const outputs = execute_result.outputs;
        defer allocator.free(outputs);
        var assigned_outputs: usize = 0;
        var outputs_transferred = false;
        defer if (!outputs_transferred) cleanupUnassignedExecuteOutputs(outputs, assigned_outputs);
        var stack_completion_event = execute_result.completion_event;
        const completion_event = if (args[0].device_complete_events) |events| blk: {
            events[device_index] = eventCreateFromRuntime(execute_result.completion_event) orelse {
                cleanupExecuteResults(args[0], output_count);
                return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate execute completion event");
            };
            break :blk eventFromC(events[device_index].?);
        } else &stack_completion_event;
        total_outputs += outputs.len;
        for (outputs, 0..) |output, output_index| {
            args[0].output_lists[device_index][output_index] = @ptrCast(output);
            assigned_outputs = output_index + 1;
            output.chainReadyAfter(completion_event) catch {
                cleanupExecuteResults(args[0], output_count);
                return makeError(c.PJRT_Error_Code_RESOURCE_EXHAUSTED, "too many output readiness dependencies for execute completion event");
            };
        }
        outputs_transferred = true;
    }
    for (donated_arguments.items) |argument| argument.markDonated();
    const backend_stats = executable.graph.backendExecutableStats() orelse backend_api.ExecutableStats{};
    trace(
        "event=execute devices={d} args={d} outputs={d} backend_candidate={d} backend_executes={d} compiled_program_executes={d} captured_program_executes={d} captured_dynamic_inputs={d} captured_static_inputs={d} donation_alias_outputs={d} donation_alias_bytes={d} last_device_index={d} last_local_hardware_id={d} resident_constants={d} resident_constant_bytes={d} fusion_groups={d} fusion_group_executes={d} materialization_evals={d} materialization_buffers={d} released_intermediates={d} borrowed_constant_nodes={d} cache_trim_bytes={d} cache_trim_evictions={d} cache_pressure_remaining_bytes={d} cache_pressure_failed={d} backend_completion_pending={d} elapsed_us={d}",
        .{
            args[0].num_devices,
            args[0].num_args,
            total_outputs,
            @intFromBool(backend_candidate),
            backend_stats.execute_count,
            backend_stats.compiled_program_execute_count,
            backend_stats.captured_program_execute_count,
            backend_stats.captured_program_dynamic_input_count,
            backend_stats.captured_program_captured_input_count,
            backend_stats.donation_alias_output_count,
            backend_stats.donation_alias_output_bytes,
            backend_stats.last_execute_device_index,
            backend_stats.last_execute_local_hardware_id,
            backend_stats.resident_constant_count,
            backend_stats.resident_constant_bytes,
            backend_stats.program_fusion_group_count,
            backend_stats.fusion_group_execute_count,
            backend_stats.materialization_eval_count,
            backend_stats.materialization_eval_buffer_count,
            backend_stats.released_intermediate_count,
            backend_stats.borrowed_constant_nodes,
            execute_cache_trim_bytes,
            execute_cache_trim_evictions,
            execute_cache_pressure_remaining_bytes,
            @intFromBool(execute_cache_pressure_failed),
            backend_completion_pending_count,
            elapsedUs(trace_start_ns),
        },
    );
    trace(
        "event=execute_backend_profile backend_executes={d} execute_us_total={d} execute_us_peak={d} schedule_us_total={d} schedule_us_peak={d} node_us_total={d} node_us_peak={d} fusion_us_total={d} fusion_us_peak={d} materialization_us_total={d} materialization_us_peak={d} output_clone_us_total={d} output_clone_us_peak={d}",
        .{
            backend_stats.execute_count,
            backend_stats.execute_wall_us_total,
            backend_stats.execute_wall_us_peak,
            backend_stats.schedule_us_total,
            backend_stats.schedule_us_peak,
            backend_stats.node_us_total,
            backend_stats.node_us_peak,
            backend_stats.fusion_group_us_total,
            backend_stats.fusion_group_us_peak,
            backend_stats.materialization_eval_us_total,
            backend_stats.materialization_eval_us_peak,
            backend_stats.output_clone_us_total,
            backend_stats.output_clone_us_peak,
        },
    );
    return null;
}

fn executableName(args: [*c]c.PJRT_Executable_Name_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].executable_name = executable.name.ptr;
    args[0].executable_name_size = executable.name.len;
    return null;
}

fn executableDestroy(args: [*c]c.PJRT_Executable_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    _ = args;
    return null;
}

fn executableNumReplicas(args: [*c]c.PJRT_Executable_NumReplicas_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].num_replicas = @intCast(executable.plan.options.num_replicas);
    return null;
}

fn executableNumPartitions(args: [*c]c.PJRT_Executable_NumPartitions_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].num_partitions = @intCast(executable.plan.options.num_partitions);
    return null;
}

fn executableNumOutputs(args: [*c]c.PJRT_Executable_NumOutputs_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].num_outputs = executable.plan.output_shardings.len;
    return null;
}

fn executableOptimizedProgram(args: [*c]c.PJRT_Executable_OptimizedProgram_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    const program = args[0].program;
    program[0].format = "mlir";
    program[0].format_size = "mlir".len;
    program[0].code_size = executable.optimized_program.len;
    if (program[0].code) |code| {
        @memcpy(code[0..executable.optimized_program.len], executable.optimized_program);
    }
    return null;
}

fn executableFingerprint(args: [*c]c.PJRT_Executable_Fingerprint_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].executable_fingerprint = executable.fingerprint.ptr;
    args[0].executable_fingerprint_size = executable.fingerprint.len;
    return null;
}

fn executableParameterMemoryKinds(args: [*c]c.PJRT_Executable_ParameterMemoryKinds_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].num_parameters = executable.parameter_memory_kinds.len;
    args[0].memory_kinds = @ptrCast(executable.parameter_memory_kinds.ptr);
    args[0].memory_kind_sizes = executable.parameter_memory_kind_sizes.ptr;
    return null;
}

fn executableOutputMemoryKinds(args: [*c]c.PJRT_Executable_OutputMemoryKinds_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].num_outputs = executable.output_memory_kinds.len;
    args[0].memory_kinds = @ptrCast(executable.output_memory_kinds.ptr);
    args[0].memory_kind_sizes = executable.output_memory_kind_sizes.ptr;
    return null;
}

fn loadedExecutableFingerprint(args: [*c]c.PJRT_LoadedExecutable_Fingerprint_Args) callconv(.c) ?*c.PJRT_Error {
    const executable = executableFromC(args[0].executable);
    args[0].executable_fingerprint = executable.fingerprint.ptr;
    args[0].executable_fingerprint_size = executable.fingerprint.len;
    return null;
}

fn bufferDestroy(args: [*c]c.PJRT_Buffer_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args[0].buffer) |buffer| bufferFromC(buffer).deinit();
    return null;
}

fn bufferElementType(args: [*c]c.PJRT_Buffer_ElementType_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].type = pjrtTypeFromRuntime(bufferFromC(args[0].buffer).element_type);
    return null;
}

fn bufferDimensions(args: [*c]c.PJRT_Buffer_Dimensions_Args) callconv(.c) ?*c.PJRT_Error {
    const buffer = bufferFromC(args[0].buffer);
    args[0].dims = buffer.dims.ptr;
    args[0].num_dims = buffer.dims.len;
    return null;
}

fn bufferOnDeviceSize(args: [*c]c.PJRT_Buffer_OnDeviceSizeInBytes_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].on_device_size_in_bytes = bufferFromC(args[0].buffer).byte_size;
    return null;
}

fn bufferDelete(args: [*c]c.PJRT_Buffer_Delete_Args) callconv(.c) ?*c.PJRT_Error {
    bufferFromC(args[0].buffer).markDeleted();
    return null;
}

fn bufferIsDeleted(args: [*c]c.PJRT_Buffer_IsDeleted_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].is_deleted = bufferFromC(args[0].buffer).deleted;
    return null;
}

fn bufferIsOnCpu(args: [*c]c.PJRT_Buffer_IsOnCpu_Args) callconv(.c) ?*c.PJRT_Error {
    _ = bufferFromC(args[0].buffer);
    args[0].is_on_cpu = false;
    return null;
}

fn bufferDynamicDimensionIndices(args: [*c]c.PJRT_Buffer_DynamicDimensionIndices_Args) callconv(.c) ?*c.PJRT_Error {
    _ = bufferFromC(args[0].buffer);
    args[0].dynamic_dim_indices = null;
    args[0].num_dynamic_dims = 0;
    return null;
}

fn bufferToHost(args: [*c]c.PJRT_Buffer_ToHostBuffer_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const buffer = bufferFromC(args[0].src);
    buffer.ensureUsable() catch return makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "buffer has been deleted or donated");
    if (args[0].dst == null) {
        args[0].dst_size = buffer.byte_size;
        return null;
    }
    buffer.ensureReady() catch |err| return switch (err) {
        error.BufferNotReady => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "buffer is not ready"),
        error.BufferReadinessFailed => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "buffer readiness failed"),
    };
    if (args[0].dst_size < buffer.byte_size) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "destination buffer is too small");
    buffer.copyToHost(@as([*]u8, @ptrCast(args[0].dst))[0..buffer.byte_size]) catch |err| {
        return switch (err) {
            error.DestinationTooSmall => makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "destination buffer is too small"),
            error.BufferDeleted, error.BufferDonated => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "buffer has been deleted or donated"),
            error.BufferNotReady => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "buffer is not ready"),
            error.BufferReadinessFailed => makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, "buffer readiness failed"),
            else => makeError(c.PJRT_Error_Code_INTERNAL, "failed to copy buffer to host"),
        };
    };
    args[0].event = eventCreatePending();
    eventSetReady(args[0].event);
    trace(
        "event=d2h bytes={d} dtype={s} rank={d} device={d} backend_storage={d} elapsed_us={d}",
        .{
            buffer.byte_size,
            @tagName(buffer.element_type),
            buffer.dims.len,
            buffer.device_id,
            @intFromBool(buffer.hasBackendStorage()),
            elapsedUs(trace_start_ns),
        },
    );
    return null;
}

fn bufferReadyEvent(args: [*c]c.PJRT_Buffer_ReadyEvent_Args) callconv(.c) ?*c.PJRT_Error {
    const buffer = bufferFromC(args[0].buffer);
    args[0].event = eventCreateFromRuntime(buffer.ready_event);
    return null;
}

fn bufferDevice(args: [*c]c.PJRT_Buffer_Device_Args) callconv(.c) ?*c.PJRT_Error {
    const buffer = bufferFromC(args[0].buffer);
    args[0].device = @ptrCast(buffer.device);
    return null;
}

fn bufferMemory(args: [*c]c.PJRT_Buffer_Memory_Args) callconv(.c) ?*c.PJRT_Error {
    const buffer = bufferFromC(args[0].buffer);
    args[0].memory = @ptrCast(buffer.memory);
    return null;
}

fn initApi() void {
    if (api_ready) return;
    api_storage = std.mem.zeroes(c.PJRT_Api);
    api_storage.struct_size = c.PJRT_Api_STRUCT_SIZE;
    gpu_custom_call_extension.base.next = null;
    api_storage.extension_start = @ptrCast(&gpu_custom_call_extension.base);
    api_storage.pjrt_api_version = .{
        .struct_size = c.PJRT_Api_Version_STRUCT_SIZE,
        .extension_start = null,
        .major_version = c.PJRT_API_MAJOR,
        .minor_version = c.PJRT_API_MINOR,
    };

    api_storage.PJRT_Error_Destroy = pjrtErrorDestroy;
    api_storage.PJRT_Error_Message = pjrtErrorMessage;
    api_storage.PJRT_Error_GetCode = pjrtErrorGetCode;
    api_storage.PJRT_Error_ForEachPayload = pjrtErrorForEachPayload;
    api_storage.PJRT_Plugin_Initialize = pjrtPluginInitialize;
    api_storage.PJRT_Plugin_Attributes = pjrtPluginAttributes;
    api_storage.PJRT_Event_Destroy = pjrtEventDestroy;
    api_storage.PJRT_Event_IsReady = pjrtEventIsReady;
    api_storage.PJRT_Event_Error = pjrtEventError;
    api_storage.PJRT_Event_Await = pjrtEventAwait;
    api_storage.PJRT_Event_OnReady = pjrtEventOnReady;
    api_storage.PJRT_Client_Create = pjrtClientCreate;
    api_storage.PJRT_Client_Destroy = pjrtClientDestroy;
    api_storage.PJRT_Client_PlatformName = pjrtClientPlatformName;
    api_storage.PJRT_Client_ProcessIndex = pjrtClientProcessIndex;
    api_storage.PJRT_Client_PlatformVersion = pjrtClientPlatformVersion;
    api_storage.PJRT_Client_TopologyDescription = pjrtClientTopologyDescription;
    api_storage.PJRT_Client_Devices = pjrtClientDevices;
    api_storage.PJRT_Client_AddressableDevices = pjrtClientAddressableDevices;
    api_storage.PJRT_Client_LookupDevice = pjrtClientLookupDevice;
    api_storage.PJRT_Client_LookupAddressableDevice = pjrtClientLookupAddressableDevice;
    api_storage.PJRT_Client_AddressableMemories = pjrtClientAddressableMemories;
    api_storage.PJRT_Client_Compile = pjrtClientCompile;
    api_storage.PJRT_Client_DefaultDeviceAssignment = pjrtClientDefaultDeviceAssignment;
    api_storage.PJRT_Client_BufferFromHostBuffer = pjrtClientBufferFromHostBuffer;
    api_storage.PJRT_Client_CreateUninitializedBuffer = pjrtClientCreateUninitializedBuffer;
    api_storage.PJRT_Client_CreateBuffersForAsyncHostToDevice = pjrtClientCreateBuffersForAsyncHostToDevice;
    api_storage.PJRT_AsyncHostToDeviceTransferManager_Destroy = asyncH2DDestroy;
    api_storage.PJRT_AsyncHostToDeviceTransferManager_TransferData = asyncH2DTransferData;
    api_storage.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer = asyncH2DRetrieveBuffer;
    api_storage.PJRT_AsyncHostToDeviceTransferManager_Device = asyncH2DDevice;
    api_storage.PJRT_AsyncHostToDeviceTransferManager_BufferCount = asyncH2DBufferCount;
    api_storage.PJRT_AsyncHostToDeviceTransferManager_BufferSize = asyncH2DBufferSize;
    api_storage.PJRT_AsyncHostToDeviceTransferManager_SetBufferError = asyncH2DSetBufferError;
    api_storage.PJRT_AsyncHostToDeviceTransferManager_AddMetadata = asyncH2DAddMetadata;
    api_storage.PJRT_AsyncHostToDeviceTransferManager_TransferLiteral = asyncH2DTransferLiteral;
    api_storage.PJRT_Client_DmaMap = pjrtClientDmaMap;
    api_storage.PJRT_Client_DmaUnmap = pjrtClientDmaUnmap;
    api_storage.PJRT_TopologyDescription_Create = topologyDescriptionCreate;
    api_storage.PJRT_TopologyDescription_Destroy = topologyDescriptionDestroy;
    api_storage.PJRT_TopologyDescription_PlatformName = topologyDescriptionPlatformName;
    api_storage.PJRT_TopologyDescription_PlatformVersion = topologyDescriptionPlatformVersion;
    api_storage.PJRT_TopologyDescription_GetDeviceDescriptions = topologyDescriptionGetDeviceDescriptions;
    api_storage.PJRT_TopologyDescription_Serialize = topologyDescriptionSerialize;
    api_storage.PJRT_TopologyDescription_Deserialize = topologyDescriptionDeserialize;
    api_storage.PJRT_TopologyDescription_Attributes = topologyDescriptionAttributes;
    api_storage.PJRT_TopologyDescription_Fingerprint = topologyDescriptionFingerprint;
    api_storage.PJRT_DeviceDescription_Id = deviceDescriptionId;
    api_storage.PJRT_DeviceDescription_ProcessIndex = deviceDescriptionProcessIndex;
    api_storage.PJRT_DeviceDescription_Attributes = deviceDescriptionAttributes;
    api_storage.PJRT_DeviceDescription_Kind = deviceDescriptionKind;
    api_storage.PJRT_DeviceDescription_DebugString = deviceDescriptionDebugString;
    api_storage.PJRT_DeviceDescription_ToString = deviceDescriptionToString;
    api_storage.PJRT_Device_GetDescription = deviceGetDescription;
    api_storage.PJRT_Device_IsAddressable = deviceIsAddressable;
    api_storage.PJRT_Device_LocalHardwareId = deviceLocalHardwareId;
    api_storage.PJRT_Device_AddressableMemories = deviceAddressableMemories;
    api_storage.PJRT_Device_DefaultMemory = deviceDefaultMemory;
    api_storage.PJRT_Device_MemoryStats = deviceMemoryStats;
    api_storage.PJRT_Device_GetAttributes = deviceGetAttributes;
    api_storage.PJRT_Memory_Id = memoryId;
    api_storage.PJRT_Memory_Kind = memoryKind;
    api_storage.PJRT_Memory_DebugString = memoryDebugString;
    api_storage.PJRT_Memory_ToString = memoryToString;
    api_storage.PJRT_Memory_AddressableByDevices = memoryAddressableByDevices;
    api_storage.PJRT_Executable_Destroy = executableDestroy;
    api_storage.PJRT_Executable_Name = executableName;
    api_storage.PJRT_Executable_NumReplicas = executableNumReplicas;
    api_storage.PJRT_Executable_NumPartitions = executableNumPartitions;
    api_storage.PJRT_Executable_NumOutputs = executableNumOutputs;
    api_storage.PJRT_Executable_OptimizedProgram = executableOptimizedProgram;
    api_storage.PJRT_Executable_Fingerprint = executableFingerprint;
    api_storage.PJRT_Executable_ParameterMemoryKinds = executableParameterMemoryKinds;
    api_storage.PJRT_Executable_OutputMemoryKinds = executableOutputMemoryKinds;
    api_storage.PJRT_LoadedExecutable_Destroy = loadedExecutableDestroy;
    api_storage.PJRT_LoadedExecutable_GetExecutable = loadedExecutableGetExecutable;
    api_storage.PJRT_LoadedExecutable_AddressableDevices = loadedExecutableAddressableDevices;
    api_storage.PJRT_LoadedExecutable_AddressableDeviceLogicalIds = loadedExecutableAddressableDeviceLogicalIds;
    api_storage.PJRT_LoadedExecutable_GetDeviceAssignment = loadedExecutableGetDeviceAssignment;
    api_storage.PJRT_LoadedExecutable_Fingerprint = loadedExecutableFingerprint;
    api_storage.PJRT_LoadedExecutable_Delete = loadedExecutableDelete;
    api_storage.PJRT_LoadedExecutable_IsDeleted = loadedExecutableIsDeleted;
    api_storage.PJRT_LoadedExecutable_Execute = loadedExecutableExecute;
    api_storage.PJRT_Buffer_Destroy = bufferDestroy;
    api_storage.PJRT_Buffer_ElementType = bufferElementType;
    api_storage.PJRT_Buffer_Dimensions = bufferDimensions;
    api_storage.PJRT_Buffer_OnDeviceSizeInBytes = bufferOnDeviceSize;
    api_storage.PJRT_Buffer_Device = bufferDevice;
    api_storage.PJRT_Buffer_Memory = bufferMemory;
    api_storage.PJRT_Buffer_DynamicDimensionIndices = bufferDynamicDimensionIndices;
    api_storage.PJRT_Buffer_Delete = bufferDelete;
    api_storage.PJRT_Buffer_IsDeleted = bufferIsDeleted;
    api_storage.PJRT_Buffer_IsOnCpu = bufferIsOnCpu;
    api_storage.PJRT_Buffer_ToHostBuffer = bufferToHost;
    api_storage.PJRT_Buffer_ReadyEvent = bufferReadyEvent;

    api_ready = true;
}

pub export fn GetPjrtApi() *const c.PJRT_Api {
    initApi();
    return &api_storage;
}

pub export fn PjRTx_RegisterCustomCallIdentity(function_name: [*c]const u8, function_name_size: usize) ?*c.PJRT_Error {
    const target = bytesFromC(function_name, function_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call target is null");
    return registerPjrtxCustomCall(.{
        .target = target,
        .kind = .identity,
    });
}

pub export fn PjRTx_RegisterCustomCallUnary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
    const target = bytesFromC(function_name, function_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call target is null");
    const op_text = bytesFromC(op_name, op_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call op name is null");
    var registration = parseUnaryCustomCallOp(op_text) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "unsupported PjRTx unary custom call op");
    registration.target = target;
    return registerPjrtxCustomCall(registration);
}

pub export fn PjRTx_RegisterCustomCallBinary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
    const target = bytesFromC(function_name, function_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call target is null");
    const op_text = bytesFromC(op_name, op_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call op name is null");
    var registration = parseBinaryCustomCallOp(op_text) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "unsupported PjRTx binary custom call op");
    registration.target = target;
    return registerPjrtxCustomCall(registration);
}

pub export fn PjRTx_CustomCall_Identity() callconv(.c) void {}
pub export fn PjRTx_CustomCall_UnarySqrt() callconv(.c) void {}
pub export fn PjRTx_CustomCall_BinaryAdd() callconv(.c) void {}

pub export fn PjRTx_UnregisterCustomCall(function_name: [*c]const u8, function_name_size: usize) void {
    const target = bytesFromC(function_name, function_name_size) orelse return;
    var backend_impl = backend_registry.create(.metal_mlx);
    backend_impl.unregisterCustomCall(target);
}

test "api table exposes bootstrap PJRT surface" {
    const api = GetPjrtApi();
    try std.testing.expectEqual(@as(usize, c.PJRT_Api_STRUCT_SIZE), api.struct_size);
    try std.testing.expect(api.PJRT_Plugin_Initialize != null);
    try std.testing.expect(api.PJRT_Client_Create != null);
    try std.testing.expect(api.PJRT_LoadedExecutable_Execute != null);
    try std.testing.expect(api.PJRT_Buffer_ToHostBuffer != null);
}

test "plugin attributes and client create expose backend selection" {
    const api = GetPjrtApi();

    var attrs_args = std.mem.zeroes(c.PJRT_Plugin_Attributes_Args);
    attrs_args.struct_size = c.PJRT_Plugin_Attributes_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Plugin_Attributes.?(&attrs_args));
    try std.testing.expectEqual(@as(usize, 5), attrs_args.num_attributes);
    try std.testing.expectEqualStrings("pjrtx_default_backend", attrs_args.attributes[4].name[0..attrs_args.attributes[4].name_size]);
    try std.testing.expectEqualStrings("metal_mlx", attrs_args.attributes[4].unnamed_0.string_value[0..attrs_args.attributes[4].value_size]);

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    try std.testing.expectEqual(runtime.BackendKind.metal_mlx, clientFromC(create_args.client).backend_kind);
}

test "PJRT memory stats include resident executable cache bytes" {
    const api = GetPjrtApi();

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }
    clientFromC(create_args.client).setExecutableCacheMaxResidentBytes(0);

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var before_stats = std.mem.zeroes(c.PJRT_Device_MemoryStats_Args);
    before_stats.struct_size = c.PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    before_stats.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_MemoryStats.?(&before_stats));
    try std.testing.expect(before_stats.peak_bytes_in_use_is_set);
    try std.testing.expect(before_stats.num_allocs_is_set);
    try std.testing.expect(before_stats.largest_alloc_size_is_set);
    if (before_stats.bytes_limit_is_set) {
        try std.testing.expect(before_stats.bytes_limit >= before_stats.bytes_in_use);
        try std.testing.expect(before_stats.largest_free_block_bytes_is_set);
        try std.testing.expect(before_stats.bytes_reservable_limit_is_set);
    }

    const module_text =
        \\module {
        \\  func.func @main() -> tensor<4xf32> {
        \\    %0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
        \\    return %0 : tensor<4xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));

    var after_compile_stats = std.mem.zeroes(c.PJRT_Device_MemoryStats_Args);
    after_compile_stats.struct_size = c.PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    after_compile_stats.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_MemoryStats.?(&after_compile_stats));
    try std.testing.expect(after_compile_stats.bytes_in_use >= before_stats.bytes_in_use + 16);
    try std.testing.expect(after_compile_stats.peak_bytes_in_use >= after_compile_stats.bytes_in_use);
    try std.testing.expect(after_compile_stats.num_allocs_is_set);
    try std.testing.expect(after_compile_stats.num_allocs >= before_stats.num_allocs + 1);
    try std.testing.expect(after_compile_stats.largest_alloc_size_is_set);
    try std.testing.expect(after_compile_stats.largest_alloc_size >= 16);
    if (after_compile_stats.bytes_limit_is_set) {
        try std.testing.expect(after_compile_stats.bytes_limit >= after_compile_stats.bytes_in_use);
        try std.testing.expect(after_compile_stats.largest_free_block_bytes_is_set);
        try std.testing.expect(after_compile_stats.bytes_reservable_limit_is_set);
    }

    var invalid_device_outputs = [_]?*c.PJRT_Buffer{null};
    var invalid_output_lists = [_][*c]?*c.PJRT_Buffer{&invalid_device_outputs};
    var wrong_arg_count_execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    wrong_arg_count_execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    wrong_arg_count_execute_args.executable = compile_args.executable;
    wrong_arg_count_execute_args.num_devices = 1;
    wrong_arg_count_execute_args.num_args = 1;
    wrong_arg_count_execute_args.output_lists = &invalid_output_lists;
    const wrong_arg_count_err = api.PJRT_LoadedExecutable_Execute.?(&wrong_arg_count_execute_args) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, wrong_arg_count_err), "argument count") != null);
    destroyError(api, wrong_arg_count_err);

    var missing_outputs_execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    missing_outputs_execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    missing_outputs_execute_args.executable = compile_args.executable;
    missing_outputs_execute_args.num_devices = 1;
    missing_outputs_execute_args.num_args = 0;
    const missing_outputs_err = api.PJRT_LoadedExecutable_Execute.?(&missing_outputs_execute_args) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, missing_outputs_err), "output_lists") != null);
    destroyError(api, missing_outputs_err);

    var executable_delete_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Delete_Args);
    executable_delete_args.struct_size = c.PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE;
    executable_delete_args.executable = compile_args.executable;
    try expectOk(api.PJRT_LoadedExecutable_Delete.?(&executable_delete_args));

    var executable_deleted_args = std.mem.zeroes(c.PJRT_LoadedExecutable_IsDeleted_Args);
    executable_deleted_args.struct_size = c.PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE;
    executable_deleted_args.executable = compile_args.executable;
    try expectOk(api.PJRT_LoadedExecutable_IsDeleted.?(&executable_deleted_args));
    try std.testing.expect(executable_deleted_args.is_deleted);

    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};
    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.num_devices = 1;
    execute_args.num_args = 0;
    execute_args.output_lists = &output_lists;
    const execute_err = api.PJRT_LoadedExecutable_Execute.?(&execute_args) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, execute_err), "deleted") != null);
    destroyError(api, execute_err);

    var after_destroy_stats = std.mem.zeroes(c.PJRT_Device_MemoryStats_Args);
    after_destroy_stats.struct_size = c.PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    after_destroy_stats.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_MemoryStats.?(&after_destroy_stats));
    try std.testing.expectEqual(before_stats.bytes_in_use, after_destroy_stats.bytes_in_use);
    try std.testing.expectEqual(before_stats.num_allocs, after_destroy_stats.num_allocs);
    try std.testing.expect(after_destroy_stats.peak_bytes_in_use >= after_compile_stats.bytes_in_use);

    var executable_destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
    executable_destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    executable_destroy_args.executable = compile_args.executable;
    try expectOk(api.PJRT_LoadedExecutable_Destroy.?(&executable_destroy_args));
}

test "PJRT compile trims idle resident executable cache under memory pressure" {
    const api = GetPjrtApi();

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    const client = clientFromC(create_args.client);
    client.setExecutableCacheMaxResidentBytes(std.math.maxInt(u64));

    const large_module =
        \\module {
        \\  func.func @main() -> tensor<8xf32> {
        \\    %0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<8xf32>
        \\    return %0 : tensor<8xf32>
        \\  }
        \\}
    ;
    var large_program = std.mem.zeroes(c.PJRT_Program);
    large_program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    large_program.code = @constCast(large_module.ptr);
    large_program.code_size = large_module.len;
    large_program.format = "mlir";
    large_program.format_size = "mlir".len;

    var large_compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    large_compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    large_compile_args.client = create_args.client;
    large_compile_args.program = &large_program;
    try expectOk(api.PJRT_Client_Compile.?(&large_compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = large_compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    const large_resident_bytes = client.executable_cache.stats.resident_bytes;
    try std.testing.expect(large_resident_bytes >= 32);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.resident_entries);

    var large_delete_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Delete_Args);
    large_delete_args.struct_size = c.PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE;
    large_delete_args.executable = large_compile_args.executable;
    try expectOk(api.PJRT_LoadedExecutable_Delete.?(&large_delete_args));
    try std.testing.expectEqual(large_resident_bytes, client.executable_cache.stats.resident_bytes);

    client.memories[0].stats.capacity_bytes = 4;

    const small_module =
        \\module {
        \\  func.func @main() -> tensor<1xf32> {
        \\    %0 = stablehlo.constant dense<[9.0]> : tensor<1xf32>
        \\    return %0 : tensor<1xf32>
        \\  }
        \\}
    ;
    var small_program = std.mem.zeroes(c.PJRT_Program);
    small_program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    small_program.code = @constCast(small_module.ptr);
    small_program.code_size = small_module.len;
    small_program.format = "mlir";
    small_program.format_size = "mlir".len;

    var small_compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    small_compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    small_compile_args.client = create_args.client;
    small_compile_args.program = &small_program;
    try expectOk(api.PJRT_Client_Compile.?(&small_compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = small_compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    const small_executable = executableFromC(small_compile_args.executable);
    const small_resident_bytes = small_executable.graph.backendExecutableStats().?.resident_constant_bytes;
    try std.testing.expectEqual(@as(u64, 4), small_resident_bytes);
    try std.testing.expectEqual(@as(u64, 4), small_executable.graph.last_compile_cache_trim.requested_bytes);
    try std.testing.expectEqual(@as(u64, 0), small_executable.graph.last_compile_cache_trim.target_resident_bytes);
    try std.testing.expectEqual(large_resident_bytes, small_executable.graph.last_compile_cache_trim.freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), small_executable.graph.last_compile_cache_trim.evicted_entries);
    try std.testing.expectEqual(@as(u64, 0), small_executable.graph.last_compile_cache_trim.remaining_resident_bytes);
    try std.testing.expect(!small_executable.graph.last_compile_cache_trim.still_over_capacity);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.pressure_trim_requests);
    try std.testing.expectEqual(large_resident_bytes, client.executable_cache.stats.pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executable_cache.stats.pressure_trim_failures);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.resident_entries);
    try std.testing.expectEqual(@as(u64, 4), client.executable_cache.stats.resident_bytes);
}

fn expectOk(err: [*c]c.PJRT_Error) !void {
    if (err) |actual| {
        const api = GetPjrtApi();
        std.debug.print("unexpected PJRT error: {s}\n", .{errorMessage(api, actual)});
        destroyError(api, actual);
        return error.TestUnexpectedResult;
    }
}

fn destroyError(api: *const c.PJRT_Api, err: [*c]c.PJRT_Error) void {
    var destroy_args = std.mem.zeroes(c.PJRT_Error_Destroy_Args);
    destroy_args.struct_size = c.PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.@"error" = err;
    api.PJRT_Error_Destroy.?(&destroy_args);
}

fn errorMessage(api: *const c.PJRT_Api, err: [*c]c.PJRT_Error) []const u8 {
    var message_args = std.mem.zeroes(c.PJRT_Error_Message_Args);
    message_args.struct_size = c.PJRT_Error_Message_Args_STRUCT_SIZE;
    message_args.@"error" = err;
    api.PJRT_Error_Message.?(&message_args);
    return message_args.message[0..message_args.message_size];
}

test "executable cache fingerprint includes plan metadata" {
    const client = try runtime.Client.init(allocator, backend_registry.create(.metal_mlx), 1);
    defer client.deinit();

    var plan = try compiler.makeReplicatedPlan(allocator, .{}, 1, 1);
    defer plan.deinit();

    const source = "module {}\n";
    const first = try allocExecutableFingerprint(allocator, client, source, &plan);
    defer allocator.free(first);
    const repeated = try allocExecutableFingerprint(allocator, client, source, &plan);
    defer allocator.free(repeated);
    try std.testing.expectEqualStrings(first, repeated);

    const custom_call_target = "pjrtx.test.cache_binary_add";
    try client.backend.registerCustomCall(.{
        .target = custom_call_target,
        .kind = .binary,
        .binary_op = .add,
    });
    defer client.backend.unregisterCustomCall(custom_call_target);
    const with_custom_call_registry = try allocExecutableFingerprint(allocator, client, source, &plan);
    defer allocator.free(with_custom_call_registry);
    try std.testing.expect(!std.mem.eql(u8, first, with_custom_call_registry));

    plan.donated_parameter_indices = try allocator.dupe(u32, &.{0});
    const with_donation = try allocExecutableFingerprint(allocator, client, source, &plan);
    defer allocator.free(with_donation);
    try std.testing.expect(!std.mem.eql(u8, first, with_donation));

    client.devices[0].local_hardware_id += 1;
    const with_target_hardware = try allocExecutableFingerprint(allocator, client, source, &plan);
    defer allocator.free(with_target_hardware);
    try std.testing.expect(!std.mem.eql(u8, first, with_target_hardware));
    client.devices[0].local_hardware_id -= 1;

    client.devices[0].memory_bytes += 4096;
    client.memories[0].stats.capacity_bytes += 4096;
    const with_target_memory = try allocExecutableFingerprint(allocator, client, source, &plan);
    defer allocator.free(with_target_memory);
    try std.testing.expect(!std.mem.eql(u8, first, with_target_memory));
    client.devices[0].memory_bytes -= 4096;
    client.memories[0].stats.capacity_bytes -= 4096;

    var partitioned_plan = try compiler.makeReplicatedPlan(allocator, .{ .num_partitions = 2 }, 1, 1);
    defer partitioned_plan.deinit();
    const with_partitions = try allocExecutableFingerprint(allocator, client, source, &partitioned_plan);
    defer allocator.free(with_partitions);
    try std.testing.expect(!std.mem.eql(u8, first, with_partitions));
}

const PjrtEventCallbackState = struct {
    count: usize = 0,
    ready_count: usize = 0,
    error_count: usize = 0,
    saw_expected_message: bool = false,
};

fn testPjrtEventCallback(err: [*c]c.PJRT_Error, user_arg: ?*anyopaque) callconv(.c) void {
    const state: *PjrtEventCallbackState = @ptrCast(@alignCast(user_arg.?));
    state.count += 1;
    if (err) |actual| {
        state.error_count += 1;
        const api = GetPjrtApi();
        const message = errorMessage(api, actual);
        if (std.mem.indexOf(u8, message, "buffer has been deleted") != null) {
            state.saw_expected_message = true;
        }
        destroyError(api, actual);
    } else {
        state.ready_count += 1;
    }
}

test "event callbacks bridge PJRT errors and pending runtime transitions" {
    const api = GetPjrtApi();

    const ready_event = eventCreateReady().?;
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        destroy_args.event = ready_event;
        _ = api.PJRT_Event_Destroy.?(&destroy_args);
    }
    var ready_state = PjrtEventCallbackState{};
    var ready_on_ready_args = std.mem.zeroes(c.PJRT_Event_OnReady_Args);
    ready_on_ready_args.struct_size = c.PJRT_Event_OnReady_Args_STRUCT_SIZE;
    ready_on_ready_args.event = ready_event;
    ready_on_ready_args.callback = testPjrtEventCallback;
    ready_on_ready_args.user_arg = &ready_state;
    try expectOk(api.PJRT_Event_OnReady.?(&ready_on_ready_args));
    try std.testing.expectEqual(@as(usize, 1), ready_state.count);
    try std.testing.expectEqual(@as(usize, 1), ready_state.ready_count);

    const failed_event = eventCreateFailed("buffer has been deleted").?;
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        destroy_args.event = failed_event;
        _ = api.PJRT_Event_Destroy.?(&destroy_args);
    }
    var failed_state = PjrtEventCallbackState{};
    var failed_on_ready_args = std.mem.zeroes(c.PJRT_Event_OnReady_Args);
    failed_on_ready_args.struct_size = c.PJRT_Event_OnReady_Args_STRUCT_SIZE;
    failed_on_ready_args.event = failed_event;
    failed_on_ready_args.callback = testPjrtEventCallback;
    failed_on_ready_args.user_arg = &failed_state;
    try expectOk(api.PJRT_Event_OnReady.?(&failed_on_ready_args));
    try std.testing.expectEqual(@as(usize, 1), failed_state.count);
    try std.testing.expectEqual(@as(usize, 1), failed_state.error_count);
    try std.testing.expect(failed_state.saw_expected_message);

    const pending_runtime_event = allocator.create(runtime.Event) catch return error.OutOfMemory;
    pending_runtime_event.* = runtime.Event.pending();
    const pending_event: ?*c.PJRT_Event = @ptrCast(pending_runtime_event);
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        destroy_args.event = pending_event;
        _ = api.PJRT_Event_Destroy.?(&destroy_args);
    }
    var pending_state = PjrtEventCallbackState{};
    var pending_on_ready_args = std.mem.zeroes(c.PJRT_Event_OnReady_Args);
    pending_on_ready_args.struct_size = c.PJRT_Event_OnReady_Args_STRUCT_SIZE;
    pending_on_ready_args.event = pending_event;
    pending_on_ready_args.callback = testPjrtEventCallback;
    pending_on_ready_args.user_arg = &pending_state;
    try expectOk(api.PJRT_Event_OnReady.?(&pending_on_ready_args));
    try std.testing.expectEqual(@as(usize, 0), pending_state.count);
    pending_runtime_event.setReady();
    try std.testing.expectEqual(@as(usize, 1), pending_state.count);
    try std.testing.expectEqual(@as(usize, 1), pending_state.ready_count);
}

test "client device memory and buffer ownership callbacks return stable handles" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));
    try std.testing.expectEqual(@as(usize, 1), devices_args.num_devices);
    try std.testing.expect(devices_args.devices[0] != null);
    try std.testing.expect(api.PJRT_Device_GetAttributes != null);

    var device_attrs_args = std.mem.zeroes(c.PJRT_Device_GetAttributes_Args);
    device_attrs_args.struct_size = c.PJRT_Device_GetAttributes_Args_STRUCT_SIZE;
    device_attrs_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_GetAttributes.?(&device_attrs_args));
    defer if (device_attrs_args.attributes_deleter) |deleter| deleter(device_attrs_args.device_attributes);
    try std.testing.expectEqual(@as(usize, 5), device_attrs_args.num_attributes);
    try std.testing.expectEqualStrings("device_name", device_attrs_args.attributes[0].name[0..device_attrs_args.attributes[0].name_size]);
    try std.testing.expectEqual(@as(@TypeOf(device_attrs_args.attributes[0].type), c.PJRT_NamedValue_kString), device_attrs_args.attributes[0].type);
    try std.testing.expect(device_attrs_args.attributes[0].value_size != 0);
    try std.testing.expectEqualStrings("pjrtx_default_memory_id", device_attrs_args.attributes[4].name[0..device_attrs_args.attributes[4].name_size]);
    try std.testing.expectEqual(@as(i64, 0), device_attrs_args.attributes[4].unnamed_0.int64_value);

    var memories_args = std.mem.zeroes(c.PJRT_Client_AddressableMemories_Args);
    memories_args.struct_size = c.PJRT_Client_AddressableMemories_Args_STRUCT_SIZE;
    memories_args.client = create_args.client;
    try expectOk(api.PJRT_Client_AddressableMemories.?(&memories_args));
    try std.testing.expectEqual(@as(usize, 1), memories_args.num_addressable_memories);
    try std.testing.expect(memories_args.addressable_memories[0] != null);

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));
    try std.testing.expectEqual(memories_args.addressable_memories[0], default_memory_args.memory);

    const async_dims = [_]i64{4};
    var async_shape = std.mem.zeroes(c.PJRT_ShapeSpec);
    async_shape.struct_size = c.PJRT_ShapeSpec_STRUCT_SIZE;
    async_shape.dims = &async_dims;
    async_shape.num_dims = async_dims.len;
    async_shape.element_type = c.PJRT_Buffer_Type_U8;
    var async_create_args = std.mem.zeroes(c.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args);
    async_create_args.struct_size = c.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args_STRUCT_SIZE;
    async_create_args.client = create_args.client;
    async_create_args.shape_specs = &async_shape;
    async_create_args.num_shape_specs = 1;
    async_create_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_CreateBuffersForAsyncHostToDevice.?(&async_create_args));
    defer {
        if (async_create_args.transfer_manager) |manager| {
            var destroy_transfer_args = std.mem.zeroes(c.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args);
            destroy_transfer_args.struct_size = c.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args_STRUCT_SIZE;
            destroy_transfer_args.transfer_manager = manager;
            _ = api.PJRT_AsyncHostToDeviceTransferManager_Destroy.?(&destroy_transfer_args);
        }
    }

    var async_size_args = std.mem.zeroes(c.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args);
    async_size_args.struct_size = c.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args_STRUCT_SIZE;
    async_size_args.transfer_manager = async_create_args.transfer_manager;
    async_size_args.buffer_index = 0;
    try expectOk(api.PJRT_AsyncHostToDeviceTransferManager_BufferSize.?(&async_size_args));
    try std.testing.expectEqual(@as(usize, 4), async_size_args.buffer_size);

    var async_retrieve_args = std.mem.zeroes(c.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args);
    async_retrieve_args.struct_size = c.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args_STRUCT_SIZE;
    async_retrieve_args.transfer_manager = async_create_args.transfer_manager;
    async_retrieve_args.buffer_index = 0;
    try expectOk(api.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer.?(&async_retrieve_args));
    defer {
        var destroy_buffer_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        destroy_buffer_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        destroy_buffer_args.buffer = async_retrieve_args.buffer_out;
        _ = api.PJRT_Buffer_Destroy.?(&destroy_buffer_args);
    }

    const async_input = [_]u8{ 9, 8, 7, 6 };
    var async_transfer_args = std.mem.zeroes(c.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args);
    async_transfer_args.struct_size = c.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args_STRUCT_SIZE;
    async_transfer_args.transfer_manager = async_create_args.transfer_manager;
    async_transfer_args.buffer_index = 0;
    async_transfer_args.data = &async_input;
    async_transfer_args.offset = 0;
    async_transfer_args.transfer_size = 2;
    async_transfer_args.is_last_transfer = false;
    try expectOk(api.PJRT_AsyncHostToDeviceTransferManager_TransferData.?(&async_transfer_args));
    if (async_transfer_args.done_with_h2d_transfer) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    }
    async_transfer_args.done_with_h2d_transfer = null;
    async_transfer_args.data = &async_input[2];
    async_transfer_args.offset = 2;
    async_transfer_args.transfer_size = 2;
    async_transfer_args.is_last_transfer = true;
    try expectOk(api.PJRT_AsyncHostToDeviceTransferManager_TransferData.?(&async_transfer_args));
    if (async_transfer_args.done_with_h2d_transfer) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    }

    var async_output: [4]u8 = undefined;
    var async_to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    async_to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    async_to_host_args.src = async_retrieve_args.buffer_out;
    async_to_host_args.dst = &async_output;
    async_to_host_args.dst_size = async_output.len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&async_to_host_args));
    try std.testing.expectEqualSlices(u8, &async_input, &async_output);

    var device_memories_args = std.mem.zeroes(c.PJRT_Device_AddressableMemories_Args);
    device_memories_args.struct_size = c.PJRT_Device_AddressableMemories_Args_STRUCT_SIZE;
    device_memories_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_AddressableMemories.?(&device_memories_args));
    try std.testing.expectEqual(@as(usize, 1), device_memories_args.num_memories);
    try std.testing.expectEqual(default_memory_args.memory, device_memories_args.memories[0]);

    var memory_devices_args = std.mem.zeroes(c.PJRT_Memory_AddressableByDevices_Args);
    memory_devices_args.struct_size = c.PJRT_Memory_AddressableByDevices_Args_STRUCT_SIZE;
    memory_devices_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Memory_AddressableByDevices.?(&memory_devices_args));
    try std.testing.expectEqual(@as(usize, 1), memory_devices_args.num_devices);
    try std.testing.expectEqual(devices_args.devices[0], memory_devices_args.devices[0]);

    const dims = [_]i64{4};
    const input = [_]u8{ 1, 2, 3, 4 };
    var from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    from_host_args.client = create_args.client;
    from_host_args.data = &input;
    from_host_args.type = c.PJRT_Buffer_Type_U8;
    from_host_args.dims = &dims;
    from_host_args.num_dims = dims.len;
    from_host_args.device = devices_args.devices[0];
    from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&from_host_args));
    defer {
        if (from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    var buffer_device_args = std.mem.zeroes(c.PJRT_Buffer_Device_Args);
    buffer_device_args.struct_size = c.PJRT_Buffer_Device_Args_STRUCT_SIZE;
    buffer_device_args.buffer = from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_Device.?(&buffer_device_args));
    try std.testing.expectEqual(devices_args.devices[0], buffer_device_args.device);

    var buffer_memory_args = std.mem.zeroes(c.PJRT_Buffer_Memory_Args);
    buffer_memory_args.struct_size = c.PJRT_Buffer_Memory_Args_STRUCT_SIZE;
    buffer_memory_args.buffer = from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_Memory.?(&buffer_memory_args));
    try std.testing.expectEqual(default_memory_args.memory, buffer_memory_args.memory);

    var memory_stats_before_delete = std.mem.zeroes(c.PJRT_Device_MemoryStats_Args);
    memory_stats_before_delete.struct_size = c.PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    memory_stats_before_delete.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_MemoryStats.?(&memory_stats_before_delete));
    try std.testing.expect(memory_stats_before_delete.bytes_in_use >= input.len);

    var delete_args = std.mem.zeroes(c.PJRT_Buffer_Delete_Args);
    delete_args.struct_size = c.PJRT_Buffer_Delete_Args_STRUCT_SIZE;
    delete_args.buffer = from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_Delete.?(&delete_args));

    var memory_stats_after_delete = std.mem.zeroes(c.PJRT_Device_MemoryStats_Args);
    memory_stats_after_delete.struct_size = c.PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    memory_stats_after_delete.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_MemoryStats.?(&memory_stats_after_delete));
    try std.testing.expectEqual(memory_stats_before_delete.bytes_in_use - @as(i64, @intCast(input.len)), memory_stats_after_delete.bytes_in_use);

    var ready_event_args = std.mem.zeroes(c.PJRT_Buffer_ReadyEvent_Args);
    ready_event_args.struct_size = c.PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    ready_event_args.buffer = from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_ReadyEvent.?(&ready_event_args));
    defer if (ready_event_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    var ready_event_error_args = std.mem.zeroes(c.PJRT_Event_Error_Args);
    ready_event_error_args.struct_size = c.PJRT_Event_Error_Args_STRUCT_SIZE;
    ready_event_error_args.event = ready_event_args.event;
    const ready_err = api.PJRT_Event_Error.?(&ready_event_error_args) orelse return error.TestUnexpectedResult;
    defer destroyError(api, ready_err);
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, ready_err), "deleted") != null);

    var output: [4]u8 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = from_host_args.buffer;
    to_host_args.dst = &output;
    to_host_args.dst_size = output.len;
    const to_host_err = api.PJRT_Buffer_ToHostBuffer.?(&to_host_args) orelse return error.TestUnexpectedResult;
    defer destroyError(api, to_host_err);
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, to_host_err), "deleted or donated") != null);
}

test "loaded executable execute chains u8 StableHLO ops through PJRT argument lists" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));
    try std.testing.expectEqual(@as(usize, 1), devices_args.num_devices);

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{4};
    const lhs = [_]u8{ 1, 2, 3, 4 };
    const rhs = [_]u8{ 10, 20, 30, 40 };

    var lhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    lhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    lhs_from_host_args.client = create_args.client;
    lhs_from_host_args.data = &lhs;
    lhs_from_host_args.type = c.PJRT_Buffer_Type_S8;
    lhs_from_host_args.dims = &dims;
    lhs_from_host_args.num_dims = dims.len;
    lhs_from_host_args.device = devices_args.devices[0];
    lhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&lhs_from_host_args));
    defer {
        if (lhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = lhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    var rhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    rhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    rhs_from_host_args.client = create_args.client;
    rhs_from_host_args.data = &rhs;
    rhs_from_host_args.type = c.PJRT_Buffer_Type_S8;
    rhs_from_host_args.dims = &dims;
    rhs_from_host_args.num_dims = dims.len;
    rhs_from_host_args.device = devices_args.devices[0];
    rhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&rhs_from_host_args));
    defer {
        if (rhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = rhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<4xi8>, %arg1: tensor<4xi8>) -> tensor<4xi8> {
        \\    %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xi8>
        \\    %1 = stablehlo.negate %0 : tensor<4xi8>
        \\    return %1 : tensor<4xi8>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    executableFromC(compile_args.executable).plan.donated_parameter_indices = try allocator.dupe(u32, &.{0});
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var deleted_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    deleted_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    deleted_from_host_args.client = create_args.client;
    deleted_from_host_args.data = &lhs;
    deleted_from_host_args.type = c.PJRT_Buffer_Type_S8;
    deleted_from_host_args.dims = &dims;
    deleted_from_host_args.num_dims = dims.len;
    deleted_from_host_args.device = devices_args.devices[0];
    deleted_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&deleted_from_host_args));
    defer {
        if (deleted_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = deleted_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }
    var delete_buffer_args = std.mem.zeroes(c.PJRT_Buffer_Delete_Args);
    delete_buffer_args.struct_size = c.PJRT_Buffer_Delete_Args_STRUCT_SIZE;
    delete_buffer_args.buffer = deleted_from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_Delete.?(&delete_buffer_args));

    var bad_device_args = [_]?*c.PJRT_Buffer{ deleted_from_host_args.buffer, rhs_from_host_args.buffer };
    var bad_argument_lists = [_][*c]const ?*c.PJRT_Buffer{&bad_device_args};
    var bad_device_outputs = [_]?*c.PJRT_Buffer{null};
    var bad_output_lists = [_][*c]?*c.PJRT_Buffer{&bad_device_outputs};
    var bad_device_events = [_]?*c.PJRT_Event{null};
    var bad_execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    bad_execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    bad_execute_args.executable = compile_args.executable;
    bad_execute_args.argument_lists = &bad_argument_lists;
    bad_execute_args.num_devices = 1;
    bad_execute_args.num_args = bad_device_args.len;
    bad_execute_args.output_lists = &bad_output_lists;
    bad_execute_args.device_complete_events = &bad_device_events;
    const bad_execute_err = api.PJRT_LoadedExecutable_Execute.?(&bad_execute_args) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, bad_execute_err), "deleted buffer") != null);
    destroyError(api, bad_execute_err);
    try std.testing.expectEqual(@as(?*c.PJRT_Buffer, null), bad_device_outputs[0]);
    try std.testing.expectEqual(@as(?*c.PJRT_Event, null), bad_device_events[0]);

    var alias_device_args = [_]?*c.PJRT_Buffer{ lhs_from_host_args.buffer, lhs_from_host_args.buffer };
    var alias_argument_lists = [_][*c]const ?*c.PJRT_Buffer{&alias_device_args};
    var alias_device_outputs = [_]?*c.PJRT_Buffer{null};
    var alias_output_lists = [_][*c]?*c.PJRT_Buffer{&alias_device_outputs};
    var alias_device_events = [_]?*c.PJRT_Event{null};
    var alias_execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    alias_execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    alias_execute_args.executable = compile_args.executable;
    alias_execute_args.argument_lists = &alias_argument_lists;
    alias_execute_args.num_devices = 1;
    alias_execute_args.num_args = alias_device_args.len;
    alias_execute_args.output_lists = &alias_output_lists;
    alias_execute_args.device_complete_events = &alias_device_events;
    const alias_execute_err = api.PJRT_LoadedExecutable_Execute.?(&alias_execute_args) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, alias_execute_err), "aliases another argument") != null);
    destroyError(api, alias_execute_err);
    try std.testing.expectEqual(@as(?*c.PJRT_Buffer, null), alias_device_outputs[0]);
    try std.testing.expectEqual(@as(?*c.PJRT_Event, null), alias_device_events[0]);

    var lhs_after_alias_args = std.mem.zeroes(c.PJRT_Buffer_IsDeleted_Args);
    lhs_after_alias_args.struct_size = c.PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
    lhs_after_alias_args.buffer = lhs_from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_IsDeleted.?(&lhs_after_alias_args));
    try std.testing.expect(!lhs_after_alias_args.is_deleted);

    var device_args = [_]?*c.PJRT_Buffer{ lhs_from_host_args.buffer, rhs_from_host_args.buffer };
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};
    var device_events = [_]?*c.PJRT_Event{null};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    execute_args.device_complete_events = &device_events;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    var execute_event_ready_args = std.mem.zeroes(c.PJRT_Event_IsReady_Args);
    execute_event_ready_args.struct_size = c.PJRT_Event_IsReady_Args_STRUCT_SIZE;
    execute_event_ready_args.event = device_events[0];
    try expectOk(api.PJRT_Event_IsReady.?(&execute_event_ready_args));
    try std.testing.expect(execute_event_ready_args.is_ready);
    var execute_event_await_args = std.mem.zeroes(c.PJRT_Event_Await_Args);
    execute_event_await_args.struct_size = c.PJRT_Event_Await_Args_STRUCT_SIZE;
    execute_event_await_args.event = device_events[0];
    try expectOk(api.PJRT_Event_Await.?(&execute_event_await_args));
    var execute_event_callback_state = PjrtEventCallbackState{};
    var execute_event_on_ready_args = std.mem.zeroes(c.PJRT_Event_OnReady_Args);
    execute_event_on_ready_args.struct_size = c.PJRT_Event_OnReady_Args_STRUCT_SIZE;
    execute_event_on_ready_args.event = device_events[0];
    execute_event_on_ready_args.callback = testPjrtEventCallback;
    execute_event_on_ready_args.user_arg = &execute_event_callback_state;
    try expectOk(api.PJRT_Event_OnReady.?(&execute_event_on_ready_args));
    try std.testing.expectEqual(@as(usize, 1), execute_event_callback_state.ready_count);
    defer if (device_events[0]) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var lhs_deleted_args = std.mem.zeroes(c.PJRT_Buffer_IsDeleted_Args);
    lhs_deleted_args.struct_size = c.PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
    lhs_deleted_args.buffer = lhs_from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_IsDeleted.?(&lhs_deleted_args));
    try std.testing.expect(lhs_deleted_args.is_deleted);

    var rhs_deleted_args = std.mem.zeroes(c.PJRT_Buffer_IsDeleted_Args);
    rhs_deleted_args.struct_size = c.PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
    rhs_deleted_args.buffer = rhs_from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_IsDeleted.?(&rhs_deleted_args));
    try std.testing.expect(!rhs_deleted_args.is_deleted);

    try std.testing.expect(device_outputs[0] != null);
    var output_ready_event_args = std.mem.zeroes(c.PJRT_Buffer_ReadyEvent_Args);
    output_ready_event_args.struct_size = c.PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    output_ready_event_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_ReadyEvent.?(&output_ready_event_args));
    defer if (output_ready_event_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    var output_ready_await_args = std.mem.zeroes(c.PJRT_Event_Await_Args);
    output_ready_await_args.struct_size = c.PJRT_Event_Await_Args_STRUCT_SIZE;
    output_ready_await_args.event = output_ready_event_args.event;
    try expectOk(api.PJRT_Event_Await.?(&output_ready_await_args));
    var output: [4]u8 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = output.len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(u8, &.{ 9, 18, 27, 36 }, &output);
}

test "loaded executable execute chains f32 unary and metadata custom_call through PJRT argument lists" {
    const api = GetPjrtApi();
    const registered_add_target = "pjrtx.test.plugin_binary_add";
    const add_op = "add";
    try expectOk(PjRTx_RegisterCustomCallBinary(registered_add_target.ptr, registered_add_target.len, add_op.ptr, add_op.len));
    defer PjRTx_UnregisterCustomCall(registered_add_target.ptr, registered_add_target.len);

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{2};
    const lhs = [_]f32{ 1.5, -2.0 };
    const rhs = [_]f32{ 2.25, 4.0 };

    var lhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    lhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    lhs_from_host_args.client = create_args.client;
    lhs_from_host_args.data = &lhs;
    lhs_from_host_args.type = c.PJRT_Buffer_Type_F32;
    lhs_from_host_args.dims = &dims;
    lhs_from_host_args.num_dims = dims.len;
    lhs_from_host_args.device = devices_args.devices[0];
    lhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&lhs_from_host_args));
    defer {
        if (lhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = lhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    var rhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    rhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    rhs_from_host_args.client = create_args.client;
    rhs_from_host_args.data = &rhs;
    rhs_from_host_args.type = c.PJRT_Buffer_Type_F32;
    rhs_from_host_args.dims = &dims;
    rhs_from_host_args.num_dims = dims.len;
    rhs_from_host_args.device = devices_args.devices[0];
    rhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&rhs_from_host_args));
    defer {
        if (rhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = rhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
        \\    %0 = "stablehlo.custom_call"(%arg0, %arg1) {call_target_name = "pjrtx.test.plugin_binary_add"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        \\    %1 = stablehlo.sqrt %0 : tensor<2xf32>
        \\    %2 = "stablehlo.custom_call"(%1) {call_target_name = "annotate_device_placement"} : (tensor<2xf32>) -> tensor<2xf32>
        \\    return %2 : tensor<2xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{ lhs_from_host_args.buffer, rhs_from_host_args.buffer };
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var output: [2]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f32, 3.75)), output[0], 0.0001);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f32, 2.0)), output[1], 0.0001);
}

test "loaded executable execute reshapes f32 buffer metadata through PJRT" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{ 2, 2 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    var from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    from_host_args.client = create_args.client;
    from_host_args.data = &input;
    from_host_args.type = c.PJRT_Buffer_Type_F32;
    from_host_args.dims = &dims;
    from_host_args.num_dims = dims.len;
    from_host_args.device = devices_args.devices[0];
    from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&from_host_args));
    defer {
        if (from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.reshape %arg0 : (tensor<2x2xf32>) -> tensor<4xf32>
        \\    return %0 : tensor<4xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{from_host_args.buffer};
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var dimensions_args = std.mem.zeroes(c.PJRT_Buffer_Dimensions_Args);
    dimensions_args.struct_size = c.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    dimensions_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_Dimensions.?(&dimensions_args));
    try std.testing.expectEqualSlices(i64, &.{4}, dimensions_args.dims[0..dimensions_args.num_dims]);

    var output: [4]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&input), std.mem.asBytes(&output));
}

test "loaded executable execute transposes f32 buffer data through PJRT" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{ 2, 3 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    var from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    from_host_args.client = create_args.client;
    from_host_args.data = &input;
    from_host_args.type = c.PJRT_Buffer_Type_F32;
    from_host_args.dims = &dims;
    from_host_args.num_dims = dims.len;
    from_host_args.device = devices_args.devices[0];
    from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&from_host_args));
    defer {
        if (from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
        \\    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
        \\    return %0 : tensor<3x2xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{from_host_args.buffer};
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var dimensions_args = std.mem.zeroes(c.PJRT_Buffer_Dimensions_Args);
    dimensions_args.struct_size = c.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    dimensions_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_Dimensions.?(&dimensions_args));
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, dimensions_args.dims[0..dimensions_args.num_dims]);

    var output: [6]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 }, &output);
}

test "loaded executable execute broadcasts f32 buffer data through PJRT" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{3};
    const input = [_]f32{ 7.0, 8.0, 9.0 };

    var from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    from_host_args.client = create_args.client;
    from_host_args.data = &input;
    from_host_args.type = c.PJRT_Buffer_Type_F32;
    from_host_args.dims = &dims;
    from_host_args.num_dims = dims.len;
    from_host_args.device = devices_args.devices[0];
    from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&from_host_args));
    defer {
        if (from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<3xf32>) -> tensor<2x3xf32> {
        \\    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
        \\    return %0 : tensor<2x3xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{from_host_args.buffer};
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var dimensions_args = std.mem.zeroes(c.PJRT_Buffer_Dimensions_Args);
    dimensions_args.struct_size = c.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    dimensions_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_Dimensions.?(&dimensions_args));
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, dimensions_args.dims[0..dimensions_args.num_dims]);

    var output: [6]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(f32, &.{ 7.0, 8.0, 9.0, 7.0, 8.0, 9.0 }, &output);
}

test "loaded executable execute slices f32 buffer data through PJRT" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{ 3, 4 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

    var from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    from_host_args.client = create_args.client;
    from_host_args.data = &input;
    from_host_args.type = c.PJRT_Buffer_Type_F32;
    from_host_args.dims = &dims;
    from_host_args.num_dims = dims.len;
    from_host_args.device = devices_args.devices[0];
    from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&from_host_args));
    defer {
        if (from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<3x4xf32>) -> tensor<2x2xf32> {
        \\    %0 = stablehlo.slice %arg0 [1:3, 0:4:2] : (tensor<3x4xf32>) -> tensor<2x2xf32>
        \\    return %0 : tensor<2x2xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{from_host_args.buffer};
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var dimensions_args = std.mem.zeroes(c.PJRT_Buffer_Dimensions_Args);
    dimensions_args.struct_size = c.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    dimensions_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_Dimensions.?(&dimensions_args));
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, dimensions_args.dims[0..dimensions_args.num_dims]);

    var output: [4]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(f32, &.{ 5.0, 7.0, 9.0, 11.0 }, &output);
}

test "loaded executable execute concatenates f32 buffers through PJRT" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const lhs_dims = [_]i64{ 2, 2 };
    const rhs_dims = [_]i64{ 2, 3 };
    const lhs = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const rhs = [_]f32{ 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

    var lhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    lhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    lhs_from_host_args.client = create_args.client;
    lhs_from_host_args.data = &lhs;
    lhs_from_host_args.type = c.PJRT_Buffer_Type_F32;
    lhs_from_host_args.dims = &lhs_dims;
    lhs_from_host_args.num_dims = lhs_dims.len;
    lhs_from_host_args.device = devices_args.devices[0];
    lhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&lhs_from_host_args));
    defer {
        if (lhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = lhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    var rhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    rhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    rhs_from_host_args.client = create_args.client;
    rhs_from_host_args.data = &rhs;
    rhs_from_host_args.type = c.PJRT_Buffer_Type_F32;
    rhs_from_host_args.dims = &rhs_dims;
    rhs_from_host_args.num_dims = rhs_dims.len;
    rhs_from_host_args.device = devices_args.devices[0];
    rhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&rhs_from_host_args));
    defer {
        if (rhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = rhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x5xf32> {
        \\    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x5xf32>
        \\    return %0 : tensor<2x5xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{ lhs_from_host_args.buffer, rhs_from_host_args.buffer };
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var dimensions_args = std.mem.zeroes(c.PJRT_Buffer_Dimensions_Args);
    dimensions_args.struct_size = c.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    dimensions_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_Dimensions.?(&dimensions_args));
    try std.testing.expectEqualSlices(i64, &.{ 2, 5 }, dimensions_args.dims[0..dimensions_args.num_dims]);

    var output: [10]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0 }, &output);
}

test "client default device assignment follows available Metal devices through PJRT" {
    const api = GetPjrtApi();

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));
    try std.testing.expect(devices_args.num_devices >= 1);

    var assignment = [_]c_int{-1};
    var assignment_args = std.mem.zeroes(c.PJRT_Client_DefaultDeviceAssignment_Args);
    assignment_args.struct_size = c.PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE;
    assignment_args.client = create_args.client;
    assignment_args.num_replicas = 1;
    assignment_args.num_partitions = 1;
    assignment_args.default_assignment_size = assignment.len;
    assignment_args.default_assignment = &assignment;
    try expectOk(api.PJRT_Client_DefaultDeviceAssignment.?(&assignment_args));
    try std.testing.expectEqualSlices(c_int, &.{0}, &assignment);
}

test "compile preserves bootstrap replicas partitions and shardy validation" {
    const api = GetPjrtApi();

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
        \\    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
        \\    return %0 : tensor<2xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    const compile_options = "replicas=1; partitions=1; use_shardy=true; assignment=0";
    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    compile_args.compile_options = compile_options;
    compile_args.compile_options_size = compile_options.len;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var get_executable_args = std.mem.zeroes(c.PJRT_LoadedExecutable_GetExecutable_Args);
    get_executable_args.struct_size = c.PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
    get_executable_args.loaded_executable = compile_args.executable;
    try expectOk(api.PJRT_LoadedExecutable_GetExecutable.?(&get_executable_args));

    var replicas_args = std.mem.zeroes(c.PJRT_Executable_NumReplicas_Args);
    replicas_args.struct_size = c.PJRT_Executable_NumReplicas_Args_STRUCT_SIZE;
    replicas_args.executable = get_executable_args.executable;
    try expectOk(api.PJRT_Executable_NumReplicas.?(&replicas_args));
    try std.testing.expectEqual(@as(usize, 1), replicas_args.num_replicas);

    var partitions_args = std.mem.zeroes(c.PJRT_Executable_NumPartitions_Args);
    partitions_args.struct_size = c.PJRT_Executable_NumPartitions_Args_STRUCT_SIZE;
    partitions_args.executable = get_executable_args.executable;
    try expectOk(api.PJRT_Executable_NumPartitions.?(&partitions_args));
    try std.testing.expectEqual(@as(usize, 1), partitions_args.num_partitions);
}

test "compile rejects unavailable multi-device shardy executable plan through PJRT" {
    const api = GetPjrtApi();

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    const module_text =
        \\sdy.mesh @mesh = <["x"=2]>
        \\func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<4xf32> {
        \\  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    const compile_options = "partitions=2; use_shardy=true; assignment=0,1";
    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    compile_args.compile_options = compile_options;
    compile_args.compile_options_size = compile_options.len;
    const err = api.PJRT_Client_Compile.?(&compile_args) orelse return error.TestUnexpectedResult;
    defer destroyError(api, err);
    try std.testing.expectEqualStrings("compile options require more devices than the client exposes", errorMessage(api, err));
}

test "compile unsupported op returns detailed PJRT diagnostic" {
    const api = GetPjrtApi();

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    const module_text =
        \\sdy.mesh @mesh = <["x"=2]>
        \\func.func @main(%arg0: tensor<f32>) -> tensor<4xf32> {
        \\  %0 = "stablehlo.broadcast"(%arg0) {
        \\    broadcast_sizes = array<i64: 4>,
        \\    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>
        \\  } : (tensor<f32>) -> tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    const err = api.PJRT_Client_Compile.?(&compile_args);
    try std.testing.expect(err != null);
    defer destroyError(api, err);

    var code_args = std.mem.zeroes(c.PJRT_Error_GetCode_Args);
    code_args.struct_size = c.PJRT_Error_GetCode_Args_STRUCT_SIZE;
    code_args.@"error" = err;
    try expectOk(api.PJRT_Error_GetCode.?(&code_args));
    try std.testing.expectEqual(@as(c.PJRT_Error_Code, @intCast(c.PJRT_Error_Code_UNIMPLEMENTED)), code_args.code);

    const message = errorMessage(api, err);
    try std.testing.expect(std.mem.indexOf(u8, message, "op=broadcast") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "dtype=f32") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "rank=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "sharding=sdy.sharding_per_value") != null);
}

test "compile rejects frontend-supported op without MLX executable lowering" {
    const api = GetPjrtApi();

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    const module_text =
        \\func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
        \\  %0 = "stablehlo.custom_call"(%arg0) {call_target_name = "pjrtx.test"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        \\  return %0 : tensor<2x2xf32>
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    const err = api.PJRT_Client_Compile.?(&compile_args);
    try std.testing.expect(err != null);
    defer destroyError(api, err);

    var code_args = std.mem.zeroes(c.PJRT_Error_GetCode_Args);
    code_args.struct_size = c.PJRT_Error_GetCode_Args_STRUCT_SIZE;
    code_args.@"error" = err;
    try expectOk(api.PJRT_Error_GetCode.?(&code_args));
    try std.testing.expectEqual(@as(c.PJRT_Error_Code, @intCast(c.PJRT_Error_Code_UNIMPLEMENTED)), code_args.code);
    const message = errorMessage(api, err);
    try std.testing.expect(std.mem.indexOf(u8, message, "pass=mlx-backend-legalization") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "op=custom_call") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "feature=pjrtx.test") != null);
}
