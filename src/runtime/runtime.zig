const std = @import("std");
const backend_api = @import("src/backend");
const core = @import("src/core");

pub const MAX_DEVICES = core.MAX_DEVICES;
pub const MemoryKind = core.MemoryKind;
pub const BackendKind = core.BackendKind;
pub const BufferType = core.BufferType;
pub const ElementwiseBinaryOp = core.ElementwiseBinaryOp;
pub const ElementwiseUnaryOp = core.ElementwiseUnaryOp;
pub const CompareOp = core.CompareOp;

pub const Device = struct {
    id: i32,
    local_hardware_id: i32,
    registry_id: u64 = 0,
    process_index: i32 = 0,
    addressable: bool = true,
    name: []const u8,
    debug_string: []const u8,
    memory_bytes: u64 = 0,
    has_unified_memory: bool = false,
    default_memory_id: i32,
    default_memory: *Memory,
    addressable_memories: []*Memory,
};

pub const Memory = struct {
    id: i32,
    kind: MemoryKind,
    debug_string: []const u8,
    addressable_device_ids: []const i32,
    addressable_devices: []*Device,
    stats: MemoryStats = .{},
};

pub const Topology = struct {
    device_assignment: []const i32,
    num_replicas: i32,
    num_partitions: i32,

    pub fn numDevices(self: Topology) usize {
        return @intCast(self.num_replicas * self.num_partitions);
    }
};

pub const MemoryStats = struct {
    capacity_bytes: u64 = 0,
    bytes_in_use: u64 = 0,
    peak_bytes_in_use: u64 = 0,
    host_to_device_bytes: u64 = 0,
    device_to_host_bytes: u64 = 0,
    executable_cache_hits: u64 = 0,
    executable_cache_misses: u64 = 0,

    pub fn retain(self: *MemoryStats, bytes: usize) void {
        self.bytes_in_use +|= @as(u64, @intCast(bytes));
        self.peak_bytes_in_use = @max(self.peak_bytes_in_use, self.bytes_in_use);
    }

    pub fn release(self: *MemoryStats, bytes: usize) void {
        const amount: u64 = @intCast(bytes);
        self.bytes_in_use = if (amount > self.bytes_in_use) 0 else self.bytes_in_use - amount;
    }
};

pub const EventState = enum {
    pending,
    ready,
    failed,
};

pub const Event = struct {
    state: EventState = .ready,
    message: []const u8 = "",

    pub fn ready() Event {
        return .{ .state = .ready };
    }

    pub fn failed(message: []const u8) Event {
        return .{ .state = .failed, .message = message };
    }

    pub fn isReady(self: Event) bool {
        return self.state != .pending;
    }
};

pub const BufferState = enum {
    live,
    deleted,
    donated,
};

pub const ExecutableCacheStats = struct {
    hits: u64 = 0,
    misses: u64 = 0,
};

pub const ExecutableCache = struct {
    allocator: std.mem.Allocator,
    fingerprints: std.StringHashMapUnmanaged(void) = .empty,
    stats: ExecutableCacheStats = .{},

    pub fn init(allocator: std.mem.Allocator) ExecutableCache {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ExecutableCache) void {
        var it = self.fingerprints.keyIterator();
        while (it.next()) |key| self.allocator.free(key.*);
        self.fingerprints.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn recordCompile(self: *ExecutableCache, fingerprint: []const u8) !bool {
        if (self.fingerprints.contains(fingerprint)) {
            self.stats.hits += 1;
            return true;
        }
        const owned = try self.allocator.dupe(u8, fingerprint);
        errdefer self.allocator.free(owned);
        try self.fingerprints.put(self.allocator, owned, {});
        self.stats.misses += 1;
        return false;
    }
};

pub const Client = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    backend_kind: BackendKind,
    devices: []Device,
    memories: []Memory,
    device_handles: []*Device,
    memory_handles: []*Memory,
    topology: Topology,
    executable_cache: ExecutableCache,

    pub fn init(allocator: std.mem.Allocator, backend_impl: backend_api.Backend, device_count: usize) !*Client {
        if (device_count == 0 or device_count > MAX_DEVICES) return error.InvalidDeviceCount;
        const backend_kind = backend_impl.kind();

        const client = try allocator.create(Client);
        errdefer allocator.destroy(client);

        const descriptors = try backend_impl.enumerateDevices(allocator, device_count);
        defer backend_impl.releaseDeviceDescriptors(allocator, descriptors);
        if (descriptors.len == 0 or descriptors.len > MAX_DEVICES) return error.InvalidDeviceCount;

        const devices = try allocator.alloc(Device, descriptors.len);
        errdefer allocator.free(devices);

        const memories = try allocator.alloc(Memory, descriptors.len);
        errdefer allocator.free(memories);

        const device_handles = try allocator.alloc(*Device, descriptors.len);
        errdefer allocator.free(device_handles);

        const memory_handles = try allocator.alloc(*Memory, descriptors.len);
        errdefer allocator.free(memory_handles);

        const assignment = try allocator.alloc(i32, descriptors.len);
        errdefer allocator.free(assignment);

        for (descriptors, 0..) |descriptor, i| {
            const name = try allocator.dupe(u8, descriptor.name);
            errdefer allocator.free(name);

            const debug_string = try allocator.dupe(u8, descriptor.debug_string);
            errdefer allocator.free(debug_string);

            const memory_debug_string = try allocator.dupe(u8, "device");
            errdefer allocator.free(memory_debug_string);

            const device_memories = try allocator.alloc(*Memory, 1);
            errdefer allocator.free(device_memories);

            const memory_devices = try allocator.alloc(*Device, 1);
            errdefer allocator.free(memory_devices);

            const ids = try allocator.alloc(i32, 1);
            errdefer allocator.free(ids);

            devices[i] = .{
                .id = descriptor.id,
                .local_hardware_id = descriptor.local_hardware_id,
                .registry_id = descriptor.registry_id,
                .process_index = descriptor.process_index,
                .addressable = descriptor.addressable,
                .name = name,
                .debug_string = debug_string,
                .memory_bytes = descriptor.memory_bytes,
                .has_unified_memory = descriptor.has_unified_memory,
                .default_memory_id = descriptor.default_memory_id,
                .default_memory = &memories[i],
                .addressable_memories = device_memories,
            };
            ids[0] = descriptor.id;
            memories[i] = .{
                .id = descriptor.default_memory_id,
                .kind = .device,
                .debug_string = memory_debug_string,
                .addressable_device_ids = ids,
                .addressable_devices = memory_devices,
                .stats = .{ .capacity_bytes = descriptor.memory_bytes },
            };
            devices[i].addressable_memories[0] = &memories[i];
            memories[i].addressable_devices[0] = &devices[i];
            device_handles[i] = &devices[i];
            memory_handles[i] = &memories[i];
            assignment[i] = descriptor.id;
        }

        client.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_kind,
            .devices = devices,
            .memories = memories,
            .device_handles = device_handles,
            .memory_handles = memory_handles,
            .topology = .{
                .device_assignment = assignment,
                .num_replicas = 1,
                .num_partitions = @intCast(descriptors.len),
            },
            .executable_cache = ExecutableCache.init(allocator),
        };
        return client;
    }

    pub fn deinit(self: *Client) void {
        self.executable_cache.deinit();
        for (self.devices) |device| {
            self.allocator.free(device.addressable_memories);
            self.allocator.free(device.name);
            self.allocator.free(device.debug_string);
        }
        for (self.memories) |memory| {
            self.allocator.free(memory.addressable_devices);
            self.allocator.free(memory.addressable_device_ids);
            self.allocator.free(memory.debug_string);
        }
        self.allocator.free(self.topology.device_assignment);
        self.allocator.free(self.memory_handles);
        self.allocator.free(self.device_handles);
        self.allocator.free(self.memories);
        self.allocator.free(self.devices);
        self.allocator.destroy(self);
    }

    pub fn lookupDevice(self: *const Client, id: i32) ?*const Device {
        for (self.devices) |*device| {
            if (device.id == id) return device;
        }
        return null;
    }

    pub fn lookupMemory(self: *const Client, id: i32) ?*const Memory {
        for (self.memories) |*memory| {
            if (memory.id == id) return memory;
        }
        return null;
    }

    pub fn recordExecutableCompile(self: *Client, fingerprint: []const u8) !bool {
        const hit = try self.executable_cache.recordCompile(fingerprint);
        for (self.memories) |*memory| {
            if (hit) memory.stats.executable_cache_hits += 1 else memory.stats.executable_cache_misses += 1;
        }
        return hit;
    }
};

pub const GraphNodeKind = enum {
    constant,
    parameter,
    compute,
    collective,
    custom_call,
    control_flow,
    structural,
};

pub const GraphNode = struct {
    instruction_index: usize,
    device_index: usize,
    device_id: i32,
    kind: GraphNodeKind,
};

pub const LoweringMode = enum {
    allow_runtime_fallback,
    require_backend_executable,
};

pub const LoweringOptions = struct {
    mode: LoweringMode = .allow_runtime_fallback,
    diagnostic_writer: ?*std.Io.Writer = null,
};

pub const LoweringPipeline = struct {
    mode: LoweringMode,
    backend_executable_ready: bool,
    lowered_instruction_count: usize,
    fallback_instruction_count: usize,
};

pub const ExecutableGraph = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    device_ids: []i32,
    device_local_hardware_ids: []i32,
    nodes: []GraphNode,
    backend_executable: ?backend_api.ExecutableHandle = null,
    lowering: LoweringPipeline,

    pub fn init(allocator: std.mem.Allocator, client: *const Client, plan: *const core.ExecutablePlan) !ExecutableGraph {
        return initWithOptions(allocator, client, plan, .{});
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, client: *const Client, plan: *const core.ExecutablePlan, options: LoweringOptions) !ExecutableGraph {
        const device_count = plan.options.numDevices();
        if (device_count == 0 or device_count > client.devices.len) return error.InvalidGraph;

        const device_ids = try allocator.alloc(i32, device_count);
        errdefer allocator.free(device_ids);
        const device_local_hardware_ids = try allocator.alloc(i32, device_count);
        errdefer allocator.free(device_local_hardware_ids);
        for (device_ids, 0..) |*device_id, i| {
            device_id.* = if (plan.options.device_assignment.len != 0)
                plan.options.device_assignment[i]
            else
                client.devices[i].id;
            const device = client.lookupDevice(device_id.*) orelse return error.InvalidGraph;
            device_local_hardware_ids[i] = device.local_hardware_id;
        }

        const node_count = std.math.mul(usize, plan.instructions.len, device_count) catch return error.InvalidGraph;
        const nodes = try allocator.alloc(GraphNode, node_count);
        errdefer allocator.free(nodes);

        var out: usize = 0;
        for (0..device_count) |device_index| {
            for (plan.instructions, 0..) |instruction, instruction_index| {
                nodes[out] = .{
                    .instruction_index = instruction_index,
                    .device_index = device_index,
                    .device_id = device_ids[device_index],
                    .kind = graphNodeKind(instruction.kind),
                };
                out += 1;
            }
        }

        const backend_executable = client.backend.compileExecutable(allocator, plan, device_local_hardware_ids) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => null,
        };
        errdefer if (backend_executable) |handle| client.backend.destroyExecutable(handle);
        if (backend_executable == null and options.mode == .require_backend_executable) {
            if (options.diagnostic_writer) |writer| {
                client.backend.writeExecutableLoweringDiagnostic(plan, device_local_hardware_ids, writer) catch {};
            }
            return error.UnsupportedRuntimeFeature;
        }
        const backend_ready = backend_executable != null;

        return .{
            .allocator = allocator,
            .backend = client.backend,
            .device_ids = device_ids,
            .device_local_hardware_ids = device_local_hardware_ids,
            .nodes = nodes,
            .backend_executable = backend_executable,
            .lowering = .{
                .mode = options.mode,
                .backend_executable_ready = backend_ready,
                .lowered_instruction_count = if (backend_ready) plan.instructions.len else 0,
                .fallback_instruction_count = if (backend_ready) 0 else plan.instructions.len,
            },
        };
    }

    pub fn deinit(self: *ExecutableGraph) void {
        if (self.backend_executable) |handle| self.backend.destroyExecutable(handle);
        self.allocator.free(self.device_local_hardware_ids);
        self.allocator.free(self.nodes);
        self.allocator.free(self.device_ids);
        self.* = undefined;
    }

    pub fn executeDevice(
        self: *const ExecutableGraph,
        allocator: std.mem.Allocator,
        client: *Client,
        plan: *const core.ExecutablePlan,
        device_index: usize,
        arguments: []const *Buffer,
    ) GraphExecuteError![]*Buffer {
        if (device_index >= self.device_ids.len or device_index >= client.devices.len) return error.InvalidArgument;
        if (plan.instructions.len == 0) return error.UnsupportedRuntimeFeature;
        if (arguments.len < plan.parameter_shardings.len) return error.InvalidArgument;
        for (arguments) |argument| argument.ensureUsable() catch |err| return mapBufferError(err);
        if (self.tryExecuteBackendExecutable(allocator, client, device_index, arguments) catch |err| return err) |outputs| {
            return outputs;
        }
        if (self.lowering.mode == .require_backend_executable) return error.UnsupportedRuntimeFeature;

        var value_buffers = try allocator.alloc(?*Buffer, plan.values.len);
        defer allocator.free(value_buffers);
        @memset(value_buffers, null);

        var value_owned = try allocator.alloc(bool, plan.values.len);
        defer allocator.free(value_owned);
        @memset(value_owned, false);
        defer destroyExecuteValues(value_buffers, value_owned);

        var parameter_index: usize = 0;
        for (plan.values) |value| {
            if (value.role != .parameter) continue;
            if (parameter_index >= arguments.len) return error.InvalidArgument;
            if (value.id.index >= value_buffers.len) return error.InvalidArgument;
            value_buffers[value.id.index] = arguments[parameter_index];
            parameter_index += 1;
        }

        for (plan.instructions) |instruction| {
            try self.executeInstruction(allocator, client, plan, instruction, value_buffers, value_owned, device_index);
        }

        const outputs = try allocator.alloc(*Buffer, plan.output_ids.len);
        errdefer allocator.free(outputs);
        var initialized_outputs: usize = 0;
        errdefer {
            for (outputs[0..initialized_outputs]) |buffer| buffer.deinit();
        }

        for (plan.output_ids, 0..) |output_id, output_index| {
            if (output_id.index >= value_buffers.len) return error.Internal;
            const output_buffer = value_buffers[output_id.index] orelse return error.Internal;
            if (value_owned[output_id.index]) {
                outputs[output_index] = output_buffer;
                value_owned[output_id.index] = false;
            } else {
                outputs[output_index] = Buffer.initDeviceCopy(allocator, output_buffer, device_index) catch |err| return mapBufferError(err);
            }
            initialized_outputs += 1;
        }

        return outputs;
    }

    fn tryExecuteBackendExecutable(
        self: *const ExecutableGraph,
        allocator: std.mem.Allocator,
        client: *Client,
        device_index: usize,
        arguments: []const *Buffer,
    ) GraphExecuteError!?[]*Buffer {
        const backend_executable = self.backend_executable orelse return null;
        var argument_handles = try allocator.alloc(backend_api.BufferHandle, arguments.len);
        defer allocator.free(argument_handles);
        for (arguments, 0..) |argument, i| {
            argument.ensureUsable() catch |err| return mapBufferError(err);
            argument_handles[i] = argument.backend_buffer orelse return null;
        }

        const backend_outputs = client.backend.executeExecutable(allocator, backend_executable, device_index, argument_handles) catch |err| return mapBufferError(err);
        const owned_backend_outputs = backend_outputs orelse return null;
        defer allocator.free(owned_backend_outputs);

        const outputs = try allocator.alloc(*Buffer, owned_backend_outputs.len);
        errdefer allocator.free(outputs);
        var initialized: usize = 0;
        errdefer {
            for (outputs[0..initialized]) |buffer| buffer.deinit();
            for (owned_backend_outputs[initialized..]) |output| client.backend.destroyBuffer(output.handle);
        }

        const device = &client.devices[device_index];
        const memory = device.default_memory;
        for (owned_backend_outputs, 0..) |output, i| {
            outputs[i] = Buffer.initBackendHandle(
                allocator,
                client.backend,
                output.element_type,
                output.dims,
                device,
                memory,
                device_index,
                output.byte_size,
                output.handle,
            ) catch |err| {
                return mapBufferError(err);
            };
            initialized += 1;
        }
        return outputs;
    }

    fn executeInstruction(
        self: *const ExecutableGraph,
        allocator: std.mem.Allocator,
        client: *Client,
        plan: *const core.ExecutablePlan,
        instruction: core.PlanInstruction,
        value_buffers: []?*Buffer,
        value_owned: []bool,
        device_index: usize,
    ) GraphExecuteError!void {
        _ = self;
        if (instruction.kind == .rng_bit_generator) {
            if (instruction.inputs.len < 1 or instruction.outputs.len != 2) return error.InvalidArgument;
            const state_input = value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument;
            const state_output = Buffer.initRngStateUpdate(allocator, state_input, device_index) catch |err| return mapBufferError(err);
            errdefer state_output.deinit();
            const bits_descriptor = plan.values[instruction.outputs[1].index].descriptor;
            const bits_output = Buffer.initRngBits(
                allocator,
                state_input,
                bits_descriptor.element_type,
                instruction.dims orelse bits_descriptor.dims,
                device_index,
            ) catch |err| return mapBufferError(err);
            value_buffers[instruction.outputs[0].index] = state_output;
            value_owned[instruction.outputs[0].index] = true;
            value_buffers[instruction.outputs[1].index] = bits_output;
            value_owned[instruction.outputs[1].index] = true;
            return;
        }

        if (instruction.outputs.len != 1) return error.InvalidArgument;
        const output_id = instruction.outputs[0];
        if (output_id.index >= value_buffers.len) return error.InvalidArgument;

        const next = switch (instruction.kind) {
            .constant => blk: {
                const descriptor = plan.values[output_id.index].descriptor;
                const device = &client.devices[device_index];
                break :blk Buffer.initHostCopyForBackend(
                    allocator,
                    client.backend,
                    descriptor.element_type,
                    descriptor.dims,
                    device,
                    device.default_memory,
                    device_index,
                    instruction.literal orelse &.{},
                ) catch |err| return mapBufferError(err);
            },
            .copy_arg0, .reduce_precision => Buffer.initDeviceCopy(allocator, value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument, device_index) catch |err| return mapBufferError(err),
            .partition_id => blk: {
                const descriptor = plan.values[output_id.index].descriptor;
                const device = &client.devices[device_index];
                break :blk Buffer.initPartitionId(
                    allocator,
                    client.backend,
                    descriptor.element_type,
                    descriptor.dims,
                    device,
                    device.default_memory,
                    @intCast(device_index),
                    device_index,
                ) catch |err| return mapBufferError(err);
            },
            .cholesky => Buffer.initCholesky(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                instruction.lower orelse true,
                instruction.dims orelse plan.values[output_id.index].descriptor.dims,
                device_index,
            ) catch |err| return mapBufferError(err),
            .rng => Buffer.initRngUniform(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                value_buffers[instruction.inputs[1].index] orelse return error.InvalidArgument,
                plan.values[output_id.index].descriptor.element_type,
                instruction.dims orelse plan.values[output_id.index].descriptor.dims,
                device_index,
            ) catch |err| return mapBufferError(err),
            .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .atan2, .remainder, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => Buffer.initElementwiseBinary(
                allocator,
                runtimeBinaryOp(instruction.kind) orelse return error.UnsupportedRuntimeFeature,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                value_buffers[instruction.inputs[1].index] orelse return error.InvalidArgument,
                device_index,
            ) catch |err| return mapBufferError(err),
            .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .abs, .cbrt, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .is_finite, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros => Buffer.initElementwiseUnaryTyped(
                allocator,
                runtimeUnaryOp(instruction.kind) orelse return error.UnsupportedRuntimeFeature,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                plan.values[output_id.index].descriptor.element_type,
                instruction.dims orelse plan.values[output_id.index].descriptor.dims,
                device_index,
            ) catch |err| return mapBufferError(err),
            .convert => Buffer.initConvert(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                plan.values[output_id.index].descriptor.element_type,
                instruction.dims orelse plan.values[output_id.index].descriptor.dims,
                device_index,
            ) catch |err| return mapBufferError(err),
            .bitcast_convert => return error.UnsupportedRuntimeFeature,
            .iota => blk: {
                const descriptor = plan.values[output_id.index].descriptor;
                const device = &client.devices[device_index];
                break :blk Buffer.initIota(
                    allocator,
                    client.backend,
                    descriptor.element_type,
                    instruction.dims orelse descriptor.dims,
                    device,
                    device.default_memory,
                    instruction.iota_dimension orelse 0,
                    device_index,
                ) catch |err| return mapBufferError(err);
            },
            .reshape => Buffer.initReshape(allocator, value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument, instruction.dims orelse &.{}, device_index) catch |err| return mapBufferError(err),
            .transpose => Buffer.initTranspose(allocator, value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument, instruction.permutation orelse &.{}, instruction.dims orelse &.{}, device_index) catch |err| return mapBufferError(err),
            .broadcast_in_dim => Buffer.initBroadcastInDim(allocator, value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument, instruction.broadcast_dimensions orelse &.{}, instruction.dims orelse &.{}, device_index) catch |err| return mapBufferError(err),
            .slice => Buffer.initSlice(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                instruction.start_indices orelse &.{},
                instruction.limit_indices orelse &.{},
                instruction.strides orelse &.{},
                instruction.dims orelse &.{},
                device_index,
            ) catch |err| return mapBufferError(err),
            .dynamic_slice => blk: {
                const starts = try allocator.alloc(*Buffer, instruction.inputs.len - 1);
                defer allocator.free(starts);
                for (instruction.inputs[1..], 0..) |input_id, i| starts[i] = value_buffers[input_id.index] orelse return error.InvalidArgument;
                break :blk Buffer.initDynamicSlice(
                    allocator,
                    value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                    starts,
                    instruction.slice_sizes orelse &.{},
                    instruction.dims orelse &.{},
                    device_index,
                ) catch |err| return mapBufferError(err);
            },
            .dynamic_update_slice => blk: {
                const starts = try allocator.alloc(*Buffer, instruction.inputs.len - 2);
                defer allocator.free(starts);
                for (instruction.inputs[2..], 0..) |input_id, i| starts[i] = value_buffers[input_id.index] orelse return error.InvalidArgument;
                break :blk Buffer.initDynamicUpdateSlice(
                    allocator,
                    value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                    value_buffers[instruction.inputs[1].index] orelse return error.InvalidArgument,
                    starts,
                    instruction.dims orelse &.{},
                    device_index,
                ) catch |err| return mapBufferError(err);
            },
            .pad => Buffer.initPad(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                value_buffers[instruction.inputs[1].index] orelse return error.InvalidArgument,
                instruction.edge_padding_low orelse &.{},
                instruction.edge_padding_high orelse &.{},
                instruction.interior_padding orelse &.{},
                instruction.dims orelse &.{},
                device_index,
            ) catch |err| return mapBufferError(err),
            .reverse => Buffer.initReverse(allocator, value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument, instruction.dimensions orelse &.{}, instruction.dims orelse &.{}, device_index) catch |err| return mapBufferError(err),
            .concatenate => Buffer.initConcatenate(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                value_buffers[instruction.inputs[1].index] orelse return error.InvalidArgument,
                instruction.dimension orelse 0,
                instruction.dims orelse &.{},
                device_index,
            ) catch |err| return mapBufferError(err),
            .dot_general => Buffer.initDotGeneral(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                value_buffers[instruction.inputs[1].index] orelse return error.InvalidArgument,
                instruction.lhs_batch_dimensions orelse &.{},
                instruction.rhs_batch_dimensions orelse &.{},
                instruction.lhs_contracting_dimensions orelse &.{},
                instruction.rhs_contracting_dimensions orelse &.{},
                instruction.dims orelse &.{},
                device_index,
            ) catch |err| return mapBufferError(err),
            .reduce_sum, .reduce_max, .reduce_and, .reduce_or => Buffer.initReduce(allocator, instruction.kind, value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument, instruction.reduce_dimensions orelse &.{}, instruction.dims orelse &.{}, device_index) catch |err| return mapBufferError(err),
            .gather => Buffer.initGather(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                value_buffers[instruction.inputs[1].index] orelse return error.InvalidArgument,
                instruction.offset_dims orelse &.{},
                instruction.collapsed_slice_dims orelse &.{},
                instruction.operand_batching_dims orelse &.{},
                instruction.start_indices_batching_dims orelse &.{},
                instruction.start_index_map orelse &.{},
                instruction.index_vector_dim orelse 0,
                instruction.slice_sizes orelse &.{},
                instruction.dims orelse &.{},
                device_index,
            ) catch |err| return mapBufferError(err),
            .sort => Buffer.initSort(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                instruction.dimension orelse return error.InvalidArgument,
                instruction.dims orelse &.{},
                instruction.compare_direction orelse .lt,
                device_index,
            ) catch |err| return mapBufferError(err),
            .compare => Buffer.initCompare(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                value_buffers[instruction.inputs[1].index] orelse return error.InvalidArgument,
                instruction.compare_direction orelse .eq,
                instruction.dims orelse &.{},
                device_index,
            ) catch |err| return mapBufferError(err),
            .select => Buffer.initSelect(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                value_buffers[instruction.inputs[1].index] orelse return error.InvalidArgument,
                value_buffers[instruction.inputs[2].index] orelse return error.InvalidArgument,
                instruction.dims orelse &.{},
                device_index,
            ) catch |err| return mapBufferError(err),
            .clamp => Buffer.initClamp(
                allocator,
                value_buffers[instruction.inputs[0].index] orelse return error.InvalidArgument,
                value_buffers[instruction.inputs[1].index] orelse return error.InvalidArgument,
                value_buffers[instruction.inputs[2].index] orelse return error.InvalidArgument,
                instruction.dims orelse &.{},
                device_index,
            ) catch |err| return mapBufferError(err),
            .rng_bit_generator => unreachable,
            .complex, .real, .imag, .fft, .convolution, .custom_call, .get_tuple_element, .scatter, .top_k, .triangular_solve, .tuple, .while_, .unsupported => return error.UnsupportedRuntimeFeature,
        };

        if (value_owned[output_id.index]) {
            if (value_buffers[output_id.index]) |old| old.deinit();
        }
        value_buffers[output_id.index] = next;
        value_owned[output_id.index] = true;
    }
};

pub const GraphExecuteError = error{
    OutOfMemory,
    InvalidArgument,
    UnsupportedElementType,
    ShapeMismatch,
    UnsupportedRuntimeFeature,
    BufferDeleted,
    BufferDonated,
    Internal,
};

fn graphNodeKind(kind: core.PlanInstructionKind) GraphNodeKind {
    return switch (kind) {
        .constant => .constant,
        .custom_call => .custom_call,
        .while_ => .control_flow,
        .tuple, .get_tuple_element => .structural,
        else => .compute,
    };
}

fn destroyOwnedBuffer(buffer: *Buffer, owned: bool) void {
    if (owned) buffer.deinit();
}

fn destroyExecuteValues(value_buffers: []?*Buffer, value_owned: []const bool) void {
    for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
}

fn mapBufferError(err: anyerror) GraphExecuteError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnsupportedElementType => error.UnsupportedElementType,
        error.ShapeMismatch => error.ShapeMismatch,
        error.InvalidArgument => error.InvalidArgument,
        error.UnsupportedRuntimeFeature => error.UnsupportedRuntimeFeature,
        error.BufferDeleted => error.BufferDeleted,
        error.BufferDonated => error.BufferDonated,
        else => error.Internal,
    };
}

fn runtimeBinaryOp(kind: core.PlanInstructionKind) ?ElementwiseBinaryOp {
    return switch (kind) {
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
        .shift_right_arithmetic => .shift_right_arithmetic,
        .shift_right_logical => .shift_right_logical,
        else => null,
    };
}

fn runtimeUnaryOp(kind: core.PlanInstructionKind) ?ElementwiseUnaryOp {
    return switch (kind) {
        .negate => .negate,
        .exp => .exp,
        .expm1 => .expm1,
        .tanh => .tanh,
        .sqrt => .sqrt,
        .rsqrt => .rsqrt,
        .abs => .abs,
        .cbrt => .cbrt,
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
        .round_nearest_afz => .round_nearest_afz,
        .round_nearest_even => .round_nearest_even,
        .popcnt => .popcnt,
        .count_leading_zeros => .count_leading_zeros,
        else => null,
    };
}

pub const Buffer = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    backend_kind: BackendKind,
    element_type: BufferType,
    dims: []i64,
    device_id: i32,
    memory_id: i32,
    device: *Device,
    memory: *Memory,
    shard_index: usize,
    byte_size: usize,
    bytes: []u8,
    backend_buffer: ?backend_api.BufferHandle = null,
    state: BufferState = .live,
    deleted: bool = false,
    accounted_bytes: usize = 0,
    ready_event: Event = Event.ready(),

    fn initBackendOnly(
        allocator: std.mem.Allocator,
        src: *Buffer,
        element_type: BufferType,
        dims_: []const i64,
        byte_size: usize,
        backend_buffer: backend_api.BufferHandle,
        shard_index: usize,
    ) !*Buffer {
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initHostCopyForBackend(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
        src: []const u8,
    ) !*Buffer {
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const backend_buffer = try backend_impl.bufferFromHost(device.local_hardware_id, element_type, dims, src);
        errdefer if (backend_buffer) |owned| backend_impl.destroyBuffer(owned);

        const bytes = if (backend_buffer != null) try allocator.alloc(u8, 0) else try allocator.dupe(u8, src);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_impl.kind(),
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = src.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        memory.stats.host_to_device_bytes += @intCast(src.len);
        return buffer.accountDeviceBytes();
    }

    pub fn initDeviceCopy(
        allocator: std.mem.Allocator,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        try src.ensureUsable();
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, src.dims);
        errdefer allocator.free(dims);

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (src.backend_buffer) |src_backend| {
            backend_buffer = try src.backend.cloneBuffer(src_backend);
        }
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);

        const bytes = if (backend_buffer != null) try allocator.alloc(u8, 0) else try allocator.dupe(u8, src.bytes);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = src.byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initPartitionId(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        output_type: BufferType,
        output_dims: []const i64,
        device: *Device,
        memory: *Memory,
        partition_id: u32,
        shard_index: usize,
    ) !*Buffer {
        if (output_dims.len != 0) return error.ShapeMismatch;
        if (output_type != .u32 and output_type != .s32) return error.UnsupportedElementType;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, output_type.byteSize());
        errdefer allocator.free(bytes);
        switch (output_type) {
            .u32 => writeU32LE(bytes, 0, partition_id),
            .s32 => writeI32LE(bytes, 0, @intCast(partition_id)),
            else => unreachable,
        }

        const backend_buffer = backend_impl.bufferFromHost(device.local_hardware_id, output_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| backend_impl.destroyBuffer(owned);
        const storage_bytes = if (backend_buffer != null) try allocator.alloc(u8, 0) else bytes;
        errdefer if (backend_buffer != null) allocator.free(storage_bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_impl.kind(),
            .element_type = output_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = storage_bytes,
            .backend_buffer = backend_buffer,
        };
        if (backend_buffer != null) allocator.free(bytes);
        return buffer.accountDeviceBytes();
    }

    pub fn initCholesky(
        allocator: std.mem.Allocator,
        src: *Buffer,
        lower: bool,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != .f32) return error.UnsupportedElementType;
        if (src.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        if (!std.mem.eql(i64, src.dims, output_dims) or output_dims.len < 2) return error.ShapeMismatch;
        const n_i64 = output_dims[output_dims.len - 1];
        const rows_i64 = output_dims[output_dims.len - 2];
        if (n_i64 <= 0 or rows_i64 != n_i64) return error.ShapeMismatch;
        const n: usize = @intCast(n_i64);
        const matrix_elems = n * n;
        const element_count = src.bytes.len / 4;
        if (element_count % matrix_elems != 0) return error.ShapeMismatch;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, src.bytes.len);
        errdefer allocator.free(bytes);
        @memset(bytes, 0);

        const batches = element_count / matrix_elems;
        for (0..batches) |batch| {
            const base = batch * matrix_elems;
            for (0..n) |i| {
                for (0..i + 1) |j| {
                    var sum = readF32LE(src.bytes, base + i * n + j);
                    for (0..j) |k| {
                        sum -= readF32LE(bytes, base + i * n + k) * readF32LE(bytes, base + j * n + k);
                    }
                    if (i == j) {
                        if (sum < 0.0) return error.ShapeMismatch;
                        writeF32LE(bytes, base + i * n + j, std.math.sqrt(sum));
                    } else {
                        const diag = readF32LE(bytes, base + j * n + j);
                        if (diag == 0.0) return error.ShapeMismatch;
                        writeF32LE(bytes, base + i * n + j, sum / diag);
                    }
                }
            }
            if (!lower) {
                for (0..n) |row| {
                    for (0..row) |col| {
                        const value = readF32LE(bytes, base + row * n + col);
                        writeF32LE(bytes, base + col * n + row, value);
                        writeF32LE(bytes, base + row * n + col, 0.0);
                    }
                }
            }
        }

        const backend_buffer = src.backend.bufferFromHost(src.device.local_hardware_id, .f32, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = .f32,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initRngUniform(
        allocator: std.mem.Allocator,
        min: *Buffer,
        max: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (output_type != .f32 and output_type != .u32 and output_type != .s32) return error.UnsupportedElementType;
        if (min.element_type != max.element_type or min.element_type != output_type) return error.UnsupportedElementType;
        if (min.bytes.len == 0 or max.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        if (min.bytes.len != output_type.byteSize() or max.bytes.len != output_type.byteSize()) return error.ShapeMismatch;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, denseByteSize(output_type, output_dims));
        errdefer allocator.free(bytes);
        var state = seedFromBytes(min.bytes) ^ rotl64(seedFromBytes(max.bytes), 17) ^ @as(u64, @intCast(shard_index + 1));
        const count = if (output_type.byteSize() == 0) 0 else bytes.len / output_type.byteSize();
        for (0..count) |i| {
            const rnd = nextRandomU32(&state);
            switch (output_type) {
                .f32 => {
                    const lo = readF32LE(min.bytes, 0);
                    const hi = readF32LE(max.bytes, 0);
                    const unit = @as(f32, @floatFromInt(rnd)) / @as(f32, @floatFromInt(std.math.maxInt(u32)));
                    writeF32LE(bytes, i, lo + (hi - lo) * unit);
                },
                .u32 => {
                    const lo = readU32LE(min.bytes, 0);
                    const hi = readU32LE(max.bytes, 0);
                    const span = if (hi > lo) hi - lo else 1;
                    writeU32LE(bytes, i, lo + rnd % span);
                },
                .s32 => {
                    const lo = readI32LE(min.bytes, 0);
                    const hi = readI32LE(max.bytes, 0);
                    const span: u32 = if (hi > lo) @intCast(hi - lo) else 1;
                    writeI32LE(bytes, i, lo + @as(i32, @intCast(rnd % span)));
                },
                else => unreachable,
            }
        }

        const backend_buffer = min.backend.bufferFromHost(min.device.local_hardware_id, output_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| min.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = min.backend,
            .backend_kind = min.backend_kind,
            .element_type = output_type,
            .dims = dims,
            .device_id = min.device_id,
            .memory_id = min.memory_id,
            .device = min.device,
            .memory = min.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initRngBits(
        allocator: std.mem.Allocator,
        state: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (output_type != .u32 and output_type != .s32) return error.UnsupportedElementType;
        if (state.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, denseByteSize(output_type, output_dims));
        errdefer allocator.free(bytes);
        var rng_state = seedFromBytes(state.bytes) ^ @as(u64, @intCast(shard_index + 1));
        const count = bytes.len / output_type.byteSize();
        for (0..count) |i| {
            const value = nextRandomU32(&rng_state);
            switch (output_type) {
                .u32 => writeU32LE(bytes, i, value),
                .s32 => writeI32LE(bytes, i, @bitCast(value)),
                else => unreachable,
            }
        }

        const backend_buffer = state.backend.bufferFromHost(state.device.local_hardware_id, output_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| state.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = state.backend,
            .backend_kind = state.backend_kind,
            .element_type = output_type,
            .dims = dims,
            .device_id = state.device_id,
            .memory_id = state.memory_id,
            .device = state.device,
            .memory = state.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initRngStateUpdate(
        allocator: std.mem.Allocator,
        state: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (state.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (state.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, state.dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, state.bytes.len);
        errdefer allocator.free(bytes);
        var rng_state = seedFromBytes(state.bytes) ^ rotl64(@as(u64, @intCast(shard_index + 1)), 17);
        var offset: usize = 0;
        while (offset < bytes.len) {
            const value = nextRandomU32(&rng_state);
            var word: [4]u8 = undefined;
            std.mem.writeInt(u32, &word, value, .little);
            const n = @min(bytes.len - offset, word.len);
            @memcpy(bytes[offset..][0..n], word[0..n]);
            offset += n;
        }

        const backend_buffer = state.backend.bufferFromHost(state.device.local_hardware_id, state.element_type, state.dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| state.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = state.backend,
            .backend_kind = state.backend_kind,
            .element_type = state.element_type,
            .dims = dims,
            .device_id = state.device_id,
            .memory_id = state.memory_id,
            .device = state.device,
            .memory = state.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initElementwiseBinary(
        allocator: std.mem.Allocator,
        op: ElementwiseBinaryOp,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (lhs.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, lhs.dims, rhs.dims) or lhs.byte_size != rhs.byte_size) return error.ShapeMismatch;
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.binary(lhs.backend_buffer.?, rhs.backend_buffer.?, op) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, lhs.element_type, lhs.dims, lhs.byte_size, backend_buffer, shard_index);
            }
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, lhs.dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, lhs.bytes.len);
        errdefer allocator.free(bytes);
        switch (lhs.element_type) {
            .pred => {
                for (lhs.bytes, rhs.bytes, 0..) |a, b, i| {
                    bytes[i] = switch (op) {
                        .and_ => if (a != 0 and b != 0) 1 else 0,
                        .or_ => if (a != 0 or b != 0) 1 else 0,
                        .xor => if ((a != 0) != (b != 0)) 1 else 0,
                        else => return error.UnsupportedElementType,
                    };
                }
            },
            .u8 => {
                for (lhs.bytes, rhs.bytes, 0..) |a, b, i| {
                    bytes[i] = switch (op) {
                        .add => a +% b,
                        .subtract => a -% b,
                        .multiply => a *% b,
                        .divide => if (b == 0) 0 else a / b,
                        .maximum => @max(a, b),
                        .minimum => @min(a, b),
                        .remainder => if (b == 0) 0 else a % b,
                        .and_ => a & b,
                        .or_ => a | b,
                        .xor => a ^ b,
                        .shift_left => a << @intCast(@min(b, 7)),
                        .shift_right_logical, .shift_right_arithmetic => a >> @intCast(@min(b, 7)),
                        else => return error.UnsupportedElementType,
                    };
                }
            },
            .s32 => {
                const element_count = lhs.bytes.len / 4;
                for (0..element_count) |i| {
                    const a = readI32LE(lhs.bytes, i);
                    const b = readI32LE(rhs.bytes, i);
                    const value: i32 = switch (op) {
                        .add => a +% b,
                        .subtract => a -% b,
                        .multiply => a *% b,
                        .divide => if (b == 0) 0 else @divTrunc(a, b),
                        .maximum => @max(a, b),
                        .minimum => @min(a, b),
                        .remainder => if (b == 0) 0 else @rem(a, b),
                        .and_ => a & b,
                        .or_ => a | b,
                        .xor => a ^ b,
                        .shift_left => a << @intCast(@min(@as(u5, @intCast(@max(b, 0))), 31)),
                        .shift_right_arithmetic => a >> @intCast(@min(@as(u5, @intCast(@max(b, 0))), 31)),
                        .shift_right_logical => @bitCast(@as(u32, @bitCast(a)) >> @intCast(@min(@as(u5, @intCast(@max(b, 0))), 31))),
                        else => return error.UnsupportedElementType,
                    };
                    writeI32LE(bytes, i, value);
                }
            },
            .u32 => {
                const element_count = lhs.bytes.len / 4;
                for (0..element_count) |i| {
                    const a = readU32LE(lhs.bytes, i);
                    const b = readU32LE(rhs.bytes, i);
                    const shift: u5 = @intCast(@min(b, 31));
                    const value: u32 = switch (op) {
                        .add => a +% b,
                        .subtract => a -% b,
                        .multiply => a *% b,
                        .divide => if (b == 0) 0 else a / b,
                        .maximum => @max(a, b),
                        .minimum => @min(a, b),
                        .remainder => if (b == 0) 0 else a % b,
                        .and_ => a & b,
                        .or_ => a | b,
                        .xor => a ^ b,
                        .shift_left => a << shift,
                        .shift_right_logical, .shift_right_arithmetic => a >> shift,
                        else => return error.UnsupportedElementType,
                    };
                    writeU32LE(bytes, i, value);
                }
            },
            .f32 => {
                var offset: usize = 0;
                while (offset < lhs.bytes.len) : (offset += 4) {
                    const a_bits = std.mem.readInt(u32, lhs.bytes[offset..][0..4], .little);
                    const b_bits = std.mem.readInt(u32, rhs.bytes[offset..][0..4], .little);
                    const a: f32 = @bitCast(a_bits);
                    const b: f32 = @bitCast(b_bits);
                    const value: f32 = switch (op) {
                        .add => a + b,
                        .subtract => a - b,
                        .multiply => a * b,
                        .divide => a / b,
                        .maximum => @max(a, b),
                        .minimum => @min(a, b),
                        .power => std.math.pow(f32, a, b),
                        .atan2 => std.math.atan2(a, b),
                        .remainder => @mod(a, b),
                        else => return error.UnsupportedElementType,
                    };
                    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(value), .little);
                }
            },
            else => return error.UnsupportedElementType,
        }

        buffer.* = .{
            .allocator = allocator,
            .backend = lhs.backend,
            .backend_kind = lhs.backend_kind,
            .element_type = lhs.element_type,
            .dims = dims,
            .device_id = lhs.device_id,
            .memory_id = lhs.memory_id,
            .device = lhs.device,
            .memory = lhs.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = null,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initU8Add(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        return initElementwiseBinary(allocator, .add, lhs, rhs, shard_index);
    }

    pub fn initU8Binary(
        allocator: std.mem.Allocator,
        op: ElementwiseBinaryOp,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != .u8 or rhs.element_type != .u8) return error.UnsupportedElementType;
        return initElementwiseBinary(allocator, op, lhs, rhs, shard_index);
    }

    pub fn initElementwiseUnary(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        return initElementwiseUnaryTyped(allocator, op, src, src.element_type, src.dims, shard_index);
    }

    pub fn initElementwiseUnaryTyped(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        if (src.element_type.byteSize() == 0 or output_type.byteSize() == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(output_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.unary(src_backend, op) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, output_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
            if (src.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, output_byte_size);
        errdefer allocator.free(bytes);
        switch (src.element_type) {
            .pred => {
                if (output_type != .pred or op != .not_) return error.UnsupportedElementType;
                for (src.bytes, 0..) |value, i| bytes[i] = if (value == 0) 1 else 0;
            },
            .u8 => {
                if (output_type != .u8 and output_type != .s8 and output_type != .pred) return error.UnsupportedElementType;
                for (src.bytes, 0..) |value, i| {
                    _ = switch (op) {
                        .negate => bytes[i] = 0 -% value,
                        .not_ => bytes[i] = ~value,
                        .popcnt => bytes[i] = @popCount(value),
                        .count_leading_zeros => bytes[i] = @clz(value),
                        .sign => bytes[i] = if (value == 0) 0 else 1,
                        .is_finite => bytes[i] = 1,
                        else => return error.UnsupportedElementType,
                    };
                }
            },
            .s32 => {
                if (output_type != .s32 and output_type != .pred) return error.UnsupportedElementType;
                const element_count = src.bytes.len / 4;
                for (0..element_count) |i| {
                    const value = readI32LE(src.bytes, i);
                    _ = switch (op) {
                        .negate => writeI32LE(bytes, i, -%value),
                        .not_ => writeI32LE(bytes, i, ~value),
                        .popcnt => writeI32LE(bytes, i, @intCast(@popCount(value))),
                        .count_leading_zeros => writeI32LE(bytes, i, @intCast(@clz(value))),
                        .sign => writeI32LE(bytes, i, if (value < 0) -1 else if (value > 0) 1 else 0),
                        .is_finite => bytes[i] = 1,
                        else => return error.UnsupportedElementType,
                    };
                }
            },
            .u32 => {
                if (output_type != .u32 and output_type != .pred) return error.UnsupportedElementType;
                const element_count = src.bytes.len / 4;
                for (0..element_count) |i| {
                    const value = readU32LE(src.bytes, i);
                    _ = switch (op) {
                        .not_ => writeU32LE(bytes, i, ~value),
                        .popcnt => writeU32LE(bytes, i, @intCast(@popCount(value))),
                        .count_leading_zeros => writeU32LE(bytes, i, @intCast(@clz(value))),
                        .sign => writeU32LE(bytes, i, if (value == 0) 0 else 1),
                        .is_finite => bytes[i] = 1,
                        else => return error.UnsupportedElementType,
                    };
                }
            },
            .f32 => {
                if (output_type != .f32 and output_type != .pred) return error.UnsupportedElementType;
                var offset: usize = 0;
                var index: usize = 0;
                while (offset < src.bytes.len) : (offset += 4) {
                    const bits = std.mem.readInt(u32, src.bytes[offset..][0..4], .little);
                    const value: f32 = @bitCast(bits);
                    if (op == .is_finite) {
                        bytes[index] = if (std.math.isFinite(value)) 1 else 0;
                        index += 1;
                    } else {
                        const out: f32 = switch (op) {
                            .negate => -value,
                            .exp => std.math.exp(value),
                            .expm1 => std.math.exp(value) - 1.0,
                            .tanh => std.math.tanh(value),
                            .sqrt => std.math.sqrt(value),
                            .rsqrt => 1.0 / std.math.sqrt(value),
                            .abs => @abs(value),
                            .cbrt => std.math.cbrt(value),
                            .ceil => @ceil(value),
                            .floor => @floor(value),
                            .log => @log(value),
                            .log1p => @log(value + 1.0),
                            .logistic => 1.0 / (1.0 + std.math.exp(-value)),
                            .sine => @sin(value),
                            .cosine => @cos(value),
                            .sign => if (value < 0.0) -1.0 else if (value > 0.0) 1.0 else 0.0,
                            .round_nearest_afz => @round(value),
                            .round_nearest_even => @round(value),
                            else => return error.UnsupportedElementType,
                        };
                        std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(out), .little);
                    }
                }
            },
            else => return error.UnsupportedElementType,
        }

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = output_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = null,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initReshape(
        allocator: std.mem.Allocator,
        src: *Buffer,
        new_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (denseByteSize(src.element_type, new_dims) != src.byte_size) return error.ShapeMismatch;
        if (src.backend_buffer) |src_backend| {
            if (src.backend.reshape(src_backend, new_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, new_dims, src.byte_size, backend_buffer, shard_index);
            }
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, new_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.dupe(u8, src.bytes);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = null,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initTranspose(
        allocator: std.mem.Allocator,
        src: *Buffer,
        permutation: []const i64,
        new_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validPermutation(permutation, src.dims.len)) return error.ShapeMismatch;
        if (new_dims.len != permutation.len) return error.ShapeMismatch;
        if (denseByteSize(src.element_type, new_dims) != src.byte_size) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer) |src_backend| {
            if (src.backend.transpose(src_backend, permutation) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, new_dims, src.byte_size, backend_buffer, shard_index);
            }
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, new_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, src.bytes.len);
        errdefer allocator.free(bytes);

        const src_strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(src_strides);
        const dst_strides = try rowMajorStrides(allocator, new_dims);
        defer allocator.free(dst_strides);
        const coords = try allocator.alloc(usize, new_dims.len);
        defer allocator.free(coords);

        const element_count = if (element_size == 0) 0 else src.bytes.len / element_size;
        for (0..element_count) |dst_index| {
            unravelIndex(dst_index, new_dims, dst_strides, coords);
            var src_index: usize = 0;
            for (permutation, 0..) |src_axis_i64, dst_axis| {
                src_index += coords[dst_axis] * src_strides[@intCast(src_axis_i64)];
            }
            @memcpy(
                bytes[dst_index * element_size ..][0..element_size],
                src.bytes[src_index * element_size ..][0..element_size],
            );
        }

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = null,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initBroadcastInDim(
        allocator: std.mem.Allocator,
        src: *Buffer,
        broadcast_dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validBroadcastDimensions(broadcast_dimensions, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(src.element_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.broadcastInDim(src_backend, broadcast_dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, output_byte_size);
        errdefer allocator.free(bytes);

        const src_strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(src_strides);
        const dst_strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(dst_strides);
        const coords = try allocator.alloc(usize, output_dims.len);
        defer allocator.free(coords);

        const element_count = if (element_size == 0) 0 else output_byte_size / element_size;
        for (0..element_count) |dst_index| {
            unravelIndex(dst_index, output_dims, dst_strides, coords);
            var src_index: usize = 0;
            for (broadcast_dimensions, 0..) |output_axis_i64, src_axis| {
                const output_axis: usize = @intCast(output_axis_i64);
                const src_coord = if (src.dims[src_axis] == 1) 0 else coords[output_axis];
                src_index += src_coord * src_strides[src_axis];
            }
            @memcpy(
                bytes[dst_index * element_size ..][0..element_size],
                src.bytes[src_index * element_size ..][0..element_size],
            );
        }

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = null,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        start_indices: []const i64,
        limit_indices: []const i64,
        strides_: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validSlice(start_indices, limit_indices, strides_, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(src.element_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.slice(src_backend, start_indices, limit_indices, strides_, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, output_byte_size);
        errdefer allocator.free(bytes);

        const src_strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(src_strides);
        const dst_strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(dst_strides);
        const coords = try allocator.alloc(usize, output_dims.len);
        defer allocator.free(coords);

        const element_count = if (element_size == 0) 0 else output_byte_size / element_size;
        for (0..element_count) |dst_index| {
            unravelIndex(dst_index, output_dims, dst_strides, coords);
            var src_index: usize = 0;
            for (coords, 0..) |coord, axis| {
                const src_coord = @as(usize, @intCast(start_indices[axis])) + coord * @as(usize, @intCast(strides_[axis]));
                src_index += src_coord * src_strides[axis];
            }
            @memcpy(
                bytes[dst_index * element_size ..][0..element_size],
                src.bytes[src_index * element_size ..][0..element_size],
            );
        }

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = null,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initConcatenate(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        dimension: i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (!validConcatenate(lhs.dims, rhs.dims, dimension, output_dims)) return error.ShapeMismatch;
        const element_size = lhs.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(lhs.element_type, output_dims);
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.concatenate(lhs.backend_buffer.?, rhs.backend_buffer.?, dimension, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, lhs.element_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, output_byte_size);
        errdefer allocator.free(bytes);

        const lhs_strides = try rowMajorStrides(allocator, lhs.dims);
        defer allocator.free(lhs_strides);
        const rhs_strides = try rowMajorStrides(allocator, rhs.dims);
        defer allocator.free(rhs_strides);
        const dst_strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(dst_strides);
        const coords = try allocator.alloc(usize, output_dims.len);
        defer allocator.free(coords);

        const concat_axis: usize = @intCast(dimension);
        const lhs_axis_len: usize = @intCast(lhs.dims[concat_axis]);
        const element_count = if (element_size == 0) 0 else output_byte_size / element_size;
        for (0..element_count) |dst_index| {
            unravelIndex(dst_index, output_dims, dst_strides, coords);
            const use_lhs = coords[concat_axis] < lhs_axis_len;
            var src_index: usize = 0;
            if (use_lhs) {
                for (coords, 0..) |coord, axis| {
                    src_index += coord * lhs_strides[axis];
                }
                @memcpy(
                    bytes[dst_index * element_size ..][0..element_size],
                    lhs.bytes[src_index * element_size ..][0..element_size],
                );
            } else {
                for (coords, 0..) |coord, axis| {
                    const src_coord = if (axis == concat_axis) coord - lhs_axis_len else coord;
                    src_index += src_coord * rhs_strides[axis];
                }
                @memcpy(
                    bytes[dst_index * element_size ..][0..element_size],
                    rhs.bytes[src_index * element_size ..][0..element_size],
                );
            }
        }

        buffer.* = .{
            .allocator = allocator,
            .backend = lhs.backend,
            .backend_kind = lhs.backend_kind,
            .element_type = lhs.element_type,
            .dims = dims,
            .device_id = lhs.device_id,
            .memory_id = lhs.memory_id,
            .device = lhs.device,
            .memory = lhs.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = null,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initDotGeneral(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        lhs_batch_dimensions: []const i64,
        rhs_batch_dimensions: []const i64,
        lhs_contracting_dimensions: []const i64,
        rhs_contracting_dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != .f32 or rhs.element_type != .f32) return error.UnsupportedElementType;
        if (!validDotGeneral(lhs.dims, rhs.dims, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims)) return error.ShapeMismatch;
        const output_byte_size = denseByteSize(.f32, output_dims);
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.dotGeneral(lhs.backend_buffer.?, rhs.backend_buffer.?, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, .f32, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, output_byte_size);
        errdefer allocator.free(bytes);

        try evalDotGeneralF32(allocator, lhs.bytes, lhs.dims, rhs.bytes, rhs.dims, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims, bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = lhs.backend,
            .backend_kind = lhs.backend_kind,
            .element_type = .f32,
            .dims = dims,
            .device_id = lhs.device_id,
            .memory_id = lhs.memory_id,
            .device = lhs.device,
            .memory = lhs.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = null,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initReduce(
        allocator: std.mem.Allocator,
        kind: core.PlanInstructionKind,
        src: *Buffer,
        dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        const output_type: BufferType = switch (kind) {
            .reduce_sum, .reduce_max => if (src.element_type == .f32) .f32 else return error.UnsupportedElementType,
            .reduce_and, .reduce_or => if (src.element_type == .pred) .pred else return error.UnsupportedElementType,
            else => return error.UnsupportedElementType,
        };
        if (!validReduce(src.dims, dimensions, output_dims)) return error.ShapeMismatch;
        const output_byte_size = denseByteSize(output_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.reduce(src_backend, kind, dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, output_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, output_byte_size);
        errdefer allocator.free(bytes);

        switch (output_type) {
            .f32 => try evalReduceF32(allocator, kind, src.bytes, src.dims, dimensions, output_dims, bytes),
            .pred => try evalReducePred(allocator, kind, src.bytes, src.dims, dimensions, output_dims, bytes),
            else => unreachable,
        }

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = output_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = null,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initCompare(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        direction: CompareOp,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, lhs.dims, rhs.dims) or !std.mem.eql(i64, lhs.dims, output_dims)) return error.ShapeMismatch;
        const output_byte_size = denseByteSize(.pred, output_dims);
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.compare(lhs.backend_buffer.?, rhs.backend_buffer.?, direction, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, .pred, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const element_size = lhs.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const element_count = lhs.bytes.len / element_size;
        const bytes = try allocator.alloc(u8, element_count);
        errdefer allocator.free(bytes);
        for (0..element_count) |i| {
            bytes[i] = if (compareElement(lhs.element_type, lhs.bytes, rhs.bytes, i, direction)) 1 else 0;
        }

        buffer.* = .{
            .allocator = allocator,
            .backend = lhs.backend,
            .backend_kind = lhs.backend_kind,
            .element_type = .pred,
            .dims = dims,
            .device_id = lhs.device_id,
            .memory_id = lhs.memory_id,
            .device = lhs.device,
            .memory = lhs.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = null,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initSelect(
        allocator: std.mem.Allocator,
        pred: *Buffer,
        on_true: *Buffer,
        on_false: *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (pred.element_type != .pred) return error.UnsupportedElementType;
        if (on_true.element_type != on_false.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, on_true.dims, on_false.dims) or !std.mem.eql(i64, on_true.dims, output_dims) or !std.mem.eql(i64, pred.dims, output_dims)) return error.ShapeMismatch;
        const element_size = on_true.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (pred.backend_buffer != null and on_true.backend_buffer != null and on_false.backend_buffer != null) {
            if (on_true.backend.select(pred.backend_buffer.?, on_true.backend_buffer.?, on_false.backend_buffer.?, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, on_true, on_true.element_type, output_dims, denseByteSize(on_true.element_type, output_dims), backend_buffer, shard_index);
            }
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, on_true.bytes.len);
        errdefer allocator.free(bytes);
        const element_count = on_true.bytes.len / element_size;
        for (0..element_count) |i| {
            const src = if (pred.bytes[i] != 0) on_true.bytes else on_false.bytes;
            @memcpy(bytes[i * element_size ..][0..element_size], src[i * element_size ..][0..element_size]);
        }

        buffer.* = .{
            .allocator = allocator,
            .backend = on_true.backend,
            .backend_kind = on_true.backend_kind,
            .element_type = on_true.element_type,
            .dims = dims,
            .device_id = on_true.device_id,
            .memory_id = on_true.memory_id,
            .device = on_true.device,
            .memory = on_true.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = null,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initConvert(
        allocator: std.mem.Allocator,
        src: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        if (src.element_type.byteSize() == 0 or output_type.byteSize() == 0) return error.UnsupportedElementType;
        const output_size = denseByteSize(output_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.convert(src_backend, output_type) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, output_type, output_dims, output_size, backend_buffer, shard_index);
            }
            if (src.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, output_size);
        errdefer allocator.free(bytes);

        const element_count = if (output_type.byteSize() == 0) 0 else output_size / output_type.byteSize();
        for (0..element_count) |i| {
            const value = readScalarAsF64(src.element_type, src.bytes, i);
            writeScalarFromF64(output_type, bytes, i, value);
        }

        const backend_buffer = src.backend.bufferFromHost(src.device.local_hardware_id, output_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = output_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initIota(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        output_dims: []const i64,
        device: *Device,
        memory: *Memory,
        iota_dimension: i64,
        shard_index: usize,
    ) !*Buffer {
        if (iota_dimension < 0 or iota_dimension >= @as(i64, @intCast(output_dims.len))) return error.ShapeMismatch;
        if (element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (backend_impl.iota(device.local_hardware_id, element_type, output_dims, iota_dimension) catch null) |backend_buffer| {
            return initBackendHandle(
                allocator,
                backend_impl,
                element_type,
                output_dims,
                device,
                memory,
                shard_index,
                denseByteSize(element_type, output_dims),
                backend_buffer,
            ) catch |err| {
                backend_impl.destroyBuffer(backend_buffer);
                return err;
            };
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, denseByteSize(element_type, output_dims));
        errdefer allocator.free(bytes);
        const strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(strides);
        const coords = try allocator.alloc(usize, output_dims.len);
        defer allocator.free(coords);
        const element_count = if (element_type.byteSize() == 0) 0 else bytes.len / element_type.byteSize();
        const axis: usize = @intCast(iota_dimension);
        for (0..element_count) |i| {
            unravelIndex(i, output_dims, strides, coords);
            writeScalarFromF64(element_type, bytes, i, @floatFromInt(coords[axis]));
        }

        const backend_buffer = backend_impl.bufferFromHost(device.local_hardware_id, element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| backend_impl.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_impl.kind(),
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initReverse(
        allocator: std.mem.Allocator,
        src: *Buffer,
        dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer) |src_backend| {
            if (src.backend.reverse(src_backend, dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, src.byte_size, backend_buffer, shard_index);
            }
            if (src.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);
        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);
        const bytes = try allocator.alloc(u8, src.bytes.len);
        errdefer allocator.free(bytes);
        const strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(strides);
        const coords = try allocator.alloc(usize, src.dims.len);
        defer allocator.free(coords);
        const element_count = src.bytes.len / element_size;
        for (0..element_count) |dst_index| {
            unravelIndex(dst_index, output_dims, strides, coords);
            var src_index: usize = 0;
            for (coords, 0..) |coord, axis| {
                const reverse_axis = axisInList(axis, dimensions);
                const src_coord = if (reverse_axis) @as(usize, @intCast(src.dims[axis])) - 1 - coord else coord;
                src_index += src_coord * strides[axis];
            }
            @memcpy(bytes[dst_index * element_size ..][0..element_size], src.bytes[src_index * element_size ..][0..element_size]);
        }
        const backend_buffer = src.backend.bufferFromHost(src.device.local_hardware_id, src.element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);
        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initClamp(
        allocator: std.mem.Allocator,
        min_buffer: *Buffer,
        value_buffer: *Buffer,
        max_buffer: *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (value_buffer.element_type != min_buffer.element_type or value_buffer.element_type != max_buffer.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, value_buffer.dims, output_dims)) return error.ShapeMismatch;
        const element_size = value_buffer.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const min_scalar = min_buffer.dims.len == 0;
        const max_scalar = max_buffer.dims.len == 0;
        if (!min_scalar and !std.mem.eql(i64, min_buffer.dims, output_dims)) return error.ShapeMismatch;
        if (!max_scalar and !std.mem.eql(i64, max_buffer.dims, output_dims)) return error.ShapeMismatch;
        if (min_buffer.backend_buffer) |min_backend| {
            if (value_buffer.backend_buffer) |value_backend| {
                if (max_buffer.backend_buffer) |max_backend| {
                    if (value_buffer.backend.clamp(min_backend, value_backend, max_backend, output_dims) catch null) |backend_buffer| {
                        return initBackendOnly(allocator, value_buffer, value_buffer.element_type, output_dims, denseByteSize(value_buffer.element_type, output_dims), backend_buffer, shard_index);
                    }
                }
            }
            if (value_buffer.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);
        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);
        const bytes = try allocator.alloc(u8, value_buffer.bytes.len);
        errdefer allocator.free(bytes);
        const element_count = value_buffer.bytes.len / element_size;
        for (0..element_count) |i| {
            const min_value = readScalarAsF64(value_buffer.element_type, min_buffer.bytes, if (min_scalar) 0 else i);
            const value = readScalarAsF64(value_buffer.element_type, value_buffer.bytes, i);
            const max_value = readScalarAsF64(value_buffer.element_type, max_buffer.bytes, if (max_scalar) 0 else i);
            writeScalarFromF64(value_buffer.element_type, bytes, i, @min(@max(value, min_value), max_value));
        }
        const backend_buffer = value_buffer.backend.bufferFromHost(value_buffer.device.local_hardware_id, value_buffer.element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| value_buffer.backend.destroyBuffer(owned);
        buffer.* = .{
            .allocator = allocator,
            .backend = value_buffer.backend,
            .backend_kind = value_buffer.backend_kind,
            .element_type = value_buffer.element_type,
            .dims = dims,
            .device_id = value_buffer.device_id,
            .memory_id = value_buffer.memory_id,
            .device = value_buffer.device,
            .memory = value_buffer.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initDynamicSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        start_buffers: []const *Buffer,
        slice_sizes: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (start_buffers.len != src.dims.len or slice_sizes.len != src.dims.len) return error.ShapeMismatch;
        if (src.backend_buffer) |src_backend| {
            const start_handles = try backendStartHandles(allocator, start_buffers);
            defer allocator.free(start_handles);
            if (start_handles.len == start_buffers.len) {
                if (src.backend.dynamicSlice(src_backend, start_handles, slice_sizes, output_dims) catch null) |backend_buffer| {
                    return initBackendOnly(allocator, src, src.element_type, output_dims, denseByteSize(src.element_type, output_dims), backend_buffer, shard_index);
                }
            }
            if (src.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        }
        var starts = try allocator.alloc(i64, start_buffers.len);
        defer allocator.free(starts);
        var limits = try allocator.alloc(i64, start_buffers.len);
        defer allocator.free(limits);
        var strides = try allocator.alloc(i64, start_buffers.len);
        defer allocator.free(strides);
        for (start_buffers, 0..) |start_buffer, i| {
            if (start_buffer.dims.len != 0) return error.ShapeMismatch;
            starts[i] = scalarIndex(start_buffer);
            if (starts[i] < 0) starts[i] = 0;
            if (starts[i] + slice_sizes[i] > src.dims[i]) starts[i] = src.dims[i] - slice_sizes[i];
            limits[i] = starts[i] + slice_sizes[i];
            strides[i] = 1;
        }
        return initSlice(allocator, src, starts, limits, strides, output_dims, shard_index);
    }

    pub fn initDynamicUpdateSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        update: *Buffer,
        start_buffers: []const *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != update.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, src.dims, output_dims) or src.dims.len != update.dims.len or start_buffers.len != src.dims.len) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer != null and update.backend_buffer != null) {
            const start_handles = try backendStartHandles(allocator, start_buffers);
            defer allocator.free(start_handles);
            if (start_handles.len == start_buffers.len) {
                if (src.backend.dynamicUpdateSlice(src.backend_buffer.?, update.backend_buffer.?, start_handles, output_dims) catch null) |backend_buffer| {
                    return initBackendOnly(allocator, src, src.element_type, output_dims, denseByteSize(src.element_type, output_dims), backend_buffer, shard_index);
                }
            }
            if (src.bytes.len == 0 or update.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);
        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);
        const bytes = try allocator.dupe(u8, src.bytes);
        errdefer allocator.free(bytes);

        var starts = try allocator.alloc(i64, start_buffers.len);
        defer allocator.free(starts);
        for (start_buffers, 0..) |start_buffer, i| {
            if (start_buffer.dims.len != 0) return error.ShapeMismatch;
            starts[i] = scalarIndex(start_buffer);
            if (starts[i] < 0) starts[i] = 0;
            if (starts[i] + update.dims[i] > src.dims[i]) starts[i] = src.dims[i] - update.dims[i];
        }

        const src_strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(src_strides);
        const update_strides = try rowMajorStrides(allocator, update.dims);
        defer allocator.free(update_strides);
        const coords = try allocator.alloc(usize, update.dims.len);
        defer allocator.free(coords);
        const update_count = update.bytes.len / element_size;
        for (0..update_count) |update_index| {
            unravelIndex(update_index, update.dims, update_strides, coords);
            var dst_index: usize = 0;
            for (coords, 0..) |coord, axis| {
                dst_index += (@as(usize, @intCast(starts[axis])) + coord) * src_strides[axis];
            }
            @memcpy(bytes[dst_index * element_size ..][0..element_size], update.bytes[update_index * element_size ..][0..element_size]);
        }
        const backend_buffer = src.backend.bufferFromHost(src.device.local_hardware_id, src.element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);
        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initPad(
        allocator: std.mem.Allocator,
        src: *Buffer,
        padding_value: *Buffer,
        edge_padding_low: []const i64,
        edge_padding_high: []const i64,
        interior_padding: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != padding_value.element_type or padding_value.dims.len != 0) return error.UnsupportedElementType;
        const rank = src.dims.len;
        if (edge_padding_low.len != rank or edge_padding_high.len != rank or interior_padding.len != rank or output_dims.len != rank) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer != null and padding_value.backend_buffer != null) {
            if (src.backend.pad(src.backend_buffer.?, padding_value.backend_buffer.?, edge_padding_low, edge_padding_high, interior_padding, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, denseByteSize(src.element_type, output_dims), backend_buffer, shard_index);
            }
            if (src.bytes.len == 0 or padding_value.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);
        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);
        const bytes = try allocator.alloc(u8, denseByteSize(src.element_type, output_dims));
        errdefer allocator.free(bytes);
        const out_count = bytes.len / element_size;
        for (0..out_count) |i| @memcpy(bytes[i * element_size ..][0..element_size], padding_value.bytes[0..element_size]);

        const src_strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(src_strides);
        const dst_strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(dst_strides);
        const coords = try allocator.alloc(usize, src.dims.len);
        defer allocator.free(coords);
        const src_count = src.bytes.len / element_size;
        for (0..src_count) |src_index| {
            unravelIndex(src_index, src.dims, src_strides, coords);
            var dst_index: usize = 0;
            for (coords, 0..) |coord, axis| {
                const interior = @as(usize, @intCast(interior_padding[axis]));
                const low = @as(usize, @intCast(edge_padding_low[axis]));
                dst_index += (low + coord * (interior + 1)) * dst_strides[axis];
            }
            @memcpy(bytes[dst_index * element_size ..][0..element_size], src.bytes[src_index * element_size ..][0..element_size]);
        }
        const backend_buffer = src.backend.bufferFromHost(src.device.local_hardware_id, src.element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);
        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initGather(
        allocator: std.mem.Allocator,
        operand: *Buffer,
        indices: *Buffer,
        offset_dims: []const i64,
        collapsed_slice_dims: []const i64,
        operand_batching_dims: []const i64,
        start_indices_batching_dims: []const i64,
        start_index_map: []const i64,
        index_vector_dim: i64,
        slice_sizes: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (operand.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (start_index_map.len == 0 or slice_sizes.len != operand.dims.len) return error.ShapeMismatch;
        if (index_vector_dim < 0) return error.ShapeMismatch;

        const element_size = operand.element_type.byteSize();
        if (operand.backend_buffer != null and indices.backend_buffer != null) {
            if (supportedGatherAxisForRuntime(collapsed_slice_dims, start_index_map, slice_sizes)) |gather_axis| {
                if (operand.backend.gatherAxis(operand.backend_buffer.?, indices.backend_buffer.?, gather_axis, index_vector_dim, output_dims) catch null) |backend_buffer| {
                    return initBackendOnly(allocator, operand, operand.element_type, output_dims, denseByteSize(operand.element_type, output_dims), backend_buffer, shard_index);
                }
            }
            if (operand.backend.gather(
                operand.backend_buffer.?,
                indices.backend_buffer.?,
                start_index_map,
                collapsed_slice_dims,
                operand_batching_dims,
                start_indices_batching_dims,
                index_vector_dim,
                slice_sizes,
                offset_dims,
                output_dims,
            ) catch null) |backend_buffer| {
                return initBackendOnly(allocator, operand, operand.element_type, output_dims, denseByteSize(operand.element_type, output_dims), backend_buffer, shard_index);
            }
            if (operand.bytes.len == 0 or indices.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        }
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);
        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);
        const bytes = try allocator.alloc(u8, denseByteSize(operand.element_type, output_dims));
        errdefer allocator.free(bytes);

        const operand_strides = try rowMajorStrides(allocator, operand.dims);
        defer allocator.free(operand_strides);
        const output_strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(output_strides);
        const out_coords = try allocator.alloc(usize, output_dims.len);
        defer allocator.free(out_coords);
        const out_count = bytes.len / element_size;

        if (collapsed_slice_dims.len == 1 and start_index_map.len == 1 and slice_sizes[@intCast(start_index_map[0])] == 1) {
            const gather_axis: usize = @intCast(start_index_map[0]);
            const index_rank = indices.dims.len;
            const index_vector_axis: usize = @intCast(index_vector_dim);
            const index_prefix_rank = if (index_rank > 0 and index_vector_axis < index_rank and indices.dims[index_vector_axis] == @as(i64, @intCast(start_index_map.len))) index_rank - 1 else index_rank;
            if (output_dims.len < index_prefix_rank + operand.dims.len - 1) return error.ShapeMismatch;
            const index_strides = try rowMajorStrides(allocator, indices.dims);
            defer allocator.free(index_strides);
            for (0..out_count) |out_index| {
                unravelIndex(out_index, output_dims, output_strides, out_coords);
                var index_flat: usize = 0;
                var prefix_axis: usize = 0;
                for (0..index_rank) |axis| {
                    const coord = if (axis == index_vector_axis and index_rank != index_prefix_rank) 0 else blk: {
                        const c = out_coords[prefix_axis];
                        prefix_axis += 1;
                        break :blk c;
                    };
                    index_flat += coord * index_strides[axis];
                }
                const gathered = scalarIndexAt(indices, index_flat);
                var operand_index: usize = 0;
                var offset_axis: usize = 0;
                for (0..operand.dims.len) |axis| {
                    const coord = if (axis == gather_axis) @as(usize, @intCast(@max(@as(i64, 0), @min(gathered, operand.dims[axis] - 1)))) else blk: {
                        const c = out_coords[index_prefix_rank + offset_axis];
                        offset_axis += 1;
                        break :blk c;
                    };
                    operand_index += coord * operand_strides[axis];
                }
                @memcpy(bytes[out_index * element_size ..][0..element_size], operand.bytes[operand_index * element_size ..][0..element_size]);
            }
        } else {
            return error.UnsupportedElementType;
        }

        const backend_buffer = operand.backend.bufferFromHost(operand.device.local_hardware_id, operand.element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| operand.backend.destroyBuffer(owned);
        buffer.* = .{
            .allocator = allocator,
            .backend = operand.backend,
            .backend_kind = operand.backend_kind,
            .element_type = operand.element_type,
            .dims = dims,
            .device_id = operand.device_id,
            .memory_id = operand.memory_id,
            .device = operand.device,
            .memory = operand.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initSort(
        allocator: std.mem.Allocator,
        src: *Buffer,
        dimension: i64,
        output_dims: []const i64,
        direction: CompareOp,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        if (dimension < 0 or dimension >= @as(i64, @intCast(src.dims.len))) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const ascending = switch (direction) {
            .lt, .le => true,
            .gt, .ge => false,
            else => return error.UnsupportedElementType,
        };
        if (src.backend_buffer) |src_backend| {
            if (src.backend.sort(src_backend, dimension, output_dims) catch null) |sorted_backend| {
                if (ascending) {
                    return initBackendOnly(allocator, src, src.element_type, output_dims, src.byte_size, sorted_backend, shard_index);
                }
                const reverse_dimensions = [_]i64{dimension};
                if (src.backend.reverse(sorted_backend, &reverse_dimensions, output_dims) catch null) |backend_buffer| {
                    src.backend.destroyBuffer(sorted_backend);
                    return initBackendOnly(allocator, src, src.element_type, output_dims, src.byte_size, backend_buffer, shard_index);
                }
                src.backend.destroyBuffer(sorted_backend);
            }
            if (src.bytes.len == 0) return error.UnsupportedRuntimeFeature;
        }

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);
        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);
        const bytes = try allocator.dupe(u8, src.bytes);
        errdefer allocator.free(bytes);
        try sortDenseBytes(allocator, src.element_type, bytes, output_dims, @intCast(dimension), ascending);
        const backend_buffer = if (ascending) src.backend.bufferFromHost(src.device.local_hardware_id, src.element_type, output_dims, bytes) catch null else null;
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);
        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = bytes.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initU8Unary(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != .u8) return error.UnsupportedElementType;
        return initElementwiseUnary(allocator, op, src, shard_index);
    }

    pub fn deinit(self: *Buffer) void {
        if (self.accounted_bytes != 0) self.memory.stats.release(self.accounted_bytes);
        if (self.backend_buffer) |backend_buffer| self.backend.destroyBuffer(backend_buffer);
        self.allocator.free(self.bytes);
        self.allocator.free(self.dims);
        self.allocator.destroy(self);
    }

    pub fn copyToHost(self: *Buffer, dst: []u8) !void {
        try self.ensureUsable();
        if (dst.len < self.byte_size) return error.DestinationTooSmall;
        if (self.backend_buffer) |backend_buffer| {
            self.backend.copyToHost(backend_buffer, dst) catch return error.BackendBufferCopyFailed;
            self.memory.stats.device_to_host_bytes += @intCast(self.byte_size);
            return;
        }
        @memcpy(dst[0..self.byte_size], self.bytes[0..self.byte_size]);
        self.memory.stats.device_to_host_bytes += @intCast(self.byte_size);
    }

    pub fn hasBackendStorage(self: *const Buffer) bool {
        return self.backend_buffer != null;
    }

    pub fn markDeleted(self: *Buffer) void {
        self.state = .deleted;
        self.deleted = true;
        self.ready_event = Event.failed("buffer has been deleted");
    }

    pub fn markDonated(self: *Buffer) void {
        self.state = .donated;
        self.deleted = true;
        self.ready_event = Event.failed("buffer has been donated");
    }

    pub fn ensureUsable(self: *const Buffer) !void {
        return switch (self.state) {
            .live => {},
            .deleted => error.BufferDeleted,
            .donated => error.BufferDonated,
        };
    }

    fn accountDeviceBytes(self: *Buffer) *Buffer {
        self.accounted_bytes = self.byte_size;
        self.memory.stats.retain(self.accounted_bytes);
        return self;
    }

    pub fn initBackendHandle(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
        byte_size: usize,
        backend_buffer: backend_api.BufferHandle,
    ) !*Buffer {
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_impl.kind(),
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }
};

fn denseByteSize(element_type: BufferType, dims: []const i64) usize {
    return core.denseByteSize(element_type, dims);
}

fn validPermutation(permutation: []const i64, rank: usize) bool {
    if (permutation.len != rank) return false;
    var seen_storage = [_]bool{false} ** 16;
    if (rank > seen_storage.len) return false;
    const seen = seen_storage[0..rank];
    for (permutation) |axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(rank))) return false;
        const axis_index: usize = @intCast(axis);
        if (seen[axis_index]) return false;
        seen[axis_index] = true;
    }
    return true;
}

fn validBroadcastDimensions(broadcast_dimensions: []const i64, input_dims: []const i64, output_dims: []const i64) bool {
    if (broadcast_dimensions.len != input_dims.len) return false;
    var seen_storage = [_]bool{false} ** 64;
    if (output_dims.len > seen_storage.len) return false;
    const seen = seen_storage[0..output_dims.len];
    for (broadcast_dimensions, 0..) |axis, input_axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(output_dims.len))) return false;
        const output_axis: usize = @intCast(axis);
        if (seen[output_axis]) return false;
        seen[output_axis] = true;
        if (input_dims[input_axis] != 1 and input_dims[input_axis] != output_dims[output_axis]) return false;
    }
    return true;
}

fn validSlice(start_indices: []const i64, limit_indices: []const i64, strides: []const i64, input_dims: []const i64, output_dims: []const i64) bool {
    const rank = input_dims.len;
    if (start_indices.len != rank or limit_indices.len != rank or strides.len != rank or output_dims.len != rank) return false;
    for (0..rank) |axis| {
        const input_dim = input_dims[axis];
        const start = start_indices[axis];
        const limit = limit_indices[axis];
        const stride = strides[axis];
        const output_dim = output_dims[axis];
        if (input_dim < 0 or output_dim < 0) return false;
        if (start < 0 or limit < start or limit > input_dim or stride <= 0) return false;
        const span = limit - start;
        const expected = if (span == 0) 0 else @divTrunc(span + stride - 1, stride);
        if (output_dim != expected) return false;
    }
    return true;
}

fn validConcatenate(lhs_dims: []const i64, rhs_dims: []const i64, dimension: i64, output_dims: []const i64) bool {
    const rank = lhs_dims.len;
    if (rank == 0 or rhs_dims.len != rank or output_dims.len != rank) return false;
    if (dimension < 0 or dimension >= @as(i64, @intCast(rank))) return false;
    const concat_axis: usize = @intCast(dimension);
    for (0..rank) |axis| {
        if (lhs_dims[axis] < 0 or rhs_dims[axis] < 0 or output_dims[axis] < 0) return false;
        const expected = if (axis == concat_axis) lhs_dims[axis] + rhs_dims[axis] else lhs_dims[axis];
        if (output_dims[axis] != expected) return false;
        if (axis != concat_axis and lhs_dims[axis] != rhs_dims[axis]) return false;
    }
    return true;
}

fn validDotGeneral(
    lhs_dims: []const i64,
    rhs_dims: []const i64,
    lhs_batch_dimensions: []const i64,
    rhs_batch_dimensions: []const i64,
    lhs_contracting_dimensions: []const i64,
    rhs_contracting_dimensions: []const i64,
    output_dims: []const i64,
) bool {
    if (lhs_contracting_dimensions.len != 1 or rhs_contracting_dimensions.len != 1) return false;
    if (lhs_batch_dimensions.len != rhs_batch_dimensions.len) return false;
    const lhs_contract: usize = if (lhs_contracting_dimensions[0] < 0) return false else @intCast(lhs_contracting_dimensions[0]);
    const rhs_contract: usize = if (rhs_contracting_dimensions[0] < 0) return false else @intCast(rhs_contracting_dimensions[0]);
    if (lhs_contract >= lhs_dims.len or rhs_contract >= rhs_dims.len) return false;
    if (lhs_dims[lhs_contract] != rhs_dims[rhs_contract]) return false;
    for (lhs_batch_dimensions, rhs_batch_dimensions) |lhs_axis_i64, rhs_axis_i64| {
        if (lhs_axis_i64 < 0 or rhs_axis_i64 < 0) return false;
        const lhs_axis: usize = @intCast(lhs_axis_i64);
        const rhs_axis: usize = @intCast(rhs_axis_i64);
        if (lhs_axis >= lhs_dims.len or rhs_axis >= rhs_dims.len or lhs_dims[lhs_axis] != rhs_dims[rhs_axis]) return false;
    }
    return denseByteSize(.f32, output_dims) != 0;
}

fn validReduce(input_dims: []const i64, dimensions: []const i64, output_dims: []const i64) bool {
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
        if (!reduced[axis]) {
            if (output_dims[out_axis] != dim) return false;
            out_axis += 1;
        }
    }
    return true;
}

fn rowMajorStrides(allocator: std.mem.Allocator, dims: []const i64) ![]usize {
    const strides = try allocator.alloc(usize, dims.len);
    var stride: usize = 1;
    var reverse_index = dims.len;
    while (reverse_index > 0) {
        reverse_index -= 1;
        strides[reverse_index] = stride;
        if (dims[reverse_index] < 0) return error.ShapeMismatch;
        stride = std.math.mul(usize, stride, @intCast(dims[reverse_index])) catch return error.ShapeMismatch;
    }
    return strides;
}

fn unravelIndex(index: usize, dims: []const i64, strides: []const usize, out: []usize) void {
    for (0..dims.len) |axis| {
        _ = dims[axis];
        out[axis] = index / strides[axis] % @as(usize, @intCast(dims[axis]));
    }
}

fn readF32LE(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little));
}

fn writeF32LE(bytes: []u8, index: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[index * 4 ..][0..4], @bitCast(value), .little);
}

fn readI32LE(bytes: []const u8, index: usize) i32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little));
}

fn writeI32LE(bytes: []u8, index: usize, value: i32) void {
    std.mem.writeInt(u32, bytes[index * 4 ..][0..4], @bitCast(value), .little);
}

fn readU32LE(bytes: []const u8, index: usize) u32 {
    return std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little);
}

fn writeU32LE(bytes: []u8, index: usize, value: u32) void {
    std.mem.writeInt(u32, bytes[index * 4 ..][0..4], value, .little);
}

fn seedFromBytes(bytes: []const u8) u64 {
    var hash: u64 = 0xcbf29ce484222325;
    for (bytes) |byte| {
        hash ^= byte;
        hash *%= 0x100000001b3;
    }
    return if (hash == 0) 0x9e3779b97f4a7c15 else hash;
}

fn rotl64(value: u64, amount: u6) u64 {
    return std.math.rotl(u64, value, amount);
}

fn nextRandomU32(state: *u64) u32 {
    var x = state.*;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state.* = if (x == 0) 0x9e3779b97f4a7c15 else x;
    return @truncate(x >> 32);
}

fn compareF32(lhs: f32, rhs: f32, direction: CompareOp) bool {
    return switch (direction) {
        .eq => lhs == rhs,
        .ne => lhs != rhs,
        .ge => lhs >= rhs,
        .gt => lhs > rhs,
        .le => lhs <= rhs,
        .lt => lhs < rhs,
    };
}

fn compareScalar(lhs: f64, rhs: f64, direction: CompareOp) bool {
    return switch (direction) {
        .eq => lhs == rhs,
        .ne => lhs != rhs,
        .ge => lhs >= rhs,
        .gt => lhs > rhs,
        .le => lhs <= rhs,
        .lt => lhs < rhs,
    };
}

fn compareElement(element_type: BufferType, lhs: []const u8, rhs: []const u8, index: usize, direction: CompareOp) bool {
    return compareScalar(readScalarAsF64(element_type, lhs, index), readScalarAsF64(element_type, rhs, index), direction);
}

fn readScalarAsF64(element_type: BufferType, bytes: []const u8, index: usize) f64 {
    return switch (element_type) {
        .pred, .u8 => @floatFromInt(bytes[index]),
        .s8 => @floatFromInt(@as(i8, @bitCast(bytes[index]))),
        .s32 => @floatFromInt(readI32LE(bytes, index)),
        .u32 => @floatFromInt(readU32LE(bytes, index)),
        .f32 => readF32LE(bytes, index),
        else => 0.0,
    };
}

fn writeScalarFromF64(element_type: BufferType, bytes: []u8, index: usize, value: f64) void {
    switch (element_type) {
        .pred => bytes[index] = if (value != 0.0) 1 else 0,
        .u8 => bytes[index] = @intFromFloat(@min(@max(value, 0.0), 255.0)),
        .s8 => bytes[index] = @bitCast(@as(i8, @intFromFloat(@min(@max(value, -128.0), 127.0)))),
        .s32 => writeI32LE(bytes, index, @intFromFloat(@min(@max(value, -2147483648.0), 2147483647.0))),
        .u32 => writeU32LE(bytes, index, @intFromFloat(@min(@max(value, 0.0), 4294967295.0))),
        .f32 => writeF32LE(bytes, index, @floatCast(value)),
        else => {},
    }
}

fn scalarIndex(buffer: *const Buffer) i64 {
    return scalarIndexAt(buffer, 0);
}

fn scalarIndexAt(buffer: *const Buffer, index: usize) i64 {
    return switch (buffer.element_type) {
        .pred, .u8 => buffer.bytes[index],
        .s8 => @as(i8, @bitCast(buffer.bytes[index])),
        .s32 => readI32LE(buffer.bytes, index),
        .u32 => @intCast(readU32LE(buffer.bytes, index)),
        .f32 => @intFromFloat(readF32LE(buffer.bytes, index)),
        else => 0,
    };
}

fn backendStartHandles(allocator: std.mem.Allocator, start_buffers: []const *Buffer) ![]backend_api.BufferHandle {
    for (start_buffers) |start_buffer| {
        if (start_buffer.backend_buffer == null) return allocator.alloc(backend_api.BufferHandle, 0);
    }
    const handles = try allocator.alloc(backend_api.BufferHandle, start_buffers.len);
    errdefer allocator.free(handles);
    for (start_buffers, 0..) |start_buffer, index| {
        handles[index] = start_buffer.backend_buffer.?;
    }
    return handles;
}

fn supportedGatherAxisForRuntime(collapsed_slice_dims: []const i64, start_index_map: []const i64, slice_sizes: []const i64) ?i64 {
    if (collapsed_slice_dims.len != 1 or start_index_map.len != 1) return null;
    const axis = start_index_map[0];
    if (axis < 0 or collapsed_slice_dims[0] != axis) return null;
    if (slice_sizes.len == 0 or axis >= @as(i64, @intCast(slice_sizes.len)) or slice_sizes[@intCast(axis)] != 1) return null;
    return axis;
}

fn sortDenseBytes(allocator: std.mem.Allocator, element_type: BufferType, bytes: []u8, dims: []const i64, axis: usize, ascending: bool) !void {
    if (axis >= dims.len or dims[axis] < 0) return error.ShapeMismatch;
    const element_size = element_type.byteSize();
    if (element_size == 0) return error.UnsupportedElementType;
    const axis_len: usize = @intCast(dims[axis]);
    if (axis_len <= 1) return;

    const strides = try rowMajorStrides(allocator, dims);
    defer allocator.free(strides);
    const element_count = bytes.len / element_size;
    const coords = try allocator.alloc(usize, dims.len);
    defer allocator.free(coords);
    const line = try allocator.alloc(usize, axis_len);
    defer allocator.free(line);
    const scratch = try allocator.alloc(u8, axis_len * element_size);
    defer allocator.free(scratch);

    for (0..element_count) |base_index| {
        unravelIndex(base_index, dims, strides, coords);
        if (coords[axis] != 0) continue;
        for (0..axis_len) |i| {
            line[i] = base_index + i * strides[axis];
        }
        std.mem.sort(usize, line, SortContext{ .bytes = bytes, .element_type = element_type, .ascending = ascending }, lessSortIndex);
        for (line, 0..) |src_index, i| {
            @memcpy(scratch[i * element_size ..][0..element_size], bytes[src_index * element_size ..][0..element_size]);
        }
        for (0..axis_len) |i| {
            const dst_index = base_index + i * strides[axis];
            @memcpy(bytes[dst_index * element_size ..][0..element_size], scratch[i * element_size ..][0..element_size]);
        }
    }
}

const SortContext = struct {
    bytes: []const u8,
    element_type: BufferType,
    ascending: bool,
};

fn lessSortIndex(context: SortContext, lhs: usize, rhs: usize) bool {
    const lhs_value = readScalarAsF64(context.element_type, context.bytes, lhs);
    const rhs_value = readScalarAsF64(context.element_type, context.bytes, rhs);
    return if (context.ascending) lhs_value < rhs_value else lhs_value > rhs_value;
}

fn axisInList(axis: usize, axes: []const i64) bool {
    for (axes) |candidate| {
        if (candidate >= 0 and @as(usize, @intCast(candidate)) == axis) return true;
    }
    return false;
}

fn outputAxesWithout(input_rank: usize, removed_axes: []const i64, allocator: std.mem.Allocator) ![]usize {
    var axes = try allocator.alloc(usize, input_rank - removed_axes.len);
    var out: usize = 0;
    for (0..input_rank) |axis| {
        if (!axisInList(axis, removed_axes)) {
            axes[out] = axis;
            out += 1;
        }
    }
    return axes;
}

fn evalReduceF32(
    allocator: std.mem.Allocator,
    kind: core.PlanInstructionKind,
    src_bytes: []const u8,
    input_dims: []const i64,
    dimensions: []const i64,
    output_dims: []const i64,
    out_bytes: []u8,
) !void {
    const input_strides = try rowMajorStrides(allocator, input_dims);
    defer allocator.free(input_strides);
    const output_strides = try rowMajorStrides(allocator, output_dims);
    defer allocator.free(output_strides);
    const coords = try allocator.alloc(usize, input_dims.len);
    defer allocator.free(coords);
    const output_axes = try outputAxesWithout(input_dims.len, dimensions, allocator);
    defer allocator.free(output_axes);

    const output_count = if (output_dims.len == 0) 1 else denseByteSize(.f32, output_dims) / 4;
    for (0..output_count) |i| {
        writeF32LE(out_bytes, i, if (kind == .reduce_sum) 0.0 else -std.math.inf(f32));
    }
    const input_count = src_bytes.len / 4;
    for (0..input_count) |input_index| {
        unravelIndex(input_index, input_dims, input_strides, coords);
        var output_index: usize = 0;
        for (output_axes, 0..) |input_axis, output_axis| {
            output_index += coords[input_axis] * output_strides[output_axis];
        }
        const current = readF32LE(out_bytes, output_index);
        const value = readF32LE(src_bytes, input_index);
        writeF32LE(out_bytes, output_index, if (kind == .reduce_sum) current + value else @max(current, value));
    }
}

fn evalReducePred(
    allocator: std.mem.Allocator,
    kind: core.PlanInstructionKind,
    src_bytes: []const u8,
    input_dims: []const i64,
    dimensions: []const i64,
    output_dims: []const i64,
    out_bytes: []u8,
) !void {
    const input_strides = try rowMajorStrides(allocator, input_dims);
    defer allocator.free(input_strides);
    const output_strides = try rowMajorStrides(allocator, output_dims);
    defer allocator.free(output_strides);
    const coords = try allocator.alloc(usize, input_dims.len);
    defer allocator.free(coords);
    const output_axes = try outputAxesWithout(input_dims.len, dimensions, allocator);
    defer allocator.free(output_axes);

    const output_count = if (output_dims.len == 0) 1 else denseByteSize(.pred, output_dims);
    @memset(out_bytes[0..output_count], if (kind == .reduce_and) 1 else 0);
    for (src_bytes, 0..) |value, input_index| {
        unravelIndex(input_index, input_dims, input_strides, coords);
        var output_index: usize = 0;
        for (output_axes, 0..) |input_axis, output_axis| {
            output_index += coords[input_axis] * output_strides[output_axis];
        }
        out_bytes[output_index] = if (kind == .reduce_and)
            @intFromBool(out_bytes[output_index] != 0 and value != 0)
        else
            @intFromBool(out_bytes[output_index] != 0 or value != 0);
    }
}

fn evalDotGeneralF32(
    allocator: std.mem.Allocator,
    lhs_bytes: []const u8,
    lhs_dims: []const i64,
    rhs_bytes: []const u8,
    rhs_dims: []const i64,
    lhs_batch_dimensions: []const i64,
    rhs_batch_dimensions: []const i64,
    lhs_contracting_dimensions: []const i64,
    rhs_contracting_dimensions: []const i64,
    output_dims: []const i64,
    out_bytes: []u8,
) !void {
    const lhs_contract: usize = @intCast(lhs_contracting_dimensions[0]);
    const rhs_contract: usize = @intCast(rhs_contracting_dimensions[0]);
    const lhs_strides = try rowMajorStrides(allocator, lhs_dims);
    defer allocator.free(lhs_strides);
    const rhs_strides = try rowMajorStrides(allocator, rhs_dims);
    defer allocator.free(rhs_strides);
    const out_strides = try rowMajorStrides(allocator, output_dims);
    defer allocator.free(out_strides);
    const out_coords = try allocator.alloc(usize, output_dims.len);
    defer allocator.free(out_coords);

    const lhs_free_axes = try outputAxesWithout(lhs_dims.len, lhs_contracting_dimensions, allocator);
    defer allocator.free(lhs_free_axes);
    const rhs_free_axes = try outputAxesWithout(rhs_dims.len, rhs_contracting_dimensions, allocator);
    defer allocator.free(rhs_free_axes);
    const batch_count = lhs_batch_dimensions.len;
    const lhs_non_batch_count = lhs_free_axes.len - batch_count;
    const rhs_non_batch_count = rhs_free_axes.len - batch_count;
    const contract_size: usize = @intCast(lhs_dims[lhs_contract]);
    const output_count = out_bytes.len / 4;

    for (0..output_count) |out_index| {
        unravelIndex(out_index, output_dims, out_strides, out_coords);
        var lhs_base: usize = 0;
        var rhs_base: usize = 0;
        var out_axis: usize = 0;
        for (0..batch_count) |i| {
            const coord = out_coords[out_axis];
            lhs_base += coord * lhs_strides[@intCast(lhs_batch_dimensions[i])];
            rhs_base += coord * rhs_strides[@intCast(rhs_batch_dimensions[i])];
            out_axis += 1;
        }
        for (lhs_free_axes[batch_count..], 0..) |lhs_axis, i| {
            lhs_base += out_coords[out_axis + i] * lhs_strides[lhs_axis];
        }
        out_axis += lhs_non_batch_count;
        for (rhs_free_axes[batch_count..], 0..) |rhs_axis, i| {
            rhs_base += out_coords[out_axis + i] * rhs_strides[rhs_axis];
        }
        _ = rhs_non_batch_count;
        var acc: f32 = 0.0;
        for (0..contract_size) |k| {
            acc += readF32LE(lhs_bytes, lhs_base + k * lhs_strides[lhs_contract]) * readF32LE(rhs_bytes, rhs_base + k * rhs_strides[rhs_contract]);
        }
        writeF32LE(out_bytes, out_index, acc);
    }
}

fn mlxMetalBackendForTest() backend_api.Backend {
    return @import("src/backend/registry").create(.metal_mlx);
}

fn initMlxMetalClientForTest() !*Client {
    return Client.init(std.testing.allocator, mlxMetalBackendForTest(), 1);
}

fn initHostCopyForTest(
    element_type: BufferType,
    dims: []const i64,
    device: *Device,
    memory: *Memory,
    shard_index: usize,
    src: []const u8,
) !*Buffer {
    return Buffer.initHostCopyForBackend(std.testing.allocator, mlxMetalBackendForTest(), element_type, dims, device, memory, shard_index, src);
}

fn expectBufferBytes(buffer: *Buffer, expected: []const u8) !void {
    const actual = try std.testing.allocator.alloc(u8, expected.len);
    defer std.testing.allocator.free(actual);
    try buffer.copyToHost(actual);
    try std.testing.expectEqualSlices(u8, expected, actual);
}

fn expectBufferF32(buffer: *Buffer, expected: []const f32) !void {
    const actual = try std.testing.allocator.alloc(u8, expected.len * @sizeOf(f32));
    defer std.testing.allocator.free(actual);
    try buffer.copyToHost(actual);
    const floats = std.mem.bytesAsSlice(f32, actual);
    for (expected, floats) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0001);
    }
}

fn testShardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !core.ShardingPlan {
    return .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, assignment),
    };
}

test "executable graph materializes per-device scheduled nodes" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;

    var plan = core.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &.{0}),
        },
        .values = &.{},
        .parameter_shardings = try allocator.alloc(core.ShardingPlan, 0),
        .output_shardings = try allocator.alloc(core.ShardingPlan, 0),
        .output_ids = try allocator.alloc(core.ValueId, 0),
        .instructions = try allocator.dupe(core.PlanInstruction, &.{
            .{ .kind = .constant },
            .{ .kind = .custom_call },
            .{ .kind = .while_ },
        }),
    };
    defer plan.deinit();

    var graph = try ExecutableGraph.init(allocator, client, &plan);
    defer graph.deinit();

    try std.testing.expectEqualSlices(i32, &.{0}, graph.device_ids);
    try std.testing.expectEqual(@as(usize, 3), graph.nodes.len);
    try std.testing.expectEqual(GraphNodeKind.constant, graph.nodes[0].kind);
    try std.testing.expectEqual(GraphNodeKind.custom_call, graph.nodes[1].kind);
    try std.testing.expectEqual(GraphNodeKind.control_flow, graph.nodes[2].kind);
    try std.testing.expectEqual(@as(usize, 0), graph.nodes[0].device_index);
    try std.testing.expectEqual(@as(i32, 0), graph.nodes[0].device_id);
    try std.testing.expectEqual(LoweringMode.allow_runtime_fallback, graph.lowering.mode);
    try std.testing.expectEqual(false, graph.lowering.backend_executable_ready);
    try std.testing.expectEqual(@as(usize, 3), graph.lowering.fallback_instruction_count);
}

test "executable graph device-only lowering rejects unsupported backend executable" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;

    var plan = core.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &.{0}),
        },
        .values = &.{},
        .parameter_shardings = try allocator.alloc(core.ShardingPlan, 0),
        .output_shardings = try allocator.alloc(core.ShardingPlan, 0),
        .output_ids = try allocator.alloc(core.ValueId, 0),
        .instructions = try allocator.dupe(core.PlanInstruction, &.{
            .{ .kind = .custom_call },
        }),
    };
    defer plan.deinit();

    try std.testing.expectError(
        error.UnsupportedRuntimeFeature,
        ExecutableGraph.initWithOptions(allocator, client, &plan, .{ .mode = .require_backend_executable }),
    );
}

test "executable graph executes through runtime buffers" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    const values = try allocator.alloc(core.Value, 3);
    errdefer allocator.free(values);
    for (values, 0..) |*value, i| {
        value.* = .{
            .id = .{ .index = @intCast(i) },
            .role = if (i < 2) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    var parameter_shardings = try allocator.alloc(core.ShardingPlan, 2);
    parameter_shardings[0] = try testShardingPlan(allocator, &assignment);
    parameter_shardings[1] = try testShardingPlan(allocator, &assignment);
    var output_shardings = try allocator.alloc(core.ShardingPlan, 1);
    output_shardings[0] = try testShardingPlan(allocator, &assignment);

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
            .kind = .add,
            .inputs = try allocator.dupe(core.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }}),
        }}),
    };
    defer plan.deinit();

    var graph = try ExecutableGraph.init(allocator, client, &plan);
    defer graph.deinit();
    try std.testing.expect(graph.backend_executable != null);
    try std.testing.expectEqual(LoweringMode.allow_runtime_fallback, graph.lowering.mode);
    try std.testing.expectEqual(true, graph.lowering.backend_executable_ready);
    try std.testing.expectEqual(@as(usize, 1), graph.lowering.lowered_instruction_count);
    try std.testing.expectEqual(@as(usize, 0), graph.lowering.fallback_instruction_count);

    const lhs_data = [_]u8{ 1, 2, 3, 4 };
    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const lhs = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &lhs_data);
    defer lhs.deinit();
    const rhs = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &rhs_data);
    defer rhs.deinit();

    const h2d_before_execute = client.memories[0].stats.host_to_device_bytes;
    const d2h_before_execute = client.memories[0].stats.device_to_host_bytes;
    const outputs = try graph.executeDevice(allocator, client, &plan, 0, &.{ lhs, rhs });
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| output.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 0), outputs[0].shard_index);
    try std.testing.expect(outputs[0].backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), outputs[0].bytes.len);
    try std.testing.expectEqual(h2d_before_execute, client.memories[0].stats.host_to_device_bytes);
    try std.testing.expectEqual(d2h_before_execute, client.memories[0].stats.device_to_host_bytes);
    try expectBufferBytes(outputs[0], &.{ 11, 22, 33, 44 });
    try std.testing.expectEqual(d2h_before_execute + outputs[0].byte_size, client.memories[0].stats.device_to_host_bytes);
}

test "buffer keeps shard/device/memory ownership metadata" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 2, 2 };
    const data = [_]u8{ 1, 2, 3, 4 };
    const buffer = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
    defer buffer.deinit();

    try std.testing.expectEqual(@as(i32, 0), buffer.device_id);
    try std.testing.expectEqual(@as(i32, 0), buffer.memory_id);
    try std.testing.expectEqual(@as(i32, 0), buffer.device.id);
    try std.testing.expectEqual(@as(i32, 0), buffer.memory.id);
    try std.testing.expectEqual(@as(usize, 0), buffer.shard_index);
    try expectBufferBytes(buffer, &data);

    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const rhs = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &rhs_data);
    defer rhs.deinit();
    const sum = try Buffer.initU8Add(std.testing.allocator, buffer, rhs, 0);
    defer sum.deinit();
    try expectBufferBytes(sum, &.{ 11, 22, 33, 44 });

    const difference = try Buffer.initU8Binary(std.testing.allocator, .subtract, rhs, buffer, 0);
    defer difference.deinit();
    try expectBufferBytes(difference, &.{ 9, 18, 27, 36 });

    const product = try Buffer.initU8Binary(std.testing.allocator, .multiply, buffer, rhs, 0);
    defer product.deinit();
    try expectBufferBytes(product, &.{ 10, 40, 90, 160 });

    const quotient = try Buffer.initU8Binary(std.testing.allocator, .divide, rhs, buffer, 0);
    defer quotient.deinit();
    try expectBufferBytes(quotient, &.{ 10, 10, 10, 10 });

    const negated = try Buffer.initU8Unary(std.testing.allocator, .negate, buffer, 0);
    defer negated.deinit();
    try expectBufferBytes(negated, &.{ 255, 254, 253, 252 });
}

test "buffer lifecycle rejects deleted and donated buffers" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const deleted = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
    defer deleted.deinit();
    deleted.markDeleted();
    try std.testing.expectEqual(BufferState.deleted, deleted.state);
    try std.testing.expect(deleted.deleted);
    try std.testing.expectError(error.BufferDeleted, deleted.ensureUsable());

    const donated = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
    defer donated.deinit();
    donated.markDonated();
    try std.testing.expectEqual(BufferState.donated, donated.state);
    try std.testing.expect(donated.deleted);
    try std.testing.expectError(error.BufferDonated, donated.ensureUsable());
}

test "client records executable cache hits and memory byte accounting" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("same-program"));
    try std.testing.expect(try client.recordExecutableCompile("same-program"));
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.misses);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.hits);

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const before = client.memories[0].stats.bytes_in_use;
    {
        const buffer = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
        defer buffer.deinit();
        try std.testing.expectEqual(before + data.len, client.memories[0].stats.bytes_in_use);
        try std.testing.expect(client.memories[0].stats.peak_bytes_in_use >= client.memories[0].stats.bytes_in_use);
        try std.testing.expect(client.memories[0].stats.host_to_device_bytes >= data.len);
    }
    try std.testing.expectEqual(before, client.memories[0].stats.bytes_in_use);
}

test "buffer elementwise arithmetic supports f32 execution" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{2};
    const lhs_values = [_]f32{ 1.5, -2.0 };
    const rhs_values = [_]f32{ 2.25, 4.0 };
    const lhs = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&lhs_values));
    defer lhs.deinit();
    const rhs = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&rhs_values));
    defer rhs.deinit();

    const sum = try Buffer.initElementwiseBinary(std.testing.allocator, .add, lhs, rhs, 0);
    defer sum.deinit();
    try expectBufferF32(sum, &.{ 3.75, 2.0 });

    const quotient = try Buffer.initElementwiseBinary(std.testing.allocator, .divide, rhs, lhs, 0);
    defer quotient.deinit();
    try expectBufferF32(quotient, &.{ 1.5, -2.0 });

    const negated = try Buffer.initElementwiseUnary(std.testing.allocator, .negate, lhs, 0);
    defer negated.deinit();
    try expectBufferF32(negated, &.{ -1.5, 2.0 });
}

test "buffer convert uses backend astype for resident buffers" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{2};
    const input = [_]u8{ 1, 255 };
    const source = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &input);
    defer source.deinit();

    const converted = try Buffer.initConvert(std.testing.allocator, source, .f32, &dims, 0);
    defer converted.deinit();
    try std.testing.expect(converted.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), converted.bytes.len);
    try expectBufferF32(converted, &.{ 1.0, 255.0 });
}

test "buffer iota uses resident mlx backend path" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 2, 3 };
    const buffer = try Buffer.initIota(
        std.testing.allocator,
        client.backend,
        .f32,
        &dims,
        &client.devices[0],
        &client.memories[0],
        1,
        0,
    );
    defer buffer.deinit();
    try std.testing.expect(buffer.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), buffer.bytes.len);
    try expectBufferF32(buffer, &.{ 0.0, 1.0, 2.0, 0.0, 1.0, 2.0 });
}

test "buffer movement ops use resident mlx backend paths" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 3, 4 };
    const values = [_]f32{
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
    };
    const source = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
    defer source.deinit();

    const start0: i32 = 1;
    const start1: i32 = 1;
    const start0_buffer = try initHostCopyForTest(.s32, &.{}, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&start0));
    defer start0_buffer.deinit();
    const start1_buffer = try initHostCopyForTest(.s32, &.{}, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&start1));
    defer start1_buffer.deinit();

    const slice_dims = [_]i64{ 2, 2 };
    const sliced = try Buffer.initDynamicSlice(std.testing.allocator, source, &.{ start0_buffer, start1_buffer }, &slice_dims, &slice_dims, 0);
    defer sliced.deinit();
    try std.testing.expect(sliced.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), sliced.bytes.len);
    try expectBufferF32(sliced, &.{ 6.0, 7.0, 10.0, 11.0 });

    const update_values = [_]f32{ 100.0, 101.0, 102.0, 103.0 };
    const update = try initHostCopyForTest(.f32, &slice_dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&update_values));
    defer update.deinit();
    const updated = try Buffer.initDynamicUpdateSlice(std.testing.allocator, source, update, &.{ start0_buffer, start1_buffer }, &dims, 0);
    defer updated.deinit();
    try std.testing.expect(updated.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), updated.bytes.len);
    try expectBufferF32(updated, &.{
        1.0, 2.0,   3.0,   4.0,
        5.0, 100.0, 101.0, 8.0,
        9.0, 102.0, 103.0, 12.0,
    });

    const pad_input_dims = [_]i64{2};
    const pad_output_dims = [_]i64{5};
    const pad_input_values = [_]f32{ 2.0, 3.0 };
    const pad_value: f32 = 0.0;
    const pad_input = try initHostCopyForTest(.f32, &pad_input_dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&pad_input_values));
    defer pad_input.deinit();
    const pad_value_buffer = try initHostCopyForTest(.f32, &.{}, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&pad_value));
    defer pad_value_buffer.deinit();
    const padded = try Buffer.initPad(std.testing.allocator, pad_input, pad_value_buffer, &.{1}, &.{2}, &.{0}, &pad_output_dims, 0);
    defer padded.deinit();
    try std.testing.expect(padded.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), padded.bytes.len);
    try expectBufferF32(padded, &.{ 0.0, 2.0, 3.0, 0.0, 0.0 });

    const reverse_dims = [_]i64{ 2, 3 };
    const reverse_values = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const reverse_source = try initHostCopyForTest(.f32, &reverse_dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&reverse_values));
    defer reverse_source.deinit();
    const reversed = try Buffer.initReverse(std.testing.allocator, reverse_source, &.{1}, &reverse_dims, 0);
    defer reversed.deinit();
    try std.testing.expect(reversed.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), reversed.bytes.len);
    try expectBufferF32(reversed, &.{ 3.0, 2.0, 1.0, 6.0, 5.0, 4.0 });

    const gather_operand_dims = [_]i64{ 3, 2 };
    const gather_output_dims = [_]i64{ 2, 2 };
    const gather_operand_values = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const gather_indices_values = [_]i32{ 2, 0 };
    const gather_operand = try initHostCopyForTest(.f32, &gather_operand_dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&gather_operand_values));
    defer gather_operand.deinit();
    const gather_indices = try initHostCopyForTest(.s32, &.{2}, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&gather_indices_values));
    defer gather_indices.deinit();
    const gathered = try Buffer.initGather(
        std.testing.allocator,
        gather_operand,
        gather_indices,
        &.{1},
        &.{0},
        &.{},
        &.{},
        &.{0},
        1,
        &.{ 1, 2 },
        &gather_output_dims,
        0,
    );
    defer gathered.deinit();
    try std.testing.expect(gathered.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), gathered.bytes.len);
    try expectBufferF32(gathered, &.{ 5.0, 6.0, 1.0, 2.0 });

    const sort_dims = [_]i64{ 2, 3 };
    const sort_values = [_]f32{ 3.0, 1.0, 2.0, 6.0, 4.0, 5.0 };
    const sort_source = try initHostCopyForTest(.f32, &sort_dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&sort_values));
    defer sort_source.deinit();
    const sorted = try Buffer.initSort(std.testing.allocator, sort_source, 1, &sort_dims, .lt, 0);
    defer sorted.deinit();
    try std.testing.expect(sorted.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), sorted.bytes.len);
    try expectBufferF32(sorted, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    const reverse_sorted = try Buffer.initSort(std.testing.allocator, sort_source, 1, &sort_dims, .gt, 0);
    defer reverse_sorted.deinit();
    try std.testing.expect(reverse_sorted.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), reverse_sorted.bytes.len);
    try expectBufferF32(reverse_sorted, &.{ 3.0, 2.0, 1.0, 6.0, 5.0, 4.0 });
}

test "buffer elementwise unary math supports f32 execution" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{2};
    const values = [_]f32{ 1.0, 4.0 };
    const input = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
    defer input.deinit();

    const exp = try Buffer.initElementwiseUnary(std.testing.allocator, .exp, input, 0);
    defer exp.deinit();
    try expectBufferF32(exp, &.{ std.math.exp(@as(f32, 1.0)), std.math.exp(@as(f32, 4.0)) });

    const tanh = try Buffer.initElementwiseUnary(std.testing.allocator, .tanh, input, 0);
    defer tanh.deinit();
    try expectBufferF32(tanh, &.{ std.math.tanh(@as(f32, 1.0)), std.math.tanh(@as(f32, 4.0)) });

    const sqrt = try Buffer.initElementwiseUnary(std.testing.allocator, .sqrt, input, 0);
    defer sqrt.deinit();
    try expectBufferF32(sqrt, &.{ 1.0, 2.0 });

    const rsqrt = try Buffer.initElementwiseUnary(std.testing.allocator, .rsqrt, input, 0);
    defer rsqrt.deinit();
    try expectBufferF32(rsqrt, &.{ 1.0, 0.5 });
}

test "buffer reshape preserves typed bytes and updates dimensions" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 2, 2 };
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
    defer input.deinit();

    const reshaped = try Buffer.initReshape(std.testing.allocator, input, &.{4}, 0);
    defer reshaped.deinit();

    try std.testing.expectEqualSlices(i64, &.{4}, reshaped.dims);
    try expectBufferF32(reshaped, &values);
}

test "buffer transpose permutes dense host bytes and updates dimensions" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 2, 3 };
    const values = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const input = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
    defer input.deinit();

    const transposed = try Buffer.initTranspose(std.testing.allocator, input, &.{ 1, 0 }, &.{ 3, 2 }, 0);
    defer transposed.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, transposed.dims);
    try expectBufferBytes(transposed, &.{ 1, 4, 2, 5, 3, 6 });
}

test "buffer broadcast_in_dim expands dense host bytes and updates dimensions" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{3};
    const values = [_]u8{ 7, 8, 9 };
    const input = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
    defer input.deinit();

    const broadcasted = try Buffer.initBroadcastInDim(std.testing.allocator, input, &.{1}, &.{ 2, 3 }, 0);
    defer broadcasted.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, broadcasted.dims);
    try expectBufferBytes(broadcasted, &.{ 7, 8, 9, 7, 8, 9 });
}

test "buffer slice copies strided dense host bytes and updates dimensions" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 3, 4 };
    const values = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const input = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
    defer input.deinit();

    const sliced = try Buffer.initSlice(
        std.testing.allocator,
        input,
        &.{ 1, 0 },
        &.{ 3, 4 },
        &.{ 1, 2 },
        &.{ 2, 2 },
        0,
    );
    defer sliced.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, sliced.dims);
    try expectBufferBytes(sliced, &.{ 5, 7, 9, 11 });
}

test "buffer concatenate joins dense host bytes along an axis" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const lhs_dims = [_]i64{ 2, 2 };
    const rhs_dims = [_]i64{ 2, 3 };
    const lhs_values = [_]u8{ 1, 2, 3, 4 };
    const rhs_values = [_]u8{ 5, 6, 7, 8, 9, 10 };
    const lhs = try initHostCopyForTest(.u8, &lhs_dims, &client.devices[0], &client.memories[0], 0, &lhs_values);
    defer lhs.deinit();
    const rhs = try initHostCopyForTest(.u8, &rhs_dims, &client.devices[0], &client.memories[0], 0, &rhs_values);
    defer rhs.deinit();

    const concatenated = try Buffer.initConcatenate(std.testing.allocator, lhs, rhs, 1, &.{ 2, 5 }, 0);
    defer concatenated.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 5 }, concatenated.dims);
    try expectBufferBytes(concatenated, &.{ 1, 2, 5, 6, 7, 3, 4, 8, 9, 10 });
}

test "buffer partition_id materializes scalar device partition" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const partition = try Buffer.initPartitionId(std.testing.allocator, mlxMetalBackendForTest(), .u32, &.{}, &client.devices[0], &client.memories[0], 0, 0);
    defer partition.deinit();

    try std.testing.expectEqual(@as(i32, 0), partition.device_id);
    var bytes: [4]u8 = undefined;
    try partition.copyToHost(&bytes);
    try std.testing.expectEqual(@as(u32, 0), readU32LE(&bytes, 0));
}

test "buffer cholesky requires backend-native lowering on MLX buffers" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 2, 2 };
    const values = [_]f32{ 4.0, 2.0, 2.0, 3.0 };
    const input = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
    defer input.deinit();

    try std.testing.expectError(error.UnsupportedRuntimeFeature, Buffer.initCholesky(std.testing.allocator, input, true, &dims, 0));
}

test "buffer rng requires backend-native lowering on MLX buffers" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const scalar = [_]i64{};
    const min_value = [_]f32{0.0};
    const max_value = [_]f32{1.0};
    const min = try initHostCopyForTest(.f32, &scalar, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&min_value));
    defer min.deinit();
    const max = try initHostCopyForTest(.f32, &scalar, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&max_value));
    defer max.deinit();

    try std.testing.expectError(error.UnsupportedRuntimeFeature, Buffer.initRngUniform(std.testing.allocator, min, max, .f32, &.{4}, 0));

    const state_words = [_]u32{ 1, 2 };
    const state = try initHostCopyForTest(.u32, &.{2}, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&state_words));
    defer state.deinit();
    try std.testing.expectError(error.UnsupportedRuntimeFeature, Buffer.initRngBits(std.testing.allocator, state, .u32, &.{4}, 0));
    try std.testing.expectError(error.UnsupportedRuntimeFeature, Buffer.initRngStateUpdate(std.testing.allocator, state, 0));
}
