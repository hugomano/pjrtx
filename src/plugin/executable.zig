const std = @import("std");

const c = @import("c");
const runtime = @import("src/runtime");

const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");
const handles = @import("pjrt_handles.zig");
const plugin = @import("plugin.zig");

const LoadedExecutableHandle = handles.LoadedExecutable(Executable);
const ExecutableHandle = handles.Executable(ExecutableMetadata);
const PjrtError = errors.Error;

const ExecutableOwned = struct {
    client: *runtime.Client,
    plan: *runtime.ExecutablePlan,
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
};

/// Opaque loaded-executable handle owning a runtime plan and resident graph.
pub const Executable = opaque {
    /// Takes ownership of a compiled runtime executable and prepares PJRT metadata.
    pub fn create(client: *runtime.Client, compiled: *runtime.CompiledExecutable) !*Executable {
        const plan = compiled.plan;

        const logical_ids = try plugin.allocator().alloc(c.PJRT_LogicalDeviceIds, plan.options.numDevices());
        errdefer plugin.allocator().free(logical_ids);
        for (logical_ids, 0..) |*id, index| {
            id.* = .{
                .replica = @intCast(index / @as(usize, @intCast(plan.options.num_partitions))),
                .partition = @intCast(index % @as(usize, @intCast(plan.options.num_partitions))),
            };
        }

        const parameter_memory_kinds = try plugin.allocator().alloc([*c]const u8, plan.parameter_shardings.len);
        errdefer plugin.allocator().free(parameter_memory_kinds);
        const parameter_memory_kind_sizes = try plugin.allocator().alloc(usize, plan.parameter_shardings.len);
        errdefer plugin.allocator().free(parameter_memory_kind_sizes);
        MemoryKinds.fillDefault(parameter_memory_kinds, parameter_memory_kind_sizes);

        const output_memory_kinds = try plugin.allocator().alloc([*c]const u8, plan.output_shardings.len);
        errdefer plugin.allocator().free(output_memory_kinds);
        const output_memory_kind_sizes = try plugin.allocator().alloc(usize, plan.output_shardings.len);
        errdefer plugin.allocator().free(output_memory_kind_sizes);
        MemoryKinds.fillDefault(output_memory_kinds, output_memory_kind_sizes);

        const executable = try plugin.allocator().create(ExecutableOwned);
        executable.* = .{
            .client = client,
            .plan = compiled.plan,
            .graph = compiled.graph,
            .logical_ids = logical_ids,
            .optimized_program = compiled.optimized_program,
            .parameter_memory_kinds = parameter_memory_kinds,
            .parameter_memory_kind_sizes = parameter_memory_kind_sizes,
            .output_memory_kinds = output_memory_kinds,
            .output_memory_kind_sizes = output_memory_kind_sizes,
            .fingerprint = compiled.fingerprint,
        };
        return @ptrCast(executable);
    }

    /// Releases the runtime plan, backend graph, and PJRT-owned metadata.
    pub fn deinit(self: *Executable) void {
        const executable = self.owned();
        self.releaseGraph();
        plugin.allocator().free(executable.fingerprint);
        plugin.allocator().free(executable.output_memory_kind_sizes);
        plugin.allocator().free(executable.output_memory_kinds);
        plugin.allocator().free(executable.parameter_memory_kind_sizes);
        plugin.allocator().free(executable.parameter_memory_kinds);
        plugin.allocator().free(executable.optimized_program);
        plugin.allocator().free(executable.logical_ids);
        executable.plan.deinit();
        plugin.allocator().destroy(executable.plan);
        plugin.allocator().destroy(executable);
    }

    /// Releases backend graph residency while keeping PJRT metadata queryable.
    pub fn releaseGraph(self: *Executable) void {
        const executable = self.owned();
        if (executable.graph_released) return;
        executable.graph.deinit();
        executable.graph_released = true;
    }

    /// Returns whether PJRT callers may still execute this loaded executable.
    pub fn isDeleted(self: *const Executable) bool {
        return self.ownedConst().deleted;
    }

    /// Returns the number of PJRT parameters expected on each execution device.
    pub fn parameterCount(self: *const Executable) usize {
        return self.ownedConst().plan.parameter_shardings.len;
    }

    /// Returns the number of PJRT outputs produced on each execution device.
    pub fn outputCount(self: *const Executable) usize {
        return self.ownedConst().plan.output_ids.len;
    }

    /// Returns the number of per-device graph plans embedded in this executable.
    pub fn graphDeviceCount(self: *const Executable) usize {
        return self.ownedConst().graph.device_ids.len;
    }

    /// Returns whether executing may consume ownership of a parameter buffer.
    pub fn donatesParameter(self: *const Executable, parameter_index: usize) bool {
        for (self.ownedConst().plan.donated_parameter_indices) |candidate| {
            if (candidate == parameter_index) return true;
        }
        return false;
    }

    /// Executes the resident backend graph for one logical device.
    pub fn executeDevice(self: *Executable, device_index: usize, arguments: []const *runtime.Buffer) runtime.GraphExecuteError!runtime.GraphExecuteResult {
        const executable = self.owned();
        return runtime.executeDevice(&executable.graph, plugin.allocator(), executable.client.executableContext(), executable.plan, device_index, arguments);
    }

    /// Narrow test access for executable invariants that are not PJRT API surface.
    pub const Testing = struct {
        /// Overrides donated parameters for focused PJRT ABI tests.
        pub fn setDonatedParameters(executable: *Executable, donated_parameter_indices: []u32) void {
            executable.owned().plan.donated_parameter_indices = donated_parameter_indices;
        }

        /// Returns backend executable statistics for focused PJRT ABI tests.
        pub fn backendExecutableStats(executable: *const Executable) ?runtime.ExecutableStats {
            return executable.ownedConst().graph.backendExecutableStats();
        }

        /// Returns the last compile-cache trim recorded by graph lowering for tests.
        pub fn lastCompileCacheTrim(executable: *const Executable) runtime.ExecutableCacheTrim {
            return executable.ownedConst().graph.last_compile_cache_trim;
        }
    };

    fn owned(self: *Executable) *ExecutableOwned {
        return @ptrCast(@alignCast(self));
    }

    fn ownedConst(self: *const Executable) *const ExecutableOwned {
        return @ptrCast(@alignCast(self));
    }
};

/// Owns PJRT_Executable metadata snapshots created from loaded executables.
pub const ExecutableMetadata = struct {
    name: []u8,
    fingerprint: []u8,
    optimized_program: []u8,
    num_replicas: usize,
    num_partitions: usize,
    num_outputs: usize,
    parameter_memory_kinds: [][*c]const u8,
    parameter_memory_kind_sizes: []usize,
    output_memory_kinds: [][*c]const u8,
    output_memory_kind_sizes: []usize,

    pub const Api = struct {
        pub const Destroy = ExecutableCallback(c.PJRT_Executable_Destroy_Args, .destroy).call;
        pub const Name = ExecutableTextCallback(c.PJRT_Executable_Name_Args, "executable_name", "executable_name_size", .name).call;
        pub const NumReplicas = ExecutableScalarCallback(c.PJRT_Executable_NumReplicas_Args, "num_replicas", .num_replicas).call;
        pub const NumPartitions = ExecutableScalarCallback(c.PJRT_Executable_NumPartitions_Args, "num_partitions", .num_partitions).call;
        pub const NumOutputs = ExecutableScalarCallback(c.PJRT_Executable_NumOutputs_Args, "num_outputs", .num_outputs).call;
        pub const OptimizedProgram = ExecutableCallback(c.PJRT_Executable_OptimizedProgram_Args, .optimized_program).call;
        pub const Fingerprint = ExecutableTextCallback(c.PJRT_Executable_Fingerprint_Args, "executable_fingerprint", "executable_fingerprint_size", .fingerprint).call;
        pub const ParameterMemoryKinds = MemoryKindsCallback(c.PJRT_Executable_ParameterMemoryKinds_Args, "num_parameters", .parameters).call;
        pub const OutputMemoryKinds = MemoryKindsCallback(c.PJRT_Executable_OutputMemoryKinds_Args, "num_outputs", .outputs).call;
    };

    /// Copies queryable executable metadata into a separately owned PJRT handle.
    fn create(source: *const Executable) !*ExecutableMetadata {
        const executable = source.ownedConst();
        const metadata = try plugin.allocator().create(ExecutableMetadata);
        errdefer plugin.allocator().destroy(metadata);

        const name = try plugin.allocator().dupe(u8, executable.name);
        errdefer plugin.allocator().free(name);
        const fingerprint = try plugin.allocator().dupe(u8, executable.fingerprint);
        errdefer plugin.allocator().free(fingerprint);
        const optimized_program = try plugin.allocator().dupe(u8, executable.optimized_program);
        errdefer plugin.allocator().free(optimized_program);

        const parameter_memory_kinds = try plugin.allocator().dupe([*c]const u8, executable.parameter_memory_kinds);
        errdefer plugin.allocator().free(parameter_memory_kinds);
        const parameter_memory_kind_sizes = try plugin.allocator().dupe(usize, executable.parameter_memory_kind_sizes);
        errdefer plugin.allocator().free(parameter_memory_kind_sizes);
        const output_memory_kinds = try plugin.allocator().dupe([*c]const u8, executable.output_memory_kinds);
        errdefer plugin.allocator().free(output_memory_kinds);
        const output_memory_kind_sizes = try plugin.allocator().dupe(usize, executable.output_memory_kind_sizes);
        errdefer plugin.allocator().free(output_memory_kind_sizes);

        metadata.* = .{
            .name = name,
            .fingerprint = fingerprint,
            .optimized_program = optimized_program,
            .num_replicas = @intCast(executable.plan.options.num_replicas),
            .num_partitions = @intCast(executable.plan.options.num_partitions),
            .num_outputs = executable.plan.output_shardings.len,
            .parameter_memory_kinds = parameter_memory_kinds,
            .parameter_memory_kind_sizes = parameter_memory_kind_sizes,
            .output_memory_kinds = output_memory_kinds,
            .output_memory_kind_sizes = output_memory_kind_sizes,
        };
        return metadata;
    }

    /// Releases the copied PJRT executable metadata.
    fn deinit(self: *ExecutableMetadata) void {
        plugin.allocator().free(self.output_memory_kind_sizes);
        plugin.allocator().free(self.output_memory_kinds);
        plugin.allocator().free(self.parameter_memory_kind_sizes);
        plugin.allocator().free(self.parameter_memory_kinds);
        plugin.allocator().free(self.optimized_program);
        plugin.allocator().free(self.fingerprint);
        plugin.allocator().free(self.name);
        plugin.allocator().destroy(self);
    }
};

const DeviceAssignment = struct {
    fn delete(_: ?*c.PJRT_DeviceAssignmentSerialized) callconv(.c) void {}
};

const ExecutableRef = struct {
    ptr: *ExecutableMetadata,

    fn at(raw: anytype) ExecutableRef {
        return .{ .ptr = ExecutableHandle.ref(raw) };
    }

    fn name(self: ExecutableRef) []const u8 {
        return self.ptr.name;
    }

    fn fingerprint(self: ExecutableRef) []const u8 {
        return self.ptr.fingerprint;
    }

    fn numReplicas(self: ExecutableRef) usize {
        return self.ptr.num_replicas;
    }

    fn numPartitions(self: ExecutableRef) usize {
        return self.ptr.num_partitions;
    }

    fn numOutputs(self: ExecutableRef) usize {
        return self.ptr.num_outputs;
    }

    fn writeText(self: ExecutableRef, comptime ptr_field: []const u8, comptime size_field: []const u8, raw: anytype, text: []const u8) void {
        _ = self;
        abi.Out.writeBytes(ptr_field, size_field, &raw[0], text);
    }
};

/// Borrowed PJRT loaded-executable reference owning lifecycle callbacks.
pub const LoadedExecutable = struct {
    ptr: *Executable,

    pub const Api = struct {
        pub const Destroy = LoadedExecutableCallback(c.PJRT_LoadedExecutable_Destroy_Args, .destroy).call;
        pub const GetExecutable = LoadedExecutableCallback(c.PJRT_LoadedExecutable_GetExecutable_Args, .get_executable).call;
        pub const AddressableDevices = LoadedExecutableCallback(c.PJRT_LoadedExecutable_AddressableDevices_Args, .addressable_devices).call;
        pub const AddressableDeviceLogicalIds = LoadedExecutableCallback(c.PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args, .logical_ids).call;
        pub const GetDeviceAssignment = LoadedExecutableCallback(c.PJRT_LoadedExecutable_GetDeviceAssignment_Args, .device_assignment).call;
        pub const Fingerprint = LoadedExecutableTextCallback(c.PJRT_LoadedExecutable_Fingerprint_Args, "executable_fingerprint", "executable_fingerprint_size", .fingerprint).call;
        pub const Delete = LoadedExecutableCallback(c.PJRT_LoadedExecutable_Delete_Args, .delete).call;
        pub const IsDeleted = LoadedExecutableCallback(c.PJRT_LoadedExecutable_IsDeleted_Args, .is_deleted).call;
    };

    fn at(raw: anytype) LoadedExecutable {
        return .{ .ptr = LoadedExecutableHandle.ref(raw) };
    }

    fn createExecutableMetadata(self: LoadedExecutable) !*c.PJRT_Executable {
        return ExecutableHandle.handle(try ExecutableMetadata.create(self.ptr));
    }

    fn addressableDeviceCount(self: LoadedExecutable) usize {
        const executable = self.ptr.ownedConst();
        return @min(executable.plan.options.numDevices(), executable.client.device_handles.len);
    }

    fn addressableDevices(self: LoadedExecutable) []const *runtime.Device {
        const executable = self.ptr.ownedConst();
        const count = self.addressableDeviceCount();
        return executable.client.device_handles[0..count];
    }

    fn delete(self: LoadedExecutable) void {
        self.ptr.owned().deleted = true;
        self.ptr.releaseGraph();
    }

    fn isDeleted(self: LoadedExecutable) bool {
        return self.ptr.isDeleted();
    }

    fn writeAddressableDevices(self: LoadedExecutable, args: anytype) void {
        const devices = self.addressableDevices();
        args.addressable_devices = handles.Device.handleSlice(devices);
        args.num_addressable_devices = devices.len;
    }

    fn writeLogicalIds(self: LoadedExecutable, args: anytype) void {
        const executable = self.ptr.ownedConst();
        args.addressable_device_logical_ids = executable.logical_ids.ptr;
        args.num_addressable_device_logical_ids = executable.logical_ids.len;
    }

    fn writeEmptyDeviceAssignment(_: LoadedExecutable, args: anytype) void {
        args.serialized_bytes = null;
        args.serialized_bytes_size = 0;
        args.serialized_device_assignment = null;
        args.serialized_device_assignment_deleter = DeviceAssignment.delete;
    }
};

const ExecutableScalar = enum {
    num_replicas,
    num_partitions,
    num_outputs,

    fn value(comptime scalar: ExecutableScalar, executable: ExecutableRef) usize {
        return switch (scalar) {
            .num_replicas => executable.numReplicas(),
            .num_partitions => executable.numPartitions(),
            .num_outputs => executable.numOutputs(),
        };
    }
};

fn ExecutableScalarCallback(comptime Args: type, comptime field: []const u8, comptime scalar: ExecutableScalar) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            @field(raw[0], field) = scalar.value(ExecutableRef.at(raw[0].executable));
            return null;
        }
    };
}

fn ExecutableTextCallback(
    comptime Args: type,
    comptime ptr_field: []const u8,
    comptime size_field: []const u8,
    comptime text: enum { name, fingerprint },
) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const executable = ExecutableRef.at(raw[0].executable);
            executable.writeText(ptr_field, size_field, raw, switch (text) {
                .name => executable.name(),
                .fingerprint => executable.fingerprint(),
            });
            return null;
        }
    };
}

fn LoadedExecutableTextCallback(
    comptime Args: type,
    comptime ptr_field: []const u8,
    comptime size_field: []const u8,
    comptime text: enum { fingerprint },
) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const loaded = LoadedExecutable.at(raw[0].executable);
            const executable = loaded.ptr.ownedConst();
            abi.Out.writeBytes(ptr_field, size_field, &raw[0], switch (text) {
                .fingerprint => executable.fingerprint,
            });
            return null;
        }
    };
}

const MemoryKinds = struct {
    kinds: [][*c]const u8,
    sizes: []usize,

    fn parameters(executable: ExecutableRef) MemoryKinds {
        return .{
            .kinds = executable.ptr.parameter_memory_kinds,
            .sizes = executable.ptr.parameter_memory_kind_sizes,
        };
    }

    fn outputs(executable: ExecutableRef) MemoryKinds {
        return .{
            .kinds = executable.ptr.output_memory_kinds,
            .sizes = executable.ptr.output_memory_kind_sizes,
        };
    }

    fn write(self: MemoryKinds, comptime count_field: []const u8, raw: anytype) void {
        @field(raw[0], count_field) = self.kinds.len;
        raw[0].memory_kinds = abi.Slice.ptrList(self.kinds);
        raw[0].memory_kind_sizes = self.sizes.ptr;
    }

    /// Writes the default plugin memory kind into parallel PJRT output arrays.
    fn fillDefault(kinds: [][*c]const u8, sizes: []usize) void {
        for (kinds, sizes) |*kind, *size| {
            kind.* = plugin.MemoryKinds.device.ptr;
            size.* = plugin.MemoryKinds.device.len;
        }
    }
};

fn MemoryKindsCallback(comptime Args: type, comptime count_field: []const u8, comptime which: enum { parameters, outputs }) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const executable = ExecutableRef.at(raw[0].executable);
            const kinds = switch (which) {
                .parameters => MemoryKinds.parameters(executable),
                .outputs => MemoryKinds.outputs(executable),
            };
            kinds.write(count_field, raw);
            return null;
        }
    };
}

const LoadedExecutableOp = enum {
    destroy,
    get_executable,
    addressable_devices,
    logical_ids,
    device_assignment,
    delete,
    is_deleted,
};

fn LoadedExecutableCallback(comptime Args: type, comptime op: LoadedExecutableOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const args = &raw[0];
            const loaded = loadedExecutable(args);
            switch (op) {
                .destroy => loaded.ptr.deinit(),
                .get_executable => args.executable = loaded.createExecutableMetadata() catch return PjrtError.resourceExhausted("failed to allocate PJRT executable metadata"),
                .addressable_devices => loaded.writeAddressableDevices(args),
                .logical_ids => loaded.writeLogicalIds(args),
                .device_assignment => loaded.writeEmptyDeviceAssignment(args),
                .delete => loaded.delete(),
                .is_deleted => args.is_deleted = loaded.isDeleted(),
            }
            return null;
        }

        fn loadedExecutable(args: anytype) LoadedExecutable {
            return switch (op) {
                .get_executable => LoadedExecutable.at(args.loaded_executable),
                .destroy,
                .addressable_devices,
                .logical_ids,
                .device_assignment,
                .delete,
                .is_deleted,
                => LoadedExecutable.at(args.executable),
            };
        }
    };
}

const ExecutableOp = enum { destroy, optimized_program };

fn ExecutableCallback(comptime Args: type, comptime op: ExecutableOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const args = &raw[0];
            switch (op) {
                .destroy => {
                    if (args.executable) |executable| ExecutableRef.at(executable).ptr.deinit();
                },
                .optimized_program => writeOptimizedProgram(ExecutableRef.at(args.executable), args),
            }
            return null;
        }

        fn writeOptimizedProgram(executable: ExecutableRef, args: anytype) void {
            const program = args.program;
            program[0].format = "mlir";
            program[0].format_size = "mlir".len;
            program[0].code_size = executable.ptr.optimized_program.len;
            if (program[0].code) |code| {
                @memcpy(code[0..executable.ptr.optimized_program.len], executable.ptr.optimized_program);
            }
        }
    };
}
