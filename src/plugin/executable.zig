const std = @import("std");

const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const state = @import("state.zig");

const Executable = state.Executable;
const ExecutableHandle = abi.Executable(Executable);
const LoadedExecutableHandle = abi.LoadedExecutable(Executable);

const ExecutableView = struct {
    ptr: *const Executable,

    fn at(raw: anytype) ExecutableView {
        return .{ .ptr = ExecutableHandle.viewConst(raw) };
    }

    fn name(self: ExecutableView) []const u8 {
        return self.ptr.name;
    }

    fn fingerprint(self: ExecutableView) []const u8 {
        return self.ptr.fingerprint;
    }

    fn numReplicas(self: ExecutableView) usize {
        return @intCast(self.ptr.plan.options.num_replicas);
    }

    fn numPartitions(self: ExecutableView) usize {
        return @intCast(self.ptr.plan.options.num_partitions);
    }

    fn numOutputs(self: ExecutableView) usize {
        return self.ptr.plan.output_shardings.len;
    }

    fn writeText(self: ExecutableView, comptime ptr_field: []const u8, comptime size_field: []const u8, raw: anytype, text: []const u8) void {
        _ = self;
        abi.writeBytes(ptr_field, size_field, &raw[0], text);
    }
};

const LoadedExecutable = struct {
    ptr: *Executable,

    fn at(raw: anytype) LoadedExecutable {
        return .{ .ptr = LoadedExecutableHandle.view(raw) };
    }

    fn asExecutable(self: LoadedExecutable) *c.PJRT_Executable {
        return ExecutableHandle.handle(self.ptr);
    }

    fn view(self: LoadedExecutable) ExecutableView {
        return .{ .ptr = self.ptr };
    }

    fn addressableDeviceCount(self: LoadedExecutable) usize {
        return @min(self.ptr.plan.options.numDevices(), self.ptr.client.device_handles.len);
    }

    fn addressableDevices(self: LoadedExecutable) []const *runtime.Device {
        const count = self.addressableDeviceCount();
        return self.ptr.client.device_handles[0..count];
    }

    fn delete(self: LoadedExecutable) void {
        self.ptr.deleted = true;
        self.ptr.releaseGraph();
    }

    fn isDeleted(self: LoadedExecutable) bool {
        return self.ptr.deleted;
    }

    fn writeAddressableDevices(self: LoadedExecutable, args: anytype) void {
        const devices = self.addressableDevices();
        args.addressable_devices = abi.Device.handleSlice(devices);
        args.num_addressable_devices = devices.len;
    }

    fn writeLogicalIds(self: LoadedExecutable, args: anytype) void {
        args.addressable_device_logical_ids = self.ptr.logical_ids.ptr;
        args.num_addressable_device_logical_ids = self.ptr.logical_ids.len;
    }

    fn writeEmptyDeviceAssignment(_: LoadedExecutable, args: anytype) void {
        args.serialized_bytes = null;
        args.serialized_bytes_size = 0;
        args.serialized_device_assignment = null;
        args.serialized_device_assignment_deleter = deviceAssignmentSerializedDeleter;
    }
};

const ExecutableScalar = enum {
    num_replicas,
    num_partitions,
    num_outputs,

    fn value(comptime scalar: ExecutableScalar, executable: ExecutableView) usize {
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
            @field(raw[0], field) = scalar.value(ExecutableView.at(raw[0].executable));
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
            const executable = ExecutableView.at(raw[0].executable);
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
            loaded.view().writeText(ptr_field, size_field, raw, switch (text) {
                .fingerprint => loaded.view().fingerprint(),
            });
            return null;
        }
    };
}

const MemoryKinds = struct {
    kinds: [][*c]const u8,
    sizes: []usize,

    fn parameters(executable: ExecutableView) MemoryKinds {
        return .{
            .kinds = executable.ptr.parameter_memory_kinds,
            .sizes = executable.ptr.parameter_memory_kind_sizes,
        };
    }

    fn outputs(executable: ExecutableView) MemoryKinds {
        return .{
            .kinds = executable.ptr.output_memory_kinds,
            .sizes = executable.ptr.output_memory_kind_sizes,
        };
    }

    fn write(self: MemoryKinds, comptime count_field: []const u8, raw: anytype) void {
        @field(raw[0], count_field) = self.kinds.len;
        raw[0].memory_kinds = abi.stringPtrList(self.kinds);
        raw[0].memory_kind_sizes = self.sizes.ptr;
    }
};

fn MemoryKindsCallback(comptime Args: type, comptime count_field: []const u8, comptime which: enum { parameters, outputs }) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const executable = ExecutableView.at(raw[0].executable);
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
            const loaded = switch (op) {
                .get_executable => LoadedExecutable.at(args.loaded_executable),
                else => LoadedExecutable.at(args.executable),
            };
            switch (op) {
                .destroy => loaded.ptr.deinit(),
                .get_executable => args.executable = loaded.asExecutable(),
                .addressable_devices => loaded.writeAddressableDevices(args),
                .logical_ids => loaded.writeLogicalIds(args),
                .device_assignment => loaded.writeEmptyDeviceAssignment(args),
                .delete => loaded.delete(),
                .is_deleted => args.is_deleted = loaded.isDeleted(),
            }
            return null;
        }
    };
}

const ExecutableOp = enum { destroy, optimized_program };

fn ExecutableCallback(comptime Args: type, comptime op: ExecutableOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const args = &raw[0];
            const executable = ExecutableView.at(args.executable);
            switch (op) {
                .destroy => {},
                .optimized_program => writeOptimizedProgram(executable, args),
            }
            return null;
        }

        fn writeOptimizedProgram(executable: ExecutableView, args: anytype) void {
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

pub fn deviceAssignmentSerializedDeleter(_: ?*c.PJRT_DeviceAssignmentSerialized) callconv(.c) void {}

pub const ExecutableApi = struct {
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

pub const LoadedExecutableApi = struct {
    pub const Destroy = LoadedExecutableCallback(c.PJRT_LoadedExecutable_Destroy_Args, .destroy).call;
    pub const GetExecutable = LoadedExecutableCallback(c.PJRT_LoadedExecutable_GetExecutable_Args, .get_executable).call;
    pub const AddressableDevices = LoadedExecutableCallback(c.PJRT_LoadedExecutable_AddressableDevices_Args, .addressable_devices).call;
    pub const AddressableDeviceLogicalIds = LoadedExecutableCallback(c.PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args, .logical_ids).call;
    pub const GetDeviceAssignment = LoadedExecutableCallback(c.PJRT_LoadedExecutable_GetDeviceAssignment_Args, .device_assignment).call;
    pub const Fingerprint = LoadedExecutableTextCallback(c.PJRT_LoadedExecutable_Fingerprint_Args, "executable_fingerprint", "executable_fingerprint_size", .fingerprint).call;
    pub const Delete = LoadedExecutableCallback(c.PJRT_LoadedExecutable_Delete_Args, .delete).call;
    pub const IsDeleted = LoadedExecutableCallback(c.PJRT_LoadedExecutable_IsDeleted_Args, .is_deleted).call;
};
