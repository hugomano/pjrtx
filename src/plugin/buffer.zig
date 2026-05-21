const c = @import("c");
const runtime = @import("src/runtime");

const abi = @import("pjrt_abi.zig");
const buffer_element = @import("buffer_element.zig");
const errors = @import("errors.zig");
const events = @import("events.zig");
const handles = @import("pjrt_handles.zig");

const PjrtError = errors.Error;
const PjrtEvent = events.Event;

/// Borrowed PJRT buffer reference backed by a runtime buffer.
pub const Buffer = struct {
    ptr: *runtime.Buffer,

    pub const Api = struct {
        pub const Destroy = BufferCallback(c.PJRT_Buffer_Destroy_Args, .destroy).call;
        pub const ElementType = BufferScalarCallback(c.PJRT_Buffer_ElementType_Args, "type", .element_type).call;
        pub const Dimensions = BufferCallback(c.PJRT_Buffer_Dimensions_Args, .dimensions).call;
        pub const OnDeviceSizeInBytes = BufferScalarCallback(c.PJRT_Buffer_OnDeviceSizeInBytes_Args, "on_device_size_in_bytes", .on_device_size).call;
        pub const Device = BufferHandleCallback(c.PJRT_Buffer_Device_Args, "device", .device).call;
        pub const Memory = BufferHandleCallback(c.PJRT_Buffer_Memory_Args, "memory", .memory).call;
        pub const DynamicDimensionIndices = BufferCallback(c.PJRT_Buffer_DynamicDimensionIndices_Args, .dynamic_dimensions).call;
        pub const Delete = BufferCallback(c.PJRT_Buffer_Delete_Args, .delete).call;
        pub const IsDeleted = BufferScalarCallback(c.PJRT_Buffer_IsDeleted_Args, "is_deleted", .is_deleted).call;
        pub const IsOnCpu = BufferScalarCallback(c.PJRT_Buffer_IsOnCpu_Args, "is_on_cpu", .is_on_cpu).call;
        pub const ToHostBuffer = BufferCallback(c.PJRT_Buffer_ToHostBuffer_Args, .to_host).call;
        pub const ReadyEvent = BufferHandleCallback(c.PJRT_Buffer_ReadyEvent_Args, "event", .ready_event).call;
    };

    fn at(raw: anytype) Buffer {
        return .{ .ptr = handles.Buffer.ref(raw) };
    }

    fn elementType(self: Buffer) c.PJRT_Buffer_Type {
        return buffer_element.ElementType.toPjrt(self.ptr.elementType());
    }

    fn onDeviceSize(self: Buffer) usize {
        return self.ptr.onDeviceSizeInBytes();
    }

    fn isDeleted(self: Buffer) bool {
        return self.ptr.isDeleted();
    }

    fn isOnCpu(_: Buffer) bool {
        return false;
    }

    fn device(self: Buffer) *c.PJRT_Device {
        return handles.Device.handle(@constCast(self.ptr.devicePlacement()));
    }

    fn memory(self: Buffer) *c.PJRT_Memory {
        return handles.Memory.handle(self.ptr.memoryPlacement());
    }

    fn readyEvent(self: Buffer) ?*c.PJRT_Event {
        return PjrtEvent.fromRuntime(self.ptr.readinessEvent());
    }

    fn destroy(raw: ?*c.PJRT_Buffer) void {
        if (raw) |buffer| Buffer.at(buffer).ptr.deinit();
    }

    fn delete(self: Buffer) void {
        self.ptr.markDeleted();
    }

    fn ensureUsable(self: Buffer) ?*c.PJRT_Error {
        self.ptr.ensureUsable() catch return PjrtError.failedPrecondition("buffer has been deleted or donated");
        return null;
    }

    fn ensureReady(self: Buffer) ?*c.PJRT_Error {
        self.ptr.ensureReady() catch |err| return switch (err) {
            error.BufferNotReady => PjrtError.failedPrecondition("buffer is not ready"),
            error.BufferReadinessFailed => PjrtError.failedPrecondition("buffer readiness failed"),
        };
        return null;
    }

    fn copyToHost(self: Buffer, dst: []u8) ?*c.PJRT_Error {
        self.ptr.copyToHost(dst) catch |err| return switch (err) {
            error.DestinationTooSmall => PjrtError.invalidArgument("destination buffer is too small"),
            error.BufferDeleted, error.BufferDonated => PjrtError.failedPrecondition("buffer has been deleted or donated"),
            error.BufferNotReady => PjrtError.failedPrecondition("buffer is not ready"),
            error.BufferReadinessFailed => PjrtError.failedPrecondition("buffer readiness failed"),
            error.UnsupportedRuntimeFeature => PjrtError.unimplemented("backend cannot copy this buffer to host"),
            error.BackendBufferCopyFailed => PjrtError.internal("failed to copy buffer to host"),
        };
        return null;
    }
};

const BufferScalar = enum {
    element_type,
    on_device_size,
    is_deleted,
    is_on_cpu,

    fn value(comptime scalar: BufferScalar, buffer: Buffer) ScalarType(scalar) {
        return switch (scalar) {
            .element_type => buffer.elementType(),
            .on_device_size => buffer.onDeviceSize(),
            .is_deleted => buffer.isDeleted(),
            .is_on_cpu => buffer.isOnCpu(),
        };
    }
};

fn ScalarType(comptime scalar: BufferScalar) type {
    return switch (scalar) {
        .element_type => c.PJRT_Buffer_Type,
        .on_device_size => usize,
        .is_deleted, .is_on_cpu => bool,
    };
}

fn BufferScalarCallback(comptime Args: type, comptime field: []const u8, comptime scalar: BufferScalar) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            @field(raw[0], field) = scalar.value(Buffer.at(raw[0].buffer));
            return null;
        }
    };
}

const BufferHandle = enum {
    device,
    memory,
    ready_event,

    fn value(comptime handle: BufferHandle, buffer: Buffer) HandleType(handle) {
        return switch (handle) {
            .device => buffer.device(),
            .memory => buffer.memory(),
            .ready_event => buffer.readyEvent(),
        };
    }
};

fn HandleType(comptime handle: BufferHandle) type {
    return switch (handle) {
        .device => *c.PJRT_Device,
        .memory => *c.PJRT_Memory,
        .ready_event => ?*c.PJRT_Event,
    };
}

fn BufferHandleCallback(comptime Args: type, comptime field: []const u8, comptime handle: BufferHandle) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            @field(raw[0], field) = handle.value(Buffer.at(raw[0].buffer));
            return null;
        }
    };
}

const BufferOp = enum {
    destroy,
    delete,
    dimensions,
    dynamic_dimensions,
    to_host,
};

fn BufferCallback(comptime Args: type, comptime op: BufferOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const args = &raw[0];
            return switch (op) {
                .destroy => destroy(args),
                .delete => delete(args),
                .dimensions => dimensions(args),
                .dynamic_dimensions => dynamicDimensions(args),
                .to_host => HostReadbackRequest.run(args),
            };
        }

        fn destroy(args: anytype) ?*c.PJRT_Error {
            Buffer.destroy(args.buffer);
            return null;
        }

        fn delete(args: anytype) ?*c.PJRT_Error {
            Buffer.at(args.buffer).delete();
            return null;
        }

        fn dimensions(args: anytype) ?*c.PJRT_Error {
            BufferDims.write(args);
            return null;
        }

        fn dynamicDimensions(args: anytype) ?*c.PJRT_Error {
            DynamicDims.write(args);
            return null;
        }
    };
}

const BufferDims = struct {
    buffer: Buffer,

    fn write(args: anytype) void {
        (BufferDims{ .buffer = Buffer.at(args.buffer) }).writeTo(args);
    }

    fn writeTo(self: BufferDims, args: anytype) void {
        const dims = self.buffer.ptr.dimensions();
        args.dims = dims.ptr;
        args.num_dims = dims.len;
    }
};

const DynamicDims = struct {
    fn write(args: anytype) void {
        _ = Buffer.at(args.buffer);
        args.dynamic_dim_indices = null;
        args.num_dynamic_dims = 0;
    }
};

const HostReadbackRequest = struct {
    raw: *allowzero c.PJRT_Buffer_ToHostBuffer_Args,
    buffer: Buffer,

    fn run(raw: *allowzero c.PJRT_Buffer_ToHostBuffer_Args) ?*c.PJRT_Error {
        return (HostReadbackRequest{
            .raw = raw,
            .buffer = Buffer.at(raw.src),
        }).copy();
    }

    fn copy(self: HostReadbackRequest) ?*c.PJRT_Error {
        if (self.buffer.ensureUsable()) |err| return err;
        if (self.raw.dst == null) return self.querySize();
        if (self.buffer.ensureReady()) |err| return err;
        if (self.raw.dst_size < self.buffer.onDeviceSize()) return PjrtError.invalidArgument("destination buffer is too small");

        const dst = abi.Slice.mutBytes(self.raw.dst, self.buffer.onDeviceSize()) orelse return PjrtError.invalidArgument("destination buffer is null");
        if (self.buffer.copyToHost(dst)) |err| return err;
        self.complete();
        return null;
    }

    fn querySize(self: HostReadbackRequest) ?*c.PJRT_Error {
        self.raw.dst_size = self.buffer.onDeviceSize();
        return null;
    }

    fn complete(self: HostReadbackRequest) void {
        self.raw.event = PjrtEvent.ready();
    }
};
