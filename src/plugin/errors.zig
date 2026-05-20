const std = @import("std");

const c = @import("c");
const abi = @import("pjrt_abi.zig");
const plugin = @import("plugin.zig");

const ErrorHandle = abi.Opaque(PjrtxError, c.PJRT_Error);

/// Plugin-owned PJRT error payload with stable message storage.
const PjrtxError = struct {
    base: c.PJRT_Error,
    code: c.PJRT_Error_Code,
    message: []u8,
};

const ErrorRef = struct {
    ptr: *PjrtxError,

    fn at(raw: *c.PJRT_Error) ErrorRef {
        return .{ .ptr = ErrorHandle.ref(raw) };
    }

    fn atConst(raw: *const c.PJRT_Error) ErrorRef {
        return .{ .ptr = @constCast(ErrorHandle.refConst(raw)) };
    }

    fn destroy(raw: ?*c.PJRT_Error) void {
        const base = raw orelse return;
        const err = ErrorRef.at(base).ptr;
        plugin.allocator().free(err.message);
        plugin.allocator().destroy(err);
    }

    fn message(self: ErrorRef) []const u8 {
        return self.ptr.message;
    }

    fn code(self: ErrorRef) c.PJRT_Error_Code {
        return self.ptr.code;
    }
};

const ErrorOp = enum { message, code };

fn ErrorVoidCallback(comptime Args: type, comptime op: ErrorOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) void {
            const args = &raw[0];
            const err = ErrorRef.atConst(args.@"error");
            switch (op) {
                .message => abi.Out.writeBytes("message", "message_size", args, err.message()),
                .code => @compileError("PJRT error code callback returns PJRT_Error"),
            }
        }
    };
}

fn ErrorCallback(comptime Args: type, comptime op: ErrorOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const args = &raw[0];
            const err = ErrorRef.atConst(args.@"error");
            switch (op) {
                .message => @compileError("PJRT error message callback returns void"),
                .code => args.code = err.code(),
            }
            return null;
        }
    };
}

fn ErrorDestroyCallback(comptime Args: type) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) void {
            ErrorRef.destroy(raw[0].@"error");
        }
    };
}

fn ErrorPayloadCallback(comptime Args: type) type {
    return struct {
        fn call(_: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            return null;
        }
    };
}

/// Allocates, inspects, and maps plugin-owned PJRT errors.
pub const Error = struct {
    pub const Api = struct {
        pub const Destroy = ErrorDestroyCallback(c.PJRT_Error_Destroy_Args).call;
        pub const Message = ErrorVoidCallback(c.PJRT_Error_Message_Args, .message).call;
        pub const GetCode = ErrorCallback(c.PJRT_Error_GetCode_Args, .code).call;
        pub const ForEachPayload = ErrorPayloadCallback(c.PJRT_Error_ForEachPayload_Args).call;
    };

    fn make(code_: c.PJRT_Error_Code, message_: []const u8) ?*c.PJRT_Error {
        const err = plugin.allocator().create(PjrtxError) catch return null;
        err.* = .{
            .base = .{ .vtable = null },
            .code = code_,
            .message = plugin.allocator().dupe(u8, message_) catch {
                plugin.allocator().destroy(err);
                return null;
            },
        };
        return ErrorHandle.handle(err);
    }

    /// Returns the PJRT status code for a nullable error handle.
    pub fn code(raw: ?*c.PJRT_Error) c.PJRT_Error_Code {
        const err = raw orelse return c.PJRT_Error_Code_OK;
        return ErrorRef.atConst(err).code();
    }

    /// Returns the borrowed message for a nullable error handle.
    pub fn message(raw: ?*c.PJRT_Error) ?[]const u8 {
        const err = raw orelse return null;
        return ErrorRef.atConst(err).message();
    }

    /// Builds an invalid-argument PJRT error for malformed caller input.
    pub fn invalidArgument(message_: []const u8) ?*c.PJRT_Error {
        return make(c.PJRT_Error_Code_INVALID_ARGUMENT, message_);
    }

    /// Builds a failed-precondition PJRT error for invalid object state.
    pub fn failedPrecondition(message_: []const u8) ?*c.PJRT_Error {
        return make(c.PJRT_Error_Code_FAILED_PRECONDITION, message_);
    }

    /// Builds an internal PJRT error for unexpected plugin failures.
    pub fn internal(message_: []const u8) ?*c.PJRT_Error {
        return make(c.PJRT_Error_Code_INTERNAL, message_);
    }

    /// Builds a not-found PJRT error for missing plugin objects.
    pub fn notFound(message_: []const u8) ?*c.PJRT_Error {
        return make(c.PJRT_Error_Code_NOT_FOUND, message_);
    }

    /// Builds a resource-exhausted PJRT error for allocation or capacity failures.
    pub fn resourceExhausted(message_: []const u8) ?*c.PJRT_Error {
        return make(c.PJRT_Error_Code_RESOURCE_EXHAUSTED, message_);
    }

    /// Builds an unimplemented PJRT error for unsupported API or op forms.
    pub fn unimplemented(message_: []const u8) ?*c.PJRT_Error {
        return make(c.PJRT_Error_Code_UNIMPLEMENTED, message_);
    }
};
