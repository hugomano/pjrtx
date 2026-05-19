const std = @import("std");

const c = @import("c");
const abi = @import("pjrt_abi.zig");
const state = @import("state.zig");

const allocator = state.allocator;
const ErrorHandle = abi.Opaque(PjrtxError, c.PJRT_Error);

pub const PjrtxError = struct {
    base: c.PJRT_Error,
    code: c.PJRT_Error_Code,
    message: []u8,
};

const ErrorView = struct {
    ptr: *PjrtxError,

    fn at(raw: *c.PJRT_Error) ErrorView {
        return .{ .ptr = ErrorHandle.view(raw) };
    }

    fn atConst(raw: *const c.PJRT_Error) ErrorView {
        return .{ .ptr = @constCast(ErrorHandle.viewConst(raw)) };
    }

    fn destroy(raw: ?*c.PJRT_Error) void {
        const base = raw orelse return;
        const err = ErrorView.at(base).ptr;
        allocator.free(err.message);
        allocator.destroy(err);
    }

    fn message(self: ErrorView) []const u8 {
        return self.ptr.message;
    }

    fn code(self: ErrorView) c.PJRT_Error_Code {
        return self.ptr.code;
    }
};

const ErrorOp = enum { message, code };

fn ErrorVoidCallback(comptime Args: type, comptime op: ErrorOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) void {
            const args = &raw[0];
            const err = ErrorView.atConst(args.@"error");
            switch (op) {
                .message => abi.writeBytes("message", "message_size", args, err.message()),
                .code => @compileError("PJRT error code callback returns PJRT_Error"),
            }
        }
    };
}

fn ErrorCallback(comptime Args: type, comptime op: ErrorOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const args = &raw[0];
            const err = ErrorView.atConst(args.@"error");
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
            ErrorView.destroy(raw[0].@"error");
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

const PluginOp = enum { initialize, attributes };

fn PluginCallback(comptime Args: type, comptime op: PluginOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            switch (op) {
                .initialize => {},
                .attributes => {
                    const attrs = state.initAttrs();
                    raw[0].attributes = abi.namedValuePtr(attrs);
                    raw[0].num_attributes = attrs.len;
                },
            }
            return null;
        }
    };
}

pub fn makeError(code: c.PJRT_Error_Code, message: []const u8) ?*c.PJRT_Error {
    const err = allocator.create(PjrtxError) catch return null;
    err.* = .{
        .base = .{ .vtable = null },
        .code = code,
        .message = allocator.dupe(u8, message) catch {
            allocator.destroy(err);
            return null;
        },
    };
    return ErrorHandle.handle(err);
}
pub fn errorCode(raw: ?*c.PJRT_Error) c.PJRT_Error_Code {
    const err = raw orelse return c.PJRT_Error_Code_OK;
    return ErrorView.atConst(err).code();
}
pub fn invalidArgument(message: []const u8) ?*c.PJRT_Error {
    return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, message);
}
pub fn failedPrecondition(message: []const u8) ?*c.PJRT_Error {
    return makeError(c.PJRT_Error_Code_FAILED_PRECONDITION, message);
}
pub fn internal(message: []const u8) ?*c.PJRT_Error {
    return makeError(c.PJRT_Error_Code_INTERNAL, message);
}
pub fn notFound(message: []const u8) ?*c.PJRT_Error {
    return makeError(c.PJRT_Error_Code_NOT_FOUND, message);
}
pub fn resourceExhausted(message: []const u8) ?*c.PJRT_Error {
    return makeError(c.PJRT_Error_Code_RESOURCE_EXHAUSTED, message);
}
pub fn unimplemented(message: []const u8) ?*c.PJRT_Error {
    return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, message);
}
pub const ErrorApi = struct {
    pub const Destroy = ErrorDestroyCallback(c.PJRT_Error_Destroy_Args).call;
    pub const Message = ErrorVoidCallback(c.PJRT_Error_Message_Args, .message).call;
    pub const GetCode = ErrorCallback(c.PJRT_Error_GetCode_Args, .code).call;
    pub const ForEachPayload = ErrorPayloadCallback(c.PJRT_Error_ForEachPayload_Args).call;
};

pub const PluginApi = struct {
    pub const Initialize = PluginCallback(c.PJRT_Plugin_Initialize_Args, .initialize).call;
    pub const Attributes = PluginCallback(c.PJRT_Plugin_Attributes_Args, .attributes).call;
};
