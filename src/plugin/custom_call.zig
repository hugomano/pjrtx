const c = @import("c");
const runtime = @import("src/runtime");

const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");

const PjrtError = errors.Error;

const RegistrationError = HandlerRegistration.DecodeError || TargetName.DecodeError || OperationName.DecodeError || runtime.CustomCallRegistrationError;

fn markerAddress(comptime marker: anytype) usize {
    return @intFromPtr(&marker);
}

const TargetName = struct {
    text: []const u8,

    fn decode(ptr: [*c]const u8, len: usize) DecodeError!TargetName {
        return .{ .text = abi.Slice.bytes(ptr, len) orelse return error.NullTarget };
    }

    const DecodeError = error{NullTarget};
};

const OperationName = struct {
    text: []const u8,

    fn decode(ptr: [*c]const u8, len: usize) DecodeError!OperationName {
        return .{ .text = abi.Slice.bytes(ptr, len) orelse return error.NullOperation };
    }

    const DecodeError = error{NullOperation};
};

const HandlerRegistration = struct {
    target: TargetName,
    api_version: c_int,
    execute: ?*anyopaque,

    fn decode(raw: [*c]c.PJRT_Gpu_Register_Custom_Call_Args) DecodeError!HandlerRegistration {
        if (raw == null) return error.NullRegistration;
        if (raw[0].struct_size < @as(usize, @intCast(c.PJRT_Gpu_Register_Custom_Call_Args_STRUCT_SIZE))) {
            return error.RegistrationTooSmall;
        }
        return .{
            .target = try TargetName.decode(raw[0].function_name, raw[0].function_name_size),
            .api_version = raw[0].api_version,
            .execute = raw[0].handler_execute,
        };
    }

    const DecodeError = error{ NullRegistration, RegistrationTooSmall } || TargetName.DecodeError;
};

const CustomCall = struct {
    fn fail(err: RegistrationError) ?*c.PJRT_Error {
        switch (err) {
            error.NullRegistration => return PjrtError.invalidArgument("custom call registration args are null"),
            error.RegistrationTooSmall => return PjrtError.invalidArgument("custom call registration args struct is too small"),
            error.NullTarget => return PjrtError.invalidArgument("custom call target is null"),
            error.NullOperation => return PjrtError.invalidArgument("custom call op name is null"),
            error.InvalidCustomCall => return PjrtError.invalidArgument("invalid PjRTx custom call registration"),
            error.OutOfMemory => return PjrtError.resourceExhausted("failed to allocate PjRTx custom call registration"),
            error.InvalidDeviceCount,
            error.InvalidProgram,
            error.UnsupportedElementType,
            error.ShapeMismatch,
            error.BufferAllocationFailed,
            error.CommandSubmissionFailed,
            error.BufferCopyFailed,
            => return PjrtError.internal("failed to register PjRTx custom call"),
        }
    }

    fn registerBinaryTarget(target: TargetName, op: OperationName) ?*c.PJRT_Error {
        runtime.registerBinaryCustomCall(target.text, op.text) catch |err| return fail(err);
        return null;
    }

    fn registerIdentityTarget(target: TargetName) ?*c.PJRT_Error {
        runtime.registerIdentityCustomCall(target.text) catch |err| return fail(err);
        return null;
    }

    fn registerUnarySqrtTarget(target: TargetName) ?*c.PJRT_Error {
        runtime.registerUnarySqrtCustomCall(target.text) catch |err| return fail(err);
        return null;
    }

    fn registerUnaryTarget(target: TargetName, op: OperationName) ?*c.PJRT_Error {
        runtime.registerUnaryCustomCall(target.text, op.text) catch |err| return fail(err);
        return null;
    }

    fn registerHandler(registration: HandlerRegistration) ?*c.PJRT_Error {
        if (registration.api_version != 0) return PjrtError.unimplemented("PjRTx custom calls currently support PJRT GPU untyped API version 0");
        const handler = registration.execute orelse return PjrtError.invalidArgument("custom call handler_execute is null");
        const handler_address = @intFromPtr(handler);

        if (handler_address == markerAddress(PjRTx_CustomCall_Identity)) {
            return registerIdentityTarget(registration.target);
        }
        if (handler_address == markerAddress(PjRTx_CustomCall_UnarySqrt)) {
            return registerUnarySqrtTarget(registration.target);
        }
        if (handler_address == markerAddress(PjRTx_CustomCall_BinaryAdd)) {
            return registerBinaryTarget(registration.target, .{ .text = "add" });
        }

        return PjrtError.unimplemented("custom call handler is not a PjRTx MLX executable handler");
    }

    fn registerIdentity(function_name: [*c]const u8, function_name_size: usize) ?*c.PJRT_Error {
        const target = TargetName.decode(function_name, function_name_size) catch |err| return fail(err);
        return registerIdentityTarget(target);
    }

    fn registerUnary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
        const target = TargetName.decode(function_name, function_name_size) catch |err| return fail(err);
        const op = OperationName.decode(op_name, op_name_size) catch |err| return fail(err);
        return registerUnaryTarget(target, op);
    }

    fn registerBinary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
        const target = TargetName.decode(function_name, function_name_size) catch |err| return fail(err);
        const op = OperationName.decode(op_name, op_name_size) catch |err| return fail(err);
        return registerBinaryTarget(target, op);
    }

    fn unregister(function_name: [*c]const u8, function_name_size: usize) void {
        const target = TargetName.decode(function_name, function_name_size) catch return;
        runtime.unregisterCustomCall(target.text);
    }

    const Extension = struct {
        fn register(args: [*c]c.PJRT_Gpu_Register_Custom_Call_Args) callconv(.c) [*c]c.PJRT_Error {
            const registration = HandlerRegistration.decode(args) catch |err| return CustomCall.fail(err);
            return CustomCall.registerHandler(registration);
        }
    };
};

var gpu_custom_call_extension: c.PJRT_Gpu_Custom_Call = .{
    .base = .{
        .struct_size = @intCast(c.PJRT_Gpu_Custom_Call_STRUCT_SIZE),
        .type = c.PJRT_Extension_Type_Gpu_Custom_Call,
        .next = null,
    },
    .custom_call = CustomCall.Extension.register,
};

/// Returns the custom-call PJRT extension header installed into the API table.
pub fn extensionBase() *c.PJRT_Extension_Base {
    gpu_custom_call_extension.base.next = null;
    return &gpu_custom_call_extension.base;
}

/// Registers a PjRTx identity custom-call target by exported C symbol.
pub export fn PjRTx_RegisterCustomCallIdentity(function_name: [*c]const u8, function_name_size: usize) ?*c.PJRT_Error {
    return CustomCall.registerIdentity(function_name, function_name_size);
}

/// Registers a PjRTx unary custom-call target by exported C symbol.
pub export fn PjRTx_RegisterCustomCallUnary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
    return CustomCall.registerUnary(function_name, function_name_size, op_name, op_name_size);
}

/// Registers a PjRTx binary custom-call target by exported C symbol.
pub export fn PjRTx_RegisterCustomCallBinary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
    return CustomCall.registerBinary(function_name, function_name_size, op_name, op_name_size);
}

/// Marker handler for identity custom calls registered through the PJRT extension.
pub export fn PjRTx_CustomCall_Identity() callconv(.c) void {}

/// Marker handler for unary sqrt custom calls registered through the PJRT extension.
pub export fn PjRTx_CustomCall_UnarySqrt() callconv(.c) void {}

/// Marker handler for binary add custom calls registered through the PJRT extension.
pub export fn PjRTx_CustomCall_BinaryAdd() callconv(.c) void {}

/// Removes a previously registered PjRTx custom-call target.
pub export fn PjRTx_UnregisterCustomCall(function_name: [*c]const u8, function_name_size: usize) void {
    CustomCall.unregister(function_name, function_name_size);
}
