const std = @import("std");

const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");

const makeError = errors.makeError;

pub const PJRT_Gpu_Register_Custom_Call_Args = extern struct {
    struct_size: usize,
    function_name: [*c]const u8,
    function_name_size: usize,
    api_version: c_int,
    handler_instantiate: ?*anyopaque,
    handler_prepare: ?*anyopaque,
    handler_initialize: ?*anyopaque,
    handler_execute: ?*anyopaque,
};

pub const PJRT_Gpu_Custom_Call = extern struct {
    base: c.PJRT_Extension_Base,
    custom_call: *const fn (args: [*c]PJRT_Gpu_Register_Custom_Call_Args) callconv(.c) ?*c.PJRT_Error,
};

pub fn parseUnaryCustomCallOp(name: []const u8) ?runtime.CustomCallRegistration {
    const op: runtime.ElementwiseUnaryOp = if (std.mem.eql(u8, name, "negate"))
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
pub fn parseBinaryCustomCallOp(name: []const u8) ?runtime.CustomCallRegistration {
    const op: runtime.ElementwiseBinaryOp = if (std.mem.eql(u8, name, "add"))
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
pub fn registerPjrtxCustomCall(registration: runtime.CustomCallRegistration) ?*c.PJRT_Error {
    runtime.registerCustomCall(registration) catch |err| switch (err) {
        error.InvalidCustomCall => return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "invalid PjRTx custom call registration"),
        error.OutOfMemory => return makeError(c.PJRT_Error_Code_RESOURCE_EXHAUSTED, "failed to allocate PjRTx custom call registration"),
        else => return makeError(c.PJRT_Error_Code_INTERNAL, "failed to register PjRTx custom call"),
    };
    return null;
}
pub fn markerAddress(comptime marker: anytype) usize {
    return @intFromPtr(&marker);
}
pub fn registerPjrtxCustomCallHandler(target: []const u8, api_version: c_int, handler_execute: ?*anyopaque) ?*c.PJRT_Error {
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
pub fn pjrtGpuRegisterCustomCall(args: [*c]PJRT_Gpu_Register_Custom_Call_Args) callconv(.c) ?*c.PJRT_Error {
    if (args == null) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call registration args are null");
    if (args[0].struct_size < @sizeOf(PJRT_Gpu_Register_Custom_Call_Args)) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call registration args struct is too small");
    }
    const target = abi.bytes(args[0].function_name, args[0].function_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call target is null");
    return registerPjrtxCustomCallHandler(target, args[0].api_version, args[0].handler_execute);
}

pub var gpu_custom_call_extension: PJRT_Gpu_Custom_Call = .{
    .base = .{
        .struct_size = @sizeOf(PJRT_Gpu_Custom_Call),
        .type = c.PJRT_Extension_Type_Gpu_Custom_Call,
        .next = null,
    },
    .custom_call = pjrtGpuRegisterCustomCall,
};

pub export fn PjRTx_RegisterCustomCallIdentity(function_name: [*c]const u8, function_name_size: usize) ?*c.PJRT_Error {
    const target = abi.bytes(function_name, function_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call target is null");
    return registerPjrtxCustomCall(.{
        .target = target,
        .kind = .identity,
    });
}

pub export fn PjRTx_RegisterCustomCallUnary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
    const target = abi.bytes(function_name, function_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call target is null");
    const op_text = abi.bytes(op_name, op_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call op name is null");
    var registration = parseUnaryCustomCallOp(op_text) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "unsupported PjRTx unary custom call op");
    registration.target = target;
    return registerPjrtxCustomCall(registration);
}

pub export fn PjRTx_RegisterCustomCallBinary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
    const target = abi.bytes(function_name, function_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call target is null");
    const op_text = abi.bytes(op_name, op_name_size) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "custom call op name is null");
    var registration = parseBinaryCustomCallOp(op_text) orelse return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "unsupported PjRTx binary custom call op");
    registration.target = target;
    return registerPjrtxCustomCall(registration);
}

pub export fn PjRTx_CustomCall_Identity() callconv(.c) void {}

pub export fn PjRTx_CustomCall_UnarySqrt() callconv(.c) void {}

pub export fn PjRTx_CustomCall_BinaryAdd() callconv(.c) void {}

pub export fn PjRTx_UnregisterCustomCall(function_name: [*c]const u8, function_name_size: usize) void {
    const target = abi.bytes(function_name, function_name_size) orelse return;
    runtime.unregisterCustomCall(target);
}
