const std = @import("std");

const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");

const PjrtError = errors.Error;

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

fn parseOp(comptime Op: type, name: []const u8, comptime entries: anytype) ?Op {
    return inline for (entries) |entry| {
        if (std.mem.eql(u8, name, entry[0])) break entry[1];
    } else null;
}

fn parseUnaryCustomCallOp(name: []const u8) ?runtime.CustomCallRegistration {
    const op = parseOp(runtime.ElementwiseUnaryOp, name, .{
        .{ "negate", runtime.ElementwiseUnaryOp.negate },
        .{ "exp", runtime.ElementwiseUnaryOp.exp },
        .{ "expm1", runtime.ElementwiseUnaryOp.expm1 },
        .{ "tanh", runtime.ElementwiseUnaryOp.tanh },
        .{ "sqrt", runtime.ElementwiseUnaryOp.sqrt },
        .{ "rsqrt", runtime.ElementwiseUnaryOp.rsqrt },
        .{ "abs", runtime.ElementwiseUnaryOp.abs },
        .{ "cbrt", runtime.ElementwiseUnaryOp.cbrt },
        .{ "ceil", runtime.ElementwiseUnaryOp.ceil },
        .{ "floor", runtime.ElementwiseUnaryOp.floor },
        .{ "log", runtime.ElementwiseUnaryOp.log },
        .{ "log1p", runtime.ElementwiseUnaryOp.log1p },
        .{ "logistic", runtime.ElementwiseUnaryOp.logistic },
        .{ "sine", runtime.ElementwiseUnaryOp.sine },
        .{ "cosine", runtime.ElementwiseUnaryOp.cosine },
        .{ "not", runtime.ElementwiseUnaryOp.not_ },
        .{ "sign", runtime.ElementwiseUnaryOp.sign },
        .{ "is_finite", runtime.ElementwiseUnaryOp.is_finite },
        .{ "round_nearest_afz", runtime.ElementwiseUnaryOp.round_nearest_afz },
        .{ "round_nearest_even", runtime.ElementwiseUnaryOp.round_nearest_even },
        .{ "popcnt", runtime.ElementwiseUnaryOp.popcnt },
        .{ "count_leading_zeros", runtime.ElementwiseUnaryOp.count_leading_zeros },
    }) orelse return null;
    return .{ .target = "", .kind = .unary, .unary_op = op };
}
fn parseBinaryCustomCallOp(name: []const u8) ?runtime.CustomCallRegistration {
    const op = parseOp(runtime.ElementwiseBinaryOp, name, .{
        .{ "add", runtime.ElementwiseBinaryOp.add },
        .{ "subtract", runtime.ElementwiseBinaryOp.subtract },
        .{ "multiply", runtime.ElementwiseBinaryOp.multiply },
        .{ "divide", runtime.ElementwiseBinaryOp.divide },
        .{ "maximum", runtime.ElementwiseBinaryOp.maximum },
        .{ "minimum", runtime.ElementwiseBinaryOp.minimum },
        .{ "power", runtime.ElementwiseBinaryOp.power },
        .{ "atan2", runtime.ElementwiseBinaryOp.atan2 },
        .{ "remainder", runtime.ElementwiseBinaryOp.remainder },
        .{ "and", runtime.ElementwiseBinaryOp.and_ },
        .{ "or", runtime.ElementwiseBinaryOp.or_ },
        .{ "xor", runtime.ElementwiseBinaryOp.xor },
        .{ "shift_left", runtime.ElementwiseBinaryOp.shift_left },
        .{ "shift_right_logical", runtime.ElementwiseBinaryOp.shift_right_logical },
        .{ "shift_right_arithmetic", runtime.ElementwiseBinaryOp.shift_right_arithmetic },
    }) orelse return null;
    return .{ .target = "", .kind = .binary, .binary_op = op };
}
fn markerAddress(comptime marker: anytype) usize {
    return @intFromPtr(&marker);
}

const CustomCall = struct {
    fn target(ptr: [*c]const u8, len: usize) ?[]const u8 {
        return abi.Slice.bytes(ptr, len);
    }

    fn op(ptr: [*c]const u8, len: usize) ?[]const u8 {
        return abi.Slice.bytes(ptr, len);
    }

    fn register(registration: runtime.CustomCallRegistration) ?*c.PJRT_Error {
        runtime.registerCustomCall(registration) catch |err| switch (err) {
            error.InvalidCustomCall => return PjrtError.invalidArgument("invalid PjRTx custom call registration"),
            error.OutOfMemory => return PjrtError.resourceExhausted("failed to allocate PjRTx custom call registration"),
            else => return PjrtError.internal("failed to register PjRTx custom call"),
        };
        return null;
    }

    fn registerHandler(target_: []const u8, api_version: c_int, handler_execute: ?*anyopaque) ?*c.PJRT_Error {
        if (api_version != 0) return PjrtError.unimplemented("PjRTx custom calls currently support PJRT GPU untyped API version 0");
        const handler = handler_execute orelse return PjrtError.invalidArgument("custom call handler_execute is null");
        const handler_address = @intFromPtr(handler);

        if (handler_address == markerAddress(PjRTx_CustomCall_Identity)) {
            return register(.{ .target = target_, .kind = .identity });
        }
        if (handler_address == markerAddress(PjRTx_CustomCall_UnarySqrt)) {
            return register(.{ .target = target_, .kind = .unary, .unary_op = .sqrt });
        }
        if (handler_address == markerAddress(PjRTx_CustomCall_BinaryAdd)) {
            return register(.{ .target = target_, .kind = .binary, .binary_op = .add });
        }

        return PjrtError.unimplemented("custom call handler is not a PjRTx MLX executable handler");
    }

    fn registerIdentity(function_name: [*c]const u8, function_name_size: usize) ?*c.PJRT_Error {
        const target_ = target(function_name, function_name_size) orelse return PjrtError.invalidArgument("custom call target is null");
        return register(.{ .target = target_, .kind = .identity });
    }

    fn registerUnary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
        const target_ = target(function_name, function_name_size) orelse return PjrtError.invalidArgument("custom call target is null");
        const op_text = op(op_name, op_name_size) orelse return PjrtError.invalidArgument("custom call op name is null");
        var registration = parseUnaryCustomCallOp(op_text) orelse return PjrtError.invalidArgument("unsupported PjRTx unary custom call op");
        registration.target = target_;
        return register(registration);
    }

    fn registerBinary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
        const target_ = target(function_name, function_name_size) orelse return PjrtError.invalidArgument("custom call target is null");
        const op_text = op(op_name, op_name_size) orelse return PjrtError.invalidArgument("custom call op name is null");
        var registration = parseBinaryCustomCallOp(op_text) orelse return PjrtError.invalidArgument("unsupported PjRTx binary custom call op");
        registration.target = target_;
        return register(registration);
    }

    fn unregister(function_name: [*c]const u8, function_name_size: usize) void {
        const target_ = target(function_name, function_name_size) orelse return;
        runtime.unregisterCustomCall(target_);
    }

    const Extension = struct {
        fn register(args: [*c]PJRT_Gpu_Register_Custom_Call_Args) callconv(.c) ?*c.PJRT_Error {
            if (args == null) return PjrtError.invalidArgument("custom call registration args are null");
            if (args[0].struct_size < @sizeOf(PJRT_Gpu_Register_Custom_Call_Args)) {
                return PjrtError.invalidArgument("custom call registration args struct is too small");
            }
            const target_ = CustomCall.target(args[0].function_name, args[0].function_name_size) orelse return PjrtError.invalidArgument("custom call target is null");
            return CustomCall.registerHandler(target_, args[0].api_version, args[0].handler_execute);
        }
    };
};

pub var gpu_custom_call_extension: PJRT_Gpu_Custom_Call = .{
    .base = .{
        .struct_size = @sizeOf(PJRT_Gpu_Custom_Call),
        .type = c.PJRT_Extension_Type_Gpu_Custom_Call,
        .next = null,
    },
    .custom_call = CustomCall.Extension.register,
};

pub export fn PjRTx_RegisterCustomCallIdentity(function_name: [*c]const u8, function_name_size: usize) ?*c.PJRT_Error {
    return CustomCall.registerIdentity(function_name, function_name_size);
}

pub export fn PjRTx_RegisterCustomCallUnary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
    return CustomCall.registerUnary(function_name, function_name_size, op_name, op_name_size);
}

pub export fn PjRTx_RegisterCustomCallBinary(function_name: [*c]const u8, function_name_size: usize, op_name: [*c]const u8, op_name_size: usize) ?*c.PJRT_Error {
    return CustomCall.registerBinary(function_name, function_name_size, op_name, op_name_size);
}

pub export fn PjRTx_CustomCall_Identity() callconv(.c) void {}

pub export fn PjRTx_CustomCall_UnarySqrt() callconv(.c) void {}

pub export fn PjRTx_CustomCall_BinaryAdd() callconv(.c) void {}

pub export fn PjRTx_UnregisterCustomCall(function_name: [*c]const u8, function_name_size: usize) void {
    CustomCall.unregister(function_name, function_name_size);
}
