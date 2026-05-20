const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

pub const CustomCallKind = backend_api.CustomCallKind;
pub const CustomCallRegistration = backend_api.CustomCallRegistration;
pub const CustomCallRegistrationError = backend_api.Error;
const ElementwiseBinaryOp = ir.ElementwiseBinaryOp;
const ElementwiseUnaryOp = ir.ElementwiseUnaryOp;

fn parseCustomCallOp(comptime Op: type, name: []const u8, comptime entries: anytype) ?Op {
    return inline for (entries) |entry| {
        if (std.mem.eql(u8, name, entry[0])) break entry[1];
    } else null;
}

fn parseBinaryCustomCallOp(name: []const u8) ?ElementwiseBinaryOp {
    return parseCustomCallOp(ElementwiseBinaryOp, name, .{
        .{ "add", ElementwiseBinaryOp.add },
        .{ "subtract", ElementwiseBinaryOp.subtract },
        .{ "multiply", ElementwiseBinaryOp.multiply },
        .{ "divide", ElementwiseBinaryOp.divide },
        .{ "maximum", ElementwiseBinaryOp.maximum },
        .{ "minimum", ElementwiseBinaryOp.minimum },
        .{ "power", ElementwiseBinaryOp.power },
        .{ "atan2", ElementwiseBinaryOp.atan2 },
        .{ "remainder", ElementwiseBinaryOp.remainder },
        .{ "and", ElementwiseBinaryOp.and_ },
        .{ "or", ElementwiseBinaryOp.or_ },
        .{ "xor", ElementwiseBinaryOp.xor },
        .{ "shift_left", ElementwiseBinaryOp.shift_left },
        .{ "shift_right_logical", ElementwiseBinaryOp.shift_right_logical },
        .{ "shift_right_arithmetic", ElementwiseBinaryOp.shift_right_arithmetic },
    });
}

fn parseUnaryCustomCallOp(name: []const u8) ?ElementwiseUnaryOp {
    return parseCustomCallOp(ElementwiseUnaryOp, name, .{
        .{ "negate", ElementwiseUnaryOp.negate },
        .{ "exp", ElementwiseUnaryOp.exp },
        .{ "expm1", ElementwiseUnaryOp.expm1 },
        .{ "tanh", ElementwiseUnaryOp.tanh },
        .{ "sqrt", ElementwiseUnaryOp.sqrt },
        .{ "rsqrt", ElementwiseUnaryOp.rsqrt },
        .{ "abs", ElementwiseUnaryOp.abs },
        .{ "cbrt", ElementwiseUnaryOp.cbrt },
        .{ "ceil", ElementwiseUnaryOp.ceil },
        .{ "floor", ElementwiseUnaryOp.floor },
        .{ "log", ElementwiseUnaryOp.log },
        .{ "log1p", ElementwiseUnaryOp.log1p },
        .{ "logistic", ElementwiseUnaryOp.logistic },
        .{ "sine", ElementwiseUnaryOp.sine },
        .{ "cosine", ElementwiseUnaryOp.cosine },
        .{ "not", ElementwiseUnaryOp.not_ },
        .{ "sign", ElementwiseUnaryOp.sign },
        .{ "is_finite", ElementwiseUnaryOp.is_finite },
        .{ "round_nearest_afz", ElementwiseUnaryOp.round_nearest_afz },
        .{ "round_nearest_even", ElementwiseUnaryOp.round_nearest_even },
        .{ "popcnt", ElementwiseUnaryOp.popcnt },
        .{ "count_leading_zeros", ElementwiseUnaryOp.count_leading_zeros },
    });
}

/// Registers a fully typed backend custom-call target with the Metal/MLX runtime.
pub fn registerCustomCall(registration: CustomCallRegistration) CustomCallRegistrationError!void {
    var backend_impl = backend_api.create();
    return backend_impl.registerCustomCall(registration);
}

/// Registers a named binary elementwise custom call after runtime validation.
pub fn registerBinaryCustomCall(target: []const u8, op_name: []const u8) CustomCallRegistrationError!void {
    const op = parseBinaryCustomCallOp(op_name) orelse return error.InvalidCustomCall;
    return registerCustomCall(.{ .target = target, .kind = .binary, .binary_op = op });
}

/// Registers an identity custom call whose execution aliases its input.
pub fn registerIdentityCustomCall(target: []const u8) CustomCallRegistrationError!void {
    return registerCustomCall(.{ .target = target, .kind = .identity });
}

/// Registers a named unary elementwise custom call after runtime validation.
pub fn registerUnaryCustomCall(target: []const u8, op_name: []const u8) CustomCallRegistrationError!void {
    const op = parseUnaryCustomCallOp(op_name) orelse return error.InvalidCustomCall;
    return registerCustomCall(.{ .target = target, .kind = .unary, .unary_op = op });
}

/// Registers the runtime-owned marker for a unary square-root custom call.
pub fn registerUnarySqrtCustomCall(target: []const u8) CustomCallRegistrationError!void {
    return registerCustomCall(.{ .target = target, .kind = .unary, .unary_op = .sqrt });
}

/// Registers the runtime-owned marker for a binary add custom call.
pub fn registerBinaryAddCustomCall(target: []const u8) CustomCallRegistrationError!void {
    return registerCustomCall(.{ .target = target, .kind = .binary, .binary_op = .add });
}

/// Removes a custom-call target from the Metal/MLX backend registry.
pub fn unregisterCustomCall(target: []const u8) void {
    var backend_impl = backend_api.create();
    backend_impl.unregisterCustomCall(target);
}
