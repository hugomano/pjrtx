const ir = @import("src/compiler/ir");

const buffer_handle = @import("buffer_handle.zig");
const encoding = @import("buffer_encoding.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const mlx_call = @import("mlx_call.zig");
const profiling = @import("profiling.zig");

pub const Error = buffer_handle.Error;

/// Combines real and imaginary buffers into a complex MLX/Metal buffer.
pub fn complex(real: anytype, imag: @TypeOf(real), output_dims: []const i64) Error!?@TypeOf(real) {
    return buffer_handle.wrap(@TypeOf(real), mlx_call.bufferComplex(real.handle, imag.handle, output_dims), error.CommandSubmissionFailed);
}

/// Extracts the real part of this complex MLX/Metal buffer.
pub fn realPart(src: anytype, output_dims: []const i64) Error!?@TypeOf(src) {
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferReal(src.handle, output_dims), error.CommandSubmissionFailed);
}

/// Extracts the imaginary part of this complex MLX/Metal buffer.
pub fn imagPart(src: anytype, output_dims: []const i64) Error!?@TypeOf(src) {
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferImag(src.handle, output_dims), error.CommandSubmissionFailed);
}

/// Converts this MLX/Metal buffer to another element type.
pub fn convert(src: anytype, output_type: ir.BufferType) Error!?@TypeOf(src) {
    const dtype = encoding.dtype(output_type) orelse return error.UnsupportedElementType;
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferAstype(src.handle, dtype), error.CommandSubmissionFailed);
}

/// Reinterprets this MLX/Metal buffer with a new element type and shape.
pub fn bitcast(src: anytype, output_type: ir.BufferType, output_dims: []const i64) Error!?@TypeOf(src) {
    const dtype = encoding.dtype(output_type) orelse return error.UnsupportedElementType;
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferViewDtype(src.handle, dtype, output_dims), error.CommandSubmissionFailed);
}

/// Applies an elementwise binary operation to two MLX/Metal buffers.
pub fn binary(lhs: anytype, rhs: @TypeOf(lhs), op: ir.ElementwiseBinaryOp) Error!?@TypeOf(lhs) {
    const code = encoding.binaryOp(op) orelse return error.CommandSubmissionFailed;
    return buffer_handle.wrap(@TypeOf(lhs), mlx_call.bufferBinary(lhs.handle, rhs.handle, code), error.CommandSubmissionFailed);
}

/// Applies an elementwise binary operation with explicit output dimensions.
pub fn binaryWithOutputDims(lhs: anytype, rhs: @TypeOf(lhs), op: ir.ElementwiseBinaryOp, output_dims: []const i64) Error!?@TypeOf(lhs) {
    const code = encoding.binaryOp(op) orelse return error.CommandSubmissionFailed;
    if (profiling.metalCppExecuteEnabled()) {
        if (metalcpp_call.denseBinaryOut(lhs.handle, rhs.handle, @intFromEnum(code), output_dims)) |handle| {
            return @TypeOf(lhs){ .handle = @ptrCast(handle) };
        }
    }
    return buffer_handle.wrap(@TypeOf(lhs), mlx_call.bufferBinaryOut(lhs.handle, rhs.handle, code, output_dims), error.CommandSubmissionFailed);
}

/// Applies an elementwise unary operation to this MLX/Metal buffer.
pub fn unary(src: anytype, op: ir.ElementwiseUnaryOp) Error!?@TypeOf(src) {
    const code = encoding.unaryOp(op) orelse return error.CommandSubmissionFailed;
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferUnary(src.handle, code), error.CommandSubmissionFailed);
}

/// Compares this MLX/Metal buffer with another buffer.
pub fn compare(lhs: anytype, rhs: @TypeOf(lhs), direction: ir.CompareOp, output_dims: []const i64) Error!?@TypeOf(lhs) {
    return buffer_handle.wrap(@TypeOf(lhs), mlx_call.bufferCompare(lhs.handle, rhs.handle, encoding.compareOp(direction), output_dims), error.CommandSubmissionFailed);
}

/// Selects between two MLX/Metal buffers using this predicate buffer.
pub fn select(pred: anytype, on_true: @TypeOf(pred), on_false: @TypeOf(pred), output_dims: []const i64) Error!?@TypeOf(pred) {
    return buffer_handle.wrap(@TypeOf(pred), mlx_call.bufferSelect(pred.handle, on_true.handle, on_false.handle, output_dims), error.CommandSubmissionFailed);
}

/// Clamps this value buffer between minimum and maximum buffers.
pub fn clamp(min: anytype, value: @TypeOf(min), max: @TypeOf(min), output_dims: []const i64) Error!?@TypeOf(min) {
    return buffer_handle.wrap(@TypeOf(min), mlx_call.bufferClamp(min.handle, value.handle, max.handle, output_dims), error.CommandSubmissionFailed);
}
