const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const encoding = @import("buffer_encoding.zig");
const mlx_call = @import("mlx_call.zig");

const Buffer = buffer_mod.Buffer;
const Error = buffer_mod.Error;

/// Combines real and imaginary buffers into a complex MLX/Metal buffer.
pub fn complex(real: Buffer, imag: Buffer, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferComplex(real.handle, imag.handle, output_dims), error.CommandSubmissionFailed);
}

/// Extracts the real part of this complex MLX/Metal buffer.
pub fn realPart(src: Buffer, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferReal(src.handle, output_dims), error.CommandSubmissionFailed);
}

/// Extracts the imaginary part of this complex MLX/Metal buffer.
pub fn imagPart(src: Buffer, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferImag(src.handle, output_dims), error.CommandSubmissionFailed);
}

/// Converts this MLX/Metal buffer to another element type.
pub fn convert(src: Buffer, output_type: ir.BufferType) Error!?Buffer {
    const dtype = encoding.dtype(output_type) orelse return error.UnsupportedElementType;
    return wrap(mlx_call.bufferAstype(src.handle, dtype), error.CommandSubmissionFailed);
}

/// Reinterprets this MLX/Metal buffer with a new element type and shape.
pub fn bitcast(src: Buffer, output_type: ir.BufferType, output_dims: []const i64) Error!?Buffer {
    const dtype = encoding.dtype(output_type) orelse return error.UnsupportedElementType;
    return wrap(mlx_call.bufferViewDtype(src.handle, dtype, output_dims), error.CommandSubmissionFailed);
}

/// Applies an elementwise binary operation to two MLX/Metal buffers.
pub fn binary(lhs: Buffer, rhs: Buffer, op: ir.ElementwiseBinaryOp) Error!?Buffer {
    const code = encoding.binaryOp(op) orelse return error.CommandSubmissionFailed;
    return wrap(mlx_call.bufferBinary(lhs.handle, rhs.handle, code), error.CommandSubmissionFailed);
}

/// Applies an elementwise binary operation with explicit output dimensions.
pub fn binaryWithOutputDims(lhs: Buffer, rhs: Buffer, op: ir.ElementwiseBinaryOp, output_dims: []const i64) Error!?Buffer {
    const code = encoding.binaryOp(op) orelse return error.CommandSubmissionFailed;
    return wrap(mlx_call.bufferBinaryOut(lhs.handle, rhs.handle, code, output_dims), error.CommandSubmissionFailed);
}

/// Applies an elementwise unary operation to this MLX/Metal buffer.
pub fn unary(src: Buffer, op: ir.ElementwiseUnaryOp) Error!?Buffer {
    const code = encoding.unaryOp(op) orelse return error.CommandSubmissionFailed;
    return wrap(mlx_call.bufferUnary(src.handle, code), error.CommandSubmissionFailed);
}

/// Compares this MLX/Metal buffer with another buffer.
pub fn compare(lhs: Buffer, rhs: Buffer, direction: ir.CompareOp, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferCompare(lhs.handle, rhs.handle, encoding.compareOp(direction), output_dims), error.CommandSubmissionFailed);
}

/// Selects between two MLX/Metal buffers using this predicate buffer.
pub fn select(pred: Buffer, on_true: Buffer, on_false: Buffer, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferSelect(pred.handle, on_true.handle, on_false.handle, output_dims), error.CommandSubmissionFailed);
}

/// Clamps this value buffer between minimum and maximum buffers.
pub fn clamp(min: Buffer, value: Buffer, max: Buffer, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferClamp(min.handle, value.handle, max.handle, output_dims), error.CommandSubmissionFailed);
}

fn wrap(handle: ?mlx_call.BufferHandle, err: Error) Error!?Buffer {
    return if (handle) |ptr| Buffer{ .handle = ptr } else err;
}
