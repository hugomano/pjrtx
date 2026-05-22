const ir = @import("src/compiler/ir");

const elementwise = @import("buffer_elementwise.zig");

/// Builds elementwise methods for the typed backend buffer owner.
pub fn Typed(comptime Buffer: type) type {
    return struct {
        /// Combines real and imaginary buffers into a complex MLX/Metal buffer.
        pub fn complex(real: Buffer, imag: Buffer, output_dims: []const i64) elementwise.Error!?Buffer { return elementwise.complex(real, imag, output_dims); }
        /// Extracts the real part of this complex MLX/Metal buffer.
        pub fn realPart(self: Buffer, output_dims: []const i64) elementwise.Error!?Buffer { return elementwise.realPart(self, output_dims); }
        /// Extracts the imaginary part of this complex MLX/Metal buffer.
        pub fn imagPart(self: Buffer, output_dims: []const i64) elementwise.Error!?Buffer { return elementwise.imagPart(self, output_dims); }
        /// Converts this MLX/Metal buffer to another element type.
        pub fn convert(self: Buffer, output_type: ir.BufferType) elementwise.Error!?Buffer { return elementwise.convert(self, output_type); }
        /// Reinterprets this MLX/Metal buffer with a new element type and shape.
        pub fn bitcast(self: Buffer, output_type: ir.BufferType, output_dims: []const i64) elementwise.Error!?Buffer { return elementwise.bitcast(self, output_type, output_dims); }
        /// Applies an elementwise binary operation to two MLX/Metal buffers.
        pub fn binary(lhs: Buffer, rhs: Buffer, op: ir.ElementwiseBinaryOp) elementwise.Error!?Buffer { return elementwise.binary(lhs, rhs, op); }
        /// Applies an elementwise binary operation with explicit output dimensions.
        pub fn binaryWithOutputDims(lhs: Buffer, rhs: Buffer, op: ir.ElementwiseBinaryOp, output_dims: []const i64) elementwise.Error!?Buffer { return elementwise.binaryWithOutputDims(lhs, rhs, op, output_dims); }
        /// Applies an elementwise unary operation to this MLX/Metal buffer.
        pub fn unary(self: Buffer, op: ir.ElementwiseUnaryOp) elementwise.Error!?Buffer { return elementwise.unary(self, op); }
        /// Compares this MLX/Metal buffer with another buffer.
        pub fn compare(lhs: Buffer, rhs: Buffer, direction: ir.CompareOp, output_dims: []const i64) elementwise.Error!?Buffer { return elementwise.compare(lhs, rhs, direction, output_dims); }
        /// Selects between two MLX/Metal buffers using this predicate buffer.
        pub fn select(pred: Buffer, on_true: Buffer, on_false: Buffer, output_dims: []const i64) elementwise.Error!?Buffer { return elementwise.select(pred, on_true, on_false, output_dims); }
        /// Clamps this value buffer between minimum and maximum buffers.
        pub fn clamp(min: Buffer, value: Buffer, max: Buffer, output_dims: []const i64) elementwise.Error!?Buffer { return elementwise.clamp(min, value, max, output_dims); }
    };
}
