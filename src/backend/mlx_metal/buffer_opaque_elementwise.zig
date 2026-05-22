const ir = @import("src/compiler/ir");

const opaque_ref = @import("buffer_opaque_ref.zig");

const Error = opaque_ref.Error;
const maybeHandle = opaque_ref.maybeHandle;
const ref = opaque_ref.ref;
const refs = opaque_ref.refs;

/// Combines real and imaginary opaque handles into a complex buffer.
pub fn complex(real: *anyopaque, imag: *anyopaque, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.complex(ref(real), ref(imag), output_dims)); }
/// Extracts the real part from an opaque complex buffer handle.
pub fn realPart(src: *anyopaque, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).realPart(output_dims)); }
/// Extracts the imaginary part from an opaque complex buffer handle.
pub fn imagPart(src: *anyopaque, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).imagPart(output_dims)); }
/// Converts an opaque buffer handle to another element type.
pub fn convert(src: *anyopaque, output_type: ir.BufferType) Error!?*anyopaque { return maybeHandle(try ref(src).convert(output_type)); }
/// Reinterprets an opaque buffer handle with a new element type and shape.
pub fn bitcast(src: *anyopaque, output_type: ir.BufferType, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).bitcast(output_type, output_dims)); }
/// Applies an elementwise binary operation to two opaque buffer handles.
pub fn binary(lhs: *anyopaque, rhs: *anyopaque, op: ir.ElementwiseBinaryOp) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.binary(ref(lhs), ref(rhs), op)); }
/// Applies an elementwise binary operation with explicit output dimensions.
pub fn binaryWithOutputDims(lhs: *anyopaque, rhs: *anyopaque, op: ir.ElementwiseBinaryOp, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.binaryWithOutputDims(ref(lhs), ref(rhs), op, output_dims)); }
/// Applies an elementwise unary operation to an opaque buffer handle.
pub fn unary(src: *anyopaque, op: ir.ElementwiseUnaryOp) Error!?*anyopaque { return maybeHandle(try ref(src).unary(op)); }
/// Compares two opaque buffer handles.
pub fn compare(lhs: *anyopaque, rhs: *anyopaque, direction: ir.CompareOp, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.compare(ref(lhs), ref(rhs), direction, output_dims)); }
/// Selects between opaque buffer handles using an opaque predicate handle.
pub fn select(pred: *anyopaque, on_true: *anyopaque, on_false: *anyopaque, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.select(ref(pred), ref(on_true), ref(on_false), output_dims)); }
/// Clamps an opaque value handle between minimum and maximum handles.
pub fn clamp(min: *anyopaque, value: *anyopaque, max: *anyopaque, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.clamp(ref(min), ref(value), ref(max), output_dims)); }
