const ir = @import("src/compiler/ir");

const opaque_ref = @import("buffer_opaque_ref.zig");

const Error = opaque_ref.Error;
const maybeHandle = opaque_ref.maybeHandle;
const ref = opaque_ref.ref;
const refs = opaque_ref.refs;

const results = @import("buffer_results.zig");

/// Runs a reduction operation on an opaque buffer handle.
pub fn reduce(src: *anyopaque, op: ir.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).reduce(op, dimensions, output_dims)); }
/// Runs max-reduction and returns opaque values plus indices handles.
pub fn reduceMaxWithIndices(values: *anyopaque, indices: *anyopaque, dimensions: []const i64, output_dims: []const i64) Error!?results.ReduceMaxWithIndicesResult { const pair = (try opaque_ref.Ref.reduceMaxWithIndices(ref(values), ref(indices), dimensions, output_dims)) orelse return null; return .{ .values = pair.first.toHandle(), .indices = pair.second.toHandle() }; }
/// Runs a windowed reduction operation on an opaque buffer handle.
pub fn reduceWindow(src: *anyopaque, op: ir.PlanInstructionKind, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).reduceWindow(op, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims)); }
/// Runs windowed max-reduction and returns opaque values plus indices handles.
pub fn reduceWindowMaxWithIndices(values: *anyopaque, indices: *anyopaque, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?results.ReduceWindowMaxWithIndicesResult { const pair = (try opaque_ref.Ref.reduceWindowMaxWithIndices(ref(values), ref(indices), window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims)) orelse return null; return .{ .values = pair.first.toHandle(), .indices = pair.second.toHandle() }; }
