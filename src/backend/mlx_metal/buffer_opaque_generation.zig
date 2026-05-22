const ir = @import("src/compiler/ir");

const generation = @import("buffer_generation.zig");
const opaque_ref = @import("buffer_opaque_ref.zig");

const Error = opaque_ref.Error;
const maybeHandle = opaque_ref.maybeHandle;
const ref = opaque_ref.ref;
const refs = opaque_ref.refs;

const results = @import("buffer_results.zig");

/// Creates an opaque iota buffer handle.
pub fn iota(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, iota_dimension: i64) Error!?*anyopaque { return maybeHandle(try generation.iota(opaque_ref.Ref, device_local_hardware_id, element_type, dims, iota_dimension)); }
/// Creates an opaque partition-id scalar buffer handle.
pub fn partitionId(device_local_hardware_id: i32, element_type: ir.BufferType, partition_id: u32) Error!?*anyopaque { return maybeHandle(try generation.partitionId(opaque_ref.Ref, device_local_hardware_id, element_type, partition_id)); }
/// Runs a random distribution operation using opaque bound handles.
pub fn rng(a: *anyopaque, b: *anyopaque, distribution: ir.RngDistribution, output_type: ir.BufferType, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.rng(ref(a), ref(b), distribution, output_type, output_dims)); }
/// Runs random bit generation and returns opaque state plus bits handles.
pub fn rngBitGenerator(state: *anyopaque, output_type: ir.BufferType, output_dims: []const i64) Error!?results.RngBitGeneratorResult { const pair = (try opaque_ref.Ref.rngBitGenerator(ref(state), output_type, output_dims)) orelse return null; return .{ .state = pair.first.toHandle(), .bits = pair.second.toHandle() }; }
