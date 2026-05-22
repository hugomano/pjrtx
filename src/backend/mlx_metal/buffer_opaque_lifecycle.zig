const ir = @import("src/compiler/ir");

const lifecycle = @import("buffer_lifecycle.zig");
const opaque_ref = @import("buffer_opaque_ref.zig");

const Error = opaque_ref.Error;
const maybeHandle = opaque_ref.maybeHandle;
const ref = opaque_ref.ref;
const refs = opaque_ref.refs;

/// Copies host bytes into a device-resident opaque buffer handle.
pub fn fromHost(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, src: []const u8) Error!?*anyopaque { return maybeHandle(try lifecycle.fromHost(opaque_ref.Ref, device_local_hardware_id, element_type, dims, src)); }
/// Allocates a zero-initialized device-resident opaque buffer handle.
pub fn zeros(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64) Error!?*anyopaque { return maybeHandle(try lifecycle.zeros(opaque_ref.Ref, device_local_hardware_id, element_type, dims)); }
/// Clones an opaque buffer handle.
pub fn clone(src: *anyopaque) Error!?*anyopaque { return maybeHandle(try ref(src).clone()); }
/// Creates an opaque zero buffer with the same shape and type as another handle.
pub fn zeroLike(src: *anyopaque) Error!?*anyopaque { return maybeHandle(try ref(src).zeroLike()); }
/// Forces an opaque buffer handle to evaluate.
pub fn eval(buffer: *anyopaque) Error!void { try ref(buffer).eval(); }
/// Forces opaque buffer handles to evaluate as a single backend call.
pub fn evalMany(buffers: []const *anyopaque) Error!void { try opaque_ref.evalMany(refs(buffers)); }
/// Copies an opaque buffer handle into host memory.
pub fn copyToHost(src: *anyopaque, dst: []u8) Error!void { try ref(src).copyToHost(dst); }
/// Reports whether an opaque buffer still owns a host shadow allocation.
pub fn hasHostShadow(src: *anyopaque) bool { return ref(src).hasHostShadow(); }
/// Destroys an opaque buffer handle.
pub fn destroy(buffer: *anyopaque) void { ref(buffer).destroy(); }
