const ir = @import("src/compiler/ir");

const lifecycle = @import("buffer_lifecycle.zig");

/// Builds lifecycle methods for the typed backend buffer owner.
pub fn Typed(comptime Buffer: type) type {
    return struct {
        /// Copies host bytes into a typed MLX/Metal buffer.
        pub fn fromHost(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, src: []const u8) lifecycle.Error!?Buffer { return lifecycle.fromHost(Buffer, device_local_hardware_id, element_type, dims, src); }
        /// Allocates a zero-initialized MLX/Metal buffer.
        pub fn zeros(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64) lifecycle.Error!?Buffer { return lifecycle.zeros(Buffer, device_local_hardware_id, element_type, dims); }
        /// Clones this buffer into a new MLX/Metal buffer.
        pub fn clone(self: Buffer) lifecycle.Error!?Buffer { return lifecycle.clone(self); }
        /// Creates a zero buffer with the same MLX/Metal shape and type as this buffer.
        pub fn zeroLike(self: Buffer) lifecycle.Error!?Buffer { return lifecycle.zeroLike(self); }
        /// Forces this MLX/Metal buffer to evaluate.
        pub fn eval(self: Buffer) lifecycle.Error!void { try lifecycle.eval(self); }
        /// Copies this MLX/Metal buffer into host memory.
        pub fn copyToHost(self: Buffer, dst: []u8) lifecycle.Error!void { try lifecycle.copyToHost(self, dst); }
        /// Reports whether this buffer still owns a host shadow allocation.
        pub fn hasHostShadow(self: Buffer) bool { return lifecycle.hasHostShadow(self); }
        /// Destroys this MLX/Metal buffer handle.
        pub fn destroy(self: Buffer) void { lifecycle.destroy(self); }
    };
}
