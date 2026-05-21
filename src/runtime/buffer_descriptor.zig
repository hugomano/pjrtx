const std = @import("std");
const ir = @import("src/compiler/ir");

/// Owns immutable logical tensor metadata for a runtime buffer.
pub const Descriptor = struct {
    element_type: ir.BufferType,
    dims: []i64,
    byte_size: usize,
    shard_index: usize,

    /// Duplicates shape metadata and records the device-resident byte size.
    pub fn init(allocator: std.mem.Allocator, element_type: ir.BufferType, dims: []const i64, shard_index: usize, byte_size: usize) !Descriptor {
        return .{
            .element_type = element_type,
            .dims = try allocator.dupe(i64, dims),
            .byte_size = byte_size,
            .shard_index = shard_index,
        };
    }

    /// Releases owned shape metadata.
    pub fn deinit(self: Descriptor, allocator: std.mem.Allocator) void {
        allocator.free(self.dims);
    }
};
