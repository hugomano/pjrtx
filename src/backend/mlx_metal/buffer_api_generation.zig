const ir = @import("src/compiler/ir");

const generation = @import("buffer_generation.zig");
const pair_mod = @import("buffer_pair.zig");

/// Builds generation methods for the typed backend buffer owner.
pub fn Typed(comptime Buffer: type) type {
    const Pair = pair_mod.Pair(Buffer);
    return struct {
        /// Creates an MLX/Metal iota buffer.
        pub fn iota(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, iota_dimension: i64) generation.Error!?Buffer { return generation.iota(Buffer, device_local_hardware_id, element_type, dims, iota_dimension); }
        /// Creates an MLX/Metal partition-id scalar buffer.
        pub fn partitionId(device_local_hardware_id: i32, element_type: ir.BufferType, partition_id: u32) generation.Error!?Buffer { return generation.partitionId(Buffer, device_local_hardware_id, element_type, partition_id); }
        /// Runs a random distribution operation using this buffer and another bound buffer.
        pub fn rng(a: Buffer, b: Buffer, distribution: ir.RngDistribution, output_type: ir.BufferType, output_dims: []const i64) generation.Error!?Buffer { return generation.rng(a, b, distribution, output_type, output_dims); }
        /// Runs random bit generation and returns updated state plus bits.
        pub fn rngBitGenerator(state: Buffer, output_type: ir.BufferType, output_dims: []const i64) generation.Error!?Pair { const pair = (try generation.rngBitGenerator(state, output_type, output_dims)) orelse return null; return .{ .first = pair.first, .second = pair.second }; }
    };
}
