/// Pair of typed MLX/Metal buffers returned by two-output operations.
pub fn Pair(comptime Buffer: type) type {
    return struct {
        /// First returned typed buffer.
        first: Buffer,
        /// Second returned typed buffer.
        second: Buffer,
    };
}
