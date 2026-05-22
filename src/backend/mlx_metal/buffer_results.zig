/// Opaque backend-buffer pair returned by max-reduction-with-indices.
pub const ReduceMaxWithIndicesResult = struct {
    /// Device buffer containing reduced values.
    values: *anyopaque,
    /// Device buffer containing selected indices.
    indices: *anyopaque,
};

/// Opaque backend-buffer pair returned by windowed max-reduction-with-indices.
pub const ReduceWindowMaxWithIndicesResult = struct {
    /// Device buffer containing reduced values.
    values: *anyopaque,
    /// Device buffer containing selected indices.
    indices: *anyopaque,
};

/// Opaque backend-buffer pair returned by random bit generation.
pub const RngBitGeneratorResult = struct {
    /// Device buffer containing the updated RNG state.
    state: *anyopaque,
    /// Device buffer containing generated random bits.
    bits: *anyopaque,
};
