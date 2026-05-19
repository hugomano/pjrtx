const compiler_facts = @import("pjrtx/compiler/facts");

const backend_kernel_rejected_alternatives = [_][]const u8{
    "custom_kernel_generation: deferred until kernel IR and device allocator exist",
    "elementwise_fusion: backend boundary kept by V0 lowering policy",
};

const elementwise_rejected_alternatives = [_][]const u8{
    "separate_backend_command: V0 prefers explainable fusible elementwise lowering",
};

const transfer_rejected_alternatives = [_][]const u8{
    "implicit_runtime_transfer: transfers must be explicit compiler regions",
};

const unsupported_rejected_alternatives = [_][]const u8{
    "fallback: forbidden by PjRTx no-fallback policy",
};

/// V0 lowering policy is intentionally explicit: no fallback, no implicit
/// runtime transfer, and no generated matmul epilogue until the kernel IR and
/// math policy are visible in MLIR.
pub const LoweringPolicy = struct {
    pub fn reasonForDecision(decision: compiler_facts.LoweringDecision) []const u8 {
        return switch (decision) {
            .backend_kernel_graph => "MLIR lowering_region_form keeps this operation as a backend kernel graph boundary",
            .elementwise_fusion => "MLIR lowering_region_form keeps elementwise-compatible work fusible before backend binding",
            .transfer => "MLIR lowering_region_form materializes an explicit transfer region",
            .unsupported => "MLIR lowering_region_form found no executable V0 lowering path",
        };
    }

    pub fn rejectedAlternativesForDecision(decision: compiler_facts.LoweringDecision) []const []const u8 {
        return switch (decision) {
            .backend_kernel_graph => &backend_kernel_rejected_alternatives,
            .elementwise_fusion => &elementwise_rejected_alternatives,
            .transfer => &transfer_rejected_alternatives,
            .unsupported => &unsupported_rejected_alternatives,
        };
    }
};
