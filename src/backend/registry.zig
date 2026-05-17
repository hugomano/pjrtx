const backend = @import("src/backend");
const core = @import("src/core");
const mlx_metal = @import("src/backend/mlx_metal");
const synthetic = @import("src/backend/synthetic");

pub fn create(kind: core.BackendKind) backend.Backend {
    return switch (kind) {
        .synthetic => synthetic.create(),
        .metal_mlx => mlx_metal.create(),
    };
}

test "registry creates public backends by kind" {
    try @import("std").testing.expectEqual(core.BackendKind.synthetic, create(.synthetic).kind());
    try @import("std").testing.expectEqual(core.BackendKind.metal_mlx, create(.metal_mlx).kind());
}
