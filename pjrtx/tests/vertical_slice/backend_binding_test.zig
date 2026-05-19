const std = @import("std");
const backend = @import("pjrtx/backend");
const core = @import("pjrtx/core");

test "backend binding test sees target kind and trace IDs" {
    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();

    try backend.writeBackendKind(&output.writer, .metal_v0);
    try std.testing.expectEqualStrings("metal_v0", output.writer.buffered());
    try std.testing.expect(core.idInBounds(0, 1));
}
