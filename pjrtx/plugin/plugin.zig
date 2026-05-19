const std = @import("std");
const compiler = @import("pjrtx/compiler");
const core = @import("pjrtx/core");
const runtime = @import("pjrtx/runtime");

pub fn writeAdapterStage(writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try compiler.writeStageName(writer, .input_setup);
}

test "plugin adapter composes new architecture packages" {
    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();

    try writeAdapterStage(&output.writer);
    try std.testing.expectEqualStrings("input_setup", output.writer.buffered());
    try std.testing.expect(core.idInBounds(0, 1));

    output.clearRetainingCapacity();
    try runtime.writeCommandKind(&output.writer, .event_record);
    try std.testing.expectEqualStrings("event_record", output.writer.buffered());
}
