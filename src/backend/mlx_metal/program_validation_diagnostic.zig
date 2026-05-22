const std = @import("std");

/// Errors reported while validating backend-program structural invariants.
pub const Error = error{
    InvalidProgram,
    OutOfMemory,
};

/// Emits a backend-program validation diagnostic and returns `InvalidProgram`.
pub fn invalidProgram(writer: ?*std.Io.Writer, comptime detail_fmt: []const u8, args: anytype) (Error || std.Io.Writer.Error)!void {
    if (writer) |w| {
        try w.writeAll("invalid backend program: pass=backend-program-verify feature=backend-program detail=\"");
        try w.print(detail_fmt, args);
        try w.writeAll("\"");
    }
    return error.InvalidProgram;
}
