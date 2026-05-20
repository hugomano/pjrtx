const std = @import("std");

const c = @import("c");
const errors = @import("errors.zig");
const PjrtError = errors.Error;
const state = @import("state.zig");
const render = @import("trace_render.zig");

const allocator = state.allocator;

pub const Timestamp = std.Io.Timestamp;
pub const io = state.io;

pub const Env = struct {
    pub fn traceEnabled() bool {
        return flag("PJRTX_TRACE") or flag("PJRTX_PROFILE");
    }

    pub fn flag(comptime name: [:0]const u8) bool {
        const raw = std.c.getenv(name) orelse return false;
        const text = std.mem.span(raw);
        return text.len != 0 and !std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false");
    }

    pub fn executableCacheMaxBytes() ?u64 {
        const raw = std.c.getenv("PJRTX_EXECUTABLE_CACHE_MAX_BYTES") orelse return null;
        const text = std.mem.span(raw);
        return if (text.len == 0) null else std.fmt.parseUnsigned(u64, text, 10) catch null;
    }
};

fn now() Timestamp {
    return Timestamp.now(io, .awake);
}

fn resultText(result: ?*c.PJRT_Error, out: *std.Io.Writer.Allocating) []const u8 {
    const code = PjrtError.code(result);
    if (code == c.PJRT_Error_Code_OK) return "ok";
    out.writer.print("error{{code={d}", .{code}) catch return "<trace_result_error>";
    if (PjrtError.message(result)) |message| {
        out.writer.writeAll(",message=") catch {};
        render.Render.tokenText(&out.writer, message) catch {};
    }
    out.writer.writeByte('}') catch {};
    return out.writer.buffered();
}

pub const Api = struct {
    pub fn Callback(comptime name: []const u8, comptime callback: anytype) type {
        const info = @typeInfo(@TypeOf(callback)).@"fn";
        const Args = info.params[0].type orelse @compileError("PJRT callback args type is missing");
        const Return = info.return_type orelse void;
        return struct {
            pub fn call(args: Args) callconv(.c) Return {
                if (!Env.traceEnabled()) return callback(args);
                const start = now();
                if (Return == void) {
                    callback(args);
                    var args_text = std.Io.Writer.Allocating.init(allocator);
                    defer args_text.deinit();
                    emit(name, start, c.PJRT_Error_Code_OK, render.Render.args(Args, args, &args_text), "ok");
                    return;
                }
                const result = callback(args);
                var args_text = std.Io.Writer.Allocating.init(allocator);
                defer args_text.deinit();
                var rendered_result = std.Io.Writer.Allocating.init(allocator);
                defer rendered_result.deinit();
                emit(name, start, PjrtError.code(result), render.Render.args(Args, args, &args_text), resultText(result, &rendered_result));
                return result;
            }
        };
    }
};

fn emit(comptime name: []const u8, start: Timestamp, code: c.PJRT_Error_Code, args: []const u8, result: []const u8) void {
    std.debug.print("pjrtx_trace event=pjrt_api_call name={s} status={s} failed={d} error_code={d} elapsed_us={d} args={s} result={s}\n", .{
        name,
        if (code == c.PJRT_Error_Code_OK) "ok" else "error",
        @intFromBool(code != c.PJRT_Error_Code_OK),
        code,
        start.durationTo(now()).toMicroseconds(),
        args,
        result,
    });
}
