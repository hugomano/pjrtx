const std = @import("std");

const c = @import("c");
const errors = @import("errors.zig");
const PjrtError = errors.Error;
const plugin_process = @import("plugin_process.zig");
const render = @import("trace_render.zig");

const Timestamp = std.Io.Timestamp;

const Config = struct {
    trace_enabled: bool,
    executable_cache_max_bytes: ?u64,

    fn init() Config {
        return .{
            .trace_enabled = flagFromEnv("PJRTX_TRACE") or flagFromEnv("PJRTX_PROFILE"),
            .executable_cache_max_bytes = integerFromEnv("PJRTX_EXECUTABLE_CACHE_MAX_BYTES"),
        };
    }
};

var config_state: enum { cold, ready } = .cold;
var config_storage: Config = undefined;
var config_mutex: std.Io.Mutex = .init;

/// Reads plugin tracing and profiling environment configuration.
pub const Env = struct {
    /// Returns whether PJRT API tracing or profiling is enabled for this process.
    pub fn traceEnabled() bool {
        return config().trace_enabled;
    }

    /// Returns the optional executable-cache residency limit requested by the environment.
    pub fn executableCacheMaxBytes() ?u64 {
        return config().executable_cache_max_bytes;
    }
};

fn config() Config {
    config_mutex.lockUncancelable(plugin_process.io());
    defer config_mutex.unlock(plugin_process.io());

    if (config_state == .cold) {
        config_storage = Config.init();
        config_state = .ready;
    }
    return config_storage;
}

fn flagFromEnv(comptime name: [:0]const u8) bool {
    const raw = std.c.getenv(name) orelse return false;
    const text = std.mem.span(raw);
    return text.len != 0 and !std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false");
}

fn integerFromEnv(comptime name: [:0]const u8) ?u64 {
    const raw = std.c.getenv(name) orelse return null;
    const text = std.mem.span(raw);
    return if (text.len == 0) null else std.fmt.parseUnsigned(u64, text, 10) catch null;
}

fn now() Timestamp {
    return Timestamp.now(plugin_process.io(), .awake);
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

/// Wraps PJRT callbacks with uniform trace rendering when tracing is enabled.
pub const Api = struct {
    /// Returns a C-callable wrapper that traces arguments, result, and elapsed time.
    pub fn Callback(comptime name: []const u8, comptime callback: anytype) type {
        const callback_fn = @typeInfo(@TypeOf(callback)).@"fn";
        const Args = callback_fn.params[0].type orelse @compileError("PJRT callback args type is missing");
        const Return = callback_fn.return_type orelse void;
        return struct {
            /// Invokes the wrapped PJRT callback and emits one trace line when enabled.
            pub fn call(args: Args) callconv(.c) Return {
                if (!Env.traceEnabled()) return callback(args);
                const start = now();
                if (Return == void) {
                    callback(args);
                    var args_text = std.Io.Writer.Allocating.init(plugin_process.allocator());
                    defer args_text.deinit();
                    emit(name, start, c.PJRT_Error_Code_OK, render.Render.args(Args, args, &args_text), "ok");
                    return;
                }
                const result = callback(args);
                var args_text = std.Io.Writer.Allocating.init(plugin_process.allocator());
                defer args_text.deinit();
                var rendered_result = std.Io.Writer.Allocating.init(plugin_process.allocator());
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
