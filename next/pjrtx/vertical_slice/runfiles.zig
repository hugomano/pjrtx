const std = @import("std");

pub fn readRunfile(allocator: std.mem.Allocator, workspace_path: []const u8) ![]u8 {
    if (try readRunfileFromEnv(allocator, "TEST_SRCDIR", "TEST_WORKSPACE", workspace_path)) |contents| {
        return contents;
    }
    if (try readRunfileFromEnv(allocator, "RUNFILES_DIR", "TEST_WORKSPACE", workspace_path)) |contents| {
        return contents;
    }
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, workspace_path, allocator, .limited(64 * 1024));
}

fn readRunfileFromEnv(
    allocator: std.mem.Allocator,
    root_env_name: []const u8,
    workspace_env_name: []const u8,
    workspace_path: []const u8,
) !?[]u8 {
    var environ: std.process.Environ.Map = try std.process.Environ.createMap(std.testing.environ, allocator);
    defer environ.deinit();

    const root = environ.get(root_env_name) orelse return null;
    const workspace = environ.get(workspace_env_name) orelse "_main";
    const path: []const u8 = try std.fs.path.join(allocator, &.{ root, workspace, workspace_path });
    defer allocator.free(path);

    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, allocator, .limited(64 * 1024)) catch |err| switch (err) {
        error.FileNotFound => null,
        else => err,
    };
}
