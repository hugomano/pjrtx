const std = @import("std");
const backend_api = @import("backend_selection.zig");

const client_mod = @import("client.zig");
const event_mod = @import("event.zig");
const executable_mod = @import("executable.zig");

const Client = client_mod.Client;
const Event = event_mod.Event;
const ExecutionContext = executable_mod.Context;

/// Converts backend completion state into a runtime readiness event.
pub fn eventFromBackendCompletion(context: ExecutionContext, completion: backend_api.ExecutionCompletion) Event {
    return switch (completion.kind) {
        .completed => Event.ready(),
        .pending => blk: {
            const backend_event = completion.backend_event orelse break :blk Event.failed("backend returned asynchronous completion without an event handle");
            defer context.destroyExecutionEvent(backend_event);
            const status = context.executionEventStatus(backend_event) catch break :blk Event.failed("backend execution event status query failed");
            break :blk switch (status.state) {
                .ready => Event.ready(),
                .failed => Event.failed(if (status.message.len == 0) "backend execution event failed" else status.message),
                .pending => Event.failed("backend execution event is pending without runtime scheduler integration"),
            };
        },
    };
}

const CompletionTestSupport = struct {
    fn mlxMetalBackend() backend_api.Backend {
        return backend_api.create();
    }

    fn initMlxMetalClient() !*Client {
        return Client.init(std.testing.allocator, mlxMetalBackend(), 1);
    }
};

test "backend pending completion without event handle fails closed" {
    const client = try CompletionTestSupport.initMlxMetalClient();
    defer client.deinit();

    const event = eventFromBackendCompletion(client.executableContext(), .{ .kind = .pending });
    try std.testing.expect(event.isReady());
    try std.testing.expectError(error.EventFailed, event.awaitReady());
}
