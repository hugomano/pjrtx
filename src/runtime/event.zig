const std = @import("std");

/// Tracks whether a runtime dependency has completed successfully or failed.
pub const EventState = enum {
    pending,
    ready,
    failed,
};

/// Caps callback storage so event ownership stays allocation-free after init.
pub const MAX_EVENT_CALLBACKS = 256;

/// Receives completion status for runtime events and dependent PJRT adapters.
pub const EventCallback = *const fn (message: ?[]const u8, user_arg: ?*anyopaque) void;

/// Stores one callback registration owned by an event until completion.
pub const EventCallbackRegistration = struct {
    callback: EventCallback,
    user_arg: ?*anyopaque = null,
};

/// Owns runtime readiness state and callback delivery for async work.
pub const Event = struct {
    state: EventState = .ready,
    message: []const u8 = "",
    callbacks: [MAX_EVENT_CALLBACKS]EventCallbackRegistration = undefined,
    callback_count: usize = 0,

    /// Creates an event that must be completed by a producer.
    pub fn pending() Event {
        return .{ .state = .pending };
    }

    /// Creates an already-completed successful event.
    pub fn ready() Event {
        return .{ .state = .ready };
    }

    /// Creates an already-completed failed event with a borrowed message.
    pub fn failed(message: []const u8) Event {
        return .{ .state = .failed, .message = message };
    }

    /// Returns true once the event will no longer transition from pending.
    pub fn isReady(self: Event) bool {
        return self.state != .pending;
    }

    /// Registers a callback or invokes it immediately when already complete.
    pub fn onReady(self: *Event, callback: EventCallback, user_arg: ?*anyopaque) !void {
        if (self.state != .pending) return callback(EventCallbacks.errorMessage(self.*), user_arg);
        if (self.callback_count >= MAX_EVENT_CALLBACKS) return error.TooManyEventCallbacks;
        self.callbacks[self.callback_count] = .{
            .callback = callback,
            .user_arg = user_arg,
        };
        self.callback_count += 1;
    }

    /// Propagates this event's eventual completion into a dependent event.
    pub fn chainTo(self: *Event, dependent: *Event) !void {
        try self.onReady(EventCallbacks.resolveChainedEvent, dependent);
    }

    /// Completes a pending event successfully and releases callbacks.
    pub fn setReady(self: *Event) void {
        if (self.state != .pending) return;
        self.state = .ready;
        self.message = "";
        EventCallbacks.invoke(self);
    }

    /// Completes or overwrites an event as failed with a borrowed message.
    pub fn setFailed(self: *Event, message: []const u8) void {
        if (self.state != .pending) {
            self.state = .failed;
            self.message = message;
            return;
        }
        self.state = .failed;
        self.message = message;
        EventCallbacks.invoke(self);
    }

    /// Fails when the event has not completed successfully.
    pub fn awaitReady(self: Event) !void {
        return switch (self.state) {
            .ready => {},
            .failed => error.EventFailed,
            .pending => error.EventPending,
        };
    }

    /// Fails pending callbacks before invalidating owned callback storage.
    pub fn deinit(self: *Event) void {
        if (self.state == .pending) {
            self.state = .failed;
            self.message = "event destroyed before completion";
            EventCallbacks.invoke(self);
        }
        self.* = undefined;
    }
};

const EventCallbacks = struct {
    fn errorMessage(event: Event) ?[]const u8 {
        return switch (event.state) {
            .failed => event.message,
            .pending, .ready => null,
        };
    }

    fn invoke(event: *Event) void {
        const callback_count = event.callback_count;
        event.callback_count = 0;
        const message = errorMessage(event.*);
        for (event.callbacks[0..callback_count]) |registration| {
            registration.callback(message, registration.user_arg);
        }
    }

    fn resolveChainedEvent(message: ?[]const u8, user_arg: ?*anyopaque) void {
        const dependent: *Event = @ptrCast(@alignCast(user_arg.?));
        if (message) |msg| {
            dependent.setFailed(msg);
        } else {
            dependent.setReady();
        }
    }
};

const CallbackTestState = struct {
    count: usize = 0,
    ready_count: usize = 0,
    failed_count: usize = 0,
    saw_expected_message: bool = false,
};

fn recordCallback(message: ?[]const u8, user_arg: ?*anyopaque) void {
    const state: *CallbackTestState = @ptrCast(@alignCast(user_arg.?));
    state.count += 1;
    if (message) |msg| {
        state.failed_count += 1;
        if (std.mem.eql(u8, msg, "boom") or std.mem.eql(u8, msg, "event destroyed before completion")) {
            state.saw_expected_message = true;
        }
    } else {
        state.ready_count += 1;
    }
}

test "callbacks run on ready and failed transitions" {
    var ready_event = Event.pending();
    var ready_state = CallbackTestState{};
    try ready_event.onReady(recordCallback, &ready_state);
    try std.testing.expect(!ready_event.isReady());
    ready_event.setReady();
    try std.testing.expect(ready_event.isReady());
    try ready_event.awaitReady();
    try std.testing.expectEqual(@as(usize, 1), ready_state.count);
    try std.testing.expectEqual(@as(usize, 1), ready_state.ready_count);

    var failed_event = Event.pending();
    var failed_state = CallbackTestState{};
    try failed_event.onReady(recordCallback, &failed_state);
    failed_event.setFailed("boom");
    try std.testing.expect(failed_event.isReady());
    try std.testing.expectError(error.EventFailed, failed_event.awaitReady());
    try std.testing.expectEqual(@as(usize, 1), failed_state.count);
    try std.testing.expectEqual(@as(usize, 1), failed_state.failed_count);
    try std.testing.expect(failed_state.saw_expected_message);
}

test "dependency chains readiness and failures" {
    var source_ready = Event.pending();
    var dependent_ready = Event.pending();
    try source_ready.chainTo(&dependent_ready);
    try std.testing.expect(!dependent_ready.isReady());
    source_ready.setReady();
    try std.testing.expect(dependent_ready.isReady());
    try dependent_ready.awaitReady();

    var source_failed = Event.pending();
    var dependent_failed = Event.pending();
    try source_failed.chainTo(&dependent_failed);
    source_failed.setFailed("boom");
    try std.testing.expect(dependent_failed.isReady());
    try std.testing.expectError(error.EventFailed, dependent_failed.awaitReady());
    try std.testing.expectEqualStrings("boom", dependent_failed.message);
}

test "callbacks run immediately for completed events and reject overflow" {
    var ready_event = Event.ready();
    var ready_state = CallbackTestState{};
    try ready_event.onReady(recordCallback, &ready_state);
    try std.testing.expectEqual(@as(usize, 1), ready_state.count);
    try std.testing.expectEqual(@as(usize, 1), ready_state.ready_count);

    var failed_event = Event.failed("boom");
    var failed_state = CallbackTestState{};
    try failed_event.onReady(recordCallback, &failed_state);
    try std.testing.expectEqual(@as(usize, 1), failed_state.count);
    try std.testing.expectEqual(@as(usize, 1), failed_state.failed_count);
    try std.testing.expect(failed_state.saw_expected_message);

    var pending_event = Event.pending();
    var pending_state = CallbackTestState{};
    for (0..MAX_EVENT_CALLBACKS) |_| {
        try pending_event.onReady(recordCallback, &pending_state);
    }
    try std.testing.expectError(error.TooManyEventCallbacks, pending_event.onReady(recordCallback, &pending_state));
    pending_event.deinit();
    try std.testing.expectEqual(@as(usize, MAX_EVENT_CALLBACKS), pending_state.count);
    try std.testing.expectEqual(@as(usize, MAX_EVENT_CALLBACKS), pending_state.failed_count);
    try std.testing.expect(pending_state.saw_expected_message);
}
