const std = @import("std");

const c = @import("c");
const runtime_mod = @import("src/runtime");

const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");
const handles = @import("pjrt_handles.zig");
const plugin = @import("plugin.zig");

const BridgeHandle = handles.UserArg(OnReadyBridge);
const PjrtError = errors.Error;

/// PJRT event reference over a runtime event owned by the plugin.
pub const Event = struct {
    ptr: *runtime_mod.Event,

    /// Mutable PJRT event output slot used by APIs that complete asynchronously.
    pub const Completion = struct {
        slot: *allowzero ?*c.PJRT_Event,

        /// Creates a pending event in a PJRT output slot.
        pub fn pending(slot: *allowzero ?*c.PJRT_Event) ?*c.PJRT_Error {
            const completion = Completion.at(slot);
            completion.slot.* = Event.create(runtime_mod.Event.pending());
            if (completion.slot.* == null) return PjrtError.resourceExhausted("failed to allocate PJRT event");
            return null;
        }

        /// Creates an event mirroring a runtime event in a PJRT output slot.
        pub fn fromRuntime(slot: *allowzero ?*c.PJRT_Event, source: runtime_mod.Event) ?Completion {
            const completion = Completion.at(slot);
            completion.slot.* = Event.create(source);
            if (completion.slot.* == null) return null;
            return completion;
        }

        /// Binds a helper to an existing PJRT event output slot.
        pub fn at(slot: *allowzero ?*c.PJRT_Event) Completion {
            return .{ .slot = slot };
        }

        /// Marks the event in this slot ready when the caller supplied one.
        pub fn setReady(self: Completion) void {
            if (self.slot.*) |event| Event.at(event).ptr.setReady();
        }

        /// Marks the event in this slot failed when the caller supplied one.
        pub fn setFailed(self: Completion, message: []const u8) void {
            if (self.slot.*) |event| Event.at(event).ptr.setFailed(message);
        }

        /// Borrows the runtime event stored in this output slot.
        pub fn runtime(self: Completion) *runtime_mod.Event {
            return Event.at(self.slot.*.?).ptr;
        }
    };

    pub const Api = struct {
        pub const Destroy = EventCallback(c.PJRT_Event_Destroy_Args, .destroy).call;
        pub const IsReady = EventCallback(c.PJRT_Event_IsReady_Args, .is_ready).call;
        pub const Error = EventCallback(c.PJRT_Event_Error_Args, .error_status).call;
        pub const Await = EventCallback(c.PJRT_Event_Await_Args, .await).call;
        pub const OnReady = EventCallback(c.PJRT_Event_OnReady_Args, .on_ready).call;
    };

    fn at(raw: *c.PJRT_Event) Event {
        return .{ .ptr = handles.Event.ref(raw) };
    }

    fn create(source: runtime_mod.Event) ?*c.PJRT_Event {
        const event = plugin.allocator().create(runtime_mod.Event) catch return null;
        event.* = source;
        return handles.Event.handle(event);
    }

    pub fn ready() ?*c.PJRT_Event {
        return create(runtime_mod.Event.ready());
    }

    pub fn pending() ?*c.PJRT_Event {
        return create(runtime_mod.Event.pending());
    }

    pub fn failed(message: []const u8) ?*c.PJRT_Event {
        return create(runtime_mod.Event.failed(message));
    }

    pub fn fromRuntime(source: runtime_mod.Event) ?*c.PJRT_Event {
        return create(source);
    }

    pub fn destroy(raw: ?*c.PJRT_Event) void {
        const event = raw orelse return;
        const event_ref = Event.at(event);
        event_ref.ptr.deinit();
        plugin.allocator().destroy(event_ref.ptr);
    }

    fn isReady(self: Event) bool {
        return self.ptr.isReady();
    }

    fn failIfNeeded(self: Event) ?*c.PJRT_Error {
        if (self.ptr.state == .failed) return PjrtError.failedPrecondition(self.ptr.message);
        return null;
    }

    fn await(self: Event) ?*c.PJRT_Error {
        self.ptr.awaitReady() catch |err| return switch (err) {
            error.EventFailed => PjrtError.failedPrecondition(self.ptr.message),
            error.EventPending => PjrtError.internal("pending event has no scheduler completion source"),
        };
        return null;
    }

    fn onReady(self: Event, callback: c.PJRT_Event_OnReadyCallback, user_arg: ?*anyopaque) ?*c.PJRT_Error {
        const bridge = OnReadyBridge.create(callback, user_arg) orelse return PjrtError.internal("failed to allocate event callback");
        self.ptr.onReady(OnReadyBridge.callback, bridge) catch |err| {
            OnReadyBridge.destroy(bridge);
            return switch (err) {
                error.TooManyEventCallbacks => PjrtError.resourceExhausted("too many callbacks registered on event"),
            };
        };
        return null;
    }
};

const OnReadyBridge = struct {
    callback_fn: c.PJRT_Event_OnReadyCallback,
    user_arg: ?*anyopaque,

    fn create(callback_fn: c.PJRT_Event_OnReadyCallback, user_arg: ?*anyopaque) ?*OnReadyBridge {
        const bridge = plugin.allocator().create(OnReadyBridge) catch return null;
        bridge.* = .{ .callback_fn = callback_fn, .user_arg = user_arg };
        return bridge;
    }

    fn fromUserArg(user_arg: ?*anyopaque) *OnReadyBridge {
        return BridgeHandle.ref(user_arg);
    }

    fn destroy(bridge: *OnReadyBridge) void {
        plugin.allocator().destroy(bridge);
    }

    fn complete(self: *OnReadyBridge, message: ?[]const u8) void {
        defer OnReadyBridge.destroy(self);
        const maybe_error = if (message) |msg| PjrtError.failedPrecondition(msg) else null;
        self.callback_fn.?(maybe_error, self.user_arg);
    }

    fn callback(message: ?[]const u8, user_arg: ?*anyopaque) void {
        OnReadyBridge.fromUserArg(user_arg).complete(message);
    }
};

const EventOp = enum {
    destroy,
    is_ready,
    error_status,
    await,
    on_ready,
};

fn EventCallback(comptime Args: type, comptime op: EventOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const args = &raw[0];
            return switch (op) {
                .destroy => destroy(args),
                .is_ready => isReady(args),
                .error_status => errorStatus(args),
                .await => awaitReady(args),
                .on_ready => onReady(args),
            };
        }

        fn event(args: anytype) Event {
            return Event.at(args.event.?);
        }

        fn destroy(args: anytype) ?*c.PJRT_Error {
            Event.destroy(args.event);
            return null;
        }

        fn isReady(args: anytype) ?*c.PJRT_Error {
            args.is_ready = event(args).isReady();
            return null;
        }

        fn errorStatus(args: anytype) ?*c.PJRT_Error {
            return event(args).failIfNeeded();
        }

        fn awaitReady(args: anytype) ?*c.PJRT_Error {
            return event(args).await();
        }

        fn onReady(args: anytype) ?*c.PJRT_Error {
            const callback = args.callback orelse return null;
            return event(args).onReady(callback, args.user_arg);
        }
    };
}
