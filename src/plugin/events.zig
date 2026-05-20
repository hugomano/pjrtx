const rt = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");
const PjrtError = errors.Error;
const state = @import("state.zig");

const allocator = state.allocator;
const BridgeHandle = abi.UserData(OnReadyBridge);

pub const Event = struct {
    ptr: *rt.Event,

    pub const Api = struct {
        pub const Destroy = EventCallback(c.PJRT_Event_Destroy_Args, .destroy).call;
        pub const IsReady = EventCallback(c.PJRT_Event_IsReady_Args, .is_ready).call;
        pub const Error = EventCallback(c.PJRT_Event_Error_Args, .error_status).call;
        pub const Await = EventCallback(c.PJRT_Event_Await_Args, .await).call;
        pub const OnReady = EventCallback(c.PJRT_Event_OnReady_Args, .on_ready).call;
    };

    pub fn at(raw: *c.PJRT_Event) Event {
        return .{ .ptr = abi.Event.view(raw) };
    }

    pub fn create(source: rt.Event) ?*c.PJRT_Event {
        const event = allocator.create(rt.Event) catch return null;
        event.* = source;
        return abi.Event.handle(event);
    }

    pub fn ready() ?*c.PJRT_Event {
        return create(rt.Event.ready());
    }

    pub fn pending() ?*c.PJRT_Event {
        return create(rt.Event.pending());
    }

    pub fn failed(message: []const u8) ?*c.PJRT_Event {
        return create(rt.Event.failed(message));
    }

    pub fn fromRuntime(source: rt.Event) ?*c.PJRT_Event {
        return switch (source.state) {
            .pending => pending(),
            .ready => ready(),
            .failed => failed(source.message),
        };
    }

    pub fn setReady(raw: ?*c.PJRT_Event) void {
        if (raw) |event| at(event).ptr.setReady();
    }

    pub fn setFailed(raw: ?*c.PJRT_Event, message: []const u8) void {
        if (raw) |event| at(event).ptr.setFailed(message);
    }

    pub fn runtime(raw: *c.PJRT_Event) *rt.Event {
        return at(raw).ptr;
    }

    fn destroy(raw: ?*c.PJRT_Event) void {
        const event = raw orelse return;
        const view = Event.at(event);
        view.ptr.deinit();
        allocator.destroy(view.ptr);
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

const OnReadyBridge = struct {
    callback_fn: c.PJRT_Event_OnReadyCallback,
    user_arg: ?*anyopaque,

    fn create(callback_fn: c.PJRT_Event_OnReadyCallback, user_arg: ?*anyopaque) ?*OnReadyBridge {
        const bridge = allocator.create(OnReadyBridge) catch return null;
        bridge.* = .{ .callback_fn = callback_fn, .user_arg = user_arg };
        return bridge;
    }

    fn fromUserArg(user_arg: ?*anyopaque) *OnReadyBridge {
        return BridgeHandle.view(user_arg);
    }

    fn destroy(bridge: *OnReadyBridge) void {
        allocator.destroy(bridge);
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

pub const Api = Event.Api;
