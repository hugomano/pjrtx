const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");
const state = @import("state.zig");

const allocator = state.allocator;
const failedPrecondition = errors.failedPrecondition;
const internal = errors.internal;
const resourceExhausted = errors.resourceExhausted;
const BridgeHandle = abi.UserData(OnReadyBridge);

const Event = struct {
    ptr: *runtime.Event,

    fn at(raw: *c.PJRT_Event) Event {
        return .{ .ptr = abi.Event.view(raw) };
    }

    fn create(source: runtime.Event) ?*c.PJRT_Event {
        const event = allocator.create(runtime.Event) catch return null;
        event.* = source;
        return abi.Event.handle(event);
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
        if (self.ptr.state == .failed) return failedPrecondition(self.ptr.message);
        return null;
    }

    fn await(self: Event) ?*c.PJRT_Error {
        self.ptr.awaitReady() catch |err| return switch (err) {
            error.EventFailed => failedPrecondition(self.ptr.message),
            error.EventPending => internal("pending event has no scheduler completion source"),
        };
        return null;
    }

    fn onReady(self: Event, callback: c.PJRT_Event_OnReadyCallback, user_arg: ?*anyopaque) ?*c.PJRT_Error {
        const bridge = OnReadyBridge.create(callback, user_arg) orelse return internal("failed to allocate event callback");
        self.ptr.onReady(onReadyBridgeCallback, bridge) catch |err| {
            OnReadyBridge.destroy(bridge);
            return switch (err) {
                error.TooManyEventCallbacks => resourceExhausted("too many callbacks registered on event"),
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
    callback: c.PJRT_Event_OnReadyCallback,
    user_arg: ?*anyopaque,

    fn create(callback: c.PJRT_Event_OnReadyCallback, user_arg: ?*anyopaque) ?*OnReadyBridge {
        const bridge = allocator.create(OnReadyBridge) catch return null;
        bridge.* = .{ .callback = callback, .user_arg = user_arg };
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
        const maybe_error = if (message) |msg| failedPrecondition(msg) else null;
        self.callback.?(maybe_error, self.user_arg);
    }
};

pub fn eventCreateReady() ?*c.PJRT_Event {
    return Event.create(runtime.Event.ready());
}
pub fn eventCreatePending() ?*c.PJRT_Event {
    return Event.create(runtime.Event.pending());
}
pub fn eventCreateFailed(message: []const u8) ?*c.PJRT_Event {
    return Event.create(runtime.Event.failed(message));
}
pub fn eventCreateFromRuntime(source: runtime.Event) ?*c.PJRT_Event {
    return switch (source.state) {
        .pending => eventCreatePending(),
        .ready => eventCreateReady(),
        .failed => eventCreateFailed(source.message),
    };
}
pub fn eventSetReady(event: ?*c.PJRT_Event) void {
    if (event) |opaque_event| {
        Event.at(opaque_event).ptr.setReady();
    }
}
pub fn eventSetFailed(event: ?*c.PJRT_Event, message: []const u8) void {
    if (event) |opaque_event| {
        Event.at(opaque_event).ptr.setFailed(message);
    }
}
pub fn runtimeEvent(event: *c.PJRT_Event) *runtime.Event {
    return Event.at(event).ptr;
}
pub fn onReadyBridgeCallback(message: ?[]const u8, user_arg: ?*anyopaque) void {
    OnReadyBridge.fromUserArg(user_arg).complete(message);
}

pub const Api = struct {
    pub const Destroy = EventCallback(c.PJRT_Event_Destroy_Args, .destroy).call;
    pub const IsReady = EventCallback(c.PJRT_Event_IsReady_Args, .is_ready).call;
    pub const Error = EventCallback(c.PJRT_Event_Error_Args, .error_status).call;
    pub const Await = EventCallback(c.PJRT_Event_Await_Args, .await).call;
    pub const OnReady = EventCallback(c.PJRT_Event_OnReady_Args, .on_ready).call;
};
