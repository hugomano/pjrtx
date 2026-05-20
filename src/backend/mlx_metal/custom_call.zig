const std = @import("std");

const ir = @import("src/compiler/ir");

/// Errors owned by the MLX custom-call registry boundary.
/// Callers should map these names into the backend package error set.
pub const Error = error{
    InvalidCustomCall,
    OutOfMemory,
};

/// Describes the public custom-call shape accepted by the MLX backend.
/// Dynamic registrations use these kinds; built-in targets are resolved to
/// internal specs by `lookup`.
pub const Kind = enum {
    identity,
    unary,
    binary,
};

/// Describes one caller-provided custom-call target registration.
/// The registry copies `target`; operation fields must match `kind`.
pub const Registration = struct {
    target: []const u8,
    kind: Kind,
    unary_op: ?ir.ElementwiseUnaryOp = null,
    binary_op: ?ir.ElementwiseBinaryOp = null,

    /// Validates the public registration and returns the executable spec.
    pub fn validate(self: Registration) Error!Spec {
        if (self.target.len == 0) return error.InvalidCustomCall;
        return switch (self.kind) {
            .identity => .{ .kind = .identity },
            .unary => .{
                .kind = .unary,
                .unary_op = self.unary_op orelse return error.InvalidCustomCall,
            },
            .binary => .{
                .kind = .binary,
                .binary_op = self.binary_op orelse return error.InvalidCustomCall,
            },
        };
    }
};

/// Names a backend-owned custom-call implementation after lookup.
/// Public registrations only produce identity/unary/binary specs; built-ins
/// are reserved by this module and must execute through the backend shim owner.
pub const SpecKind = enum {
    identity,
    unary,
    binary,
    metal_kernel_binary_add_f32,
    scaled_dot_product_attention,
};

/// Captures the executable lowering and execution contract for a custom-call.
/// The spec borrows operation enum values and contains no raw C handles.
pub const Spec = struct {
    kind: SpecKind,
    unary_op: ?ir.ElementwiseUnaryOp = null,
    binary_op: ?ir.ElementwiseBinaryOp = null,
};

const Entry = struct {
    target: []const u8,
    spec: Spec,
};

/// Built-in target implemented by the MLX/Metal custom kernel shim.
pub const BuiltinBinaryAddF32Target = "pjrtx.mlx_metal.custom_binary_add_f32";

/// Built-in target implemented by the MLX/Metal attention shim.
pub const ScaledDotProductAttentionTarget = "pjrtx.mlx_metal.scaled_dot_product_attention";

var mutex: std.atomic.Mutex = .unlocked;
var registry: std.StringHashMapUnmanaged(Entry) = .empty;
var registry_version: u64 = 0;

/// Installs or replaces a custom-call target in the process registry.
/// The target bytes are owned by the registry until `unregister` replaces or
/// removes them.
pub fn register(registration: Registration) Error!void {
    const spec = try registration.validate();
    const target = std.heap.page_allocator.dupe(u8, registration.target) catch return error.OutOfMemory;
    errdefer std.heap.page_allocator.free(target);

    lock();
    defer unlock();

    if (registry.fetchRemove(registration.target)) |removed| {
        std.heap.page_allocator.free(removed.value.target);
    }
    registry.put(std.heap.page_allocator, target, .{
        .target = target,
        .spec = spec,
    }) catch return error.OutOfMemory;
    registry_version +|= 1;
}

/// Removes a dynamic custom-call target from the process registry.
/// Built-in targets are immutable and are not affected by this operation.
pub fn unregister(target: []const u8) void {
    lock();
    defer unlock();

    if (registry.fetchRemove(target)) |removed| {
        std.heap.page_allocator.free(removed.value.target);
        registry_version +|= 1;
    }
}

/// Returns the registry mutation counter used by executable caching.
/// The counter saturates on overflow and includes dynamic registrations only.
pub fn version() u64 {
    lock();
    defer unlock();
    return registry_version;
}

/// Resolves a target to either a built-in implementation or a dynamic entry.
/// The returned spec is a value copy and does not borrow registry storage.
pub fn lookup(target: []const u8) ?Spec {
    if (std.mem.eql(u8, target, "annotate_device_placement")) return .{ .kind = .identity };
    if (std.mem.eql(u8, target, BuiltinBinaryAddF32Target)) return .{ .kind = .metal_kernel_binary_add_f32 };
    if (std.mem.eql(u8, target, ScaledDotProductAttentionTarget)) return .{ .kind = .scaled_dot_product_attention };

    lock();
    defer unlock();
    const entry = registry.get(target) orelse return null;
    return entry.spec;
}

fn lock() void {
    while (!mutex.tryLock()) std.atomic.spinLoopHint();
}

fn unlock() void {
    mutex.unlock();
}

test "custom-call registration validates operation payloads" {
    try std.testing.expectError(error.InvalidCustomCall, (Registration{
        .target = "",
        .kind = .identity,
    }).validate());
    try std.testing.expectError(error.InvalidCustomCall, (Registration{
        .target = "missing_unary",
        .kind = .unary,
    }).validate());
    try std.testing.expectEqual(SpecKind.binary, (try (Registration{
        .target = "binary",
        .kind = .binary,
        .binary_op = .add,
    }).validate()).kind);
}

test "custom-call registry resolves built-ins and dynamic targets" {
    try std.testing.expectEqual(SpecKind.identity, lookup("annotate_device_placement").?.kind);
    try std.testing.expectEqual(SpecKind.metal_kernel_binary_add_f32, lookup(BuiltinBinaryAddF32Target).?.kind);
    try std.testing.expectEqual(SpecKind.scaled_dot_product_attention, lookup(ScaledDotProductAttentionTarget).?.kind);

    const target = "pjrtx.test.custom_call.dynamic";
    unregister(target);
    const before = version();
    try register(.{
        .target = target,
        .kind = .unary,
        .unary_op = .exp,
    });
    defer unregister(target);

    try std.testing.expect(version() != before);
    const spec = lookup(target).?;
    try std.testing.expectEqual(SpecKind.unary, spec.kind);
    try std.testing.expectEqual(ir.ElementwiseUnaryOp.exp, spec.unary_op.?);
}
