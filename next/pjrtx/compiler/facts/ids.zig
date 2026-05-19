const std = @import("std");

/// Stable compiler fact IDs are value wrappers, not bare integers, so reports
/// and verifier joins cannot accidentally mix source, graph, cost, and lowering
/// namespaces.
pub const SourceId = struct {
    index: u32,

    pub fn eql(self: SourceId, other: SourceId) bool {
        return self.index == other.index;
    }
};

pub const GraphValueId = struct {
    index: u32,

    pub fn eql(self: GraphValueId, other: GraphValueId) bool {
        return self.index == other.index;
    }
};

pub const GraphInstructionId = struct {
    index: u32,

    pub fn eql(self: GraphInstructionId, other: GraphInstructionId) bool {
        return self.index == other.index;
    }
};

pub const CostLedgerId = struct {
    index: u32,

    pub fn eql(self: CostLedgerId, other: CostLedgerId) bool {
        return self.index == other.index;
    }
};

pub const MemoryTrafficId = struct {
    index: u32,

    pub fn eql(self: MemoryTrafficId, other: MemoryTrafficId) bool {
        return self.index == other.index;
    }
};

pub const LoweringRecordId = struct {
    index: u32,

    pub fn eql(self: LoweringRecordId, other: LoweringRecordId) bool {
        return self.index == other.index;
    }
};

test "compiler fact IDs compare by index within their namespace" {
    const source_zero: SourceId = .{ .index = 0 };
    const source_zero_again: SourceId = .{ .index = 0 };
    const source_one: SourceId = .{ .index = 1 };
    const cost_zero: CostLedgerId = .{ .index = 0 };
    const cost_one: CostLedgerId = .{ .index = 1 };

    try std.testing.expect(source_zero.eql(source_zero_again));
    try std.testing.expect(!source_zero.eql(source_one));
    try std.testing.expect(!cost_zero.eql(cost_one));
}
