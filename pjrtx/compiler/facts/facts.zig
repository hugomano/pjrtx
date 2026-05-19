const std = @import("std");

pub const ids = @import("ids.zig");
pub const graph = @import("graph.zig");
pub const middle = @import("middle.zig");

/// Compiler facts are a small façade over owned fact families. Keep logic in
/// the owning family structs, not in this package root.
pub const ValidationError = graph.ValidationError;

pub const SourceId = ids.SourceId;
pub const GraphValueId = ids.GraphValueId;
pub const GraphInstructionId = ids.GraphInstructionId;
pub const CostLedgerId = ids.CostLedgerId;
pub const MemoryTrafficId = ids.MemoryTrafficId;
pub const LoweringRecordId = ids.LoweringRecordId;

pub const LayoutKind = graph.LayoutKind;
pub const SourceFrontend = graph.SourceFrontend;
pub const SourceRef = graph.SourceRef;
pub const TensorType = graph.TensorType;
pub const TensorFacts = graph.TensorFacts;
pub const GraphValueRole = graph.GraphValueRole;
pub const GraphValue = graph.GraphValue;
pub const GraphInstructionKind = graph.GraphInstructionKind;
pub const ElementwiseUnaryOp = graph.ElementwiseUnaryOp;
pub const ElementwiseBinaryOp = graph.ElementwiseBinaryOp;
pub const DotGeneralSpec = graph.DotGeneralSpec;
pub const ElementwiseUnarySpec = graph.ElementwiseUnarySpec;
pub const ElementwiseBinarySpec = graph.ElementwiseBinarySpec;
pub const BroadcastSpec = graph.BroadcastSpec;
pub const ReshapeSpec = graph.ReshapeSpec;
pub const TransposeSpec = graph.TransposeSpec;
pub const CollectiveOp = graph.CollectiveOp;
pub const CollectiveReduction = graph.CollectiveReduction;
pub const CollectiveSpec = graph.CollectiveSpec;
pub const ReturnSpec = graph.ReturnSpec;
pub const GraphPayload = graph.GraphPayload;
pub const GraphPayloadFacts = graph.GraphPayloadFacts;
pub const GraphInstruction = graph.GraphInstruction;

pub const MlirPassStatus = middle.MlirPassStatus;
pub const MlirPassRecord = middle.MlirPassRecord;
pub const GraphRewriteDecision = middle.GraphRewriteDecision;
pub const GraphRewriteRecord = middle.GraphRewriteRecord;
pub const FusionDecision = middle.FusionDecision;
pub const FusionPressureDelta = middle.FusionPressureDelta;
pub const FusionGroup = middle.FusionGroup;
pub const PlacementRecord = middle.PlacementRecord;
pub const CollectivePlanDecision = middle.CollectivePlanDecision;
pub const CollectiveAlgorithm = middle.CollectiveAlgorithm;
pub const CollectivePlanRecord = middle.CollectivePlanRecord;
pub const CostOpClass = middle.CostOpClass;
pub const CostLedgerEntry = middle.CostLedgerEntry;
pub const MemoryTrafficKind = middle.MemoryTrafficKind;
pub const MemoryTrafficRecord = middle.MemoryTrafficRecord;
pub const LoweringDecision = middle.LoweringDecision;
pub const LoweringRecord = middle.LoweringRecord;

test "compiler fact families are reachable from the package facade" {
    const source_id: SourceId = .{ .index = 0 };
    const instruction_id: GraphInstructionId = .{ .index = 1 };
    const lowering_id: LoweringRecordId = .{ .index = 2 };

    try std.testing.expect(source_id.eql(.{ .index = 0 }));
    try std.testing.expect(instruction_id.eql(.{ .index = 1 }));
    try std.testing.expect(lowering_id.eql(.{ .index = 2 }));
    try std.testing.expect(GraphPayloadFacts.matchesKind(.return_, .{ .return_ = .{} }));
}
