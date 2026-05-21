const std = @import("std");

const ir = @import("src/compiler/ir");

const control_flow = @import("lowering_control_flow.zig");
const diagnostic = @import("lowering_diagnostic.zig");
const rules = @import("lowering_rules.zig");
const shapes = @import("lowering_shapes.zig");

/// Describes a rejected MLX backend lowering form with diagnostic context.
pub const Issue = diagnostic.Issue;
/// Names a region operand used by the recognized device-side while pattern.
pub const WhilePatternOperand = control_flow.WhilePatternOperand;
/// Recognized device-side while pattern used by MLX lowering and execution.
pub const WhileF32LtAddPattern = control_flow.WhileF32LtAddPattern;

/// Returns the first MLX backend lowering issue for a compiler executable plan.
pub const executableIssue = rules.executableIssue;

/// Writes a stable executable-level lowering diagnostic for the MLX backend.
pub fn writeExecutableDiagnostic(plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    if (rules.executableIssue(plan, device_local_hardware_ids)) |issue| {
        try diagnostic.writeIssue(plan, issue, writer);
        return;
    }
    try diagnostic.writeRejectedWithoutSpecificIssue(plan, writer);
}

/// Looks up a region value by id inside a cloned backend subprogram.
pub const regionValueById = control_flow.regionValueById;
/// Recognizes the currently supported device-side while compare/add pattern.
pub const matchWhileF32LtAddPattern = control_flow.matchWhileF32LtAddPattern;
/// Writes a stable human-readable lowering diagnostic for one issue.
pub const writeIssue = diagnostic.writeIssue;
/// Returns the axis for scatter forms executable by the MLX axis fast path.
pub const supportedScatterAxis = shapes.supportedScatterAxis;
/// Maps a compiler instruction kind to the MLX executable binary op payload.
pub const executableBinaryOp = rules.executableBinaryOp;
/// Maps a compiler instruction kind to the MLX executable unary op payload.
pub const executableUnaryOp = rules.executableUnaryOp;
