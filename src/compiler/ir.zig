/// Tensor element types, layouts, descriptors, and dense storage sizing.
pub const tensor = @import("tensor.zig");
/// Device, memory, placement, topology, compile option, and sharding vocabulary.
pub const topology = @import("topology.zig");
/// Shared instruction operation vocabulary used by executable and region plans.
pub const operation = @import("operation.zig");
/// Nested computation region identifiers, values, and instruction records.
pub const region = @import("region.zig");
/// Top-level executable plan values, aliases, instructions, and ownership.
pub const executable_plan = @import("executable_plan.zig");

/// Maximum number of devices the compiler IR topology vocabulary can describe.
pub const MAX_DEVICES = topology.MAX_DEVICES;
/// Memory class used by compiler-owned placement and topology descriptions.
pub const MemoryKind = topology.MemoryKind;
/// Scalar element type carried by compiler tensor descriptors.
pub const BufferType = tensor.BufferType;
/// Physical layout tag for compiler tensor descriptors.
pub const LayoutKind = tensor.LayoutKind;
/// Operation tag for binary elementwise instructions.
pub const ElementwiseBinaryOp = operation.ElementwiseBinaryOp;
/// Operation tag for unary elementwise instructions.
pub const ElementwiseUnaryOp = operation.ElementwiseUnaryOp;
/// Comparison direction for compare instructions and comparator regions.
pub const CompareOp = operation.CompareOp;
/// Update behavior for scatter instructions and scatter-update regions.
pub const ScatterUpdateKind = operation.ScatterUpdateKind;
/// Fast Fourier transform variant requested by an executable instruction.
pub const FftKind = operation.FftKind;
/// Random number distribution requested by RNG instructions.
pub const RngDistribution = operation.RngDistribution;
/// Matrix orientation requested by triangular solve instructions.
pub const TriangularSolveTranspose = operation.TriangularSolveTranspose;
/// Shared operation kind used by top-level and region plan instructions.
pub const PlanInstructionKind = operation.PlanInstructionKind;
/// Device facts discovered before compilation and consumed by placement.
pub const DeviceDescriptor = topology.DeviceDescriptor;
/// Memory facts discovered before compilation and consumed by placement.
pub const MemoryDescriptor = topology.MemoryDescriptor;
/// Tensor buffer description used by values and placement-aware runtime calls.
pub const BufferDescriptor = tensor.BufferDescriptor;
/// Concrete device-memory placement for a compiler value.
pub const Placement = topology.Placement;
/// Replica and partition device assignment for a compiled program.
pub const Topology = topology.Topology;
/// User and runtime compile options normalized for compiler planning.
pub const CompileOptions = topology.CompileOptions;
/// Sharding strategy class attached to parameters and outputs.
pub const ShardingKind = topology.ShardingKind;
/// Owned sharding metadata decoded from program attributes.
pub const ShardingMetadata = topology.ShardingMetadata;
/// Executable-plan sharding assignment for one parameter or output.
pub const ShardingPlan = topology.ShardingPlan;
/// Stable identifier for a value in an executable plan.
pub const ValueId = executable_plan.ValueId;
/// Role a value plays in the top-level executable plan.
pub const ValueRole = executable_plan.ValueRole;
/// Storage shape of a top-level executable-plan value.
pub const ValueStorageKind = executable_plan.ValueStorageKind;
/// Top-level executable-plan value with tensor descriptor and ownership role.
pub const Value = executable_plan.Value;
/// Output aliasing relationship class for result and parameter storage.
pub const OutputAliasKind = executable_plan.OutputAliasKind;
/// Output-to-parameter aliasing record for executable residency planning.
pub const OutputAlias = executable_plan.OutputAlias;
/// Stable identifier for a nested executable-plan region.
pub const RegionId = region.RegionId;
/// Stable identifier for a value scoped to a nested executable-plan region.
pub const RegionValueId = region.RegionValueId;
/// Semantic role of a nested executable-plan region.
pub const RegionKind = region.RegionKind;
/// Role a value plays inside a nested executable-plan region.
pub const RegionValueRole = region.RegionValueRole;
/// Region-scoped value with tensor descriptor and optional literal payload.
pub const RegionValue = region.RegionValue;
/// Nested executable-plan region used by control-flow and reducer operations.
pub const PlanRegion = region.PlanRegion;
/// Region-scoped instruction record used by nested computation bodies.
pub const RegionInstruction = region.RegionInstruction;
/// Top-level instruction record used by executable plans.
pub const PlanInstruction = executable_plan.PlanInstruction;
/// Owned compiler executable plan handed to runtime and backend layers.
pub const ExecutablePlan = executable_plan.ExecutablePlan;
/// Returns the byte size of a dense tensor, or zero for invalid/overflowing shapes.
pub const denseByteSize = tensor.denseByteSize;
