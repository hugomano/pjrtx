const std = @import("std");
const mlir_session = @import("mlir_session.zig");
const decode = @import("stablehlo_decode.zig");
const diagnostic = @import("stablehlo_diagnostic.zig");
const module_import = @import("stablehlo_module_import.zig");
const sharding = @import("stablehlo_sharding.zig");
const model = @import("compiler_model.zig");

const AnalyzeError = model.AnalyzeError;
const ModuleAnalysis = model.ModuleAnalysis;
const ProgramFormat = model.ProgramFormat;

/// Creates a compiler buffer descriptor from an MLIR type.
pub const descriptorFromType = decode.descriptorFromType;
/// Adds a dialect to an owned dialect list if it is not already present.
pub const addDialect = decode.addDialect;
/// Returns whether a StableHLO operation name is currently imported by PjRTx.
pub const stablehloOpSupported = decode.stablehloOpSupported;
/// Writes an operation-scoped import diagnostic.
pub const writeOpDiagnostic = diagnostic.writeOpDiagnostic;
/// Writes a location-scoped import diagnostic.
pub const writeSimpleDiagnostic = diagnostic.writeSimpleDiagnostic;
/// Decodes source coordinates from an MLIR location.
pub const mlirLocationLineColumn = decode.mlirLocationLineColumn;
/// Decodes the element dtype text for an MLIR type.
pub const typeDtype = decode.typeDtype;
/// Decodes the rank of an MLIR shaped type.
pub const typeRank = decode.typeRank;
/// Decodes the shape dimensions of an MLIR shaped type.
pub const typeDims = decode.typeDims;
/// Decodes an MLIR integer-list attribute.
pub const intListAttribute = decode.intListAttribute;
/// Decodes one side of a flattened padding-pair list.
pub const paddingPairList = decode.paddingPairList;
/// Allocates an integer list filled with one value.
pub const filledIntList = decode.filledIntList;
/// Decodes an integer-list attribute or allocates a filled default.
pub const intListAttributeOrFill = decode.intListAttributeOrFill;
/// Decodes an MLIR boolean-list attribute.
pub const boolListAttribute = decode.boolListAttribute;
/// Decodes a boolean-list attribute or allocates a filled default.
pub const boolListAttributeOrFill = decode.boolListAttributeOrFill;
/// Decodes a scalar integer attribute.
pub const intAttribute = decode.intAttribute;
/// Decodes a scalar boolean attribute.
pub const boolAttribute = decode.boolAttribute;
/// Decodes a string attribute into owned bytes.
pub const stringAttribute = decode.stringAttribute;
/// Decodes StableHLO FFT kind attributes.
pub const fftKindFromAttr = decode.fftKindFromAttr;
/// Decodes StableHLO RNG distribution attributes.
pub const rngDistributionFromAttr = decode.rngDistributionFromAttr;
/// Decodes StableHLO triangular-solve transpose attributes.
pub const triangularTransposeFromAttr = decode.triangularTransposeFromAttr;
/// Returns a representative result or operand MLIR type for an operation.
pub const resultOrOperandType = decode.resultOrOperandType;
/// Decodes the sharding label used in unsupported-op diagnostics.
pub const operationShardingLabel = sharding.operationShardingLabel;
/// Decodes a Shardy mesh name from a tensor-sharding attribute.
pub const meshNameFromTensorSharding = sharding.meshNameFromTensorSharding;
/// Allocates sharding metadata with an owned mesh name.
pub const makeShardingMetadata = sharding.makeShardingMetadata;
/// Decodes sharding metadata from a tensor-sharding attribute.
pub const metadataFromTensorSharding = sharding.metadataFromTensorSharding;
/// Decodes sharding metadata from a value sharding attribute.
pub const metadataFromShardingAttribute = sharding.metadataFromShardingAttribute;
/// Decodes sharding metadata from a dictionary attribute.
pub const metadataFromDictionarySharding = sharding.metadataFromDictionarySharding;
/// Decodes sharding metadata from an indexed dictionary array.
pub const metadataFromArrayDictionarySharding = sharding.metadataFromArrayDictionarySharding;
/// Decodes donated output aliases from function argument attributes.
pub const aliasingOutputFromArrayDictionary = sharding.aliasingOutputFromArrayDictionary;
/// Decodes a function symbol name if present.
pub const functionSymbolName = decode.functionSymbolName;
/// Returns whether an MLIR function operation is the main entrypoint.
pub const isEntryFunction = decode.isEntryFunction;
/// Deep-copies sharding metadata.
pub const copyShardingMetadata = sharding.copyShardingMetadata;

/// Imports StableHLO/VHLO bytes from a reader and returns owned module analysis.
pub fn analyzeProgramFromReader(
    allocator: std.mem.Allocator,
    format_text: []const u8,
    program_reader: *std.Io.Reader,
    diagnostic_writer: *std.Io.Writer,
) AnalyzeError!ModuleAnalysis {
    const format = ProgramFormat.parse(format_text);
    if (format == .unknown) {
        try diagnostic_writer.print("unsupported program format: {s}", .{format_text});
        return error.UnsupportedProgramFormat;
    }

    var source = try program_reader.allocRemaining(allocator, .limited(2 * 1024 * 1024 * 1024));
    errdefer allocator.free(source);
    const use_portable_artifact = format == .stablehlo_bytecode or !mlir_session.isLikelyTextMlir(source);
    var session = if (use_portable_artifact) blk: {
        const artifact = source;
        var deserialized_text = std.Io.Writer.Allocating.init(allocator);
        errdefer deserialized_text.deinit();
        const deserialized = try mlir_session.deserializeStablehloPortableArtifactWithCapi(artifact, diagnostic_writer);
        try mlir_session.writeModuleText(deserialized, &deserialized_text.writer);
        source = try deserialized_text.toOwnedSlice();
        allocator.free(artifact);
        break :blk deserialized;
    } else try mlir_session.parseAndRunMlirWithCapi(source, diagnostic_writer);
    defer session.deinit();
    return module_import.analyzeMlirSessionWithCapi(allocator, source, session, diagnostic_writer);
}
