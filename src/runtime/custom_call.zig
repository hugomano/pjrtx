const backend_api = @import("src/backend/mlx_metal");

/// Public custom-call registration kind accepted by the Metal/MLX runtime.
pub const CustomCallKind = backend_api.CustomCallKind;

/// Public custom-call registration record forwarded to the Metal/MLX backend.
pub const CustomCallRegistration = backend_api.CustomCallRegistration;

/// Errors returned while mutating the Metal/MLX custom-call registry.
pub const CustomCallRegistrationError = backend_api.Error;

/// Registers a fully typed custom-call target with the Metal/MLX backend.
pub fn registerCustomCall(registration: CustomCallRegistration) CustomCallRegistrationError!void {
    var backend_impl = backend_api.create();
    return backend_impl.registerCustomCall(registration);
}

/// Registers a named binary custom-call target with backend-owned validation.
pub fn registerBinaryCustomCall(target: []const u8, op_name: []const u8) CustomCallRegistrationError!void {
    var backend_impl = backend_api.create();
    return backend_impl.registerBinaryCustomCall(target, op_name);
}

/// Registers an identity custom-call target whose execution aliases its input.
pub fn registerIdentityCustomCall(target: []const u8) CustomCallRegistrationError!void {
    var backend_impl = backend_api.create();
    return backend_impl.registerIdentityCustomCall(target);
}

/// Registers a named unary custom-call target with backend-owned validation.
pub fn registerUnaryCustomCall(target: []const u8, op_name: []const u8) CustomCallRegistrationError!void {
    var backend_impl = backend_api.create();
    return backend_impl.registerUnaryCustomCall(target, op_name);
}

/// Registers the built-in square-root custom-call marker.
pub fn registerUnarySqrtCustomCall(target: []const u8) CustomCallRegistrationError!void {
    var backend_impl = backend_api.create();
    return backend_impl.registerUnarySqrtCustomCall(target);
}

/// Registers the built-in binary add custom-call marker.
pub fn registerBinaryAddCustomCall(target: []const u8) CustomCallRegistrationError!void {
    var backend_impl = backend_api.create();
    return backend_impl.registerBinaryAddCustomCall(target);
}

/// Removes a custom-call target from the Metal/MLX backend registry.
pub fn unregisterCustomCall(target: []const u8) void {
    var backend_impl = backend_api.create();
    backend_impl.unregisterCustomCall(target);
}
