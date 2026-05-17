#include "src/backend/mlx_vendor_probe.h"

#include "mlx/backend/metal/metal.h"
#include "mlx/device.h"
#include "mlx/version.h"

int pjrtx_mlx_vendor_version_major(void) { return MLX_VERSION_MAJOR; }

int pjrtx_mlx_vendor_version_minor(void) { return MLX_VERSION_MINOR; }

int pjrtx_mlx_vendor_version_patch(void) { return MLX_VERSION_PATCH; }

int pjrtx_mlx_vendor_has_metal_headers(void) {
  return sizeof(&mlx::core::metal::is_available) > 0;
}

int pjrtx_mlx_vendor_can_query_metal_device(void) {
  try {
    return mlx::core::device_count(mlx::core::Device::gpu) > 0 ? 1 : 0;
  } catch (...) {
    return 0;
  }
}
