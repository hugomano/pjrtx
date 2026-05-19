#include "src/backend/mlx_metal/vendor_probe.h"

#include <cassert>

int main() {
  assert(pjrtx_mlx_vendor_version_major() == 0);
  assert(pjrtx_mlx_vendor_version_minor() >= 31);
  assert(pjrtx_mlx_vendor_version_patch() >= 0);
  assert(pjrtx_mlx_vendor_has_metal_headers() == 1);
  const int device_query = pjrtx_mlx_vendor_can_query_metal_device();
  assert(device_query == 0 || device_query == 1);
  return 0;
}
