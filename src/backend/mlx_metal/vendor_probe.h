#ifndef PJRTX_BACKEND_MLX_VENDOR_PROBE_H_
#define PJRTX_BACKEND_MLX_VENDOR_PROBE_H_

#ifdef __cplusplus
extern "C" {
#endif

int pjrtx_mlx_vendor_version_major(void);
int pjrtx_mlx_vendor_version_minor(void);
int pjrtx_mlx_vendor_version_patch(void);
int pjrtx_mlx_vendor_has_metal_headers(void);
int pjrtx_mlx_vendor_can_query_metal_device(void);

#ifdef __cplusplus
}
#endif

#endif  // PJRTX_BACKEND_MLX_VENDOR_PROBE_H_
