#ifndef PJRTX_BACKEND_MLX_METAL_API_H_
#define PJRTX_BACKEND_MLX_METAL_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PJRTX_MLX_METAL_DEVICE_NAME_BYTES 128

typedef struct PjrtxMlxMetalDeviceInfo {
  int32_t ordinal;
  uint64_t registry_id;
  uint64_t recommended_max_working_set_size;
  uint8_t has_unified_memory;
  char name[PJRTX_MLX_METAL_DEVICE_NAME_BYTES];
} PjrtxMlxMetalDeviceInfo;

typedef struct PjrtxMlxMetalBuffer PjrtxMlxMetalBuffer;

enum {
  PJRTX_MLX_METAL_DTYPE_INVALID = 0,
  PJRTX_MLX_METAL_DTYPE_U8 = 1,
  PJRTX_MLX_METAL_DTYPE_F32 = 2,
};

enum {
  PJRTX_MLX_METAL_U8_BINARY_ADD = 0,
  PJRTX_MLX_METAL_U8_BINARY_SUBTRACT = 1,
  PJRTX_MLX_METAL_U8_BINARY_MULTIPLY = 2,
  PJRTX_MLX_METAL_U8_BINARY_DIVIDE = 3,
};

enum {
  PJRTX_MLX_METAL_U8_UNARY_NEGATE = 0,
  PJRTX_MLX_METAL_UNARY_EXP = 1,
  PJRTX_MLX_METAL_UNARY_TANH = 2,
  PJRTX_MLX_METAL_UNARY_SQRT = 3,
  PJRTX_MLX_METAL_UNARY_RSQRT = 4,
};

int pjrtx_mlx_metal_version_major(void);
int pjrtx_mlx_metal_version_minor(void);
int pjrtx_mlx_metal_version_patch(void);
int pjrtx_mlx_metal_has_upstream_mlx_metal_api(void);
int pjrtx_mlx_metal_device_count(void);
int pjrtx_mlx_metal_copy_devices(PjrtxMlxMetalDeviceInfo* out_devices,
                                 int max_devices);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_from_host(int device_ordinal,
                                                      const void* data,
                                                      uint64_t byte_size);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_from_host_typed(
    int device_ordinal, const void* data, uint64_t byte_size, int dtype,
    const int64_t* dims, uint64_t rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_clone(PjrtxMlxMetalBuffer* src);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_add_u8(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_binary_u8(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs, int op);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_unary_u8(
    PjrtxMlxMetalBuffer* src, int op);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_binary(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs, int op);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_unary(
    PjrtxMlxMetalBuffer* src, int op);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_transpose(
    PjrtxMlxMetalBuffer* src, const int64_t* permutation, uint64_t rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_broadcast_in_dim(
    PjrtxMlxMetalBuffer* src, const int64_t* broadcast_dimensions,
    uint64_t operand_rank, const int64_t* output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_slice(
    PjrtxMlxMetalBuffer* src, const int64_t* start_indices,
    const int64_t* limit_indices, const int64_t* strides, uint64_t rank,
    const int64_t* output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_concatenate(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs, int64_t dimension,
    const int64_t* output_dims, uint64_t output_rank);
uint64_t pjrtx_mlx_metal_buffer_size(PjrtxMlxMetalBuffer* buffer);
int pjrtx_mlx_metal_buffer_copy_to_host(PjrtxMlxMetalBuffer* buffer, void* dst,
                                        uint64_t dst_size);
void pjrtx_mlx_metal_buffer_destroy(PjrtxMlxMetalBuffer* buffer);

#ifdef __cplusplus
}
#endif

#endif  // PJRTX_BACKEND_MLX_METAL_API_H_
