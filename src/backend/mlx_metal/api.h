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
  PJRTX_MLX_METAL_DTYPE_PRED = 3,
  PJRTX_MLX_METAL_DTYPE_S32 = 4,
  PJRTX_MLX_METAL_DTYPE_U32 = 5,
  PJRTX_MLX_METAL_DTYPE_S8 = 6,
  PJRTX_MLX_METAL_DTYPE_F16 = 7,
  PJRTX_MLX_METAL_DTYPE_BF16 = 8,
};

enum {
  PJRTX_MLX_METAL_U8_BINARY_ADD = 0,
  PJRTX_MLX_METAL_U8_BINARY_SUBTRACT = 1,
  PJRTX_MLX_METAL_U8_BINARY_MULTIPLY = 2,
  PJRTX_MLX_METAL_U8_BINARY_DIVIDE = 3,
  PJRTX_MLX_METAL_BINARY_MAXIMUM = 4,
  PJRTX_MLX_METAL_BINARY_MINIMUM = 5,
  PJRTX_MLX_METAL_BINARY_POWER = 6,
  PJRTX_MLX_METAL_BINARY_REMAINDER = 7,
};

enum {
  PJRTX_MLX_METAL_U8_UNARY_NEGATE = 0,
  PJRTX_MLX_METAL_UNARY_EXP = 1,
  PJRTX_MLX_METAL_UNARY_TANH = 2,
  PJRTX_MLX_METAL_UNARY_SQRT = 3,
  PJRTX_MLX_METAL_UNARY_RSQRT = 4,
  PJRTX_MLX_METAL_UNARY_ABS = 5,
  PJRTX_MLX_METAL_UNARY_CEIL = 6,
  PJRTX_MLX_METAL_UNARY_FLOOR = 7,
  PJRTX_MLX_METAL_UNARY_LOG = 8,
  PJRTX_MLX_METAL_UNARY_LOG1P = 9,
  PJRTX_MLX_METAL_UNARY_LOGISTIC = 10,
  PJRTX_MLX_METAL_UNARY_SIN = 11,
  PJRTX_MLX_METAL_UNARY_COS = 12,
  PJRTX_MLX_METAL_UNARY_SIGN = 13,
};

enum {
  PJRTX_MLX_METAL_REDUCE_SUM = 0,
  PJRTX_MLX_METAL_REDUCE_MAX = 1,
};

enum {
  PJRTX_MLX_METAL_COMPARE_EQ = 0,
  PJRTX_MLX_METAL_COMPARE_NE = 1,
  PJRTX_MLX_METAL_COMPARE_GE = 2,
  PJRTX_MLX_METAL_COMPARE_GT = 3,
  PJRTX_MLX_METAL_COMPARE_LE = 4,
  PJRTX_MLX_METAL_COMPARE_LT = 5,
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
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_astype(
    PjrtxMlxMetalBuffer* src, int dtype);
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
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_reshape(
    PjrtxMlxMetalBuffer* src, const int64_t* dims, uint64_t rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_transpose(
    PjrtxMlxMetalBuffer* src, const int64_t* permutation, uint64_t rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_broadcast_in_dim(
    PjrtxMlxMetalBuffer* src, const int64_t* broadcast_dimensions,
    uint64_t operand_rank, const int64_t* output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_slice(
    PjrtxMlxMetalBuffer* src, const int64_t* start_indices,
    const int64_t* limit_indices, const int64_t* strides, uint64_t rank,
    const int64_t* output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_dynamic_slice(
    PjrtxMlxMetalBuffer* src, PjrtxMlxMetalBuffer* const* start_buffers,
    uint64_t num_start_buffers, const int64_t* slice_sizes, uint64_t rank,
    const int64_t* output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_dynamic_update_slice(
    PjrtxMlxMetalBuffer* src, PjrtxMlxMetalBuffer* update,
    PjrtxMlxMetalBuffer* const* start_buffers, uint64_t num_start_buffers,
    const int64_t* output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_pad(
    PjrtxMlxMetalBuffer* src, PjrtxMlxMetalBuffer* padding_value,
    const int64_t* edge_padding_low, const int64_t* edge_padding_high,
    const int64_t* interior_padding, uint64_t rank, const int64_t* output_dims,
    uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_reverse(
    PjrtxMlxMetalBuffer* src, const int64_t* dimensions,
    uint64_t num_dimensions, const int64_t* output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_concatenate(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs, int64_t dimension,
    const int64_t* output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_gather_axis(
    PjrtxMlxMetalBuffer* operand, PjrtxMlxMetalBuffer* indices, int64_t axis,
    int64_t index_vector_dim, const int64_t* output_dims,
    uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_sort(
    PjrtxMlxMetalBuffer* src, int64_t dimension, const int64_t* output_dims,
    uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_dot_general(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs,
    const int64_t* lhs_batch_dimensions, uint64_t lhs_batch_rank,
    const int64_t* rhs_batch_dimensions, uint64_t rhs_batch_rank,
    const int64_t* lhs_contracting_dimensions, uint64_t lhs_contracting_rank,
    const int64_t* rhs_contracting_dimensions, uint64_t rhs_contracting_rank,
    const int64_t* output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_reduce(
    PjrtxMlxMetalBuffer* src, int op, const int64_t* dimensions,
    uint64_t num_dimensions, const int64_t* output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_compare(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs, int direction,
    const int64_t* output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_select(
    PjrtxMlxMetalBuffer* pred, PjrtxMlxMetalBuffer* on_true,
    PjrtxMlxMetalBuffer* on_false, const int64_t* output_dims,
    uint64_t output_rank);
uint64_t pjrtx_mlx_metal_buffer_size(PjrtxMlxMetalBuffer* buffer);
int pjrtx_mlx_metal_buffer_has_host_shadow(PjrtxMlxMetalBuffer* buffer);
int pjrtx_mlx_metal_buffer_copy_to_host(PjrtxMlxMetalBuffer* buffer, void* dst,
                                        uint64_t dst_size);
void pjrtx_mlx_metal_buffer_destroy(PjrtxMlxMetalBuffer* buffer);

#ifdef __cplusplus
}
#endif

#endif  // PJRTX_BACKEND_MLX_METAL_API_H_
