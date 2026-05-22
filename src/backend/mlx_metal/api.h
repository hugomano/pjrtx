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
typedef struct PjrtxMlxMetalProgram PjrtxMlxMetalProgram;
typedef struct PjrtxMlxMetalAsyncHostToDeviceTransfer
    PjrtxMlxMetalAsyncHostToDeviceTransfer;

typedef struct PjrtxMlxMetalProgramExecuteProfile {
  uint64_t host_enqueue_us;
  uint64_t device_sync_wait_us;
  uint64_t output_count;
  uint64_t donated_input_count;
  uint8_t measured_device_sync;
  uint8_t first_execute;
} PjrtxMlxMetalProgramExecuteProfile;

typedef struct PjrtxMetalCppFusionKernel PjrtxMetalCppFusionKernel;
typedef struct PjrtxMetalCppExecutableProgram PjrtxMetalCppExecutableProgram;

typedef struct PjrtxMetalCppTensorSpec {
  int dtype;
  const int64_t *dims;
  uint64_t rank;
} PjrtxMetalCppTensorSpec;

typedef struct PjrtxMetalCppExecutableValue {
  PjrtxMetalCppTensorSpec spec;
} PjrtxMetalCppExecutableValue;

typedef struct PjrtxMetalCppExecutableStep {
  const char *kernel_name;
  uint64_t kernel_name_size;
  const char *source;
  uint64_t source_size;
  const uint64_t *inputs;
  uint64_t input_count;
  const uint64_t *outputs;
  uint64_t output_count;
  const uint64_t *release_values;
  uint64_t release_value_count;
  uint64_t element_count;
  uint32_t threads_per_threadgroup;
  uint64_t in_place_input_plus_one;
} PjrtxMetalCppExecutableStep;

typedef struct PjrtxMetalCppFusionKernelCreateArgs {
  uint64_t struct_size;
  int device_ordinal;
  const char *kernel_name;
  uint64_t kernel_name_size;
  const char *source;
  uint64_t source_size;
  const PjrtxMetalCppTensorSpec *inputs;
  uint64_t input_count;
  const PjrtxMetalCppTensorSpec *outputs;
  uint64_t output_count;
  uint64_t element_count;
  uint32_t threads_per_threadgroup;
} PjrtxMetalCppFusionKernelCreateArgs;

typedef struct PjrtxMetalCppExecutableProgramCreateArgs {
  uint64_t struct_size;
  int device_ordinal;
  const char *entry_kernel_name;
  uint64_t entry_kernel_name_size;
  const char *source;
  uint64_t source_size;
  const PjrtxMetalCppTensorSpec *inputs;
  uint64_t input_count;
  const PjrtxMetalCppTensorSpec *outputs;
  uint64_t output_count;
  uint64_t element_count;
  uint32_t threads_per_threadgroup;
  const PjrtxMetalCppExecutableValue *values;
  uint64_t value_count;
  const uint64_t *input_values;
  uint64_t input_value_count;
  const uint64_t *output_values;
  uint64_t output_value_count;
  const PjrtxMetalCppExecutableStep *steps;
  uint64_t step_count;
} PjrtxMetalCppExecutableProgramCreateArgs;

typedef int (*PjrtxMlxMetalProgramBuildFn)(void *user_data,
                                           PjrtxMlxMetalBuffer *const *inputs,
                                           uint64_t input_count,
                                           PjrtxMlxMetalBuffer **outputs,
                                           uint64_t output_count);

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
  PJRTX_MLX_METAL_DTYPE_C64 = 9,
  PJRTX_MLX_METAL_DTYPE_U16 = 10,
  PJRTX_MLX_METAL_DTYPE_U64 = 11,
};

enum {
  PJRTX_MLX_METAL_FFT = 0,
  PJRTX_MLX_METAL_IFFT = 1,
  PJRTX_MLX_METAL_RFFT = 2,
  PJRTX_MLX_METAL_IRFFT = 3,
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
  PJRTX_MLX_METAL_BINARY_ATAN2 = 8,
  PJRTX_MLX_METAL_BINARY_AND = 9,
  PJRTX_MLX_METAL_BINARY_OR = 10,
  PJRTX_MLX_METAL_BINARY_XOR = 11,
  PJRTX_MLX_METAL_BINARY_SHIFT_LEFT = 12,
  PJRTX_MLX_METAL_BINARY_SHIFT_RIGHT = 13,
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
  PJRTX_MLX_METAL_UNARY_EXPM1 = 14,
  PJRTX_MLX_METAL_UNARY_NOT = 15,
  PJRTX_MLX_METAL_UNARY_ISFINITE = 16,
  PJRTX_MLX_METAL_UNARY_ROUND = 17,
  PJRTX_MLX_METAL_UNARY_CBRT = 18,
  PJRTX_MLX_METAL_UNARY_ROUND_AFZ = 19,
  PJRTX_MLX_METAL_UNARY_POPCNT = 20,
  PJRTX_MLX_METAL_UNARY_CLZ = 21,
};

enum {
  PJRTX_MLX_METAL_REDUCE_SUM = 0,
  PJRTX_MLX_METAL_REDUCE_MAX = 1,
  PJRTX_MLX_METAL_REDUCE_AND = 2,
  PJRTX_MLX_METAL_REDUCE_OR = 3,
  PJRTX_MLX_METAL_REDUCE_MIN = 4,
};

enum {
  PJRTX_MLX_METAL_SCATTER_SET = 0,
  PJRTX_MLX_METAL_SCATTER_ADD = 1,
};

enum {
  PJRTX_MLX_METAL_COMPARE_EQ = 0,
  PJRTX_MLX_METAL_COMPARE_NE = 1,
  PJRTX_MLX_METAL_COMPARE_GE = 2,
  PJRTX_MLX_METAL_COMPARE_GT = 3,
  PJRTX_MLX_METAL_COMPARE_LE = 4,
  PJRTX_MLX_METAL_COMPARE_LT = 5,
};

enum {
  PJRTX_MLX_METAL_RNG_UNIFORM = 0,
  PJRTX_MLX_METAL_RNG_NORMAL = 1,
};

int pjrtx_mlx_metal_version_major(void);
int pjrtx_mlx_metal_version_minor(void);
int pjrtx_mlx_metal_version_patch(void);
int pjrtx_mlx_metal_has_upstream_mlx_metal_api(void);
int pjrtx_mlx_metal_device_count(void);
int pjrtx_mlx_metal_copy_devices(PjrtxMlxMetalDeviceInfo *out_devices,
                                 int max_devices);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_from_host(int device_ordinal,
                                                      const void *data,
                                                      uint64_t byte_size);
PjrtxMlxMetalBuffer *
pjrtx_mlx_metal_buffer_from_host_typed(int device_ordinal, const void *data,
                                       uint64_t byte_size, int dtype,
                                       const int64_t *dims, uint64_t rank);
PjrtxMlxMetalAsyncHostToDeviceTransfer *
pjrtx_mlx_metal_async_h2d_create(int device_ordinal, int dtype,
                                 const int64_t *dims, uint64_t rank,
                                 uint64_t byte_size);
int pjrtx_mlx_metal_async_h2d_write(
    PjrtxMlxMetalAsyncHostToDeviceTransfer *transfer, uint64_t offset,
    const void *data, uint64_t byte_size);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_async_h2d_finish(
    PjrtxMlxMetalAsyncHostToDeviceTransfer *transfer);
void pjrtx_mlx_metal_async_h2d_destroy(
    PjrtxMlxMetalAsyncHostToDeviceTransfer *transfer);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_zeros(int device_ordinal, int dtype,
                                                  const int64_t *dims,
                                                  uint64_t rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_iota(int device_ordinal, int dtype,
                                                 const int64_t *dims,
                                                 uint64_t rank,
                                                 int64_t iota_dimension);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_partition_id(int device_ordinal,
                                                         int dtype,
                                                         uint32_t partition_id);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_clone(PjrtxMlxMetalBuffer *src);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_zero_like(PjrtxMlxMetalBuffer *src);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_complex(PjrtxMlxMetalBuffer *real,
                                                    PjrtxMlxMetalBuffer *imag,
                                                    const int64_t *output_dims,
                                                    uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_real(PjrtxMlxMetalBuffer *src,
                                                 const int64_t *output_dims,
                                                 uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_imag(PjrtxMlxMetalBuffer *src,
                                                 const int64_t *output_dims,
                                                 uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_astype(PjrtxMlxMetalBuffer *src,
                                                   int dtype);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_view_dtype(PjrtxMlxMetalBuffer *src,
                                                       int dtype,
                                                       const int64_t *dims,
                                                       uint64_t rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_add_u8(PjrtxMlxMetalBuffer *lhs,
                                                   PjrtxMlxMetalBuffer *rhs);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_binary_u8(PjrtxMlxMetalBuffer *lhs,
                                                      PjrtxMlxMetalBuffer *rhs,
                                                      int op);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_unary_u8(PjrtxMlxMetalBuffer *src,
                                                     int op);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_binary(PjrtxMlxMetalBuffer *lhs,
                                                   PjrtxMlxMetalBuffer *rhs,
                                                   int op);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_binary_out(
    PjrtxMlxMetalBuffer *lhs, PjrtxMlxMetalBuffer *rhs, int op,
    const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_metalcpp_buffer_dense_binary_out(
    PjrtxMlxMetalBuffer *lhs, PjrtxMlxMetalBuffer *rhs, int op,
    const int64_t *output_dims, uint64_t output_rank);
PjrtxMetalCppFusionKernel *pjrtx_metalcpp_fusion_kernel_create(
    const PjrtxMetalCppFusionKernelCreateArgs *args);
int pjrtx_metalcpp_fusion_kernel_execute(PjrtxMetalCppFusionKernel *kernel,
                                         PjrtxMlxMetalBuffer *const *inputs,
                                         uint64_t input_count,
                                         PjrtxMlxMetalBuffer ***out_outputs,
                                         uint64_t *out_output_count);
void pjrtx_metalcpp_fusion_output_array_destroy(PjrtxMlxMetalBuffer **outputs);
void pjrtx_metalcpp_fusion_kernel_destroy(PjrtxMetalCppFusionKernel *kernel);
PjrtxMetalCppExecutableProgram *pjrtx_metalcpp_executable_program_create(
    const PjrtxMetalCppExecutableProgramCreateArgs *args);
int pjrtx_metalcpp_executable_program_execute(
    PjrtxMetalCppExecutableProgram *program, PjrtxMlxMetalBuffer *const *inputs,
    uint64_t input_count, PjrtxMlxMetalBuffer ***out_outputs,
    uint64_t *out_output_count);
void pjrtx_metalcpp_executable_program_output_array_destroy(
    PjrtxMlxMetalBuffer **outputs);
void pjrtx_metalcpp_executable_program_destroy(
    PjrtxMetalCppExecutableProgram *program);
const char *pjrtx_metalcpp_last_error(void);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_unary(PjrtxMlxMetalBuffer *src,
                                                  int op);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_reshape(PjrtxMlxMetalBuffer *src,
                                                    const int64_t *dims,
                                                    uint64_t rank);
PjrtxMlxMetalBuffer *
pjrtx_mlx_metal_buffer_transpose(PjrtxMlxMetalBuffer *src,
                                 const int64_t *permutation, uint64_t rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_broadcast_in_dim(
    PjrtxMlxMetalBuffer *src, const int64_t *broadcast_dimensions,
    uint64_t operand_rank, const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_slice(
    PjrtxMlxMetalBuffer *src, const int64_t *start_indices,
    const int64_t *limit_indices, const int64_t *strides, uint64_t rank,
    const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_dynamic_slice(
    PjrtxMlxMetalBuffer *src, PjrtxMlxMetalBuffer *const *start_buffers,
    uint64_t num_start_buffers, const int64_t *slice_sizes, uint64_t rank,
    const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_dynamic_update_slice(
    PjrtxMlxMetalBuffer *src, PjrtxMlxMetalBuffer *update,
    PjrtxMlxMetalBuffer *const *start_buffers, uint64_t num_start_buffers,
    const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_pad(
    PjrtxMlxMetalBuffer *src, PjrtxMlxMetalBuffer *padding_value,
    const int64_t *edge_padding_low, const int64_t *edge_padding_high,
    const int64_t *interior_padding, uint64_t rank, const int64_t *output_dims,
    uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_reverse(PjrtxMlxMetalBuffer *src,
                                                    const int64_t *dimensions,
                                                    uint64_t num_dimensions,
                                                    const int64_t *output_dims,
                                                    uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_concatenate(
    PjrtxMlxMetalBuffer *lhs, PjrtxMlxMetalBuffer *rhs, int64_t dimension,
    const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_gather_axis(
    PjrtxMlxMetalBuffer *operand, PjrtxMlxMetalBuffer *indices, int64_t axis,
    int64_t index_vector_dim, const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_gather(
    PjrtxMlxMetalBuffer *operand, PjrtxMlxMetalBuffer *indices,
    const int64_t *start_index_map, uint64_t num_start_axes,
    const int64_t *collapsed_slice_dims, uint64_t num_collapsed_slice_dims,
    const int64_t *operand_batching_dims, uint64_t num_operand_batching_dims,
    const int64_t *start_indices_batching_dims,
    uint64_t num_start_indices_batching_dims, int64_t index_vector_dim,
    const int64_t *slice_sizes, uint64_t slice_rank, const int64_t *offset_dims,
    uint64_t num_offset_dims, const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_scatter_axis(
    PjrtxMlxMetalBuffer *operand, PjrtxMlxMetalBuffer *indices,
    PjrtxMlxMetalBuffer *updates, int64_t axis, int64_t index_vector_dim,
    int update_kind, const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_scatter(
    PjrtxMlxMetalBuffer *operand, PjrtxMlxMetalBuffer *indices,
    PjrtxMlxMetalBuffer *updates, const int64_t *scatter_dims_to_operand_dims,
    uint64_t num_scatter_axes, const int64_t *inserted_window_dims,
    uint64_t num_inserted_window_dims, const int64_t *update_window_dims,
    uint64_t num_update_window_dims, const int64_t *input_batching_dims,
    uint64_t num_input_batching_dims,
    const int64_t *scatter_indices_batching_dims,
    uint64_t num_scatter_indices_batching_dims, int64_t index_vector_dim,
    int update_kind, const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_sort(PjrtxMlxMetalBuffer *src,
                                                 int64_t dimension,
                                                 const int64_t *output_dims,
                                                 uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_argsort(PjrtxMlxMetalBuffer *src,
                                                    int64_t dimension,
                                                    int output_dtype,
                                                    const int64_t *output_dims,
                                                    uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_take_along_axis(
    PjrtxMlxMetalBuffer *src, PjrtxMlxMetalBuffer *indices, int64_t dimension,
    const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_dot_general(
    PjrtxMlxMetalBuffer *lhs, PjrtxMlxMetalBuffer *rhs,
    const int64_t *lhs_batch_dimensions, uint64_t lhs_batch_rank,
    const int64_t *rhs_batch_dimensions, uint64_t rhs_batch_rank,
    const int64_t *lhs_contracting_dimensions, uint64_t lhs_contracting_rank,
    const int64_t *rhs_contracting_dimensions, uint64_t rhs_contracting_rank,
    const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_convolution(
    PjrtxMlxMetalBuffer *lhs, PjrtxMlxMetalBuffer *rhs,
    const int64_t *window_strides, const int64_t *padding_low,
    const int64_t *padding_high, const int64_t *lhs_dilation,
    const int64_t *rhs_dilation, const uint8_t *window_reversal,
    uint64_t spatial_rank, int64_t feature_group_count,
    const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_cholesky(PjrtxMlxMetalBuffer *src,
                                                     int lower,
                                                     const int64_t *output_dims,
                                                     uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_triangular_solve(
    PjrtxMlxMetalBuffer *a, PjrtxMlxMetalBuffer *b, int left_side, int lower,
    int unit_diagonal, int transpose_a, const int64_t *output_dims,
    uint64_t output_rank);
PjrtxMlxMetalBuffer *
pjrtx_mlx_metal_buffer_fft(PjrtxMlxMetalBuffer *src, int fft_kind,
                           const int64_t *fft_lengths, uint64_t fft_rank,
                           const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *
pjrtx_mlx_metal_buffer_rng(PjrtxMlxMetalBuffer *a, PjrtxMlxMetalBuffer *b,
                           int distribution, int output_dtype,
                           const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_reduce(
    PjrtxMlxMetalBuffer *src, int op, const int64_t *dimensions,
    uint64_t num_dimensions, const int64_t *output_dims, uint64_t output_rank);
int pjrtx_mlx_metal_buffer_reduce_max_with_indices(
    PjrtxMlxMetalBuffer *values, PjrtxMlxMetalBuffer *indices,
    const int64_t *dimensions, uint64_t num_dimensions,
    const int64_t *output_dims, uint64_t output_rank,
    PjrtxMlxMetalBuffer **out_values, PjrtxMlxMetalBuffer **out_indices);
int pjrtx_mlx_metal_buffer_rng_bit_generator(PjrtxMlxMetalBuffer *state,
                                             int output_dtype,
                                             const int64_t *output_dims,
                                             uint64_t output_rank,
                                             PjrtxMlxMetalBuffer **out_state,
                                             PjrtxMlxMetalBuffer **out_bits);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_reduce_window(
    PjrtxMlxMetalBuffer *src, int op, const int64_t *window_dimensions,
    const int64_t *window_strides, const int64_t *base_dilations,
    const int64_t *window_dilations, const int64_t *padding_low,
    const int64_t *padding_high, uint64_t rank, const int64_t *output_dims,
    uint64_t output_rank);
int pjrtx_mlx_metal_buffer_reduce_window_max_with_indices(
    PjrtxMlxMetalBuffer *values, PjrtxMlxMetalBuffer *indices,
    const int64_t *window_dimensions, const int64_t *window_strides,
    const int64_t *base_dilations, const int64_t *window_dilations,
    const int64_t *padding_low, const int64_t *padding_high, uint64_t rank,
    const int64_t *output_dims, uint64_t output_rank,
    PjrtxMlxMetalBuffer **out_values, PjrtxMlxMetalBuffer **out_indices);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_compare(PjrtxMlxMetalBuffer *lhs,
                                                    PjrtxMlxMetalBuffer *rhs,
                                                    int direction,
                                                    const int64_t *output_dims,
                                                    uint64_t output_rank);
PjrtxMlxMetalBuffer *
pjrtx_mlx_metal_buffer_select(PjrtxMlxMetalBuffer *pred,
                              PjrtxMlxMetalBuffer *on_true,
                              PjrtxMlxMetalBuffer *on_false,
                              const int64_t *output_dims, uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_clamp(PjrtxMlxMetalBuffer *min,
                                                  PjrtxMlxMetalBuffer *value,
                                                  PjrtxMlxMetalBuffer *max,
                                                  const int64_t *output_dims,
                                                  uint64_t output_rank);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_while_f32_lt_add(
    PjrtxMlxMetalBuffer *state, PjrtxMlxMetalBuffer *limit,
    PjrtxMlxMetalBuffer *step, const int64_t *output_dims, uint64_t output_rank,
    uint64_t max_iterations);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_buffer_while_f32_compare_add(
    PjrtxMlxMetalBuffer *state, PjrtxMlxMetalBuffer *limit,
    PjrtxMlxMetalBuffer *step, int compare_direction, int update_op,
    const int64_t *output_dims, uint64_t output_rank, uint64_t max_iterations);
PjrtxMlxMetalBuffer *
pjrtx_mlx_metal_custom_call_binary_add_f32(PjrtxMlxMetalBuffer *lhs,
                                           PjrtxMlxMetalBuffer *rhs);
PjrtxMlxMetalBuffer *pjrtx_mlx_metal_custom_call_scaled_dot_product_attention(
    PjrtxMlxMetalBuffer *q, PjrtxMlxMetalBuffer *k, PjrtxMlxMetalBuffer *v,
    PjrtxMlxMetalBuffer *token_index);
uint64_t pjrtx_mlx_metal_buffer_size(PjrtxMlxMetalBuffer *buffer);
int pjrtx_mlx_metal_buffer_has_host_shadow(PjrtxMlxMetalBuffer *buffer);
int pjrtx_mlx_metal_buffer_eval(PjrtxMlxMetalBuffer *buffer);
int pjrtx_mlx_metal_buffer_eval_many(PjrtxMlxMetalBuffer *const *buffers,
                                     uint64_t count);
int pjrtx_mlx_metal_buffer_copy_to_host(PjrtxMlxMetalBuffer *buffer, void *dst,
                                        uint64_t dst_size);
void pjrtx_mlx_metal_buffer_destroy(PjrtxMlxMetalBuffer *buffer);
PjrtxMlxMetalProgram *
pjrtx_mlx_metal_program_create(void *user_data, uint64_t input_count,
                               uint64_t output_count,
                               PjrtxMlxMetalProgramBuildFn build_fn);
PjrtxMlxMetalProgram *pjrtx_mlx_metal_program_create_with_captures(
    void *user_data, uint64_t full_input_count, uint64_t output_count,
    PjrtxMlxMetalProgramBuildFn build_fn,
    PjrtxMlxMetalBuffer *const *captured_inputs,
    const uint64_t *dynamic_indices, uint64_t dynamic_count);
int pjrtx_mlx_metal_program_execute(PjrtxMlxMetalProgram *program,
                                    PjrtxMlxMetalBuffer *const *inputs,
                                    uint64_t input_count,
                                    PjrtxMlxMetalBuffer ***out_outputs,
                                    uint64_t *out_output_count);
int pjrtx_mlx_metal_program_execute_with_donation(
    PjrtxMlxMetalProgram *program, PjrtxMlxMetalBuffer *const *inputs,
    uint64_t input_count, const uint64_t *donated_input_indices,
    uint64_t donated_input_count, PjrtxMlxMetalBuffer ***out_outputs,
    uint64_t *out_output_count,
    PjrtxMlxMetalProgramExecuteProfile *out_profile);
void pjrtx_mlx_metal_program_output_array_destroy(
    PjrtxMlxMetalBuffer **outputs);
void pjrtx_mlx_metal_program_destroy(PjrtxMlxMetalProgram *program);

#ifdef __cplusplus
}
#endif

#endif // PJRTX_BACKEND_MLX_METAL_API_H_
