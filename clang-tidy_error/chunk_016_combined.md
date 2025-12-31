# Clang-Tidy Narrowing Conversion Errors - chunk_016_combined

**Total Errors in this chunk:** 50

**Files in this chunk:** 13

---

## File: `/workspace/Paddle/paddle/cinn/adt/schedule_mesh.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/adt/schedule_mesh.cc:297:14: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/adt/schedule_mesh.cc:300:14: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/adt/schedule_mesh.cc:310:14: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/adt/schedule_mesh.cc:313:14: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.cc:392:24: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.cc:394:17: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.cc:405:24: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/ir/generate_shape_util.cc:406:17: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/merge_reshape_with_broadcast_pass.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/merge_reshape_with_broadcast_pass.cc:46:17: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/merge_reshape_with_broadcast_pass.cc:47:36: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/merge_reshape_with_broadcast_pass.cc:52:18: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/merge_reshape_with_broadcast_pass.cc:53:38: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/general/auto_layout_pass.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/auto_layout_pass.cc:134:77: error: narrowing conversion from constant 'double' to 'int' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/auto_layout_pass.cc:143:59: error: narrowing conversion from 'float' to 'int' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/auto_layout_pass.cc:144:12: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/auto_layout_pass.cc:144:45: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/general/params_sync_among_devices_pass.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/params_sync_among_devices_pass.cc:113:47: error: narrowing conversion from 'unsigned long' to signed type '__gnu_cxx::__normal_iterator<phi::DenseTensor **, std::vector<phi::DenseTensor *>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/params_sync_among_devices_pass.cc:114:32: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type '__gnu_cxx::__normal_iterator<phi::DenseTensor **, std::vector<phi::DenseTensor *>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/params_sync_among_devices_pass.cc:126:64: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::reverse_iterator<__gnu_cxx::__normal_iterator<phi::DenseTensor **, std::vector<phi::DenseTensor *>>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/params_sync_among_devices_pass.cc:97:21: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/gpu/fused_dot_product_attention_pass.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/gpu/fused_dot_product_attention_pass.cc:113:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/gpu/fused_dot_product_attention_pass.cc:268:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/gpu/fused_dot_product_attention_pass.cc:406:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/gpu/fused_dot_product_attention_pass.cc:576:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.cc:41:28: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.cc:73:21: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/s_to_r_reshard_function.cc:90:11: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/store/tcp_store_libuv.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/store/tcp_store_libuv.cc:204:30: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/distributed/store/tcp_store_libuv.cc:361:40: error: narrowing conversion from 'ssize_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/distributed/store/tcp_store_libuv.cc:361:74: error: narrowing conversion from 'ssize_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/memory/allocation/cuda_malloc_async_allocator.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/memory/allocation/cuda_malloc_async_allocator.cc:203:43: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/memory/allocation/cuda_malloc_async_allocator.cc:205:43: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/memory/allocation/cuda_malloc_async_allocator.cc:258:50: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/memory/allocation/cuda_malloc_async_allocator.cc:295:58: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/platform/collective_helper.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/platform/collective_helper.cc:135:24: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/platform/collective_helper.cc:147:30: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/platform/collective_helper.cc:147:46: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/platform/collective_helper.cc:168:24: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/expand.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/expand.cc:23:14: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/expand.cc:24:16: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/expand.cc:50:25: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/expand.cc:50:40: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/crf_decoding_kernel.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/crf_decoding_kernel.cc:109:31: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/crf_decoding_kernel.cc:54:15: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/crf_decoding_kernel.cc:58:16: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/crf_decoding_kernel.cc:94:27: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/interpolate_kernel.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_kernel.cc:134:23: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_kernel.cc:61:55: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_kernel.cc:67:55: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_kernel.cc:67:59: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
