# Clang-Tidy Narrowing Conversion Errors - chunk_019_combined

**Total Errors in this chunk:** 50

**Files in this chunk:** 25

---

## File: `/workspace/Paddle/paddle/cinn/backends/llvm/codegen_x86.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_x86.cc:59:64: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_x86.cc:71:67: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/framework/pir/utils.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/utils.cc:193:32: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/utils.cc:893:43: error: narrowing conversion from 'unsigned long' to signed type '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/op/contrib/sort.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/op/contrib/sort.cc:140:17: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/op/contrib/sort.cc:81:17: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/op/nn.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/op/nn.cc:1174:18: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/op/nn.cc:1189:19: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/group_schedule/config/file_database.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/config/file_database.cc:202:47: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/config/file_database.cc:50:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/schedule/schedule_desc.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/schedule/schedule_desc.cc:781:36: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/schedule/schedule_desc.cc:793:20: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/ir/delete_weight_dequant_linear_op_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/ir/delete_weight_dequant_linear_op_pass.cc:145:40: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/ir/delete_weight_dequant_linear_op_pass.cc:146:51: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/ir/embedding_eltwise_layernorm_fuse_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/ir/embedding_eltwise_layernorm_fuse_pass.cc:298:13: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/ir/embedding_eltwise_layernorm_fuse_pass.cc:299:13: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/ir/fusion_group/fusion_group_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/ir/fusion_group/fusion_group_pass.cc:53:15: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/ir/fusion_group/fusion_group_pass.cc:69:26: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/ir/onednn/int8_scale_calculation_onednn_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/ir/onednn/int8_scale_calculation_onednn_pass.cc:154:16: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/ir/onednn/int8_scale_calculation_onednn_pass.cc:156:18: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/imperative/heter_ccl_context.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/imperative/heter_ccl_context.cc:64:22: error: narrowing conversion from 'std::set<std::basic_string<char>>::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/imperative/heter_ccl_context.cc:90:31: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/same_operands_result.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/same_operands_result.cc:247:22: error: narrowing conversion from 'int64_t' (aka 'long') to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/same_operands_result.cc:247:54: error: narrowing conversion from 'int64_t' (aka 'long') to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/tensorrt_op.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/tensorrt_op.cc:204:34: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'std::vector<int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/tensorrt_op.cc:240:46: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/primitive/base/decomp_trans.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/primitive/base/decomp_trans.cc:430:47: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/primitive/base/decomp_trans.cc:430:56: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pybind/op_function_common.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pybind/op_function_common.cc:460:39: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pybind/op_function_common.cc:460:47: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/api/include/compat/torch/csrc/api/include/torch/cuda.cpp`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/api/include/compat/torch/csrc/api/include/torch/cuda.cpp:26:10: error: narrowing conversion from 'int' to signed type 'c10::DeviceIndex' (aka 'signed char') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/api/include/compat/torch/csrc/api/include/torch/cuda.cpp:43:33: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/global_and_sub_mesh_reshard_function.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/global_and_sub_mesh_reshard_function.cc:117:32: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/global_and_sub_mesh_reshard_function.cc:134:30: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/nd_mesh_reshard_function.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/nd_mesh_reshard_function.cc:377:29: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/nd_mesh_reshard_function.cc:387:24: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/p_to_s_reshard_function.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/p_to_s_reshard_function.cc:103:31: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/p_to_s_reshard_function.cc:90:27: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/comm_task_manager.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/comm_task_manager.cc:121:21: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/distributed/comm_task_manager.cc:127:16: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/c_softmax_with_multi_label_cross_entropy.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/c_softmax_with_multi_label_cross_entropy.cc:159:40: error: narrowing conversion from 'int' to signed type '__gnu_cxx::__alloc_traits<std::allocator<char>, char>::value_type' (aka 'char') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/c_softmax_with_multi_label_cross_entropy.cc:39:35: error: narrowing conversion from 'int' to signed type '__gnu_cxx::__alloc_traits<std::allocator<char>, char>::value_type' (aka 'char') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/cross_entropy_with_softmax.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/cross_entropy_with_softmax.cc:246:20: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/cross_entropy_with_softmax.cc:269:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/stack.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/stack.cc:109:41: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/stack.cc:64:46: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/lookup_table_dequant_kernel.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/lookup_table_dequant_kernel.cc:32:31: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/lookup_table_dequant_kernel.cc:34:19: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/fusion/cpu/fused_rms_norm_quant_avx_kernel.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/fusion/cpu/fused_rms_norm_quant_avx_kernel.cc:49:13: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/fusion/cpu/fused_rms_norm_quant_avx_kernel.cc:52:13: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
