# Clang-Tidy Narrowing Conversion Errors - chunk_ 23_combined

**Total Errors in this chunk:** 35

**Files in this chunk:** 35

---

## File: `/workspace/Paddle/paddle/ap/src/axpr/serializable_value_helper.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/ap/src/axpr/serializable_value_helper.cc:213:32: error: narrowing conversion from 'std::size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ast_gen_ius/tensor_group.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ast_gen_ius/tensor_group.cc:100:33: error: narrowing conversion from 'std::unordered_set<std::basic_string<char>>::size_type' (aka 'unsigned long') to signed type 'std::unordered_map<std::basic_string<char>, int>::mapped_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/common/axis.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/common/axis.cc:53:20: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.cc:384:52: error: narrowing conversion from 'uint32_t' (aka 'unsigned int') to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/dynamic_reshape_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/dynamic_reshape_pass.cc:52:35: error: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/pe/schedule.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/pe/schedule.cc:146:21: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/stmt.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/stmt.cc:178:12: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/lang/compute.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/lang/compute.cc:155:44: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/optim/if_fusion_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/optim/if_fusion_pass.cc:77:23: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/optim/replace_cross_block_reduction.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/optim/replace_cross_block_reduction.cc:119:19: error: narrowing conversion from 'typename iterator_traits<__normal_iterator<Argument *, vector<Argument>>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/optim/resize_buffer.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/optim/resize_buffer.cc:280:15: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/runtime/cpu/thread_backend.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/runtime/cpu/thread_backend.cc:39:23: error: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/utils/event.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/utils/event.cc:86:38: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/utils/multi_threading.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/utils/multi_threading.cc:52:19: error: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/utils/timer.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/utils/timer.cc:23:14: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/compiled_program.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/compiled_program.cc:431:17: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/ir/quant_linear_fuse_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/ir/quant_linear_fuse_pass.cc:217:27: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/new_executor/standalone_executor.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/new_executor/standalone_executor.cc:93:38: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/tensor_util.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/tensor_util.cc:503:29: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::streamsize' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/ir_adaptor/translator/op_translator.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/ir_adaptor/translator/op_translator.cc:1609:22: error: narrowing conversion from 'int' to signed type 'int8_t' (aka 'signed char') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/onednn/operator_scale_onednn_fuse_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/operator_scale_onednn_fuse_pass.cc:164:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/onednn/shuffle_channel_detect_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/shuffle_channel_detect_pass.cc:210:18: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pybind/pybind.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pybind/pybind.cc:1910:26: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/r_to_x_reshard_function.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/r_to_x_reshard_function.cc:78:29: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'long' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/conv3d.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/conv3d.cc:248:28: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/layer_norm.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/layer_norm.cc:364:34: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/numel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/numel.cc:28:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/graph_khop_sampler_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/graph_khop_sampler_kernel.cc:251:20: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/roi_align_grad_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/roi_align_grad_kernel.cc:110:34: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/roi_align_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/roi_align_kernel.cc:227:34: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/set_value_grad_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/set_value_grad_kernel.cc:108:35: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/onednn/multi_gru_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/onednn/multi_gru_kernel.cc:154:25: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/onednn/reduce_mean_grad_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/onednn/reduce_mean_grad_kernel.cc:41:30: error: narrowing conversion from 'unsigned long' to signed type '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/onednn/shuffle_channel_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/onednn/shuffle_channel_kernel.cc:47:25: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/pir/src/dialect/shape/transforms/shape_optimization_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/pir/src/dialect/shape/transforms/shape_optimization_pass.cc:44:29: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::unordered_set<int>::key_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
