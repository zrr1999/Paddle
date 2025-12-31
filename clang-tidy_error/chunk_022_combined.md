# Clang-Tidy Narrowing Conversion Errors - chunk_022_combined

**Total Errors in this chunk:** 50

**Files in this chunk:** 50

---

## File: `/workspace/Paddle/paddle/cinn/adt/generate_map_expr.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/adt/generate_map_expr.cc:67:20: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/common/simplify_special_pattern.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/common/simplify_special_pattern.cc:292:27: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/group_merge/single_op_fallback_to_phi.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/group_merge/single_op_fallback_to_phi.cc:255:33: error: narrowing conversion from 'typename iterator_traits<__normal_iterator<const long *, vector<long>>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/replace_dynamic_expand_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/replace_dynamic_expand_pass.cc:47:29: error: narrowing conversion from 'unsigned long' to signed type '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/framework/pir/op_mapper.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_mapper.cc:33:19: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'std::vector<int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/framework/pir/trivial_op_impl.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/trivial_op_impl.cc:583:26: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/op/reduction.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/op/reduction.cc:84:25: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/ir.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/ir.cc:857:12: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/operation.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/operation.cc:65:42: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/optim/call_arg_list_to_pod_value.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/optim/call_arg_list_to_pod_value.cc:44:50: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/optim/map_extern_call.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/optim/map_extern_call.cc:159:30: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/optim/unroll_loops.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/optim/unroll_loops.cc:95:18: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/runtime/cpu/onednn_math.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/runtime/cpu/onednn_math.cc:41:14: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/utils/string.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/utils/string.cc:67:9: error: narrowing conversion from 'int' to signed type 'char' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/common/ddim.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/common/ddim.cc:34:31: error: narrowing conversion from 'std::initializer_list::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/eager/activation_offloader.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/eager/activation_offloader.cc:211:24: error: narrowing conversion from 'std::chrono::duration<long, std::ratio<1, 1000000000>>::rep' (aka 'long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/operators/fused/fused_bn_activation_op.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/operators/fused/fused_bn_activation_op.cc:287:17: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/operators/fused/fused_bn_add_activation_op.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/operators/fused/fused_bn_add_activation_op.cc:252:17: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/operators/nccl/nccl_gpu_common.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/operators/nccl/nccl_gpu_common.cc:50:31: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::unordered_map<int, int>::mapped_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/distributed/ir/dist_op.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/distributed/ir/dist_op.cc:571:27: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/element_wise_binary.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/element_wise_binary.cc:32:14: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_slice_utils.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/infer_sym_slice_utils.cc:49:20: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/operator/utils/utils.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/utils/utils.cc:434:45: error: narrowing conversion from 'double' to 'std::vector<long>::value_type' (aka 'long') [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/general/auto_layout_insert_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/auto_layout_insert_pass.cc:179:26: error: narrowing conversion from 'uint32_t' (aka 'unsigned int') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/general/constant_folding_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/constant_folding_pass.cc:532:20: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/sub_graph_extract_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/sub_graph_extract_pass.cc:57:19: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pybind/args_mapper.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pybind/args_mapper.cc:142:35: error: narrowing conversion from 'int64_t' (aka 'long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pybind/eager_properties.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pybind/eager_properties.cc:663:56: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'Py_ssize_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/auto_parallel/dist_attr.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/dist_attr.cc:367:35: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/reshard/reshard_utils.cc:258:31: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/framework/dense_tensor_tostream.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/framework/dense_tensor_tostream.cc:151:29: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::streamsize' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/tensor_array.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/tensor_array.cc:128:37: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type '__gnu_cxx::__normal_iterator<phi::DenseTensor *, std::vector<phi::DenseTensor>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/binary.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/binary.cc:2242:37: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/fusion.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/fusion.cc:1013:17: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/conv2d.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/conv2d.cc:296:28: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/index_select.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/index_select.cc:82:27: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/reduction.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/reduction.cc:253:14: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'long' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/array_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/array_kernel.cc:151:14: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/linspace_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/linspace_kernel.cc:62:20: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/multiclass_nms3_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/multiclass_nms3_kernel.cc:81:35: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/nce_grad_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/nce_grad_kernel.cc:149:61: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/nce_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/nce_kernel.cc:238:48: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/fusion/cpu/fused_softmax_mask_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/fusion/cpu/fused_softmax_mask_kernel.cc:39:27: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/fusion/onednn/fused_transpose_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/fusion/onednn/fused_transpose_kernel.cc:35:33: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::set<long>::key_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/onednn/concat_grad_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/onednn/concat_grad_kernel.cc:43:52: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/onednn/flatten_grad_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/onednn/flatten_grad_kernel.cc:40:35: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/onednn/quantize_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/onednn/quantize_kernel.cc:35:27: error: narrowing conversion from 'int32_t' (aka 'int') to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/onednn/reshape_grad_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/onednn/reshape_grad_kernel.cc:42:35: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/onednn/squeeze_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/onednn/squeeze_kernel.cc:72:21: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/stride/strided_slice_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/stride/strided_slice_kernel.cc:73:22: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
