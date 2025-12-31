# Clang-Tidy Narrowing Conversion Errors - chunk_021_combined

**Total Errors in this chunk:** 50

**Files in this chunk:** 32

---

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/accuracy_check_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/accuracy_check_pass.cc:203:20: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/check_infer_symbolic_pass.cc:211:30: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'long' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/fuse_shape_ops_into_generate_shape_op_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/fuse_shape_ops_into_generate_shape_op_pass.cc:389:30: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'long' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/compute_at_reduction_tactic.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/compute_at_reduction_tactic.cc:297:22: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/optimize_reduction_tactic.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/optimize_reduction_tactic.cc:136:31: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/optimize_reduction_tactic.cc:145:7: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/schedule/impl/storage.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/schedule/impl/storage.cc:120:16: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/operator_fusion/pattern_graph.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/operator_fusion/pattern_graph.cc:119:20: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'std::map<std::shared_ptr<cinn::fusion::PatternNode>, int>::mapped_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/operator_fusion/pattern_graph.cc:93:20: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'std::map<std::shared_ptr<cinn::fusion::PatternNode>, int>::mapped_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/runtime/cuda/cuda_util.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/runtime/cuda/cuda_util.cc:2932:14: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/runtime/cuda/cuda_util.cc:2946:10: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/common/performance_statistician.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/common/performance_statistician.cc:82:19: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/common/performance_statistician.cc:98:19: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/fleet/gloo_wrapper.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/fleet/gloo_wrapper.cc:325:37: error: narrowing conversion from 'std::chrono::duration<long>::rep' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/fleet/gloo_wrapper.cc:337:37: error: narrowing conversion from 'std::chrono::duration<long>::rep' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.cc:429:36: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::set<int>::key_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/onednn/onednn_instruction.cc:456:40: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::set<int>::key_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/manual_op.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/manual_op.cc:3460:25: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/manual_op.cc:3597:47: error: narrowing conversion from 'double' to 'std::vector<long>::value_type' (aka 'long') [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/serialize_deserialize/src/ir_deserialize.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/serialize_deserialize/src/ir_deserialize.cc:116:21: error: narrowing conversion from 'std::map<long, pir::Value>::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/general/auto_mixed_precision_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/auto_mixed_precision_pass.cc:126:19: error: narrowing conversion from 'std::unordered_set<pir::Operation *>::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/auto_mixed_precision_pass.cc:590:24: error: narrowing conversion from 'uint32_t' (aka 'unsigned int') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/general/delete_weight_dequant_linear_op_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/delete_weight_dequant_linear_op_pass.cc:115:35: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/delete_weight_dequant_linear_op_pass.cc:115:59: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/gpu/add_norm_fuse_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/gpu/add_norm_fuse_pass.cc:118:34: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/gpu/add_norm_fuse_pass.cc:119:34: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/onednn/conv_concat_activation_onednn_fuse_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/conv_concat_activation_onednn_fuse_pass.cc:881:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/conv_concat_activation_onednn_fuse_pass.cc:885:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/onednn/fc_activation_fuse_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/fc_activation_fuse_pass.cc:308:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/fc_activation_fuse_pass.cc:312:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/onednn/self_attention_fuse_pass.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/self_attention_fuse_pass.cc:147:18: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pybind/eager_method.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pybind/eager_method.cc:1597:20: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pybind/eager_method.cc:1848:20: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/store/tcp_utils.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/store/tcp_utils.cc:181:20: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/platform/profiler.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/platform/profiler.cc:784:24: error: narrowing conversion from 'std::uniform_int_distribution<unsigned long>::result_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/conv2d_transpose.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/conv2d_transpose.cc:262:28: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/elementwise.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/elementwise.cc:502:19: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/elementwise.cc:516:19: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/fused_gemm_epilogue.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/fused_gemm_epilogue.cc:257:58: error: narrowing conversion from 'std::basic_string<char>::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/rms_norm.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/rms_norm.cc:169:30: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/unbind.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/unbind.cc:68:18: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/sequence_expand_kernel.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/sequence_expand_kernel.cc:28:25: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/sequence_expand_kernel.cc:39:23: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<unsigned long>, unsigned long>::value_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/funcs/gather_scatter_functor.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/funcs/gather_scatter_functor.cc:282:32: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/funcs/gather_scatter_functor.cc:283:31: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/fusion/onednn/fusion_lstm_kernel.cc`

**Number of errors:** 1

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/fusion/onednn/fusion_lstm_kernel.cc:376:21: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/onednn/concat_kernel.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/onednn/concat_kernel.cc:78:54: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/onednn/concat_kernel.cc:78:60: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/stride/view_kernel.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/stride/view_kernel.cc:39:28: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/stride/view_kernel.cc:47:19: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
