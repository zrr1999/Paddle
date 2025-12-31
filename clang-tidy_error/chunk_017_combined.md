# Clang-Tidy Narrowing Conversion Errors - chunk_017_combined

**Total Errors in this chunk:** 51

**Files in this chunk:** 17

---

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/fold_output_data_derivable_ops_pass.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/fold_output_data_derivable_ops_pass.cc:153:38: error: narrowing conversion from 'long' to signed type 'std::vector<int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/fold_output_data_derivable_ops_pass.cc:155:28: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/fold_output_data_derivable_ops_pass.cc:185:29: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/op/elementwise.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/op/elementwise.cc:1550:22: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/op/elementwise.cc:1620:22: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/op/elementwise.cc:815:22: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/pe/broadcast.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/pe/broadcast.cc:135:14: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/broadcast.cc:149:27: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/broadcast.cc:181:28: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/pe/nn_util.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/pe/nn_util.cc:381:13: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/nn_util.cc:382:13: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/nn_util.cc:477:25: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/group_schedule/search/config_searcher.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/search/config_searcher.cc:115:21: error: narrowing conversion from 'std::chrono::duration<double, std::ratio<1, 1000000>>::rep' (aka 'double') to 'cinn::ir::search::ScoreType' (aka 'float') [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/search/config_searcher.cc:138:14: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/search/config_searcher.cc:181:10: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/schedule/impl/base.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/schedule/impl/base.cc:373:26: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/schedule/impl/base.cc:420:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/schedule/impl/base.cc:652:10: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/imperative/reducer.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/imperative/reducer.cc:677:36: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::vector<long>::value_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/imperative/reducer.cc:849:34: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/imperative/reducer.cc:861:46: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/inference/analysis/passes/ir_params_sync_among_devices_pass.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/inference/analysis/passes/ir_params_sync_among_devices_pass.cc:149:45: error: narrowing conversion from 'unsigned long' to signed type '__gnu_cxx::__normal_iterator<phi::DenseTensor **, std::vector<phi::DenseTensor *>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/inference/analysis/passes/ir_params_sync_among_devices_pass.cc:150:30: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type '__gnu_cxx::__normal_iterator<phi::DenseTensor **, std::vector<phi::DenseTensor *>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/inference/analysis/passes/ir_params_sync_among_devices_pass.cc:161:62: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::reverse_iterator<__gnu_cxx::__normal_iterator<phi::DenseTensor **, std::vector<phi::DenseTensor *>>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/inference/io.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/inference/io.cc:189:48: error: narrowing conversion from 'unsigned long' to signed type '__gnu_cxx::__normal_iterator<paddle::framework::VarDesc **, std::vector<paddle::framework::VarDesc *>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/inference/io.cc:190:30: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type '__gnu_cxx::__normal_iterator<paddle::framework::VarDesc **, std::vector<paddle::framework::VarDesc *>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/inference/io.cc:202:52: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::reverse_iterator<__gnu_cxx::__normal_iterator<paddle::framework::VarDesc **, std::vector<paddle::framework::VarDesc *>>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/nullary_infer_sym.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/nullary_infer_sym.cc:340:24: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/nullary_infer_sym.cc:355:9: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/nullary_infer_sym.cc:413:36: error: narrowing conversion from 'int64_t' (aka 'long') to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/drr/src/ir_operation_factory.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/drr/src/ir_operation_factory.cc:307:15: error: narrowing conversion from 'float' to 'bool' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/drr/src/ir_operation_factory.cc:315:15: error: narrowing conversion from 'float' to 'bool' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/drr/src/ir_operation_factory.cc:877:23: error: narrowing conversion from 'float' to 'bool' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pybind/arg_pre_process.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pybind/arg_pre_process.cc:109:30: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pybind/arg_pre_process.cc:111:24: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pybind/arg_pre_process.cc:92:24: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/auto_parallel/process_mesh.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/process_mesh.cc:154:28: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/process_mesh.cc:205:23: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/process_mesh.cc:229:13: error: narrowing conversion from 'typename __normal_iterator<long *, vector<long>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/platform/device/gpu/gpu_info.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/platform/device/gpu/gpu_info.cc:100:24: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/platform/device/gpu/gpu_info.cc:100:7: error: narrowing conversion from 'double' to 'size_t' (aka 'unsigned long') [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/platform/device/gpu/gpu_info.cc:101:24: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/gather_nd.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/gather_nd.cc:136:22: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/gather_nd.cc:43:22: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/gather_nd.cc:96:22: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/reshape.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/reshape.cc:222:44: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/reshape.cc:305:9: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/reshape.cc:313:9: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/pir/src/dialect/shape/utils/dim_expr_util.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/pir/src/dialect/shape/utils/dim_expr_util.cc:1035:18: error: narrowing conversion from 'std::size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/pir/src/dialect/shape/utils/dim_expr_util.cc:1098:18: error: narrowing conversion from 'std::size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/pir/src/dialect/shape/utils/dim_expr_util.cc:1100:18: error: narrowing conversion from 'std::size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
