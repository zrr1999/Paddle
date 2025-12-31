# Clang-Tidy Narrowing Conversion Errors - chunk_020_combined

**Total Errors in this chunk:** 50

**Files in this chunk:** 25

---

## File: `/workspace/Paddle/paddle/cinn/backends/codegen_gpu_dev.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/backends/codegen_gpu_dev.cc:502:15: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/codegen_gpu_dev.cc:542:15: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/common/dim_expr_converter.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/common/dim_expr_converter.cc:151:24: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/common/dim_expr_converter.cc:172:24: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/common/graph_utils.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/common/graph_utils.cc:48:21: error: narrowing conversion from 'std::set<cinn::common::GraphNode *>::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/common/graph_utils.cc:82:25: error: narrowing conversion from 'std::set<cinn::common::Shared<cinn::common::GraphEdge>, cinn::common::GraphEdgeCompare>::size_type' (aka 'unsigned long') to signed type 'std::map<std::basic_string<char>, int>::mapped_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/tensor.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/tensor.cc:237:40: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/tensor.cc:328:65: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/runtime/tiny_runtime.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/runtime/tiny_runtime.cc:27:23: error: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/runtime/tiny_runtime.cc:43:15: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/new_executor/feed_fetch_utils.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/new_executor/feed_fetch_utils.cc:109:15: error: narrowing conversion from 'typename __normal_iterator<const basic_string<char> *, vector<basic_string<char>>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/feed_fetch_utils.cc:257:17: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/new_executor/instruction/instruction_base.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/instruction_base.cc:122:12: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/instruction_base.cc:124:12: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/operators/generator/get_expected_kernel_func.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/operators/generator/get_expected_kernel_func.cc:101:75: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/operators/generator/get_expected_kernel_func.cc:102:48: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/control_flow_op.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/control_flow_op.cc:1015:34: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/control_flow_op.cc:935:20: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/build_cinn_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/build_cinn_pass.cc:105:32: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/build_cinn_pass.cc:59:19: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/onednn/conv_activation_onednn_fuse_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/conv_activation_onednn_fuse_pass.cc:466:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/conv_activation_onednn_fuse_pass.cc:470:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/onednn/elementwise_act_onednn_fuse_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/elementwise_act_onednn_fuse_pass.cc:235:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/elementwise_act_onednn_fuse_pass.cc:239:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/onednn/scale_matmul_fuse_pass.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/scale_matmul_fuse_pass.cc:106:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/scale_matmul_fuse_pass.cc:234:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/utils/general_functions.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/utils/general_functions.cc:113:37: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/utils/general_functions.cc:128:36: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pybind/imperative.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pybind/imperative.cc:1355:15: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pybind/imperative.cc:1356:35: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pybind/sot/guards.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pybind/sot/guards.cc:161:18: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pybind/sot/guards.cc:214:14: error: narrowing conversion from 'pybind11::ssize_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/api/lib/op_meta_info.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/api/lib/op_meta_info.cc:381:29: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/api/lib/op_meta_info.cc:398:37: error: narrowing conversion from 'std::unordered_map<unsigned long, unsigned long>::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/distributed/auto_parallel/dist_tensor.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/dist_tensor.cc:52:44: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::vector<long>::value_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/distributed/auto_parallel/dist_tensor.cc:53:32: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/gather.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/gather.cc:183:20: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/gather.cc:38:20: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/index_put.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/index_put.cc:139:22: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/index_put.cc:31:22: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/ternary.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/ternary.cc:2669:20: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/ternary.cc:2672:44: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_put_grad_kernel.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_put_grad_kernel.cc:62:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_put_grad_kernel.cc:68:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/match_matrix_tensor_kernel.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/match_matrix_tensor_kernel.cc:100:21: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/match_matrix_tensor_kernel.cc:99:21: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/funcs/jit/kernel_key.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/funcs/jit/kernel_key.cc:73:31: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/funcs/jit/kernel_key.cc:80:31: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/utils/md5.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/utils/md5.cc:248:24: error: narrowing conversion from 'int' to signed type 'char' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/utils/md5.cc:248:39: error: narrowing conversion from 'int' to signed type 'char' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
