# Clang-Tidy Narrowing Conversion Errors - chunk_018_combined

**Total Errors in this chunk:** 51

**Files in this chunk:** 19

---

## File: `/workspace/Paddle/paddle/cinn/backends/nvrtc/nvrtc_util.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/backends/nvrtc/nvrtc_util.cc:210:33: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/nvrtc/nvrtc_util.cc:228:35: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/common/integer_set.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/common/integer_set.cc:45:9: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/common/integer_set.cc:46:9: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.cc:56:58: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.cc:64:58: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/add_broadcast_to_elementwise_pass.cc:76:17: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.cc:274:15: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.cc:283:15: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/split_generate_shape_into_shape_ops_pass.cc:377:38: error: narrowing conversion from 'typename iterator_traits<__normal_iterator<const Value *, vector<Value>>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/framework/pir/compilation_task.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/compilation_task.cc:130:26: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/compilation_task.cc:141:29: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/compilation_task.cc:188:44: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/group_schedule/search/measurer.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/search/measurer.cc:76:16: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/search/measurer.cc:94:40: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/new_executor/program_interpreter.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/new_executor/program_interpreter.cc:1080:50: error: narrowing conversion from 'uint64_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/program_interpreter.cc:1292:18: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/program_interpreter.cc:1486:19: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/general/matmul_transpose_fuse_pass.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/matmul_transpose_fuse_pass.cc:152:29: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/matmul_transpose_fuse_pass.cc:47:29: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/matmul_transpose_fuse_pass.cc:99:29: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/general/transfer_layout_pass.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/transfer_layout_pass.cc:254:27: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/transfer_layout_pass.cc:579:19: error: narrowing conversion from 'float' to 'bool' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/transfer_layout_pass.cc:708:21: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::unordered_map<char, int>::mapped_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/gpu/horizontal_fuse_pass.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/gpu/horizontal_fuse_pass.cc:239:50: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/gpu/horizontal_fuse_pass.cc:239:70: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/gpu/horizontal_fuse_pass.cc:307:41: error: narrowing conversion from 'uint32_t' (aka 'unsigned int') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/sub_graph_detector.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/sub_graph_detector.cc:489:27: error: narrowing conversion from 'std::set<std::shared_ptr<pir::SubGraph>, pir::SubGraph::CompareById>::size_type' (aka 'unsigned long') to signed type 'std::unordered_map<std::shared_ptr<pir::SubGraph>, int>::mapped_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/sub_graph_detector.cc:800:53: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type '__gnu_cxx::__normal_iterator<std::shared_ptr<pir::SubGraph> *, std::vector<std::shared_ptr<pir::SubGraph>>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/backends/device_base.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/backends/device_base.cc:271:24: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/backends/device_base.cc:271:7: error: narrowing conversion from 'double' to 'size_t' (aka 'unsigned long') [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/backends/device_base.cc:272:24: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/memory/allocation/system_allocator.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/memory/allocation/system_allocator.cc:216:41: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/memory/allocation/system_allocator.cc:218:39: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/transpose.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/transpose.cc:209:39: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/transpose.cc:209:63: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/transpose.cc:37:18: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type '__gnu_cxx::__alloc_traits<std::allocator<int>, int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/triu.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/triu.cc:121:18: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/triu.cc:27:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/triu.cc:74:18: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_get_kernel.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_get_kernel.cc:51:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_get_kernel.cc:54:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_get_kernel.cc:57:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/qr_kernel.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/qr_kernel.cc:45:42: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/qr_kernel.cc:45:62: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/qr_kernel.cc:48:37: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/yolo_box_kernel.cc`

**Number of errors:** 3

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/yolo_box_kernel.cc:101:25: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/yolo_box_kernel.cc:127:27: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/yolo_box_kernel.cc:87:25: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/selected_rows/cpu/nce_grad_kernel.cc`

**Number of errors:** 2

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/selected_rows/cpu/nce_grad_kernel.cc:151:61: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/selected_rows/cpu/nce_grad_kernel.cc:73:22: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
