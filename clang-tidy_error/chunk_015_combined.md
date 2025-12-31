# Clang-Tidy Narrowing Conversion Errors - chunk_015_combined

**Total Errors in this chunk:** 52

**Files in this chunk:** 13

---

## File: `/workspace/Paddle/paddle/cinn/ast_gen_ius/ast_gen.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ast_gen_ius/ast_gen.cc:191:63: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ast_gen_ius/ast_gen.cc:223:70: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ast_gen_ius/ast_gen.cc:235:19: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ast_gen_ius/ast_gen.cc:308:63: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/fuse_parallel_matmul_pass.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/fuse_parallel_matmul_pass.cc:117:28: error: narrowing conversion from 'typename iterator_traits<PointerListIterator<Operation>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/fuse_parallel_matmul_pass.cc:119:28: error: narrowing conversion from 'typename iterator_traits<PointerListIterator<Operation>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/fuse_parallel_matmul_pass.cc:271:28: error: narrowing conversion from 'typename iterator_traits<PointerListIterator<Operation>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/fuse_parallel_matmul_pass.cc:273:28: error: narrowing conversion from 'typename iterator_traits<PointerListIterator<Operation>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/pe/elementwise.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/pe/elementwise.cc:139:17: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/elementwise.cc:182:56: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/elementwise.cc:232:18: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/elementwise.cc:233:18: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/pe/map_expr_to_ir.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/pe/map_expr_to_ir.cc:736:18: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/map_expr_to_ir.cc:914:56: error: narrowing conversion from 'std::size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/map_expr_to_ir.cc:930:44: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/map_expr_to_ir.cc:931:44: error: narrowing conversion from 'std::int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/pe/nn.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/pe/nn.cc:1120:12: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/nn.cc:1171:12: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/nn.cc:1361:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/nn.cc:1362:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/data_set.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/data_set.cc:2121:29: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/data_set.cc:409:26: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/data_set.cc:413:24: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/data_set.cc:450:29: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/ir/onednn/cpu_quantize_pass.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/ir/onednn/cpu_quantize_pass.cc:100:52: error: narrowing conversion from 'unsigned int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/ir/onednn/cpu_quantize_pass.cc:180:49: error: narrowing conversion from 'unsigned int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/ir/onednn/cpu_quantize_pass.cc:255:52: error: narrowing conversion from 'unsigned int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/ir/onednn/cpu_quantize_pass.cc:304:52: error: narrowing conversion from 'unsigned int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/ir/onednn/shuffle_channel_onednn_detect_pass.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/ir/onednn/shuffle_channel_onednn_detect_pass.cc:126:31: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type '__gnu_cxx::__alloc_traits<std::allocator<int>, int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/ir/onednn/shuffle_channel_onednn_detect_pass.cc:131:31: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type '__gnu_cxx::__alloc_traits<std::allocator<int>, int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/ir/onednn/shuffle_channel_onednn_detect_pass.cc:143:15: error: narrowing conversion from 'long' to signed type '__gnu_cxx::__alloc_traits<std::allocator<int>, int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/ir/onednn/shuffle_channel_onednn_detect_pass.cc:163:15: error: narrowing conversion from 'long' to signed type '__gnu_cxx::__alloc_traits<std::allocator<int>, int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/general/identity_op_clean_pass.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/identity_op_clean_pass.cc:101:26: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/identity_op_clean_pass.cc:108:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/identity_op_clean_pass.cc:94:26: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/identity_op_clean_pass.cc:98:26: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/core/platform/profiler/cpu_utilization.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/core/platform/profiler/cpu_utilization.cc:140:21: error: narrowing conversion from 'unsigned long' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/platform/profiler/cpu_utilization.cc:145:21: error: narrowing conversion from 'unsigned long' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/platform/profiler/cpu_utilization.cc:174:7: error: narrowing conversion from 'long' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/core/platform/profiler/cpu_utilization.cc:177:43: error: narrowing conversion from 'long' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/expand_as.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/expand_as.cc:103:17: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/expand_as.cc:104:17: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/expand_as.cc:51:19: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/expand_as.cc:52:16: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/pad.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/pad.cc:34:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/pad.cc:47:33: error: narrowing conversion from 'unsigned long' to signed type 'std::vector<long>::value_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/pad.cc:67:18: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/pad.cc:81:33: error: narrowing conversion from 'unsigned long' to signed type 'std::vector<long>::value_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/squeeze.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/squeeze.cc:100:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/squeeze.cc:171:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/squeeze.cc:173:18: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/squeeze.cc:231:27: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
