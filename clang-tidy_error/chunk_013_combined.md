# Clang-Tidy Narrowing Conversion Errors - chunk_013_combined

**Total Errors in this chunk:** 50

**Files in this chunk:** 9

---

## File: `/workspace/Paddle/paddle/cinn/hlir/framework/pir_compiler.cc`

**Number of errors:** 6

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir_compiler.cc:109:15: error: narrowing conversion from 'unsigned long' to signed type 'std::streamsize' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir_compiler.cc:167:19: error: narrowing conversion from 'unsigned long' to signed type 'std::streamsize' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir_compiler.cc:304:54: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir_compiler.cc:305:40: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir_compiler.cc:343:56: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir_compiler.cc:344:42: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/schedule/impl/loop_transformation.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/schedule/impl/loop_transformation.cc:106:22: error: narrowing conversion from 'double' to 'int' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/schedule/impl/loop_transformation.cc:139:18: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/schedule/impl/loop_transformation.cc:249:16: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/schedule/impl/loop_transformation.cc:357:16: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/schedule/impl/loop_transformation.cc:442:22: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/schedule/ir_schedule.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/schedule/ir_schedule.cc:112:11: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/schedule/ir_schedule.cc:196:16: error: narrowing conversion from 'std::set<cinn::ir::Var, cinn::ir::CompVar>::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/schedule/ir_schedule.cc:380:13: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/schedule/ir_schedule.cc:380:29: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/schedule/ir_schedule.cc:409:48: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'std::vector<int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/runtime/cinn_runtime.cc`

**Number of errors:** 6

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/runtime/cinn_runtime.cc:239:24: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/runtime/cinn_runtime.cc:286:10: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/runtime/cinn_runtime.cc:308:10: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int8_t' (aka 'signed char') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/runtime/cinn_runtime.cc:312:10: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int16_t' (aka 'short') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/runtime/cinn_runtime.cc:316:10: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/runtime/cinn_runtime.cc:395:20: error: narrowing conversion from 'uint64_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/serialize_deserialize/src/version_compat.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/serialize_deserialize/src/version_compat.cc:117:16: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/serialize_deserialize/src/version_compat.cc:161:55: error: narrowing conversion from 'std::unordered_map<std::basic_string<char>, int>::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/serialize_deserialize/src/version_compat.cc:161:64: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/serialize_deserialize/src/version_compat.cc:178:55: error: narrowing conversion from 'std::unordered_map<std::basic_string<char>, int>::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/serialize_deserialize/src/version_compat.cc:178:64: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.cc`

**Number of errors:** 6

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.cc:132:20: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::map<int, int>::key_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.cc:132:25: error: narrowing conversion from 'std::map<std::basic_string<char>, unsigned int>::mapped_type' (aka 'unsigned int') to signed type 'std::map<int, int>::mapped_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.cc:136:20: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::map<int, int>::key_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.cc:136:25: error: narrowing conversion from 'std::map<std::basic_string<char>, unsigned int>::mapped_type' (aka 'unsigned int') to signed type 'std::map<int, int>::mapped_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.cc:623:39: error: narrowing conversion from 'int64_t' (aka 'long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/general/common_subexpression_elimination_pass.cc:623:61: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/api/lib/api_gen_utils.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/api/lib/api_gen_utils.cc:842:19: error: narrowing conversion from constant value 1073741824 of type 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/api/lib/api_gen_utils.cc:876:40: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/api/lib/api_gen_utils.cc:911:51: error: narrowing conversion from 'int64_t' (aka 'long') to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/api/lib/api_gen_utils.cc:912:41: error: narrowing conversion from 'int64_t' (aka 'long') to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/api/lib/api_gen_utils.cc:913:41: error: narrowing conversion from 'int64_t' (aka 'long') to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/dim_trans.cc`

**Number of errors:** 6

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/dim_trans.cc:519:25: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'std::unordered_map<int, std::vector<int>>::key_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/dim_trans.cc:519:38: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'std::vector<int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/dim_trans.cc:539:21: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/dim_trans.cc:612:25: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'std::unordered_map<int, std::vector<int>>::key_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/dim_trans.cc:612:38: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'std::vector<int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/dim_trans.cc:641:21: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/add_position_encoding_kernel.cc`

**Number of errors:** 6

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/add_position_encoding_kernel.cc:64:18: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/add_position_encoding_kernel.cc:80:25: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/add_position_encoding_kernel.cc:80:55: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/add_position_encoding_kernel.cc:85:19: error: narrowing conversion from 'int64_t' (aka 'long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/add_position_encoding_kernel.cc:85:61: error: narrowing conversion from 'long' to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/add_position_encoding_kernel.cc:86:19: error: narrowing conversion from 'int64_t' (aka 'long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
