# Clang-Tidy Narrowing Conversion Errors - chunk_009_combined

**Total Errors in this chunk:** 51

**Files in this chunk:** 5

---

## File: `/workspace/Paddle/paddle/cinn/backends/llvm/codegen_llvm.cc`

**Number of errors:** 10

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_llvm.cc:128:7: error: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_llvm.cc:1471:7: error: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_llvm.cc:1544:22: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_llvm.cc:1545:16: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_llvm.cc:154:7: error: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_llvm.cc:1561:21: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_llvm.cc:170:14: error: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_llvm.cc:182:34: error: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_llvm.cc:184:34: error: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/backends/llvm/codegen_llvm.cc:413:26: error: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc`

**Number of errors:** 11

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc:1247:21: error: narrowing conversion from 'typename __normal_iterator<Argument *, vector<Argument>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc:1260:39: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc:287:20: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc:303:15: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc:348:12: error: narrowing conversion from 'int' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc:430:22: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc:529:49: error: narrowing conversion from 'unsigned long' to signed type '__gnu_cxx::__normal_iterator<cinn::ir::Expr *, std::vector<cinn::ir::Expr>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc:574:18: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc:596:19: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'std::vector<int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc:783:24: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/framework/pir/op_lowering_util.cc:992:15: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc`

**Number of errors:** 10

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc:137:54: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc:155:52: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc:161:52: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc:164:52: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc:189:9: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc:226:17: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc:268:23: error: narrowing conversion from 'uint32_t' (aka 'unsigned int') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc:269:24: error: narrowing conversion from 'uint32_t' (aka 'unsigned int') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc:317:36: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.cc:338:25: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/interpolate_grad_kernel.cc`

**Number of errors:** 10

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_grad_kernel.cc:110:42: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_grad_kernel.cc:110:54: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_grad_kernel.cc:264:40: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_grad_kernel.cc:264:52: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_grad_kernel.cc:277:42: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_grad_kernel.cc:277:54: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_grad_kernel.cc:290:44: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_grad_kernel.cc:290:56: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_grad_kernel.cc:97:40: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/interpolate_grad_kernel.cc:97:52: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/funcs/dense_tensor_iterator.cc`

**Number of errors:** 10

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/funcs/dense_tensor_iterator.cc:160:32: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/funcs/dense_tensor_iterator.cc:166:32: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/funcs/dense_tensor_iterator.cc:171:11: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/funcs/dense_tensor_iterator.cc:275:20: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/funcs/dense_tensor_iterator.cc:276:22: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/funcs/dense_tensor_iterator.cc:277:20: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/funcs/dense_tensor_iterator.cc:278:20: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/funcs/dense_tensor_iterator.cc:324:32: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/funcs/dense_tensor_iterator.cc:342:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/funcs/dense_tensor_iterator.cc:434:15: error: narrowing conversion from 'unsigned long' to signed type '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
