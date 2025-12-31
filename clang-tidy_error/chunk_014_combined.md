# Clang-Tidy Narrowing Conversion Errors - chunk_014_combined

**Total Errors in this chunk:** 53

**Files in this chunk:** 11

---

## File: `/workspace/Paddle/paddle/cinn/hlir/pe/reduction.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/pe/reduction.cc:371:39: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/reduction.cc:387:57: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type '__gnu_cxx::__normal_iterator<const cinn::ir::Expr *, std::vector<cinn::ir::Expr>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/reduction.cc:477:18: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/reduction.cc:490:59: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type '__gnu_cxx::__normal_iterator<const cinn::ir::Expr *, std::vector<cinn::ir::Expr>>::difference_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/pe/reduction.cc:790:22: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/ir_printer.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/ir_printer.cc:177:41: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/ir_printer.cc:288:24: error: narrowing conversion from 'int' to signed type 'char' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/ir_printer.cc:617:22: error: narrowing conversion from 'std::size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/ir_printer.cc:627:22: error: narrowing conversion from 'std::size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/ir_printer.cc:920:24: error: narrowing conversion from 'int' to signed type 'char' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/framework/new_executor/pir_interpreter.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/framework/new_executor/pir_interpreter.cc:1257:19: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/pir_interpreter.cc:1676:24: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/pir_interpreter.cc:1722:18: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/pir_interpreter.cc:1780:18: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/framework/new_executor/pir_interpreter.cc:753:17: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/cinn_op_infer_sym.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/cinn_op_infer_sym.cc:415:25: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/cinn_op_infer_sym.cc:43:25: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/cinn_op_infer_sym.cc:86:24: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/cinn_op_infer_sym.cc:86:31: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/infer_symbolic_shape/cinn_op_infer_sym.cc:97:40: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/op_dialect.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/op_dialect.cc:1129:21: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/op_dialect.cc:1142:23: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/op_dialect.cc:799:13: error: narrowing conversion from 'typename iterator_traits<__normal_iterator<const basic_string<char> *, vector<basic_string<char>>>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/op_dialect.cc:803:13: error: narrowing conversion from 'typename iterator_traits<__normal_iterator<const basic_string<char> *, vector<basic_string<char>>>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/ir/op_dialect.cc:812:15: error: narrowing conversion from 'typename iterator_traits<__normal_iterator<const basic_string<char> *, vector<basic_string<char>>>>::difference_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/onednn/conv2d_transpose_bn_fuse_pass.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/conv2d_transpose_bn_fuse_pass.cc:147:22: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/conv2d_transpose_bn_fuse_pass.cc:288:24: error: narrowing conversion from 'float' to 'bool' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/conv2d_transpose_bn_fuse_pass.cc:340:22: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/conv2d_transpose_bn_fuse_pass.cc:95:24: error: narrowing conversion from 'float' to 'bool' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/transforms/onednn/matmul_activation_fuse_pass.cc`

**Number of errors:** 4

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/matmul_activation_fuse_pass.cc:284:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/matmul_activation_fuse_pass.cc:288:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/matmul_activation_fuse_pass.cc:614:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/transforms/onednn/matmul_activation_fuse_pass.cc:618:18: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/fused_rope.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/fused_rope.cc:116:14: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/fused_rope.cc:32:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/fused_rope.cc:34:29: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/fused_rope.cc:54:14: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/fused_rope.cc:55:27: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/unary.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/unary.cc:1343:18: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/unary.cc:205:26: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/unary.cc:5430:20: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/unary.cc:5466:23: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/unary.cc:5467:23: error: narrowing conversion from 'long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_put_kernel.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_put_kernel.cc:141:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_put_kernel.cc:147:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_put_kernel.cc:64:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_put_kernel.cc:67:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/cpu/index_elementwise_put_kernel.cc:70:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/kernels/fusion/onednn/fused_matmul_kernel.cc`

**Number of errors:** 5

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/kernels/fusion/onednn/fused_matmul_kernel.cc:171:22: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/fusion/onednn/fused_matmul_kernel.cc:234:14: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/fusion/onednn/fused_matmul_kernel.cc:235:14: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/fusion/onednn/fused_matmul_kernel.cc:236:14: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/kernels/fusion/onednn/fused_matmul_kernel.cc:245:28: error: narrowing conversion from 'size_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
