# 任务目标
API 单测 test_errors 相关 case 适配 PIR

目前 test_errors 函数总共有47个，分布在 30 个不同文件中。

其中有 11 个（分布在7个文件中）已经添加了 `@test_with_pir_api`。

其余部分可以使用 ast-grep 统一添加 `@test_with_pir_api`
```yaml
id: test_errors
message: Add a decorator to test_errors functions.
severity: info
language: Python
rule:
  kind: function_definition
  pattern: $F
  has:
    kind: identifier
    regex: ^test_errors
  follows:
    not:
      kind: decorator
      has:
        kind: identifier
        regex: ^test_with_pir_api

fix: |
  @test_with_pir_api
  $F
```

# 推进方案
1. 修复报错类型问题。（[PR](https://github.com/PaddlePaddle/Paddle/pull/60487)）
2. 汇总全部添加 `@test_with_pir_api` 的单测无法通过遇到的问题。（[PR](https://github.com/PaddlePaddle/Paddle/pull/60488)）
3. 对于已经适配过 pir 的 `test_errors` 单测，先直接修改为与静态图单测统一的格式。
4. 对于未适配过 pir 的 `test_errors` 单测（或已经统一的单测），添加 `@test_with_pir_api`（仅替换测试可以直接通过的部分）。
5. 针对一些遇到特殊问题的单测，逐步修改 `test_errors` 单测，使其通过 PIR 单测。

# 已知问题
## 报错类型不统一
一些 `[A-z<>:0-9]+\sCastPyArg2[A-z0-9]+\(` 函数 platform::errors::InvalidArgument 需要改为 platform::errors::InvalidType

`paddle/fluid/pybind/eager_utils.cc` 34处
`paddle/fluid/pybind/op_function_common.cc` 33 处

这里可以使用 ast-grep 替换
```yaml
id: cast-pyarg
message: Find a function definition used to cast a Python argument.
severity: info
language: C++
rule:
  kind: qualified_identifier
  regex: ^(platform::errors::InvalidArgument)
  inside:
    kind: function_definition
    has:
      kind: function_declarator
      regex: 'CastPyArg2[A-z0-9]+'
    stopBy:
      kind: function_declarator

fix:
  platform::errors::InvalidType
```

## 类型支持情况不统一
部分静态图的类型检查有滞后（或支持情况不一致），跟实际 PIR 支持的类型不一致。

## 部分出现段错误


# 第一期算子文件名备忘录

文件名一致：
test_mean_op.py test_sum_op.py test_fill_constant_op.py test_dropout_op.py test_gelu_op.py test_cast_op.py test_subtract_op.py test_erf_op.py test_greater_equal_op.py test_reshape_op.py test_scale_op.py test_concat_op.py test_expand_op.py test_matmul_op.py test_split_op.py test_add_n_op.py test_transpose_op.py test_softmax_op.py test_rms_norm_op.py test_full_like_op.py test_gather_nd_op.py test_shape_op.py test_numel_op.py test_slice_op.py test_stack_op.py test_where_op.py test_fill_any_like_op.py test_pad_op.py test_tile_op.py test_assign_op.py

elementwise:
test_elementwise_div_op.py test_elementwise_add_op.py test_elementwise_pow_op.py test_elementwise_mul_op.py

test_activation_op.py:
silu_op.py exp_op.py tanh_op.py rsqrt

test_reduce_op.py:
reduce_max_op.py

to：
test_squeeze2_op.py test_unsqueeze2_op.py

test_uniform_random_op.py:
uniform_op

其他：
embedding: test_lookup_table_v2_op.py
bitwise_and: test_bitwise_op.py
arange: test_arange.py
equal: test_compare_op.py

tril:test_tril_triu_op.py

所有文件：
test_mean_op.py test_sum_op.py test_fill_constant_op.py test_dropout_op.py test_gelu_op.py test_cast_op.py test_subtract_op.py test_erf_op.py test_greater_equal_op.py test_reshape_op.py test_scale_op.py test_concat_op.py test_expand_op.py test_matmul_op.py test_split_op.py test_add_n_op.py test_transpose_op.py test_softmax_op.py test_rms_norm_op.py test_full_like_op.py test_gather_nd_op.py test_shape_op.py test_numel_op.py test_slice_op.py test_stack_op.py test_where_op.py test_fill_any_like_op.py test_pad_op.py test_tile_op.py test_assign_op.py
test_elementwise_div_op.py test_elementwise_add_op.py test_elementwise_pow_op.py test_elementwise_mul_op.py
test_activation_op.py
test_reduce_op.py
test_squeeze2_op.py test_unsqueeze2_op.py
test_uniform_random_op.py

test failed test_activation_op.py
test failed test_reduce_op.py
test failed test_squeeze2_op.py
test failed test_lookup_table_v2_op.py
test failed test_full_like_op.py
test failed test_gather_nd_op.py
test failed test_matmul_op.py
test failed test_split_op.py
test failed test_transpose_op.py
test failed test_softmax_op.py
test failed test_mean_op.py
test failed test_fill_constant_op.py
test failed test_dropout_op.py
test failed test_concat_op.py
test failed test_stack_op.py
test failed test_pad_op.py

git add test_elementwise_div_op.py
