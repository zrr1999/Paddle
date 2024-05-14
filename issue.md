### 一，背景和任务列表
在之前 API 适配 PIR 的时候有⼀些 `test_errors` 的单测，在 PIR 下并没有适配，
因为当时 PIR 下没有`check dtype` 相关的逻辑。
现在我们已经完成了`check dtype`下沉到 cpp API 的⼯作，
所以需要将`test_errors` 的测试也适配⼀下 PIR。

 ⭐️ **提交PR 模版** ⭐️：
+ **// ------- PR 标题  --------**
```python
[PIR] A-1 Adapt mean test_errors
```

本期需要升级的 API 如下：
> 按 merge 的时间顺序，排名不分先后
> 第一批（整体进展：7/30）

| 序号 | Python API | 所在文件| 贡献者 | PR链接 |
|:----|:----|:----|:----|:----:|
|A-1|mean|test/legacy_test/test_mean_op.py|@zrr1999|#60695|
|A-2|sum|test/legacy_test/test_sum_op.py|@zrr1999|#60693|
|A-3|fill_constant|test/legacy_test/test_fill_constant_op.py|@zrr1999|#60695|
|A-4|dropout|test/legacy_test/test_dropout_op.py|||
|A-5|gelu|test/legacy_test/test_gelu_op.py|||
|A-6|cast|test/legacy_test/test_cast_op.py|@zrr1999|#60693|
|A-7|subtract|test/legacy_test/test_subtract_op.py|||
|A-8|erf|test/legacy_test/test_erf_op.py|||
|A-9|greater_equal|test/legacy_test/test_greater_equal_op.py|||
|A-10|reshape|test/legacy_test/test_reshape_op.py|||
|A-11|scale|test/legacy_test/test_scale_op.py|@zrr1999|#60693|
|A-12|concat|test/legacy_test/test_concat_op.py|||
|A-13|expand|test/legacy_test/test_expand_op.py|||
|A-14|matmul|test/legacy_test/test_matmul_op.py|||
|A-15|split|test/legacy_test/test_split_op.py|||
|A-16|add_n|test/legacy_test/test_add_n_op.py|||
|A-17|transpose|test/legacy_test/test_transpose_op.py|||
|A-18|softmax|test/legacy_test/test_softmax_op.py|||
|A-19|rms_norm|test/legacy_test/test_rms_norm_op.py|||
|A-20|full_like|test/legacy_test/test_full_like_op.py|||
|A-21|gather_nd|test/legacy_test/test_gather_nd_op.py|||
|A-22|shape|test/legacy_test/test_shape_op.py|||
|A-23|numel|test/legacy_test/test_numel_op.py|||
|A-24|slice|test/legacy_test/test_slice_op.py|||
|A-25|stack|test/legacy_test/test_stack_op.py|||
|A-26|where|test/legacy_test/test_where_op.py|@zrr1999|#60693|
|A-27|fill_any_like|test/legacy_test/test_fill_any_like_op.py|||
|A-28|pad|test/legacy_test/test_pad_op.py|||
|A-29|tile|test/legacy_test/test_tile_op.py|||
|A-30|assign|test/legacy_test/test_assign_op.py|@zrr1999|#60693|

> 第二批（整体进展：0/8）

| 序号 | Python API | 所在文件| 贡献者 | PR链接 |
|:----|:----|:----|:----|:----:|
|B-1|div|test/legacy_test/test_elementwise_div_op.py|||
|B-2|add|test/legacy_test/test_elementwise_add_op.py|||
|B-3|pow|test/legacy_test/test_elementwise_pow_op.py|||
|B-4|mul|test/legacy_test/test_elementwise_mul_op.py|||
|B-5|silu|test/legacy_test/test_activation_op.py|||
|B-6|exp|test/legacy_test/test_activation_op.py|||
|B-7|tanh|test/legacy_test/test_activation_op.py|||
|B-8|rsqrt|test/legacy_test/test_activation_op.py|||

> 第三批（整体进展：1/9）

| 序号 | Python API | 所在文件| 贡献者 | PR链接 |
|:----|:----|:----|:----|:----:|
|C-1|reduce_max|test/legacy_test/test_reduce_max_op.py|||
|C-2|squeeze |test/legacy_test/test_squeeze_squeeze2_op.py|||
|C-3|unsqueeze |test/legacy_test/test_unsqueeze_unsqueeze2_op.py|||
|C-4|uniform|test/legacy_test/test_uniform_random_op.py|@zrr1999|#60693|
|C-5|embedding|test/legacy_test/test_lookup_table_v2_op.py|||
|C-6|bitwise_and|test/legacy_test/test_bitwise_op.py|||
|C-7|arange|test/legacy_test/test_arange.py|||
|C-8|equal|test/legacy_test/test_compare_op.py|||
|C-9|tril|test/legacy_test/test_tril_triu_op.py|||

### 二，任务详情
可参考 [PR](https://github.com/PaddlePaddle/Paddle/pull/60695)
需要对相关单测进行修改合并，以mean为例，测试改api的单测在文件`test_mean_op.py`中，在单测函数中，将`with program_guard(Program(), Program())`
与 `with paddle.pir_utils.IrGuard(), program_guard(Program(), Program())` 合并，并添加`test_with_pir_api`装饰器。
```python
@test_with_pir_api
def test_errors(self):
    with program_guard(Program(), Program()):
        # The input type of mean_op must be Variable.
        input1 = 12
        self.assertRaises(TypeError, paddle.mean, input1)
                   # The input dtype of mean_op must be float16, float32, float64.
            input2 = paddle.static.data(
                name='input2', shape=[-1, 12, 10], dtype="int32"
            )
            self.assertRaises(TypeError, paddle.mean, input2)
            input3 = paddle.static.data(
                name='input3', shape=[-1, 4], dtype="float16"
            )
            paddle.nn.functional.softmax(input3)

    with paddle.pir_utils.IrGuard(), program_guard(Program(), Program()):
        input1 = 12
        self.assertRaises(TypeError, paddle.mean, input1)

        input2 = paddle.static.data(
            name='input2', shape=[2, 3, 4, 5], dtype="int32"
        )

        out = paddle.mean(input2)

        exe = paddle.static.Executor(self.place)
        res = exe.run(feed={'input2': self.x}, fetch_list=[out])

```

```python
@test_with_pir_api
def test_errors(self):
    with program_guard(Program(), Program()):
        # The input type of mean_op must be Variable.
        input1 = 12
        self.assertRaises(TypeError, paddle.mean, input1)
        if not in_pir_mode():
            # The input dtype of mean_op must be float16, float32, float64.
            input2 = paddle.static.data(
                name='input2', shape=[-1, 12, 10], dtype="int32"
            )
            self.assertRaises(TypeError, paddle.mean, input2)
            input3 = paddle.static.data(
                name='input3', shape=[-1, 4], dtype="float16"
            )
            paddle.nn.functional.softmax(input3)
        else:
            input2 = paddle.static.data(
                name='input2', shape=[2, 3, 4, 5], dtype="int32"
            )

            out = paddle.mean(input2)

            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'input2': self.x}, fetch_list=[out])
```

#### 三、参考PR
https://github.com/PaddlePaddle/Paddle/pull/60693
https://github.com/PaddlePaddle/Paddle/pull/60695
