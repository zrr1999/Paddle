# Welford算法在PaddlePaddle方差计算中的应用

## 概述

本次更新将 `paddle/phi/kernels/gpu/var_kernel.cu` 中的方差计算从传统的两遍算法改为使用 Welford 算法（在线算法）。

## 改动内容

### 1. 新增公共函数文件
**文件**: `paddle/phi/kernels/funcs/welford.h`

提供了 Welford 算法的可复用实现，包括：
- `WelfordOnline`: 单个值的在线更新
- `WelfordCombine`: 合并两组统计数据（用于并行归约）
- `WelfordWarpReduce`: warp级别的归约（使用shuffle指令）
- `WelfordBlockReduce`: block级别的归约（使用共享内存）

### 2. 更新方差计算内核
**文件**: `paddle/phi/kernels/gpu/var_kernel.cu`

- 引入 `welford.h` 头文件
- 实现两个 Welford 方差内核：
  - `WelfordVarKernel`: 用于单轴归约
  - `WelfordVarReduceAllKernel`: 用于全局归约（reduce_all）
- 移除了原有的多遍计算逻辑（mean, subtract, square, mean again）

## Welford算法优势

### 1. 数值稳定性
传统方法：`Var(X) = E[X²] - E[X]²` 在数据有大偏移时会导致精度损失。

Welford算法通过增量更新避免了大数相减的问题：
```
count = count + 1
delta = x - mean
mean = mean + delta / count
delta2 = x - mean
M2 = M2 + delta * delta2
variance = M2 / count (或 M2 / (count-1) 用于无偏估计)
```

### 2. 内存效率
- **旧方法**: 需要分配多个中间张量（mean_tensor, diff, squared_diff）
- **新方法**: 仅需三个累加器（mean, m2, count），大大减少内存占用

### 3. 单遍计算
- **旧方法**: 需要多次遍历数据和调用其他kernel（MeanKernel, SubtractKernel, MultiplyKernel）
- **新方法**: 一次遍历即可完成，减少kernel调用开销

## 实现特点

### 支持的功能
- ✅ reduce_all（全局方差）
- ✅ 单轴归约（axis参数为单个维度）
- ✅ biased/unbiased 方差（通过 `unbiased` 参数控制）
- ✅ 数据类型：float, double, float16

### 限制
- 当前实现仅支持单轴归约或reduce_all
- 多轴归约会抛出 `Unimplemented` 错误（可在后续版本中扩展）

## 性能对比

### 内存使用
```
旧方法: O(3 * input_size) - 需要存储 mean_tensor, diff, squared_diff
新方法: O(output_size) - 只需存储输出结果
```

### Kernel调用
```
旧方法: 5次kernel调用 (2次MeanKernel + SubtractKernel + MultiplyKernel + ScaleKernel)
新方法: 1次kernel调用 (WelfordVarKernel 或 WelfordVarReduceAllKernel)
```

## 算法参考
- [Welford's online algorithm - Wikipedia](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
- Chan, Tony F.; Golub, Gene H.; LeVeque, Randall J. (1979). "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances."

## 测试

可使用 `test_welford_var.py` 进行基本功能测试，包括：
- reduce_all 测试
- 单轴归约测试
- biased/unbiased 方差测试
- 数值稳定性测试（大偏移量数据）

运行测试：
```bash
python test_welford_var.py
```

## 未来改进方向

1. **多轴归约支持**: 扩展实现以支持多个维度的同时归约
2. **性能优化**:
   - 使用 warp shuffle 进行更高效的归约
   - 针对不同数据规模优化 block/grid 配置
3. **更多数据类型**: 支持 bfloat16, int32, int64 等
