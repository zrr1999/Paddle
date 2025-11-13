# Numpy参考实现修复说明

## 问题描述

1. 原始的`bilinear_interp_np`函数在处理边界索引时存在问题，可能导致数组越界
2. **参数顺序错误**：函数签名中`scale_w`和`scale_h`的顺序与调用时相反

## 修复内容

### test_bilinear_interp_v2_op.py

#### 1. 修正参数顺序

**问题**：
```python
# 函数签名（修复前）
def bilinear_interp_np(..., scale_w=0, scale_h=0, ...):

# 调用处（setUp函数）
bilinear_interp_np(..., scale_h, scale_w, ...)  # 参数顺序相反！
```

当`scale=[2.0, 0.5]`时：
- setUp传递：`scale_h=2.0, scale_w=0.5`
- 函数接收：`scale_w=2.0, scale_h=0.5`（错位！）
- 导致：`ratio_h = 1.0/0.5 = 2.0`（错误，应该是0.5）
- 导致：`ratio_w = 1.0/2.0 = 0.5`（错误，应该是2.0）

**修复**：
```python
# 函数签名（修复后）
def bilinear_interp_np(..., scale_h=0, scale_w=0, ...):  # 交换顺序
```

现在参数顺序与调用一致，ratio计算正确。

#### 2. 修复边界索引逻辑

对齐CUDA实现`PreCalculatorForLinearInterpInputIndex`：

1. **正确的源位置计算**：先计算src_h和src_w
2. **索引限制**：使用`max(0, min(h, in_h - 1))`限制索引
3. **lambda计算**：基于原始src和clamped索引计算，并限制在[0, 1]范围
4. **安全的偏移**：使用`h_next = min(h + hid, in_h - 1)`确保不越界

```cuda
// CUDA参考实现
*in_img_idx = min(static_cast<size_t>(floorf(src_x)), in_img_x - 1);
*x_id = (*in_img_idx < in_img_x - 1) ? 1 : 0;
MT raw_lambda = static_cast<MT>(src_x) - static_cast<MT>(*in_img_idx);
*lambda1 = static_cast<T>(std::min(std::max(raw_lambda, MT(0)), MT(1)));
```

### test_bicubic_interp_v2_op.py

bicubic的numpy实现已经正确使用了`max(min(...))`模式，无需修改。

## 测试结果

- ✅ **96/96 bilinear测试全部通过**（包括TestBilinearInterpCase7）
- ✅ 所有bicubic测试通过
- ✅ 17个独立抗锯齿测试通过
- ✅ 8个集成抗锯齿测试通过

**总计：121个测试全部通过！**

## 关键修复

### 修复前：
```python
# 参数顺序错误
def bilinear_interp_np(..., scale_w=0, scale_h=0, ...):
    ...
    if scale_h > 0:
        ratio_h = 1.0 / scale_h  # 使用了错误的scale值！
    ...
    h = max(0, h)
    out[:, :, i, j] = ... input[:, :, h + hid, w + wid]  # 可能越界!
```

### 修复后：
```python
# 参数顺序正确
def bilinear_interp_np(..., scale_h=0, scale_w=0, ...):
    ...
    if scale_h > 0:
        ratio_h = 1.0 / scale_h  # 现在使用正确的scale值
    ...
    h = max(0, min(h, in_h - 1))
    h1lambda = max(0.0, min(1.0, src_h - h))
    h_next = min(h + hid, in_h - 1)  # 安全!
    out[:, :, i, j] = ... input[:, :, h_next, w_next]
```

## 根本原因

TestBilinearInterpCase7失败是因为：
1. 使用了不同的高宽缩放比例`scale=[2.0, 0.5]`
2. 参数顺序错误导致scale值交换
3. ratio计算完全相反，导致插值结果错误
