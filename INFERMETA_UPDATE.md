# VarInferMeta 和 StdInferMeta 实现说明

## 背景

由于 `var` 和 `std` 操作的参数与标准的 reduce 操作不同，需要为它们创建独立的 InferMeta 函数，而不是复用 `ReduceIntArrayAxisInferMeta`。

## 参数差异

### ReduceIntArrayAxisInferMeta
```cpp
void ReduceIntArrayAxisInferMeta(const MetaTensor& x,
                                 const IntArray& axis,
                                 bool keep_dim,    // 注意参数名
                                 MetaTensor* out);
```

### VarInferMeta
```cpp
void VarInferMeta(const MetaTensor& x,
                  const IntArray& axis,
                  bool keepdim,        // 参数名不同
                  bool unbiased,       // 额外参数
                  double correction,   // 额外参数
                  MetaTensor* out);
```

### StdInferMeta
```cpp
void StdInferMeta(const MetaTensor& x,
                  const IntArray& axis,
                  bool keepdim,        // 参数名不同
                  bool unbiased,       // 额外参数
                  MetaTensor* out);
```

## 修改的文件

### 1. paddle/phi/infermeta/unary.h

添加函数声明：

```cpp
PADDLE_API void VarInferMeta(const MetaTensor& x,
                             const IntArray& axis,
                             bool keepdim,
                             bool unbiased,
                             double correction,
                             MetaTensor* out,
                             MetaConfig config = MetaConfig());

PADDLE_API void StdInferMeta(const MetaTensor& x,
                             const IntArray& axis,
                             bool keepdim,
                             bool unbiased,
                             MetaTensor* out,
                             MetaConfig config = MetaConfig());
```

**位置**: 在 `StrictReduceIntArrayAxisInferMeta` 之后

### 2. paddle/phi/infermeta/unary.cc

添加函数实现：

```cpp
void VarInferMeta(const MetaTensor& x,
                  const IntArray& axis,
                  bool keepdim,
                  bool unbiased,
                  double correction,
                  MetaTensor* out,
                  MetaConfig config) {
  bool reduce_all = false;
  if (axis.size() == 0) {
    reduce_all = true;
  }
  ReduceIntArrayAxisInferMetaBase(x, axis, keepdim, reduce_all, out, config);
}

void StdInferMeta(const MetaTensor& x,
                  const IntArray& axis,
                  bool keepdim,
                  bool unbiased,
                  MetaTensor* out,
                  MetaConfig config) {
  bool reduce_all = false;
  if (axis.size() == 0) {
    reduce_all = true;
  }
  ReduceIntArrayAxisInferMetaBase(x, axis, keepdim, reduce_all, out, config);
}
```

**位置**: 在 `StrictReduceIntArrayAxisInferMeta` 之后

## 实现要点

### 1. 复用基础逻辑

虽然参数不同，但输出形状的推断逻辑是相同的，因此仍然调用 `ReduceIntArrayAxisInferMetaBase`：

```cpp
ReduceIntArrayAxisInferMetaBase(x, axis, keepdim, reduce_all, out, config);
```

### 2. 处理 reduce_all

```cpp
bool reduce_all = false;
if (axis.size() == 0) {
  reduce_all = true;
}
```

当 `axis` 为空时，表示对所有维度进行归约。

### 3. 忽略统计参数

InferMeta 只负责推断输出的形状和类型，不需要关心 `unbiased` 和 `correction` 参数的具体值，这些参数在 Kernel 中处理。

## ops.yaml 配置

ops.yaml 中已经正确配置：

```yaml
- op : var
  args : (Tensor x, IntArray axis={}, bool keepdim=false, bool unbiased=true, double correction=1)
  output : Tensor(out)
  infer_meta :
    func : VarInferMeta
  kernel :
    func : var

- op : std
  args : (Tensor x, IntArray axis={}, bool keepdim=false, bool unbiased=true)
  output : Tensor(out)
  infer_meta :
    func : StdInferMeta
  kernel :
    func : std
```

## 验证清单

- [x] 在 unary.h 中声明 VarInferMeta
- [x] 在 unary.h 中声明 StdInferMeta
- [x] 在 unary.cc 中实现 VarInferMeta
- [x] 在 unary.cc 中实现 StdInferMeta
- [x] 参数名与 kernel 保持一致 (keepdim)
- [x] 参数顺序与 kernel 保持一致
- [x] 正确处理 reduce_all 情况
- [x] 复用 ReduceIntArrayAxisInferMetaBase
- [x] ops.yaml 配置正确

## 编译验证

编译后应该能够正确生成以下代码：

1. C++ API 函数签名匹配
2. Python API 函数签名匹配
3. InferMeta 函数被正确调用
4. 输出形状推断正确

## 与旧实现的区别

### 旧方式（错误）
可能直接使用 `ReduceIntArrayAxisInferMeta`，但参数名不匹配：
- Kernel 使用 `keepdim`
- InferMeta 使用 `keep_dim`
- 导致编译错误或参数不匹配

### 新方式（正确）
创建专用的 InferMeta 函数：
- 参数名与 Kernel 完全一致
- 包含所有必要的参数（即使 InferMeta 不使用）
- API 生成工具能够正确匹配和生成代码

## 总结

通过创建独立的 `VarInferMeta` 和 `StdInferMeta`，我们确保了：

1. **参数一致性**: 参数名和顺序与 Kernel 完全一致
2. **类型安全**: 编译时检查参数类型
3. **代码清晰**: 每个操作有自己的 InferMeta 函数
4. **易于维护**: 未来修改参数时只需修改一处

这是 PaddlePaddle Phi 算子开发的标准做法。
