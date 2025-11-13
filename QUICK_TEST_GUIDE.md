# 插值抗锯齿功能 - 快速测试指南

## 快速验证

### 1. 编译
```bash
cd /workspace/paddle/build
ninja -j128 phi_gpu
```

### 2. 运行单元测试
```bash
cd /workspace/paddle

# 独立的抗锯齿测试
python test/legacy_test/test_interpolate_antialias.py

# 双线性插值抗锯齿测试
python test/legacy_test/test_bilinear_interp_v2_op.py TestBilinearInterpAntiAlias

# 双三次插值抗锯齿测试
python test/legacy_test/test_bicubic_interp_v2_op.py TestBicubicInterpAntiAlias
```

预期输出:
```
# test_interpolate_antialias.py
Ran 17 tests in 0.437s
OK

# test_bilinear_interp_v2_op.py (AA tests)
Ran 4 tests in 0.217s
OK

# test_bicubic_interp_v2_op.py (AA tests)
Ran 4 tests in 0.245s
OK
```

### 3. 快速功能测试
```python
import paddle

paddle.device.set_device('gpu:0')
x = paddle.randn([2, 3, 64, 64])

# 双线性抗锯齿
out = paddle.nn.functional.interpolate(
    x, size=(32, 32), mode='bilinear', antialias=True
)
print(f"Bilinear AA: {x.shape} -> {out.shape}")

# 双三次抗锯齿
out = paddle.nn.functional.interpolate(
    x, size=(32, 32), mode='bicubic', antialias=True
)
print(f"Bicubic AA: {x.shape} -> {out.shape}")
```

## 测试覆盖

### 独立测试 (test_interpolate_antialias.py)
- ✓ 双线性插值抗锯齿 (7个测试)
- ✓ 双三次插值抗锯齿 (4个测试)
- ✓ 边界情况 (4个测试)
- ✓ 一致性测试 (1个测试)
- ✓ 数据布局测试 (1个测试)

### 集成测试
- ✓ test_bilinear_interp_v2_op.py (4个AA测试)
- ✓ test_bicubic_interp_v2_op.py (4个AA测试)

**总计: 25个测试用例**

## 支持的特性

| 特性 | 支持情况 |
|------|----------|
| 双线性插值 | ✓ |
| 双三次插值 | ✓ |
| float32 | ✓ |
| float16 | ✓ |
| bfloat16 | ✓ |
| NCHW布局 | ✓ |
| NHWC布局 | ✗ |
| align_corners | ✓ |
| scale_factor | ✓ |

## 相关文件

- **实现**: `paddle/phi/kernels/gpu/interpolate_kernel.cu`
- **独立测试**: `test/legacy_test/test_interpolate_antialias.py`
- **集成测试**:
  - `test/legacy_test/test_bilinear_interp_v2_op.py` (含AA测试)
  - `test/legacy_test/test_bicubic_interp_v2_op.py` (含AA测试)
- **文档**: `INTERPOLATE_AA_IMPLEMENTATION.md`
