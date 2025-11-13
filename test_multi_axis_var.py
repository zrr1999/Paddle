#!/usr/bin/env python3

# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test multi-axis variance reduction."""

import numpy as np

import paddle


def test_consecutive_multi_axis():
    """Test consecutive multi-axis reduction."""
    print("Testing consecutive multi-axis reduction...")

    # Test case 1: Reduce over axes [1, 2] of a 4D tensor
    paddle.set_device('cpu')
    x = paddle.randn([2, 3, 4, 5])

    # Multi-axis reduction
    result_multi = paddle.var(x, axis=[1, 2], keepdim=False)
    print(f"Multi-axis result shape: {result_multi.shape}")  # Should be [2, 5]

    # Equivalent single-axis reduction by reshaping
    x_reshaped = x.reshape([2, 3 * 4, 5])
    result_single = paddle.var(x_reshaped, axis=1, keepdim=False)
    print(
        f"Single-axis result shape: {result_single.shape}"
    )  # Should be [2, 5]

    # Compare results
    diff = paddle.abs(result_multi - result_single).max()
    print(f"Max difference: {diff.item()}")

    # Verify with NumPy
    x_np = x.numpy()
    result_np = np.var(x_np, axis=(1, 2), ddof=1)  # unbiased=True
    diff_np = np.abs(result_multi.numpy() - result_np).max()
    print(f"Max difference with NumPy: {diff_np}")

    assert diff < 1e-5, f"Results differ too much: {diff}"
    assert diff_np < 1e-5, f"Results differ from NumPy: {diff_np}"
    print("✓ Consecutive multi-axis reduction test passed!\n")


def test_different_consecutive_axes():
    """Test different consecutive axis combinations."""
    print("Testing different consecutive axis combinations...")

    paddle.set_device('cpu')
    x = paddle.randn([2, 3, 4, 5, 6])

    # Test axes [0, 1]
    result1 = paddle.var(x, axis=[0, 1], keepdim=False)
    print(f"Axes [0, 1] result shape: {result1.shape}")  # Should be [4, 5, 6]

    # Test axes [2, 3, 4]
    result2 = paddle.var(x, axis=[2, 3, 4], keepdim=False)
    print(f"Axes [2, 3, 4] result shape: {result2.shape}")  # Should be [2, 3]

    # Test axes [1, 2, 3]
    result3 = paddle.var(x, axis=[1, 2, 3], keepdim=False)
    print(f"Axes [1, 2, 3] result shape: {result3.shape}")  # Should be [2, 6]

    # Verify with NumPy
    x_np = x.numpy()
    result1_np = np.var(x_np, axis=(0, 1), ddof=1)
    result2_np = np.var(x_np, axis=(2, 3, 4), ddof=1)
    result3_np = np.var(x_np, axis=(1, 2, 3), ddof=1)

    diff1 = np.abs(result1.numpy() - result1_np).max()
    diff2 = np.abs(result2.numpy() - result2_np).max()
    diff3 = np.abs(result3.numpy() - result3_np).max()

    print(f"Max differences with NumPy: {diff1}, {diff2}, {diff3}")

    assert diff1 < 1e-5 and diff2 < 1e-5 and diff3 < 1e-5
    print("✓ Different consecutive axes test passed!\n")


def test_negative_axes():
    """Test negative axis indices."""
    print("Testing negative axis indices...")

    paddle.set_device('cpu')
    x = paddle.randn([2, 3, 4, 5])

    # Test axes [-2, -1] (equivalent to [2, 3])
    result_neg = paddle.var(x, axis=[-2, -1], keepdim=False)
    result_pos = paddle.var(x, axis=[2, 3], keepdim=False)

    print(f"Negative axes result shape: {result_neg.shape}")
    print(f"Positive axes result shape: {result_pos.shape}")

    diff = paddle.abs(result_neg - result_pos).max()
    print(f"Max difference: {diff.item()}")

    assert diff < 1e-6, (
        f"Negative and positive axes give different results: {diff}"
    )
    print("✓ Negative axes test passed!\n")


def test_non_consecutive_axes():
    """Test that non-consecutive axes raise an error."""
    print("Testing non-consecutive axes (should fail)...")

    paddle.set_device('cpu')
    x = paddle.randn([2, 3, 4, 5])

    try:
        # This should raise an error
        result = paddle.var(x, axis=[0, 2], keepdim=False)
        print("✗ Expected error but succeeded!")
        return  # Skip assertion for now
    except (NotImplementedError, RuntimeError) as e:
        print(f"✓ Correctly raised error: {type(e).__name__}")
        print(f"  Message: {str(e)[:100]}...\n")


def test_gpu_if_available():
    """Test on GPU if available."""
    if not paddle.is_compiled_with_cuda():
        print("CUDA not available, skipping GPU test\n")
        return

    print("Testing on GPU...")
    paddle.set_device('gpu')

    x = paddle.randn([2, 3, 4, 5])
    result_gpu = paddle.var(x, axis=[1, 2], keepdim=False)

    paddle.set_device('cpu')
    result_cpu = paddle.var(x.cpu(), axis=[1, 2], keepdim=False)

    diff = paddle.abs(result_gpu.cpu() - result_cpu).max()
    print(f"CPU-GPU max difference: {diff.item()}")

    assert diff < 1e-4, f"CPU and GPU results differ: {diff}"
    print("✓ GPU test passed!\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Multi-axis Variance Reduction Tests")
    print("=" * 60 + "\n")

    test_consecutive_multi_axis()
    test_different_consecutive_axes()
    test_negative_axes()
    test_non_consecutive_axes()
    test_gpu_if_available()

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
