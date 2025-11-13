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

import numpy as np

import paddle

# Test Case 2: input_shape = [3, 3, 9, 6], out_h = 10, out_w = 8, align_corners = True
np.random.seed(0)
input_data = np.random.random((3, 3, 9, 6)).astype(np.float64)

paddle.disable_static()
place = (
    paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
)

# Create input tensor with gradient tracking
input_tensor = paddle.to_tensor(input_data, stop_gradient=False)

# Forward pass
output = paddle.nn.functional.interpolate(
    input_tensor, size=[10, 8], mode='bicubic', align_corners=True
)

# Compute gradient
grad_output = paddle.ones_like(output)
output.backward(grad_output)

# Get analytical gradient
analytical_grad = input_tensor.grad.numpy()


# Compute numerical gradient
def compute_numerical_gradient(input_np, idx, eps=1e-5):
    """Compute numerical gradient at specific index"""
    # Forward with +eps
    input_plus = input_np.copy()
    input_plus.flat[idx] += eps
    input_tensor_plus = paddle.to_tensor(input_plus)
    output_plus = paddle.nn.functional.interpolate(
        input_tensor_plus, size=[10, 8], mode='bicubic', align_corners=True
    )

    # Forward with -eps
    input_minus = input_np.copy()
    input_minus.flat[idx] -= eps
    input_tensor_minus = paddle.to_tensor(input_minus)
    output_minus = paddle.nn.functional.interpolate(
        input_tensor_minus, size=[10, 8], mode='bicubic', align_corners=True
    )

    # Compute gradient
    grad = (output_plus.numpy().sum() - output_minus.numpy().sum()) / (2 * eps)
    return grad


print("Testing numerical gradient vs analytical gradient...")
print("Input shape:", input_data.shape)
print("Output shape:", output.shape)
print()

# Test a few random indices
test_indices = [0, 10, 50, 100, 150]
for idx in test_indices:
    numerical_grad = compute_numerical_gradient(input_data, idx)
    analytical_val = analytical_grad.flat[idx]
    diff = abs(numerical_grad - analytical_val)
    rel_diff = diff / (abs(analytical_val) + 1e-10)

    print(f"Index {idx}:")
    print(f"  Numerical:  {numerical_grad:.10f}")
    print(f"  Analytical: {analytical_val:.10f}")
    print(f"  Abs diff:   {diff:.2e}")
    print(f"  Rel diff:   {rel_diff:.2e}")
    print()

# Find maximum gradient difference
print("Computing full numerical gradient (this may take a while)...")
numerical_grad_full = np.zeros_like(input_data)
for idx in range(min(20, input_data.size)):  # Test first 20 elements
    numerical_grad_full.flat[idx] = compute_numerical_gradient(input_data, idx)

max_diff = np.max(
    np.abs(numerical_grad_full.flat[:20] - analytical_grad.flat[:20])
)
print(f"\nMax absolute difference (first 20 elements): {max_diff:.2e}")
