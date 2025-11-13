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

# Test CPU variance kernel
paddle.set_device('cpu')

# Test 1: Simple case
x = paddle.to_tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32')
print("Input tensor:")
print(x)

# Test variance along axis 0
var_axis0 = paddle.var(x, axis=0, unbiased=True)
print("\nVariance along axis 0 (unbiased):")
print(var_axis0)

# Test variance along axis 1
var_axis1 = paddle.var(x, axis=1, unbiased=True)
print("\nVariance along axis 1 (unbiased):")
print(var_axis1)

# Test variance all
var_all = paddle.var(x, unbiased=True)
print("\nVariance all (unbiased):")
print(var_all)

# Compare with numpy
x_np = x.numpy()
print("\n--- Comparison with NumPy ---")
print("NumPy var axis 0:", np.var(x_np, axis=0, ddof=1))
print("NumPy var axis 1:", np.var(x_np, axis=1, ddof=1))
print("NumPy var all:", np.var(x_np, ddof=1))

# Test 2: Edge case with correction
print("\n--- Test with custom correction ---")
var_correction = paddle.var(x, axis=0, unbiased=False, correction=0.5)
print("Variance with correction=0.5:", var_correction)

print("\nTest completed successfully!")
