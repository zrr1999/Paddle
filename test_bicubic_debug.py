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


def cubic_1(x, a):
    return ((a + 2) * x - (a + 3)) * x * x + 1


def cubic_2(x, a):
    return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a


def cubic_interp1d(x0, x1, x2, x3, t):
    param = [0, 0, 0, 0]
    a = -0.75
    x_1 = t
    x_2 = 1.0 - t
    param[0] = cubic_2(x_1 + 1.0, a)
    param[1] = cubic_1(x_1, a)
    param[2] = cubic_1(x_2, a)
    param[3] = cubic_2(x_2 + 1.0, a)
    return x0 * param[0] + x1 * param[1] + x2 * param[2] + x3 * param[3]


def bicubic_interp_np(input, out_h, out_w, align_corners=True):
    """Original NumPy implementation from test"""
    batch_size, channel, in_h, in_w = input.shape

    ratio_h = 0.0
    if align_corners:
        if out_h > 1:
            ratio_h = (in_h - 1.0) / (out_h - 1.0)
    else:
        ratio_h = 1.0 * in_h / out_h

    ratio_w = 0.0
    if align_corners:
        if out_w > 1:
            ratio_w = (in_w - 1.0) / (out_w - 1.0)
    else:
        ratio_w = 1.0 * in_w / out_w

    out = np.zeros((batch_size, channel, out_h, out_w))

    for k in range(out_h):
        if align_corners:
            h = ratio_h * k
        else:
            h = ratio_h * (k + 0.5) - 0.5
        input_y = np.floor(h)
        y_t = h - input_y
        for l in range(out_w):
            if align_corners:
                w = ratio_w * l
            else:
                w = ratio_w * (l + 0.5) - 0.5
            input_x = np.floor(w)
            x_t = w - input_x
            for i in range(batch_size):
                for j in range(channel):
                    coefficients = [0, 0, 0, 0]
                    for ii in range(4):
                        access_x_0 = int(max(min(input_x - 1, in_w - 1), 0))
                        access_x_1 = int(max(min(input_x + 0, in_w - 1), 0))
                        access_x_2 = int(max(min(input_x + 1, in_w - 1), 0))
                        access_x_3 = int(max(min(input_x + 2, in_w - 1), 0))
                        access_y = int(max(min(input_y - 1 + ii, in_h - 1), 0))

                        coefficients[ii] = cubic_interp1d(
                            input[i, j, access_y, access_x_0],
                            input[i, j, access_y, access_x_1],
                            input[i, j, access_y, access_x_2],
                            input[i, j, access_y, access_x_3],
                            x_t,
                        )
                    out[i, j, k, l] = cubic_interp1d(
                        coefficients[0],
                        coefficients[1],
                        coefficients[2],
                        coefficients[3],
                        y_t,
                    )
    return out.astype(input.dtype)


# Test Case 2: input_shape = [3, 3, 9, 6], out_h = 10, out_w = 8, align_corners = True
np.random.seed(0)
input_data = np.random.random((3, 3, 9, 6)).astype(np.float64)

# NumPy implementation
output_np = bicubic_interp_np(input_data, 10, 8, align_corners=True)

# Paddle implementation
paddle.enable_static()
from paddle import base

place = (
    base.CUDAPlace(0) if base.core.is_compiled_with_cuda() else base.CPUPlace()
)
with base.dygraph.guard(place):
    input_tensor = paddle.to_tensor(input_data.astype(np.float64))
    output_paddle = paddle.nn.functional.interpolate(
        input_tensor, size=[10, 8], mode='bicubic', align_corners=True
    )
    output_paddle_np = output_paddle.numpy()

# Compare
print("NumPy output shape:", output_np.shape)
print("Paddle output shape:", output_paddle_np.shape)
print(
    "\nMax absolute difference:", np.max(np.abs(output_np - output_paddle_np))
)
print(
    "Max relative difference:",
    np.max(np.abs((output_np - output_paddle_np) / (output_np + 1e-10))),
)

# Print some sample values
print("\nSample comparisons:")
for i in range(min(5, output_np.shape[2])):
    for j in range(min(5, output_np.shape[3])):
        np_val = output_np[0, 0, i, j]
        pd_val = output_paddle_np[0, 0, i, j]
        diff = abs(np_val - pd_val)
        print(
            f"[{i},{j}] NumPy: {np_val:.10f}, Paddle: {pd_val:.10f}, Diff: {diff:.2e}"
        )
