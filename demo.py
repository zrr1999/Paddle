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

import paddle

init_data_list = [[[[10, 50], [100, 200]]]]
tensor_uint8 = paddle.to_tensor(init_data_list, dtype='uint8')
tensor_fp32 = tensor_uint8.astype('float32')

print(tensor_fp32)
output_size = [4, 4]
interpolated_tensor_fp32 = paddle.nn.functional.interpolate(
    tensor_fp32, size=output_size, mode='bilinear', align_corners=False
)
print(interpolated_tensor_fp32)

interpolated_tensor_uint8 = interpolated_tensor_fp32.astype('uint8')
print(interpolated_tensor_uint8)
