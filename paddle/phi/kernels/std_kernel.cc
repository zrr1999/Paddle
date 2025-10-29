// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/std_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/var_kernel.h"

namespace phi {

template <typename T, typename Context>
void StdKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const IntArray& axis,
               bool keepdim,
               bool unbiased,
               DenseTensor* out) {
  VarKernel<T, Context>(dev_ctx, x, axis, keepdim, unbiased, 1, out);
  SqrtKernel<T, Context>(dev_ctx, *out, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(std, CPU, ALL_LAYOUT, phi::StdKernel, float, double) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(
    std, GPU, ALL_LAYOUT, phi::StdKernel, float, double, phi::dtype::float16) {}
#endif
