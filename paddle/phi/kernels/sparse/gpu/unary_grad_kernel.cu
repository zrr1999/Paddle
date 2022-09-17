// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/sparse/unary_grad_kernel.h"
#include "paddle/phi/kernels/sparse/unary_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/sparse/empty_kernel.h"
#include "paddle/phi/kernels/sparse/impl/unary_grad_kernel_impl.h"

namespace phi {
namespace sparse {

std::vector<int> get_gpu_grad_perm(std::vector<int> perm) {
  std::vector<int> grad_perm(perm.size());
  for (unsigned int i = 0; i < perm.size(); ++i) {
    grad_perm[perm[i]] = i;
  }
  return grad_perm;
}

template <typename T, typename Context>
void TransposeCooGradKernel(const Context& dev_ctx,
                            const SparseCooTensor& x,
                            const SparseCooTensor& dout,
                            const std::vector<int>& perm,
                            SparseCooTensor* dx) {
  EmptyLikeCooKernel<T, Context>(dev_ctx, x, dx);
  std::vector<int> grad_perm = get_gpu_grad_perm(perm);
  TransposeCooKernel<T, Context>(dev_ctx, dout, grad_perm, dx);
}

template <typename T, typename Context>
void TransposeCsrGradKernel(const Context& dev_ctx,
                            const SparseCsrTensor& x,
                            const SparseCsrTensor& dout,
                            const std::vector<int>& perm,
                            SparseCsrTensor* dx) {
  EmptyLikeCsrKernel<T, Context>(dev_ctx, x, dx);
  std::vector<int> grad_perm = get_gpu_grad_perm(perm);
  TransposeCsrKernel<T, Context>(dev_ctx, dout, grad_perm, dx);
}
}  // namespace sparse
}  // namespace phi

#define PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(name, prefix)     \
  PD_REGISTER_KERNEL(name##_coo_grad,                              \
                     GPU,                                          \
                     ALL_LAYOUT,                                   \
                     phi::sparse::prefix##CooGradKernel,           \
                     phi::dtype::float16,                          \
                     float,                                        \
                     double) {                                     \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_COO); \
  }                                                                \
                                                                   \
  PD_REGISTER_KERNEL(name##_csr_grad,                              \
                     GPU,                                          \
                     ALL_LAYOUT,                                   \
                     phi::sparse::prefix##CsrGradKernel,           \
                     phi::dtype::float16,                          \
                     float,                                        \
                     double) {                                     \
    kernel->InputAt(0).SetDataLayout(phi::DataLayout::SPARSE_CSR); \
  }

PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(sin, Sin)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(tan, Tan)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(asin, Asin)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(atan, Atan)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(sinh, Sinh)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(tanh, Tanh)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(asinh, Asinh)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(atanh, Atanh)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(sqrt, Sqrt)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(square, Square)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(log1p, Log1p)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(relu, Relu)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(abs, Abs)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(pow, Pow)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(expm1, Expm1)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(relu6, Relu6)
PD_REGISTER_SPARSE_UNARY_GPU_GRAD_KERNEL(leaky_relu, LeakyRelu)

PD_REGISTER_KERNEL(cast_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CastCooGradKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(cast_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::CastCsrGradKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(transpose_coo_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::TransposeCooGradKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}

PD_REGISTER_KERNEL(transpose_csr_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::sparse::TransposeCsrGradKernel,
                   phi::dtype::float16,
                   float,
                   double,
                   int8_t,
                   uint8_t,
                   int16_t,
                   int,
                   int64_t,
                   bool) {}
