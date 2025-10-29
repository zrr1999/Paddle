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

#include "paddle/phi/kernels/var_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/kernel_utils.h"
#include "paddle/phi/kernels/funcs/welford.h"

namespace phi {

template <typename T, typename Context>
void VarKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const IntArray& axis,
               bool keepdim,
               bool unbiased,
               double correction,
               DenseTensor* out) {
  int64_t size = x.numel();

  if (size == 0) {
    dev_ctx.template Alloc<T>(out);
    auto* out_data = out->data<T>();
    *out_data = std::numeric_limits<T>::quiet_NaN();
    return;
  }

  auto reduce_dims = axis.GetData();
  bool reduce_all = recompute_reduce_all(x, axis);

  // Determine the actual correction value
  // When unbiased=None in Python, it's converted to the YAML default (true)
  // So we check: if unbiased is true, use correction; otherwise use 0
  double actual_correction = unbiased ? correction : 0.0;

  // Calculate reduction factor and warn if DOF <= 0
  int64_t reduction_factor = 1;
  if (reduce_all) {
    reduction_factor = size;
  } else {
    const auto& input_shape = x.dims();
    for (int64_t dim : reduce_dims) {
      int64_t actual_dim = dim < 0 ? dim + input_shape.size() : dim;
      if (actual_dim >= 0 && actual_dim < input_shape.size()) {
        reduction_factor *= input_shape[actual_dim];
      }
    }
  }

  if (reduction_factor - actual_correction <= 0) {
    LOG(WARNING) << "WARNING: degrees of freedom is <= 0. Correction ("
                 << actual_correction
                 << ") should be strictly less than the reduction factor ("
                 << reduction_factor << ").";
  }

  dev_ctx.template Alloc<T>(out);
  auto* out_data = out->data<T>();
  T correction_val = static_cast<T>(actual_correction);

  // Reshape to [outer_size, reduce_size, inner_size]
  // where reduce_size is the product of all reduction dimensions
  int64_t outer_size = 1;
  int64_t reduce_size = 1;
  int64_t inner_size = 1;

  if (reduce_all) {
    // reduce_all: shape becomes [1, numel, 1]
    outer_size = 1;
    reduce_size = size;
    inner_size = 1;
  } else if (reduce_dims.size() == 1) {
    // Single-axis reduction
    int64_t reduce_dim = reduce_dims[0];
    if (reduce_dim < 0) {
      reduce_dim += x.dims().size();
    }

    // Shape: [d0, d1, ..., d_reduce, ..., dn]
    // Reshape to: [d0*d1*...*d(reduce-1), d_reduce, d(reduce+1)*...*dn]
    for (int i = 0; i < reduce_dim; ++i) {
      outer_size *= x.dims()[i];
    }
    reduce_size = x.dims()[reduce_dim];
    for (int i = reduce_dim + 1; i < x.dims().size(); ++i) {
      inner_size *= x.dims()[i];
    }
  } else {
    // Multi-axis reduction
    // Normalize reduce_dims
    std::vector<int64_t> normalized_dims;
    for (int64_t dim : reduce_dims) {
      int64_t actual_dim = dim < 0 ? dim + x.dims().size() : dim;
      normalized_dims.push_back(actual_dim);
    }
    std::sort(normalized_dims.begin(), normalized_dims.end());

    // Check if reduce_dims are consecutive
    bool is_consecutive = true;
    for (size_t i = 1; i < normalized_dims.size(); ++i) {
      if (normalized_dims[i] != normalized_dims[i - 1] + 1) {
        is_consecutive = false;
        break;
      }
    }

    if (is_consecutive) {
      // Consecutive dimensions: can directly merge
      int64_t first_dim = normalized_dims[0];
      int64_t last_dim = normalized_dims.back();

      for (int i = 0; i < first_dim; ++i) {
        outer_size *= x.dims()[i];
      }
      for (int i = first_dim; i <= last_dim; ++i) {
        reduce_size *= x.dims()[i];
      }
      for (int i = last_dim + 1; i < x.dims().size(); ++i) {
        inner_size *= x.dims()[i];
      }
    } else {
      // Non-consecutive dimensions: need transpose
      PADDLE_THROW(
          phi::errors::Unimplemented("Welford variance currently only supports "
                                     "consecutive multi-axis reduction. "
                                     "Got %d non-consecutive axes.",
                                     reduce_dims.size()));
    }
  }

  // Call CPU implementation from welford.h
  funcs::WelfordVarCPU<T>(x.data<T>(),
                          out_data,
                          outer_size,
                          reduce_size,
                          inner_size,
                          correction_val);
}

}  // namespace phi

PD_REGISTER_KERNEL(var, CPU, ALL_LAYOUT, phi::VarKernel, float, double) {}
