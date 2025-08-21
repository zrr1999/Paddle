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

#pragma once

#include "paddle/common/errors.h"
#include "paddle/phi/common/type_promotion.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/kernels/cast_kernel.h"
namespace phi {
namespace funcs {

static inline int64_t ComputeAxis(int64_t axis, int64_t rank) {
  PADDLE_ENFORCE_EQ(
      axis >= -rank && axis < rank,
      true,
      common::errors::InvalidArgument(
          "The axis is expected to be in range of [%d, %d), but got %d",
          -rank,
          rank,
          axis));
  if (axis < 0) {
    axis = axis + rank;
  }
  return axis > 0 ? axis : 0;
}

static inline phi::DDim ComputeAndCheckShape(
    bool is_runtime,
    const std::vector<phi::DDim>& inputs_dims,
    const size_t axis) {
  const size_t n = inputs_dims.size();
  auto out_dims = inputs_dims[0];
  size_t in_zero_dims_size = out_dims.size();
  for (size_t i = 1; i < n; i++) {
    PADDLE_ENFORCE_EQ(
        inputs_dims[i].size(),
        out_dims.size(),
        common::errors::InvalidArgument("The shape of input[0] and input[%d] "
                                        "is expected to be equal."
                                        "But received input[0]'s shape = "
                                        "[%s], input[%d]'s shape = [%s].",
                                        i,
                                        inputs_dims[0],
                                        i,
                                        inputs_dims[i]));
    for (size_t j = 0; j < in_zero_dims_size; j++) {
      if (j == axis) {
        if (is_runtime) {
          out_dims[axis] += inputs_dims[i][j];
        } else {
          if (inputs_dims[i][j] == -1 || out_dims[j] == -1) {
            out_dims[axis] = -1;
          } else {
            out_dims[axis] += inputs_dims[i][j];
          }
        }
      } else {
        bool check_shape =
            is_runtime || (inputs_dims[0][j] > 0 && inputs_dims[i][j] > 0);
        if (check_shape) {
          // check all shape in run time
          PADDLE_ENFORCE_EQ(inputs_dims[0][j],
                            inputs_dims[i][j],
                            common::errors::InvalidArgument(
                                "The %d-th dimension of input[0] and input[%d] "
                                "is expected to be equal."
                                "But received input[0]'s shape = "
                                "[%s], input[%d]'s shape = [%s].",
                                j,
                                i,
                                inputs_dims[0],
                                i,
                                inputs_dims[i]));
        }
        if (!is_runtime && out_dims[j] == -1 && inputs_dims[i][j] > 0) {
          out_dims[j] = inputs_dims[i][j];
        }
      }
    }
  }
  return out_dims;
}

// From a multi-input, gather only nonempty inputs
static const std::vector<const DenseTensor*> ReduceMultiInput(
    const std::vector<const DenseTensor*>& inputs) {
  std::vector<const DenseTensor*> reduced(inputs.size());
  auto end_it = std::copy_if(
      inputs.begin(), inputs.end(), reduced.begin(), [](const DenseTensor* t) {
        return t->numel() > 0;
      });
  reduced.resize(std::distance(reduced.begin(), end_it));
  return reduced;
}

static DataType FindHighestPrecisionType(
    const std::vector<const MetaTensor*>& inputs) {
  if (inputs.empty()) {
    return DataType::UNDEFINED;
  }

  DataType highest_type = inputs[0]->dtype();
  for (size_t i = 1; i < inputs.size(); ++i) {
    highest_type = promoteTypes(highest_type, inputs[i]->dtype());
  }
  return highest_type;
}

template <typename T, typename Context>
static std::vector<DenseTensor> PromoteTensorTypes(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& inputs,
    DataType target_type) {
  auto non_empty_inputs = ReduceMultiInput(inputs);

  if (inputs.empty()) {
    return {};
  }

  std::vector<DenseTensor> promoted_tensors;
  promoted_tensors.reserve(non_empty_inputs.size());

  // Convert each tensor to the target type
  for (const auto* tensor : non_empty_inputs) {
    if (tensor->dtype() == target_type) {
      // Same type, copy directly
      promoted_tensors.emplace_back(*tensor);
    } else {
      DenseTensor promoted_tensor;
      CastKernel<T, Context>(dev_ctx, *tensor, target_type, &promoted_tensor);
      promoted_tensors.push_back(promoted_tensor);
    }
  }

  return promoted_tensors;
}

}  // namespace funcs
}  // namespace phi
