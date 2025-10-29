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

#pragma once

#include <limits>

namespace phi {
namespace funcs {

// Welford's online algorithm for computing mean and variance
// Reference:
// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
//
// For a new value x:
//   count = count + 1
//   delta = x - mean
//   mean = mean + delta / count
//   delta2 = x - mean
//   M2 = M2 + delta * delta2
//
// Variance = M2 / count (biased)
// Variance = M2 / (count - 1) (unbiased, sample variance)

// WelfordAccumulator: A class-based implementation for accumulating statistics
template <typename T>
struct WelfordAccumulator {
  T mean;
  T m2;
  T count;

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  __host__ __device__
#endif
  WelfordAccumulator()
      : mean(static_cast<T>(0)),
        m2(static_cast<T>(0)),
        count(static_cast<T>(0)) {
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  __host__ __device__
#endif
  WelfordAccumulator(T mean_val, T m2_val, T count_val)
      : mean(mean_val), m2(m2_val), count(count_val) {
  }

  // Update with a single value using Welford's online algorithm
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  __host__ __device__
#endif
      inline void
      Update(T val) {
    count += static_cast<T>(1);
    T delta = val - mean;
    mean += delta / count;
    T delta2 = val - mean;
    m2 += delta * delta2;
  }

  // Combine with another accumulator using Chan's parallel algorithm
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  __host__ __device__
#endif
      inline void
      Combine(const WelfordAccumulator<T>& other) {
    if (other.count == static_cast<T>(0)) {
      return;
    }
    T new_count = count + other.count;
    T delta = other.mean - mean;
    mean = (mean * count + other.mean * other.count) / new_count;
    m2 += other.m2 + delta * delta * (count * other.count) / new_count;
    count = new_count;
  }

  // Get variance with specified correction (ddof)
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  __host__ __device__
#endif
      inline T
      GetVariance(T correction = static_cast<T>(0)) const {
    T denominator = count - correction;
    if (denominator <= static_cast<T>(0)) {
      return std::numeric_limits<T>::quiet_NaN();
    }
    return m2 / denominator;
  }

  // Reset the accumulator
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  __host__ __device__
#endif
      inline void
      Reset() {
    mean = static_cast<T>(0);
    m2 = static_cast<T>(0);
    count = static_cast<T>(0);
  }
};

// CPU implementation: Compute variance for a batch of elements
template <typename T>
void WelfordVarCPU(const T* input,
                   T* output,
                   int64_t outer_size,
                   int64_t reduce_size,
                   int64_t inner_size,
                   T correction) {
  for (int64_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    for (int64_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
      WelfordAccumulator<T> acc;
      int64_t base_offset = outer_idx * reduce_size * inner_size + inner_idx;

      // Accumulate values using Welford's online algorithm
      for (int64_t i = 0; i < reduce_size; ++i) {
        T val = input[base_offset + i * inner_size];
        acc.Update(val);
      }

      output[outer_idx * inner_size + inner_idx] = acc.GetVariance(correction);
    }
  }
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

// Legacy function-based API for backward compatibility
// Update running statistics with a single new value
template <typename T>
__device__ inline void WelfordOnline(T val, T* mean, T* m2, T* count) {
  *count += static_cast<T>(1);
  T delta = val - *mean;
  *mean += delta / (*count);
  T delta2 = val - *mean;
  *m2 += delta * delta2;
}

// Combine two sets of Welford statistics (parallel reduction)
// Chan's parallel algorithm for combining statistics
template <typename T>
__device__ inline void WelfordCombine(
    T b_mean, T b_m2, T b_count, T* mean, T* m2, T* count) {
  if (b_count == static_cast<T>(0)) {
    return;
  }
  T new_count = *count + b_count;
  T delta = b_mean - *mean;
  *mean = (*mean * *count + b_mean * b_count) / new_count;
  *m2 += b_m2 + delta * delta * (*count * b_count) / new_count;
  *count = new_count;
}

// Warp-level reduction using shuffle operations
template <typename T>
__device__ inline void WelfordWarpReduce(T* mean, T* m2, T* count) {
  constexpr int kWarpSize = 32;
#pragma unroll
  for (int mask = 1; mask < kWarpSize; mask *= 2) {
    T b_mean = __shfl_down_sync(0xffffffff, *mean, mask);
    T b_m2 = __shfl_down_sync(0xffffffff, *m2, mask);
    T b_count = __shfl_down_sync(0xffffffff, *count, mask);
    WelfordCombine(b_mean, b_m2, b_count, mean, m2, count);
  }

  *mean = __shfl_sync(0xffffffff, *mean, 0, kWarpSize);
  *m2 = __shfl_sync(0xffffffff, *m2, 0, kWarpSize);
  *count = __shfl_sync(0xffffffff, *count, 0, kWarpSize);
}

// Block-level reduction using shared memory
template <typename T, int BlockSize>
__device__ inline void WelfordBlockReduce(
    T* mean, T* m2, T* count, T* shared_mean, T* shared_m2, T* shared_count) {
  int tid = threadIdx.x;

  shared_mean[tid] = *mean;
  shared_m2[tid] = *m2;
  shared_count[tid] = *count;
  __syncthreads();

  for (int stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      WelfordCombine(shared_mean[tid + stride],
                     shared_m2[tid + stride],
                     shared_count[tid + stride],
                     &shared_mean[tid],
                     &shared_m2[tid],
                     &shared_count[tid]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    *mean = shared_mean[0];
    *m2 = shared_m2[0];
    *count = shared_count[0];
  }
}

#endif  // PADDLE_WITH_CUDA || PADDLE_WITH_HIP

}  // namespace funcs
}  // namespace phi
