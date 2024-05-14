# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

# cd test/legacy_test
# for file in test_mean_op.py test_sum_op.py test_divide_op.py test_fill_constant_op.py test_dropout_op.py test_gelu_op.py test_cast_op.py test_add_op.py test_multiply_op.py test_pow_op.py test_subtract_op.py test_erf_op.py test_greater_equal_op.py test_reshape_op.py test_scale_op.py test_uniform_op.py test_concat_op.py test_expand_op.py test_embedding_op.py test_matmul_op.py test_rsqrt_op.py test_split_op.py test_add_n_op.py test_tanh_op.py test_transpose_op.py test_softmax_op.py test_rms_norm_op.py test_silu_op.py test_exp_op.py test_reduce_max_op.py test_bitwise_and_op.py test_equal_op.py test_full_like_op.py test_gather_nd_op.py test_arange_op.py test_shape_op.py test_numel_op.py test_slice_op.py test_squeeze_op.py test_stack_op.py test_unsqueeze_op.py test_where_op.py test_fill_any_like_op.py test_pad_op.py test_tile_op.py test_tril_op.py test_assign_op.py # do
#     echo $file
#     python3.10 $file
#     echo $file
# done

# python3.10  test_layer_norm_op.py # for file in test_mean_op.py test_sum_op.py test_divide_op.py test_fill_constant_op.py test_dropout_op.py test_gelu_op.py test_cast_op.py test_add_op.py test_multiply_op.py test_pow_op.py test_subtract_op.py test_erf_op.py test_greater_equal_op.py test_reshape_op.py test_scale_op.py test_uniform_op.py test_concat_op.py test_expand_op.py test_embedding_op.py test_matmul_op.py test_rsqrt_op.py test_split_op.py test_add_n_op.py test_tanh_op.py test_transpose_op.py test_softmax_op.py test_rms_norm_op.py test_silu_op.py test_exp_op.py test_reduce_max_op.py test_bitwise_and_op.py test_equal_op.py test_full_like_op.py test_gather_nd_op.py test_arange_op.py test_shape_op.py test_numel_op.py test_slice_op.py test_squeeze_op.py test_stack_op.py test_unsqueeze_op.py test_where_op.py test_fill_any_like_op.py test_pad_op.py test_tile_op.py test_tril_op.py test_assign_op.py # do
#     # 如果存在
#     if [ -f $file ]; then
#         git add $file
#     # else
#         echo $file
#     fi
# done

# 已通过
# for file in test_mean_op.py test_sum_op.py test_fill_constant_op.py test_gelu_op.py test_cast_op.py test_subtract_op.py test_erf_op.py test_greater_equal_op.py test_reshape_op.py test_scale_op.py 
# do
#     for no_file in test_mean_op test_fill_constant_op test_reduce_op test_activation_op test_pad_op test_concat_op test_full_like_op test_stack_op test_matmul_op test_split_op test_transpose_op test_split_op test_transpose_op test_matmul_op_static_build test_matmul_op_static_build 
#     do
#         if [ $file == "$no_file.py" ]; then
#             continue 2
#         fi
#     done
#     # 如果存在
#     if [ -f $file ]; then
#         ## test
#         python3.10 $file
#         if [ $? -ne 0 ]; then
#             echo "test failed $file"
#             exit 1
#         else 
#             echo "git add $file"
#             git add $file
#         fi
#     else
#         echo "no file $file"
#     fi
# done

# for file in test_concat_op.py test_expand_op.py test_matmul_op.py test_split_op.py test_add_n_op.py test_transpose_op.py test_softmax_op.py test_rms_norm_op.py test_full_like_op.py test_gather_nd_op.py test_shape_op.py test_numel_op.py test_slice_op.py test_stack_op.py test_where_op.py test_fill_any_like_op.py test_pad_op.py test_tile_op.py test_assign_op.py  test_elementwise_div_op.py test_elementwise_add_op.py test_elementwise_pow_op.py test_elementwise_mul_op.py  test_activation_op.py test_reduce_op.py test_squeeze2_op.py test_unsqueeze2_op.py test_uniform_random_op.py 
# do
#     for no_file in test_reduce_op test_activation_op test_pad_op test_concat_op test_full_like_op test_stack_op test_matmul_op test_split_op test_transpose_op test_split_op test_transpose_op test_matmul_op_static_build test_matmul_op_static_build 
#     do
#         if [ $file == "$no_file.py" ]; then
#             continue 2
#         fi
#     done
#     # 如果存在
#     if [ -f $file ]; then
#         ## test
#         python3.10 $file
#         if [ $? -ne 0 ]; then
#             echo "test failed $file"
#             exit 1
#         else 
#             echo "git add $file"
#             git add $file
#         fi
#     else
#         echo "no file $file"
#     fi
# done

# failed 
# test_mean_op test_fill_constant_op test_reduce_op test_activation_op test_pad_op test_concat_op test_full_like_op test_stack_op test_matmul_op test_split_op test_transpose_op test_split_op test_transpose_op test_matmul_op_static_build test_matmul_op_static_build 
for file in test_concat_op.py test_expand_op.py test_matmul_op.py test_split_op.py test_add_n_op.py test_transpose_op.py test_softmax_op.py test_rms_norm_op.py test_full_like_op.py test_gather_nd_op.py test_shape_op.py test_numel_op.py test_slice_op.py test_stack_op.py test_where_op.py test_fill_any_like_op.py test_pad_op.py test_tile_op.py test_assign_op.py  test_elementwise_div_op.py test_elementwise_add_op.py test_elementwise_pow_op.py test_elementwise_mul_op.py  test_activation_op.py test_reduce_op.py test_squeeze2_op.py test_unsqueeze2_op.py test_uniform_random_op.py 
do
    for no_file in test_reduce_op test_activation_op test_pad_op test_concat_op test_full_like_op test_stack_op test_matmul_op test_split_op test_transpose_op test_split_op test_transpose_op test_matmul_op_static_build test_matmul_op_static_build 
    do
        if [ $file == "$no_file.py" ]; then
            continue 2
        fi
    done
    # 如果存在
    if [ -f $file ]; then
        ## test
        python3.10 $file
        if [ $? -ne 0 ]; then
            echo "test failed $file"
            exit 1
        else 
            echo "git add $file"
            git add $file
        fi
    else
        echo "no file $file"
    fi
done
