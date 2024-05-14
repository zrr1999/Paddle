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

function test_file() {
    file=$1
    python=$2

    if [ -f $file ]; then
        if [ $(git status --porcelain $file | wc -l) -ne 0 ]; then
            $python $file
            if [ $? -ne 0 ]; then
                echo "test failed $file"
            else 
                echo "git add $file"
                git add $file
            fi
        else
            echo "File $file is unmodified"
        fi
        
    else
        echo "no file $file"
    fi
}


# part 1

# for file in test_mean_op.py test_sum_op.py test_fill_constant_op.py test_dropout_op.py test_gelu_op.py test_cast_op.py test_subtract_op.py test_erf_op.py test_greater_equal_op.py test_reshape_op.py test_scale_op.py test_concat_op.py test_expand_op.py test_matmul_op.py test_split_op.py test_add_n_op.py test_transpose_op.py test_softmax_op.py test_rms_norm_op.py test_full_like_op.py test_gather_nd_op.py test_shape_op.py test_numel_op.py test_slice_op.py test_stack_op.py test_where_op.py test_fill_any_like_op.py test_pad_op.py test_tile_op.py test_assign_op.py test_elementwise_div_op.py test_elementwise_add_op.py test_elementwise_pow_op.py test_elementwise_mul_op.py test_activation_op.py test_reduce_op.py test_squeeze2_op.py test_unsqueeze2_op.py test_uniform_random_op.py test_lookup_table_v2_op.py test_bitwise_op.py test_arange.py test_compare_op.py test_tril_triu_op.py
# do
#     # 如果存在
#     if [ -f $file ]; then
#         ## test
#         python3.10 $file
#         if [ $? -ne 0 ]; then
#             echo "test failed $file"
#         else 
#             echo "git add $file"
#             git add $file
#         fi
#     else
#         echo "no file $file"
#     fi
# done

# part 2

for file in test_allclose_op.py test_transpose_op.py test_pad_op.py test_conv2d_op.py test_depthwise_conv2d_op.py test_sqrt_op.py test_reduce_all_op.py test_flatten_op.py test_relu_op.py test_abs_op.py test_log_op.py test_clip_op.py test_ceil_op.py test_frobenius_norm_op.py test_p_norm_op.py test_maximum_op.py test_argsort_op.py test_argmax_op.py test_min_op.py test_max_pool2d_with_index_op.py test_pool2d_op.py test_minimum_op.py test_adam_op.py test_prod_op.py test_round_op.py test_sin_op.py test_COS_op.py test_dot_op.py test_floor_op.py test_accuracy_op.py test_topk_op.py test_pool2d_op.py test_sgd_op.py test_remainder_op.py test_square_op.py test_gather_op.py test_unique_op.py test_adamw_op.py test_one_hot_op.py test_smooth_label_op.py test_cross_entropy_with_softmax_op.py test_mean_all_op.py test_any_op.py test_cumsum_op.py test_linear_interp_op.py test_bilinear_interp_op.py test_trilinear_interp_op.py test_nearest_interp_op.py test_bicubic_interp_op.py test_pool2d_op.py test_less_than_op.py test_assign_op.py test_assign_out__op.py test_assign_value__op.py test_isclose_op.py test_conv2d_op.py test_depthwise_conv2d_op.py test_momentum_op.py test_real_op.py test_flip_op.py test_equal_op.py test_logical_and_op.py test_nonzero_op.py test_full_op.py test_full_op.py test_isnan_op.py test_softmax_op.py test_expand_op.py test_randint_op.py test_conv2d_transpose_op.py test_depthwise_conv2d_transpose_op.py test_increment_op.py test_sigmoid_op.py test_einsum_op.py test_leaky_relu_op.py test_square_op.py test_topk_op.py test_log10_op.py test_conv2d_op.py test_depthwise_conv2d_op.py test_conv3d_op.py test_reshape__op.py test_solve_op.py test_diag_op.py test_linspace_op.py test_empty_op.py test_gelu_op.py test_trace_op.py test_exponential_op.py test_gaussian_random_op.py test_multinomial_op.py test_poIssOn_op.py test_gumbe_softmax_op.py test_eig_op.py test_eigvals_op.py test_logcumsumexp_op.py test_randperm_op.py test_greaterthan_op.py test_less_equal_op.py test_not_equal_op.py
do
    test_file $file python3.11
done
