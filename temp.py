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

ops = [
    "mean",
    "sum",
    "fill_constant",
    "dropout",
    "gelu",
    "cast",
    "subtract",
    "erf",
    "greater_equal",
    "reshape",
    "scale",
    "concat",
    "expand",
    "matmul",
    "split",
    "add_n",
    "transpose",
    "softmax",
    "rms_norm",
    "full_like",
    "gather_nd",
    "shape",
    "numel",
    "slice",
    "stack",
    "where",
    "fill_any_like",
    "pad",
    "tile",
    "assign",
]


for i, op in enumerate(ops):
    print(f"|A-{i+1}|{op}|test/legacy_test/test_{op}_op.py|||")

ops = ["div", "add", "pow", "mul"]

for i, op in enumerate(ops):
    print(f"|B-{i+1}|{op}|test/legacy_test/test_elementwise_{op}_op.py|||")

ops = ["silu", "exp", "tanh", "rsqrt"]

for i, op in enumerate(ops):
    print(f"|B-{i+5}|{op}|test/legacy_test/test_activation_op.py|||")


ops = {
    "reduce_max": "test_reduce_op.py",
    "squeeze ": "test_squeeze2_op.py",
    "unsqueeze ": "test_unsqueeze2_op.py",
    "uniform": "test_uniform_random_op.py",
    "embedding": "test_lookup_table_v2_op.py",
    "bitwise_and": "test_bitwise_op.py",
    "arange": "test_arange.py",
    "equal": "test_compare_op.py",
    "tril": "test_tril_triu_op.py",
}
for i, (op, path) in enumerate(ops.items()):
    print(f"|C-{i+1}|{op}|test/legacy_test/{path}|||")


# part II

ops = ["clip", "min", "prod", "gather", "dot"]


for i, op in enumerate(ops):
    print(f"|🚧 D-{i+1}|{op}|test/legacy_test/test_{op}_op.py|@zrr1999|#61289|")

ops = [
    "transpose",
    "pad",
    "conv2d",
    "dot",
    "cumsum",
    "linear_interp",
    "bicubic_interp",
    "assign",
    "conv2d",
    "full",
    "full",
    "softmax",
    "randint",
    "conv2d",
    "solve",
    "eig",
    "eigvals",
    "randperm",
    "batch_norm",
]
for i, op in enumerate(ops):
    print(f"|D-{i+6}|{op}|test/legacy_test/test_{op}_op.py|||")
