#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import os

os.environ["FLAGS_print_ir"] = "1"

import paddle

paddle.base.core.set_vlog_level(
    {
        "phi_kernel_instruction": 10,
        "op_translator": 10,
    }
)

import os
import unittest
from contextlib import contextmanager

import numpy as np
from op_test import (
    OpTest,
    get_device,
)

import paddle

devices = ['cpu', get_device()]


@contextmanager
def dynamic_guard():
    paddle.disable_static()
    try:
        yield
    finally:
        paddle.enable_static()


def ref_leaky_relu(x, alpha=0.01):
    out = np.copy(x)
    out[out < 0] *= alpha
    return out


class TestLeakyRelu(OpTest):
    def init_dtype(self):
        self.dtype = np.float64

    def init_shape(self):
        self.shape = [11, 17]

    def init_kernel_type(self):
        pass

    def convert_input_output(self):
        pass

    def get_alpha(self):
        return 0.02

    def setUp(self):
        self.op_type = "leaky_relu"
        self.python_api = paddle.nn.functional.leaky_relu
        self.public_python_api = paddle.nn.functional.leaky_relu
        self.prim_op_type = "comp"
        self.init_dtype()
        self.init_shape()
        self.if_enable_cinn()
        alpha = self.get_alpha()
        np.set_printoptions(precision=20)

        np.random.seed(1024)
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.05
        out = ref_leaky_relu(x, alpha)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'alpha': alpha}
        self.convert_input_output()

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(
            check_prim=False,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
            check_symbol_infer=False,
        )

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'],
            'Out',
            check_prim=False,
            check_pir=True,
            check_prim_pir=True,
            check_pir_onednn=self.check_pir_onednn,
        )


if __name__ == "__main__":
    unittest.main()
