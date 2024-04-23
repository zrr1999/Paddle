# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle


def foo(x):
    return x + 1


class TestOpcodeExecutorDynamicShapeCache(TestCaseBase):
    def test_cache_hit(self):
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(foo, paddle.randn([2, 3]))
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(foo, paddle.randn([3, 3]))
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(foo, paddle.randn([4, 3]))
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(foo, paddle.randn([4, 4]))
            self.assertEqual(ctx.translate_count, 3)
            self.assert_results(foo, paddle.randn([5, 5]))
            self.assertEqual(ctx.translate_count, 3)


if __name__ == '__main__':
    unittest.main()
