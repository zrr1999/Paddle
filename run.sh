# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

/workspace/paddle/.venv/bin/python /workspace/paddle/paddle/fluid/eager/auto_code_generator/generator/eager_gen.py \
    --api_yaml_path=/workspace/paddle/paddle/phi/ops/yaml/ops.yaml,/workspace/paddle/paddle/phi/ops/yaml/inconsistent/dygraph_ops.yaml,/workspace/paddle/paddle/phi/ops/yaml/sparse_ops.yaml,/workspace/paddle/paddle/phi/ops/yaml/fused_ops.yaml,/workspace/paddle/paddle/phi/ops/yaml/strings_ops.yaml \
    --backward_yaml_path=/workspace/paddle/paddle/phi/ops/yaml/backward.yaml,/workspace/paddle/paddle/phi/ops/yaml/inconsistent/dygraph_backward.yaml,/workspace/paddle/paddle/phi/ops/yaml/sparse_backward.yaml,/workspace/paddle/paddle/phi/ops/yaml/fused_backward.yaml \
    --forwards_cc_path=/workspace/paddle/paddle/fluid/eager/api/generated/eager_generated/forwards/tmp_dygraph_functions.cc \
    --forwards_h_path=/workspace/paddle/paddle/fluid/eager/api/generated/eager_generated/forwards/tmp_dygraph_functions.h \
    --backwards_cc_path=/workspace/paddle/paddle/fluid/eager/api/generated/eager_generated/forwards/tmp_dygraph_grad_functions.cc \
    --backwards_h_path=/workspace/paddle/paddle/fluid/eager/api/generated/eager_generated/forwards/tmp_dygraph_grad_functions.h \
    --nodes_cc_path=/workspace/paddle/paddle/fluid/eager/api/generated/eager_generated/backwards/tmp_nodes.cc \
    --nodes_h_path=/workspace/paddle/paddle/fluid/eager/api/generated/eager_generated/backwards/tmp_nodes.h

# python test/ir/pir/fused_pass/onednn/test_fc_activation_fuse_pass.py
# python test/ir/pir/fused_pass/onednn/test_matmul_activation_fuse_pass.py
python test/ir/pir/fused_pass/onednn/test_elementwise_act_fuse_pass.py
python test/ir/pir/fused_pass/onednn/test_softplus_activation_fuse_pass.py
