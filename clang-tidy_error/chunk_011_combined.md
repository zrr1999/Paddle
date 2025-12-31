# Clang-Tidy Narrowing Conversion Errors - chunk_011_combined

**Total Errors in this chunk:** 54

**Files in this chunk:** 7

---

## File: `/workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/reduce_as_to_sum_pass.cc`

**Number of errors:** 7

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/reduce_as_to_sum_pass.cc:153:30: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/reduce_as_to_sum_pass.cc:157:32: error: narrowing conversion from 'unsigned long' to signed type 'std::vector<long>::value_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/reduce_as_to_sum_pass.cc:164:34: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::set<long>::key_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/reduce_as_to_sum_pass.cc:192:32: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/reduce_as_to_sum_pass.cc:201:34: error: narrowing conversion from 'unsigned long' to signed type 'std::vector<long>::value_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/reduce_as_to_sum_pass.cc:212:36: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'std::set<long>::key_type' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/hlir/dialect/operator/transforms/reduce_as_to_sum_pass.cc:63:30: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.cc`

**Number of errors:** 8

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.cc:477:16: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.cc:555:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.cc:574:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.cc:597:17: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.cc:597:39: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.cc:599:17: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.cc:599:39: error: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/group_schedule/tactic/tile_transpose_tactic.cc:608:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/ir/ir_base.cc`

**Number of errors:** 7

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/ir/ir_base.cc:155:10: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int8_t' (aka 'signed char') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/ir_base.cc:163:10: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int16_t' (aka 'short') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/ir_base.cc:172:10: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/ir_base.cc:236:10: error: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/ir_base.cc:281:18: error: narrowing conversion from 'int64_t' (aka 'long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/ir_base.cc:282:18: error: narrowing conversion from 'uint64_t' (aka 'unsigned long') to 'double' [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/ir/ir_base.cc:578:10: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/cinn/operator_fusion/utils.cc`

**Number of errors:** 8

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/cinn/operator_fusion/utils.cc:135:15: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/operator_fusion/utils.cc:214:26: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/operator_fusion/utils.cc:215:27: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/operator_fusion/utils.cc:256:51: error: narrowing conversion from 'size_t' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/operator_fusion/utils.cc:288:31: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/operator_fusion/utils.cc:288:51: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/operator_fusion/utils.cc:298:20: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/cinn/operator_fusion/utils.cc:299:21: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/operators/fused/fused_gate_attention_op.cc`

**Number of errors:** 8

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/operators/fused/fused_gate_attention_op.cc:50:22: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/operators/fused/fused_gate_attention_op.cc:51:21: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/operators/fused/fused_gate_attention_op.cc:52:21: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/operators/fused/fused_gate_attention_op.cc:69:18: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/operators/fused/fused_gate_attention_op.cc:70:18: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/operators/fused/fused_gate_attention_op.cc:93:18: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/operators/fused/fused_gate_attention_op.cc:94:18: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/operators/fused/fused_gate_attention_op.cc:95:16: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/layout_transformation.cc`

**Number of errors:** 8

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/layout_transformation.cc:224:27: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'std::vector<int>::value_type' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/layout_transformation.cc:424:9: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/layout_transformation.cc:425:9: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/layout_transformation.cc:426:9: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/layout_transformation.cc:428:9: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/layout_transformation.cc:429:9: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/layout_transformation.cc:430:9: error: narrowing conversion from 'int64_t' (aka 'long') to signed type 'int32_t' (aka 'int') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/fluid/pir/dialect/operator/interface/layout_transformation.cc:458:13: error: narrowing conversion from '__gnu_cxx::__alloc_traits<std::allocator<long>, long>::value_type' (aka 'long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |

## File: `/workspace/Paddle/paddle/phi/infermeta/spmd_rules/tile.cc`

**Number of errors:** 8

| Line | Description  |
|------|-------------|
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/tile.cc:102:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/tile.cc:116:18: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/tile.cc:136:28: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/tile.cc:178:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/tile.cc:192:23: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/tile.cc:213:28: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/tile.cc:43:16: error: narrowing conversion from 'std::vector::size_type' (aka 'unsigned long') to signed type 'int' is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
| | /workspace/Paddle/paddle/phi/infermeta/spmd_rules/tile.cc:56:28: error: narrowing conversion from 'unsigned long' to signed type 'int64_t' (aka 'long') is implementation-defined [bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions,-warnings-as-errors] |
