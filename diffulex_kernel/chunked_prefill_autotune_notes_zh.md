# Chunked Prefill Autotune 精度问题记录

## 背景

本记录针对 `diffulex_kernel/python/chunked_prefill_triton.py` 里的 unified chunked prefill Triton kernel。

关注配置范围是：

- `page_size ∈ {4, 8, 16, 32}`
- `block_size ∈ {4, 8, 16, 32}`
- 只考虑 `page_size % block_size == 0`

用户预期是：在 `block_size` 固定时，`page_size` 只是 KV page 的存储粒度，**不应该改变输出 token 和分数**。

## 问题现象

修复前，`autotune off` 和 `autotune on` 的表现不一致：

- `autotune off`：
  同一个 `block_size` 下，不同 `page_size` 可以对齐。
- `autotune on`：
  某些组合会出现 token 级分叉。

最典型的复现是：

- `block_size=16, page_size=16`
- `block_size=16, page_size=32`

在同一个 prompt、`temperature=0.0`、`max_num_reqs=1`、`enforce_eager=True` 下，修复前两边会生成不同 token，首个分叉很早就出现，说明不是后处理或调度问题，而是 forward 路径已经发生偏移。

## 根因

问题不是单纯的 `page_size` bookkeeping 错误，而是 **autotune 选到不同 kernel launch shape 后，cache-stage softmax 的归约顺序发生变化**，这种数值差异在 greedy decode 下被放大成了不同 token。

具体链路分两层：

### 1. 语义层问题

早期 kernel 的 Stage-1 cache attention 是按 page 组织遍历的，这会让 `page_size` 直接参与 softmax 归约顺序，导致不同 `page_size` 下数值行为不等价。

这个问题后来改成了：

- 按全局 `BLOCK_N` tile 遍历 cache token
- 再把全局 KV 位置映射到 `(page_rel_id, page_off)`

也就是：`page_size` 只影响寻址，不影响归约顺序。

### 2. autotune 层问题

即使语义层修正之后，`autotune on` 仍然会根据 `PAGE_SIZE` 选到不同的 launch shape。

修复前，本地抓到过这样的真实分叉：

- `block_size=16, page_size=16`
  选到 `BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2`
- `block_size=16, page_size=32`
  选到 `BLOCK_M=64, BLOCK_N=128, num_warps=4, num_stages=1`

这两种 tile shape 的 softmax / `acc` 累加顺序不同，足以把 `temperature=0.0` 的 greedy decode 翻成另一条输出。

## 修复方案

当前采用的是 **保守 correctness 优先策略**。

在 [`chunked_prefill_triton.py`](./python/chunked_prefill_triton.py) 中：

1. 保留 Triton autotune。
2. 对 `DLLM_BLOCK_SIZE <= 16` 的 case，提前 prune autotune 候选。
3. 将这类小 block 的候选固定为稳定 launch shape：
   - `BLOCK_M=64`
   - `BLOCK_N=64`
   - `num_warps=4`
   - `num_stages=2`
4. 对 `DLLM_BLOCK_SIZE > 16`，仍保留完整搜索空间。

对应代码入口：

- [`_prune_chunked_prefill_configs`](./python/chunked_prefill_triton.py)
- `triton.autotune(..., prune_configs_by={"early_config_prune": ...})`

## 为什么这样修

目标不是完全关闭 autotune，而是：

- 保留大 block 的性能搜索空间
- 只在已经确认会影响 correctness 的小 block case 上收紧搜索空间

换句话说，当前策略是：

- `small block`: correctness first
- `large block`: performance first

## 当前验证结论

### 1. 轻量单测

已补：

- [`test/python/engine/test_chunked_prefill_autotune_prune.py`](../test/python/engine/test_chunked_prefill_autotune_prune.py)

验证：

- `DLLM_BLOCK_SIZE=16` 时，只保留稳定 config
- `DLLM_BLOCK_SIZE=32` 时，不收紧搜索空间

### 2. 重度矩阵测试

已补：

- [`test/python/engine/test_page_block_pow4_matrix.py`](../test/python/engine/test_page_block_pow4_matrix.py)

覆盖范围：

- `page_size ∈ {4, 8, 16, 32}`
- `block_size ∈ {4, 8, 16, 32}`
- `page_size % block_size == 0`

检查项：

- `same_text_as_baseline`
- `same_tokens_as_baseline`
- `same_steps_as_baseline`

运行方式：

```bash
DIFFULEX_RUN_PAGE_BLOCK_POW4_MATRIX_AUTOTUNE=1 \
DIFFULEX_MATRIX_MAX_TOKENS=32 \
pytest -q test/python/engine/test_page_block_pow4_matrix.py::test_page_block_pow4_matrix_autotune_on -s --forked
```

本地当前结果：

- `block_size=4`: `page=4/8/16/32` 全对齐
- `block_size=8`: `page=8/16/32` 全对齐
- `block_size=16`: `page=16/32` 全对齐
- `block_size=32`: 只有 `32/32`

也就是在当前关注的 `pow4` 合法组合中，**`autotune on` 已重新打通 correctness**。

## 当前约束

这次修复是一个明确的工程 tradeoff：

- 优点：
  小 block 的 `autotune on` correctness 已恢复
- 代价：
  `DLLM_BLOCK_SIZE <= 16` 时，autotune 不再搜索完整 tile 空间，理论上会损失一部分峰值性能

因此，如果后续要继续做性能优化，正确方向不是把这个 prune 直接删掉，而是：

1. 重新扩大搜索空间
2. 同时保证不同 `page_size` 下的数值稳定性不再破坏 greedy decode

## 后续可继续做的事

如果后续需要把小 block 的性能再抬高，建议按下面顺序继续：

1. 比较 `BLOCK_M/BLOCK_N/num_stages` 对数值漂移的敏感度
2. 尝试只放开一部分安全候选，而不是一次性恢复完整搜索空间
3. 若仍需更大搜索空间，再评估：
   - softmax 累加顺序
   - `acc`/`v` 的精度路径
   - 是否要进一步限制 `BLOCK_N`

## 一句话结论

这次问题的根因是：

> `autotune on` 时，不同 `page_size` 触发了不同 cache-stage tile shape，进而改变了 softmax 归约顺序，最终把数值差异放大成了不同 token。

当前修复方案是：

> 对 `DLLM_BLOCK_SIZE <= 16` 的 case，保留 autotune 框架，但把候选收缩到已经验证稳定的一组 launch shape。
