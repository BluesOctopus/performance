# Stage3 离线字面量压缩系统（方案 A）

本项目实现了一个可运行、可测试、可扩展的离线字面量压缩系统，目标是在**固定 Tokenizer** 下，最小化字段字面量的期望 token 成本，同时保证严格可逆（semantic loss = 0）。

## 1. 项目目标

- 对多个离散字段进行独立建模（field-level dictionary）。
- 对高频且原始 token 成本较高的字面量分配短码。
- 短码分配由自信息/分布驱动，并满足前缀可解码约束。
- 输出可复现的离线构建产物：`codebook.json` 与 `report.json`。

## 2. 方案 A 数学定义

对于字段 `X_i`，值域 `V_i = {x_1, ..., x_k}`，计数 `n(x)`，总样本数 `N`：

- 经验分布：`p_hat(x) = n(x) / N`
- Lidstone/Laplace 平滑分布：`p_tilde(x) = (n(x) + alpha) / (N + alpha * |V_i|)`
- 自信息：`I(x) = -log2(p(x))`
- 熵：`H(X_i) = -sum_x p(x) log2 p(x)`

说明：
- `H(X_i)` 是 bit 维度上最优前缀码平均码长理论下界（信息论基线）。
- 在固定 Tokenizer 下，优化目标是实际 token 数，不是 bit 长度本身。
- 因此还需要计算经验期望 token 成本：
  - `E_raw = sum_x p(x) * TokLen(raw(x))`

## 3. token 成本与期望收益

对字面量 `x` 及候选短码 `z`：

- `c_raw(x) = TokLen(raw(x))`
- `c_code(z) = TokLen(z)`
- `saving(x, z) = c_raw(x) - c_code(z)`
- `gain(x, z) = p(x) * max(0, saving(x, z))`

字段总收益（可有未分配项）：

- `total_gain = sum_x p(x) * (c_raw(x) - c_assigned(x))`

## 4. 贪心 prefix-free 分配策略

默认 `GreedyPrefixFreeAssigner`：

1. 统计字段分布与 `raw_token_cost`。
2. 只保留 `raw_token_cost > min_code_token_cost` 的候选压缩字面量。
3. 计算权重（默认 `weight = p(x) * c_raw(x)`，可切换 `weight = p(x)`）。
4. 候选短码池按 `(token_cost, char_length, lexicographic)` 升序搜索。
5. 通过 Trie 过滤前缀冲突，得到 prefix-feasible 短码集合。
6. 将高权重字面量与低成本短码一一配对。
7. 若某对 `gain <= 0`，则跳过分配，保持原样输出。

交换论证（简要）：
- 若有两个字面量 `x_a, x_b`，权重 `w_a >= w_b`，两短码成本 `c_1 <= c_2`。
- 赋值 A：`x_a->c_1, x_b->c_2`；赋值 B：`x_a->c_2, x_b->c_1`。
- A 相对 B 的收益差为 `(w_a-w_b) * (c_2-c_1) >= 0`，故高权重配低成本不劣。

局限：
- 这是高效近似，不保证全局最优。
- 候选池质量与 Tokenizer 特性会影响最终效果。

## 5. 前缀可解码与转义机制

编码容器使用 `escape_prefix`（默认 `__L__`）：

- 命中字典：`value -> "__L__" + code`
- 原文若本身以 `__L__` 开头：转义为 `__L____L__ + suffix`
- 解码时：
  - `__L____L__...` 还原成原文 `__L__...`
  - `__L__<code>` 按码表反查原文

这样可防止原文与短码混淆，保证严格可逆。

## 6. 复杂度分析（默认实现）

- 字段统计：`O(N)`
- 熵/自信息计算：`O(|V|)`
- 候选池搜索（best-first + heap）：约 `O(M log M)`，`M` 为探索候选数
- 前缀检查（Trie 插入）：`O(L)`，`L` 为码字字符长度
- 贪心分配：`O(|V| log |V| + K log K)`

空间主要由计数表、码表、候选缓存组成，约 `O(|V| + K + M)`。

## 7. 运行方式

### 安装

```bash
cd stage3
python -m pip install -r requirements.txt
```

### 构建 codebook + 报告（CLI）

```bash
python -m literal_codec.pipeline.offline_builder \
  --input examples/sample_literals.csv \
  --fields service_name env tag_prefix \
  --output_dir artifacts
```

### 运行 demo

```bash
python examples/build_codebook_demo.py
```

### 运行测试

```bash
pytest -q
```

## 8. 产物说明

- `artifacts/codebook.json`：字段级码表（可用于编码/解码）。
- `artifacts/report.json`：字段统计、收益、覆盖率、汇总指标。

## 9. 未来 B/C 扩展口

当前仅实现 A，且 semantic loss = 0。

- B（近似压缩）预留：
  - `SemanticLossModel` 抽象接口，用于定义 `loss(raw, compressed)`。
- C（在线漂移与切换）预留：
  - `DriftDetector` 抽象接口，用于触发重训练。
  - `CodebookSwitchPolicy` 抽象接口，用于版本切换策略。

可在不破坏现有 A 的前提下，扩展更复杂的近似编码、漂移检测与字典发布链路。
