# Stage3 Hybrid AB (Exact + Semantic)

`hybrid_ab` 是 Stage3 的双通道路由后端：

- **A 通道（Exact）**：对 `variable` / `attribute` / 精确型 `string` 做请求内 aliasing，目标是近似无损且 `gain > 0` 才替换。
- **B 通道（Semantic）**：只处理自由文本 `string`，做轻量语义聚类，满足相似度/风险阈值且净收益为正才替换。
- **fallback**：无法判定、收益不足、风险过高全部回退原文。

## 为什么需要分流

单一 Stage3 策略会把精确符号与自由文本混在一起，导致收益和风险都不可控。`hybrid_ab` 强制按 literal 语义类型分层：

- 精确对象走 A：保守、可回放（A 可精确 decode）。
- 自由文本走 B：可控有损，按阈值严格筛选。

## 配置项

- `ET_STAGE3_BACKEND=hybrid_ab`
- `ET_STAGE3_AB_FREE_TEXT_MIN_CHARS`（默认 `24`）
- `ET_STAGE3_AB_FREE_TEXT_MIN_WORDS`（默认 `4`）
- `ET_STAGE3_AB_B_SIMILARITY_THRESHOLD`（默认 `0.82`，gpt4 默认更严格）
- `ET_STAGE3_AB_B_RISK_THRESHOLD`（默认 `0.72`，gpt4 默认更严格）
- `ET_STAGE3_AB_B_MIN_CLUSTER_SIZE`（默认 `2`）
- `ET_STAGE3_AB_ENABLE_B`（默认 `1`，置 `0` 可只跑 A）

## 评测输出

`v2_eval.py` 会输出 A/B 分项：

- `stage3_ab_a_*`：A 候选、选中、intro、sequence saved、effective net
- `stage3_ab_b_*`：B 候选、簇数、使用簇、intro、sequence saved、fallback、avg_similarity
- `stage3_ab_fallback_count`

并保持总账一致：

- `stage3_vocab_intro_tokens = stage3_ab_a_intro_tokens + stage3_ab_b_intro_tokens`
- `stage3_component_saved = stage3_ab_a_sequence_saved + stage3_ab_b_sequence_saved`
