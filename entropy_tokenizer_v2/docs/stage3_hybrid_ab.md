# Stage3 Hybrid AB (Exact + Lexical Baseline)

`hybrid_ab` 是 Stage3 的实验性双通道路由后端：

- **A 通道（Exact）**：对 `variable` / `attribute` / 精确型 `string` 做请求内 aliasing，目标是近似无损且 `gain > 0` 才替换。
- **B 通道（Lexical）**：只处理自由文本 `string`，使用 bag-of-words + cosine 的词汇聚类基线；满足阈值且净收益为正才替换。
- **fallback**：无法判定、收益不足、风险过高全部回退原文。

## 默认行为（保守）

- 默认 `ET_STAGE3_AB_MODE=exact_only`（gpt4/gpt2 都是）。
- 默认不会启用 B 通道；需要显式 `ET_STAGE3_AB_MODE=hybrid` 且 `ET_STAGE3_AB_ENABLE_B=1`。
- A 通道保持近似无损（可精确 decode）。

## 配置项

- `ET_STAGE3_BACKEND=hybrid_ab`
- `ET_STAGE3_AB_MODE=exact_only|hybrid`（默认 `exact_only`）
- `ET_STAGE3_AB_FREE_TEXT_MIN_CHARS`（默认 `24`）
- `ET_STAGE3_AB_FREE_TEXT_MIN_WORDS`（默认 `4`）
- `ET_STAGE3_AB_B_SIMILARITY_THRESHOLD`（默认 `0.82`，gpt4 默认更严格）
- `ET_STAGE3_AB_B_RISK_THRESHOLD`（默认 `0.72`，gpt4 默认更严格）
- `ET_STAGE3_AB_B_MIN_CLUSTER_SIZE`（默认 `2`）
- `ET_STAGE3_AB_KEY_LIKE_PATTERNS`（key-like 正则，`||` 分隔多条；用于路由器将其归入 A 通道）
- `ET_STAGE3_AB_ENABLE_B`（兼容开关，主控制由 `ET_STAGE3_AB_MODE`）
- `ET_STAGE3_AB_A_MIN_OCC`（默认 `2`）
- `ET_STAGE3_AB_A_MIN_NET_GAIN`（默认 `1`）
- `ET_STAGE3_AB_A_ALIAS_STYLE=short|mnemonic`

## 评测输出

`v2_eval.py` 会输出 A/B 分项：

- `stage3_ab_a_*`：A 候选、选中、intro、sequence saved、effective net
- `stage3_ab_b_*`：B 候选、簇数、使用簇、intro、sequence saved、fallback、avg_similarity
- `stage3_ab_fallback_count`
- `stage3_ab_similarity_kind=lexical_bow_cosine`
- `stage3_ab_b_mode=lexical_free_text_baseline`（B 通道是 lexical baseline）
- `stage3_ab_mode=exact_only|hybrid`

并保持总账一致：

- `stage3_vocab_intro_tokens = stage3_ab_a_intro_tokens + stage3_ab_b_intro_tokens`
- `stage3_component_saved = stage3_ab_a_sequence_saved + stage3_ab_b_sequence_saved`
