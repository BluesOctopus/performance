# stage3ab_starcoder_200k 实验报告

## 1. 总压缩率

- **sequence_reduction_pct**（全序列相对 baseline）: **15.1325%**
- **effective_total_reduction_pct**（含 Stage1+Stage3 词表 intro）: **14.6215%**

## 2. Stage1 / Stage2 / Stage3 各贡献多少（占 baseline 比例）

- **syntax_pct（Stage1）**: 0.9065%
- **cleaning_pct（Stage2）**: 13.3460%
- **replacement_pct（仅 Stage3 序列段： (Stage2输出−Stage3输出)/baseline）**: 0.8800%

## 3. Stage3 内部 A 真实压了多少（pipeline token 计数）

- A **input**（Stage2 输出 token 总和）: 171495
- A **output**（A+guardrail 后）: 170315
- A **saved_tokens** (input−output): 1180
- A **intro_tokens**（请求内求和）: 439
- A **net_saved_tokens** (saved−intro): 741

## 4. Stage3 内部 B 真实压了多少

- B **input**（A 输出 token 总和）: 170315
- B **output**（最终 Stage3）: 169735
- B **saved_tokens**: 580
- B **intro_tokens**: 155
- B **net_saved_tokens**: 425

## 5. 主力是不是 A？

- **是。** A 的序列 saved（1180）占 Stage3 总序列 saved（1760）的主体；B saved=580。

## 6. B 通道主因 / 瓶颈归类

- **判断**: B 有净序列收益（真实 saved_tokens=580）；funnel 上 formed=649，too_small 拒 634，最终入选 cluster=6，相对 A（saved=1180）仍偏弱。
- **telemetry**: visible=685, clusters_formed=649, too_small=634, sim/q=0, intro=9, selected=6

## 7. 是否支持「主要靠 A，B 还没起量」？

- **支持。** 以 **真实 token 计数**计：A saved=1180，B saved=580，Stage3 序列段合计 1760（应与 corpus 级 replacement_saved=1760 一致）。telemetry 估算字段 A/B sequence_saved=2803/536 与上式可能不完全同口径，本报告以 input−output 为准。

## 生效配置（本实验）

- stage2_profile=stage2_hybrid_ab_aggressive, stage2_mode=blockwise, resolution=hybrid_ab_default
- stage3_ab_mode=hybrid
