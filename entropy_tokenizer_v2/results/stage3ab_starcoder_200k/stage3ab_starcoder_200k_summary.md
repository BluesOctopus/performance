# stage3ab_starcoder_200k summary

- **dataset**: `zhensuuu/starcoderdata_100star_py`
- **tokenizer**: `gpt4`
- **baseline token budget**: 200000
- **actual baseline tokens (sum raw encode)**: 200000
- **n_sources**: 132

## Token chain (corpus sums, same counters as v2_eval / apply_pipeline)

| Stage | Tokens |
|------|--------|
| Baseline (raw source) | 200,000 |
| After Stage1 (syntax) | 198,187 |
| After Stage2 (cleaning) | 171,495 |
| After Stage3 (sequence final) | 169,735 |
| Effective total (seq + S1voc + S3voc) | 170,757 |

## Percentages (vs baseline)

- **sequence_reduction_pct**: 15.1325%
- **effective_total_reduction_pct**: 14.6215%
- **syntax_pct** (Stage1): 0.9065%
- **cleaning_pct** (Stage2): 13.3460%
- **replacement_pct** ((Stage2out−Stage3out)/baseline): 0.8800%

## Vocab intro

- stage1_vocab_intro_tokens: 428
- stage3_vocab_intro_tokens: 594
