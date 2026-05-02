# Offline Stage1/Stage3 Diagnostics

This branch keeps Stage1/Stage3 offline diagnostics isolated from training, HumanEval, and historical experiment outputs.

## Supported tokenizers

- `gpt4`
- `cl100k_base`
- `Qwen/Qwen2.5-Coder-1.5B`

## Supported Stage2 profiles

- `safe`
- `aggressive`

`aggressive` is the default for Stage2 baseline comparisons.

## Minimal smoke run

```bash
python tools/build_py_chunks.py --input <sample_dir> --output tmp/chunks.jsonl --tokenizer <tokenizer>
python analysis/stage1_roundtrip_eval.py --chunks tmp/chunks.jsonl --out tmp/stage1.csv --tokenizer <tokenizer>
python analysis/stage3_gain_eval.py --chunks tmp/chunks.jsonl --out tmp/stage3.csv --tokenizer <tokenizer> --stage2-profile aggressive
python analysis/run_offline_stage_ablation.py --chunks tmp/chunks.jsonl --out_dir tmp/ablation --tokenizer <tokenizer> --stage2-profile aggressive
```

## Stage1 marker stress smoke

Fixed source snippets live in:

- `entropy_tokenizer_v2/eval/data/stage1_marker_stress_samples.json`

One simple local workflow is:

```bash
python tools/build_py_chunks.py --input tmp/stage1_marker_stress_src --output tmp/stage1_marker_stress_chunks.jsonl --tokenizer Qwen/Qwen2.5-Coder-1.5B
python analysis/run_stage1_marker_ablation.py --chunks tmp/stage1_marker_stress_chunks.jsonl --out_dir tmp/stage1_marker_stress --tokenizer Qwen/Qwen2.5-Coder-1.5B --stage2-profile aggressive
```

## Alpha-renaming stress smoke

Fixed source snippets live in:

- `entropy_tokenizer_v2/eval/data/alpha_rename_stress_samples.json`

One simple local workflow is:

```bash
python tools/build_py_chunks.py --input tmp/alpha_rename_stress_src --output tmp/alpha_rename_stress_chunks.jsonl --tokenizer Qwen/Qwen2.5-Coder-1.5B
python analysis/run_alpha_rename_oracle_eval.py --chunks tmp/alpha_rename_stress_chunks.jsonl --out_dir tmp/alpha_rename_stress --tokenizer Qwen/Qwen2.5-Coder-1.5B --stage2-profile aggressive
```

## Notes

- `codebook_accounting_mode` defaults to `per_chunk` for conservative offline accounting.
- `global_once` is supported for summary reporting when you want a corpus-once codebook view.
- Stage3 decode fields are never faked. When a deterministic decoder is unavailable, outputs report:
  - `decode_success=false`
  - `roundtrip_ok=false`
  - `ast_equivalent=false`
  - `error_type=stage3_decoder_missing`
