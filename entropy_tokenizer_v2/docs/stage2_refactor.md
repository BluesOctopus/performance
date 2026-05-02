# Stage2 Refactor Notes

## Goal
- Make Stage2 behavior explicit, configurable, and evaluable without mixing Stage3.

## Single Source of Truth
- Stage2 profile defaults and flags are defined in `config.py`:
  - `STAGE2_DEFAULT_PROFILE`
  - `STAGE2_DEFAULT_MODE`
  - `STAGE2_PROFILE_FLAGS`
- Runtime builder lives in `stage2/config.py` via `build_stage2_config(...)`.

## Mode Split
- `linewise`: clean each non-`<SYN_n>` line independently.
- `blockwise`: clean each contiguous non-`<SYN_n>` block as a whole.
- Implementation is in `stage2/cleaning.py`.

## Profile Split
- `stage2_parseable`: `remove_indentation=False` (parser-friendlier).
- `stage2_aggressive`: `remove_indentation=True` (more aggressive compression).

## Blank-Line Decoupling
- Outer filtering is controlled explicitly by `drop_empty_cleaned_lines`.
- Default in new Stage2 helpers is `False`, so blank-line behavior comes from Stage2 rule itself.

## Stage2-only Eval
- New entry: `eval/eval_stage2_only.py`.
- Outputs include:
  - `all_samples`
  - `original_parseable_subset`
- Output summary always records `mode`, `profile`, and effective flags.
- Optional reconstruction integration:
  - `--with-reconstruction`
  - reuses `llm_reconstruction_eval` (`run_with_precompressed`) instead of duplicating logic
  - reports reconstruction metrics for all samples and original-parseable subset.

## Result Identity
- Stage2-only result directories include:
  - version tag (`stage2_eval_v2_1`)
  - profile
  - mode
  - source tag
  - sample count
  - reconstruction flag (`recon_on` / `recon_off`)

## Why report two subsets
- Real corpora include original parse-invalid samples.
- Without splitting subsets, parseability conclusions for Stage2 are polluted.
