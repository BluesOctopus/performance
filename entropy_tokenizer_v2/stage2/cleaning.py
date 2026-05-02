"""Stage2 cleaning implementations with explicit linewise/blockwise modes."""

from __future__ import annotations

from lossy_cleaner import CleaningConfig, CleaningStats, clean_code
from markers import is_syn_line

STAGE2_PRE_BLOCKWISE = "blockwise"
STAGE2_POST_BLOCKWISE = "blockwise"


def _validate_mode_config(mode: str, cfg: CleaningConfig) -> None:
    if mode == "linewise" and cfg.remove_docstrings:
        raise ValueError("linewise mode does not support remove_docstrings=True")


def clean_stage2_linewise(
    text: str,
    cfg: CleaningConfig,
    *,
    drop_empty_cleaned_lines: bool = False,
    path: str | None = None,
) -> tuple[str, CleaningStats]:
    _validate_mode_config("linewise", cfg)
    out_lines: list[str] = []
    total_stats = CleaningStats()
    for line in text.splitlines():
        if is_syn_line(line):
            out_lines.append(line.rstrip())
            continue
        cleaned_line, s = clean_code(line, cfg, path=path)
        total_stats = total_stats + s
        if drop_empty_cleaned_lines:
            if cleaned_line.strip():
                out_lines.append(cleaned_line)
        else:
            out_lines.append(cleaned_line)
    return "\n".join(out_lines), total_stats


def clean_stage2_blockwise(
    text: str,
    cfg: CleaningConfig,
    *,
    path: str | None = None,
) -> tuple[str, CleaningStats]:
    out_lines: list[str] = []
    total_stats = CleaningStats()
    block_lines: list[str] = []

    def flush_block() -> None:
        nonlocal total_stats
        if not block_lines:
            return
        block_text = "\n".join(block_lines)
        cleaned_block, s = clean_code(block_text, cfg, path=path)
        total_stats = total_stats + s
        out_lines.extend(cleaned_block.split("\n"))
        block_lines.clear()

    for line in text.splitlines():
        if is_syn_line(line):
            flush_block()
            out_lines.append(line.rstrip())
        else:
            block_lines.append(line)

    flush_block()
    return "\n".join(out_lines), total_stats


def stage2_clean_skip_syn_and_stats(
    text: str,
    cfg: CleaningConfig,
    *,
    mode: str = "linewise",
    drop_empty_cleaned_lines: bool = False,
    path: str | None = None,
) -> tuple[str, CleaningStats]:
    _validate_mode_config(mode, cfg)
    if mode == "linewise":
        return clean_stage2_linewise(
            text,
            cfg,
            drop_empty_cleaned_lines=drop_empty_cleaned_lines,
            path=path,
        )
    if mode == "blockwise":
        return clean_stage2_blockwise(text, cfg, path=path)
    raise ValueError(f"unknown stage2 mode: {mode}")


def stage2_clean_skip_syn(
    text: str,
    cfg: CleaningConfig,
    *,
    mode: str = "linewise",
    drop_empty_cleaned_lines: bool = False,
    path: str | None = None,
) -> str:
    cleaned, _ = stage2_clean_skip_syn_and_stats(
        text,
        cfg,
        mode=mode,
        drop_empty_cleaned_lines=drop_empty_cleaned_lines,
        path=path,
    )
    return cleaned


def run_stage2_pre_safe(
    text: str,
    cfg: CleaningConfig,
    *,
    path: str | None = None,
) -> tuple[str, CleaningStats]:
    """Parseable-source phase: docstrings, comments, blanks, ws; never strip indent."""
    _validate_mode_config(STAGE2_PRE_BLOCKWISE, cfg)
    if cfg.remove_indentation:
        raise ValueError("run_stage2_pre_safe: remove_indentation must be False")
    return clean_stage2_blockwise(text, cfg, path=path)


def run_stage2_post_surface(
    text: str,
    cfg: CleaningConfig,
    *,
    path: str | None = None,
) -> tuple[str, CleaningStats]:
    """After Stage1: surface-only; no AST docstrings or comment stripping."""
    if cfg.remove_docstrings:
        raise ValueError("run_stage2_post_surface: remove_docstrings must be False")
    if cfg.remove_comments:
        raise ValueError("run_stage2_post_surface: remove_comments must be False")
    _validate_mode_config(STAGE2_POST_BLOCKWISE, cfg)
    return clean_stage2_blockwise(text, cfg, path=path)
