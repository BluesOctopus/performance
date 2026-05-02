"""Stage2 single source of truth for mode/profile config."""

from __future__ import annotations

from dataclasses import dataclass

from config import STAGE2_DEFAULT_MODE, STAGE2_DEFAULT_PROFILE, STAGE2_PROFILE_FLAGS
from lossy_cleaner import CleaningConfig

STAGE2_PROFILES: dict[str, dict[str, bool]] = dict(STAGE2_PROFILE_FLAGS)

STAGE2_ADAPTED_ORDER_LABEL = "pre_safe -> stage1 -> post_surface"
STAGE2_LAYOUT_EXPERIMENTAL_ORDER_LABEL = "pre_safe -> stage1 -> post_layout_encode"


@dataclass(frozen=True)
class Stage2ExecutionPlan:
    """Pre-phase (parseable source) + post-phase (Stage1 output, surface-only)."""

    profile_name: str
    pre_cfg: CleaningConfig
    post_cfg: CleaningConfig
    pre_mode: str
    post_mode: str
    order_label: str = STAGE2_ADAPTED_ORDER_LABEL


def build_stage2_execution_plan(profile: str) -> Stage2ExecutionPlan:
    """
    Build adapted Stage2 plan for ``safe`` or ``aggressive_upper_bound``.

    * Pre: AST docstrings + directive-preserving comment removal + blank/ws; never indent strip.
    * Post: blank/ws only (+ indent strip for aggressive upper bound); no docstrings/comments.
    """
    if profile == "safe":
        pre_cfg = CleaningConfig(
            remove_comments=True,
            remove_blank_lines=True,
            remove_trailing_whitespace=True,
            remove_docstrings=True,
            docstring_removal_mode="safe_only",
            remove_indentation=False,
        )
        post_cfg = CleaningConfig(
            remove_comments=False,
            remove_blank_lines=True,
            remove_trailing_whitespace=True,
            remove_docstrings=False,
            remove_indentation=False,
        )
        return Stage2ExecutionPlan(
            profile_name=profile,
            pre_cfg=pre_cfg,
            post_cfg=post_cfg,
            pre_mode="blockwise",
            post_mode="blockwise",
        )
    if profile == "aggressive_upper_bound":
        pre_cfg = CleaningConfig(
            remove_comments=True,
            remove_blank_lines=True,
            remove_trailing_whitespace=True,
            remove_docstrings=True,
            docstring_removal_mode="safe_only",
            remove_indentation=False,
        )
        post_cfg = CleaningConfig(
            remove_comments=False,
            remove_blank_lines=True,
            remove_trailing_whitespace=True,
            remove_docstrings=False,
            remove_indentation=True,
        )
        return Stage2ExecutionPlan(
            profile_name=profile,
            pre_cfg=pre_cfg,
            post_cfg=post_cfg,
            pre_mode="blockwise",
            post_mode="blockwise",
        )
    raise ValueError(f"unknown adapted stage2 profile: {profile!r}")


@dataclass(frozen=True)
class Stage2Config:
    profile: str
    mode: str
    cleaning: CleaningConfig

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "mode": self.mode,
            "remove_comments": self.cleaning.remove_comments,
            "remove_blank_lines": self.cleaning.remove_blank_lines,
            "remove_trailing_whitespace": self.cleaning.remove_trailing_whitespace,
            "remove_docstrings": self.cleaning.remove_docstrings,
            "docstring_removal_mode": self.cleaning.docstring_removal_mode,
            "remove_indentation": self.cleaning.remove_indentation,
        }


def build_stage2_config(
    *,
    profile: str = STAGE2_DEFAULT_PROFILE,
    mode: str = STAGE2_DEFAULT_MODE,
    overrides: dict | None = None,
) -> Stage2Config:
    if profile not in STAGE2_PROFILES:
        raise ValueError(f"unknown stage2 profile: {profile}")
    if mode not in ("linewise", "blockwise"):
        raise ValueError(f"unknown stage2 mode: {mode}")

    flags = dict(STAGE2_PROFILES[profile])
    if overrides:
        for k, v in overrides.items():
            if k in flags and v is not None:
                flags[k] = bool(v)

    if mode == "linewise" and flags.get("remove_docstrings"):
        raise ValueError("linewise mode does not support remove_docstrings=True")

    return Stage2Config(
        profile=profile,
        mode=mode,
        cleaning=CleaningConfig(**flags),
    )
