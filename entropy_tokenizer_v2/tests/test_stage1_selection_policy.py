import io
import tokenize
from collections import Counter

from marker_count import count_augmented_text_fragment, count_syn_marker
from syntax_compressor import SkeletonCandidate, build_candidate_pool, greedy_mdl_select, score_skeleton_occurrence


class Stage1TestTokenizer:
    """Tokenize-like counter that is sensitive to indentation/newlines."""

    def encode(self, text: str, add_special_tokens: bool = False, allowed_special: str = "all"):
        del add_special_tokens, allowed_special
        out = []
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.ENDMARKER:
                continue
            out.append((tok.type, tok.string))
        return out


def test_common_templates_can_be_selected_when_net_saving_positive() -> None:
    candidates = [
        SkeletonCandidate(
            skeleton="if {0}:",
            frequency=10,
            fixed_tokens=2,
            num_slots=1,
            savings_per_instance=1.2,
            codebook_cost=2,
            mdl_net_benefit=10.0,
            empirical_total_savings=12,
            avg_net_saving=1.2,
            total_net_saving=12,
            effective_total_net_saving=12,
        ),
        SkeletonCandidate(
            skeleton="return {0}",
            frequency=10,
            fixed_tokens=2,
            num_slots=1,
            savings_per_instance=1.0,
            codebook_cost=2,
            mdl_net_benefit=10.0,
            empirical_total_savings=10,
            avg_net_saving=1.0,
            total_net_saving=10,
            effective_total_net_saving=10,
        ),
    ]
    selected = greedy_mdl_select(candidates, N_baseline=80, V0=4)
    selected_set = {c.skeleton for c in selected}
    assert "if {0}:" in selected_set
    assert "return {0}" in selected_set


def test_common_template_rejected_when_net_saving_non_positive() -> None:
    candidates = [
        SkeletonCandidate(
            skeleton="if {0}:",
            frequency=10,
            fixed_tokens=2,
            num_slots=1,
            savings_per_instance=0.0,
            codebook_cost=2,
            mdl_net_benefit=0.0,
            empirical_total_savings=0,
            avg_net_saving=0.0,
            total_net_saving=0,
        )
    ]
    selected = greedy_mdl_select(candidates, N_baseline=80, V0=4)
    assert selected == []


def test_stage1_scoring_uses_augmented_marker_counting() -> None:
    tok = Stage1TestTokenizer()
    s = score_skeleton_occurrence(
        match_text="    if value > 0:\n",
        skeleton="if {0}:",
        slots=["value > 0"],
        marker="<SYN_7>",
        tokenizer=tok,
        tok_type="hf",
    )
    marker_cost = count_syn_marker("<SYN_7>")
    assert marker_cost == 1
    serialized = "<SYN_7> value > 0"
    compressed = count_augmented_text_fragment(serialized, tok, "hf")
    assert s["compressed_cost"] == compressed


def test_stage1_selection_is_deterministic() -> None:
    sources = [
        "def f(a):\n    if a:\n        return a\n",
        "def g(b):\n    for i in b:\n        return i\n",
    ]
    skeleton_counts = Counter({"if {0}:": 2, "return {0}": 2, "for {0} in {1}:": 2})
    tok = Stage1TestTokenizer()
    c1 = build_candidate_pool(skeleton_counts, tok, "hf", sources)
    c2 = build_candidate_pool(skeleton_counts, tok, "hf", sources)
    s1 = [c.skeleton for c in greedy_mdl_select(c1, N_baseline=300, V0=256)]
    s2 = [c.skeleton for c in greedy_mdl_select(c2, N_baseline=300, V0=256)]
    assert s1 == s2
