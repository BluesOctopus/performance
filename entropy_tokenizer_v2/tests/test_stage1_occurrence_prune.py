from unittest.mock import patch

from syntax_compressor import SkeletonCandidate, compress_source_syntax


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False, allowed_special="all"):
        del add_special_tokens, allowed_special
        return text.split()


def test_selected_skeleton_skips_nonpositive_occurrence() -> None:
    source = "def f():\n    return total\n    return other\n"
    selected = [
        SkeletonCandidate(
            skeleton="return {0}",
            frequency=2,
            fixed_tokens=2,
            num_slots=1,
            savings_per_instance=1.0,
            codebook_cost=2,
            mdl_net_benefit=1.0,
            empirical_total_savings=2,
            avg_net_saving=1.0,
            total_net_saving=2,
            effective_total_net_saving=2,
            selected=True,
        )
    ]
    tok = TinyTokenizer()

    def _oc(seq_net: int):
        return {
            "baseline_sequence_tokens": 0,
            "compressed_sequence_tokens": 0,
            "sequence_net_saving": seq_net,
            "marker_sequence_tokens": 1,
            "slot_sequence_tokens": 0,
            "compressed_text": "",
        }

    # First occurrence non-positive, second positive.
    with patch(
        "syntax_compressor.estimate_stage1_occurrence_sequence_cost",
        side_effect=[_oc(0), _oc(2)],
    ):
        out, stats = compress_source_syntax(
            source,
            selected,
            tokenizer=tok,
            tok_type="hf",
            prune_nonpositive=True,
            return_stats=True,
        )

    assert out != ""
    sk = "return {0}"
    assert stats[sk]["candidate_occurrences"] == 2
    assert stats[sk]["replaced_occurrences"] == 1
    assert stats[sk]["skipped_nonpositive_occurrences"] == 1
