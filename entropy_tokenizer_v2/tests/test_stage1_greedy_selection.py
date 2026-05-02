from syntax_compressor import SkeletonCandidate, greedy_mdl_select


def test_greedy_selection_keeps_searching_after_rejected_candidate() -> None:
    candidates = [
        SkeletonCandidate(
            skeleton="bad",
            frequency=10,
            fixed_tokens=1,
            num_slots=1,
            savings_per_instance=0.1,
            codebook_cost=10,
            mdl_net_benefit=-1.0,
            empirical_total_savings=1,
            effective_total_net_saving=-1,
        ),
        SkeletonCandidate(
            skeleton="good",
            frequency=10,
            fixed_tokens=1,
            num_slots=1,
            savings_per_instance=9.0,
            codebook_cost=1,
            mdl_net_benefit=10.0,
            empirical_total_savings=90,
            effective_total_net_saving=90,
        ),
    ]

    selected = greedy_mdl_select(candidates, N_baseline=100, V0=2)
    assert [c.skeleton for c in selected] == ["good"]
