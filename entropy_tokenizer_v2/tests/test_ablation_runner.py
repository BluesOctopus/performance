import csv
import subprocess
import sys
from pathlib import Path


def test_ablation_runner_generates_complete_deterministic_table(tmp_path) -> None:
    summary1 = tmp_path / "ablation_summary_1.csv"
    summary2 = tmp_path / "ablation_summary_2.csv"
    per_file1 = tmp_path / "per_file_1.csv"
    per_file2 = tmp_path / "per_file_2.csv"
    stage1_sel1 = tmp_path / "stage1_sel_1.csv"
    stage1_sel2 = tmp_path / "stage1_sel_2.csv"
    j1a = tmp_path / "s1v1.json"
    j1b = tmp_path / "s1v2.json"
    j3a = tmp_path / "s3v1.json"
    j3b = tmp_path / "s3v2.json"
    cmd1 = [
        sys.executable,
        "-m",
        "scripts.run_ablation",
        "--summary-output",
        str(summary1),
        "--per-file-output",
        str(per_file1),
        "--stage1-selected-output",
        str(stage1_sel1),
        "--stage1-vocab-json",
        str(j1a),
        "--stage3-vocab-json",
        str(j3a),
    ]
    cmd2 = [
        sys.executable,
        "-m",
        "scripts.run_ablation",
        "--summary-output",
        str(summary2),
        "--per-file-output",
        str(per_file2),
        "--stage1-selected-output",
        str(stage1_sel2),
        "--stage1-vocab-json",
        str(j1b),
        "--stage3-vocab-json",
        str(j3b),
    ]

    p1 = subprocess.run(cmd1, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])
    p2 = subprocess.run(cmd2, capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1])

    assert p1.returncode == 0, p1.stdout + p1.stderr
    assert p2.returncode == 0, p2.stdout + p2.stderr
    assert summary1.exists()
    assert summary2.exists()
    assert per_file1.exists()
    assert per_file2.exists()
    assert stage1_sel1.exists()
    assert stage1_sel2.exists()
    assert j1a.read_text(encoding="utf-8") == j1b.read_text(encoding="utf-8")

    expected_fields = {
        "experiment_name",
        "num_files",
        "baseline_sequence_tokens",
        "baseline_vocab_intro_tokens",
        "baseline_effective_total_tokens",
        "stage1_sequence_tokens",
        "stage1_vocab_intro_tokens",
        "stage1_effective_total_tokens",
        "stage2_sequence_tokens",
        "stage2_vocab_intro_tokens",
        "stage2_effective_total_tokens",
        "stage3_sequence_tokens",
        "stage3_vocab_intro_tokens",
        "stage3_effective_total_tokens",
        "token_count_mode",
        "vocab_cost_mode",
        "vocab_cost_scope",
        "stage1_effective_reduction_ratio",
        "stage3_effective_reduction_ratio",
        "compressed_sequence_tokens",
        "token_reduction_ratio",
        "parse_success_rate",
        "identity_preservation_rate",
        "selected_skeleton_count",
        "selected_skeletons",
        "stage1_total_net_saving",
        "notes",
    }
    with summary1.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert set(reader.fieldnames) == expected_fields
        assert len(rows) == 6
        stage1_only = next(r for r in rows if r["experiment_name"] == "stage1_only")
        assert int(stage1_only["baseline_sequence_tokens"]) >= int(stage1_only["stage1_sequence_tokens"])
        assert all(r["identity_preservation_rate"] != "n/a" for r in rows)

    with per_file1.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert "stage1_text" in reader.fieldnames
        assert "stage2_text" in reader.fieldnames
        assert "stage3_text" in reader.fieldnames
        assert "original_sequence_tokens" in reader.fieldnames
        assert len(rows) == 42

    assert summary1.read_text(encoding="utf-8") == summary2.read_text(encoding="utf-8")
    assert per_file1.read_text(encoding="utf-8") == per_file2.read_text(encoding="utf-8")
    assert stage1_sel1.read_text(encoding="utf-8") == stage1_sel2.read_text(encoding="utf-8")

    with stage1_sel1.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert "marker" in reader.fieldnames
        assert "candidate_occurrences" in reader.fieldnames
        assert "replaced_occurrences" in reader.fieldnames
        assert "skipped_nonpositive_occurrences" in reader.fieldnames
        assert "effective_total_net_saving" in reader.fieldnames
        assert len(rows) >= 0
