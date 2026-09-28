"""Tests for the Issue #230 confirmation plot."""

from training.plot_issue230_freeze import load_summary, plot_confirmation


def test_confirmation_plot_is_created(tmp_path) -> None:
    rows = load_summary()
    output = tmp_path / "confirmation.png"

    plot_confirmation(rows, output)

    assert len(rows) == 6
    assert output.is_file()
    assert output.stat().st_size > 0
