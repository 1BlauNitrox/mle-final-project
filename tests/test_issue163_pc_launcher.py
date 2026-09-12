"""The PC supervisor uses fixed commands and never overwrites previous output attempts."""

from scripts.run_issue163_pc import command, next_output


def test_command_and_retained_attempts(tmp_path):
    first = next_output(tmp_path, "analysis")
    first.mkdir()
    assert next_output(tmp_path, "analysis").name == "analysis-002"
    cmd = command("run", tmp_path, "--resume")
    assert cmd[1:4] == ["-m", "training.task3_attack_campaign", "run"]
    assert cmd[-1] == "--resume"
    assert str(tmp_path / "binding") in cmd
    assert str(tmp_path / "runs") in cmd
