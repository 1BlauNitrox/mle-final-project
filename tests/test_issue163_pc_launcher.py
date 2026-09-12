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


def test_supervisor_cleanup_only_uses_owned_tree(monkeypatch):
    from scripts import run_issue163_pc as launcher

    killed = []

    class Child:
        def __init__(self, pid):
            self.pid = pid

        def kill(self):
            killed.append(self.pid)

    class Parent(Child):
        def children(self, recursive):
            assert recursive
            return [Child(22)]

    class Handle:
        pid = 11

        def poll(self):
            return None

    monkeypatch.setattr(launcher.psutil, "Process", lambda pid: Parent(pid))
    monkeypatch.setattr(launcher.psutil, "wait_procs", lambda *a, **k: None)
    launcher.stop_owned_process(Handle())
    assert killed == [22, 11]


def test_hardware_record_has_fixed_serial_contract():
    from scripts.run_issue163_pc import hardware_record

    record = hardware_record("owner decision")
    assert record["training_workers"] == record["evaluation_workers"] == 1
    assert record["total_memory_bytes"] > 0
    assert record["platform"] and record["python"] and record["processor"]
