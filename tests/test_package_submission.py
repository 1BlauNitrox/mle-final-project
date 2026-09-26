"""The submission zip holds exactly what the agent needs to play, and nothing else."""

from __future__ import annotations

import ast
from pathlib import Path

from scripts import package_submission as package

ROOT = Path(__file__).resolve().parents[1]
AGENT = ROOT / "agent_code/Bomb-omb"


def test_the_zip_list_leaves_out_training_code_logs_and_metadata(tmp_path):
    (tmp_path / "features").mkdir()
    for name in ("__init__.py", "assemble.py"):
        (tmp_path / "features" / name).write_text("", encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "Bomb-omb.log").write_text("x", encoding="utf-8")
    names = package.shipped_names(tmp_path)
    for unwanted in (
        "train.py",
        "rewards.py",
        "migration.py",
        "artifact.json",
        "logs/Bomb-omb.log",
    ):
        assert unwanted not in names
    assert {"callbacks.py", "checkpoint.pt", "requirements.txt", "features/assemble.py"} <= set(
        names
    )


def test_everything_the_shipped_agent_imports_is_shipped():
    # Follow relative imports from callbacks.py; every module reached must be on the list.
    shipped = set(package.CODE_FILES) | {
        f"features/{p.name}" for p in (AGENT / "features").glob("*.py")
    }
    seen, queue = set(), ["callbacks.py"]
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        package_dir = Path(module).parent
        for node in ast.walk(ast.parse((AGENT / module).read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level:
                base = package_dir
                for _ in range(node.level - 1):
                    base = base.parent
                target = base / (node.module or "").replace(".", "/")
                candidate = f"{target.as_posix()}.py".lstrip("./")
                package_init = f"{target.as_posix()}/__init__.py".lstrip("./")
                if (AGENT / candidate).is_file():
                    queue.append(candidate)
                elif (AGENT / package_init).is_file():
                    queue.append(package_init)
    assert seen <= shipped, sorted(seen - shipped)
    assert not {"train.py", "rewards.py", "migration.py"} & seen
