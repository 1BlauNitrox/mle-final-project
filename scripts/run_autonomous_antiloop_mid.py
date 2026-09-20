"""Bind iteration 5 to its registration and delegate to the verified runner."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_autonomous_antiloop as runner  # noqa: E402
from scripts.curriculum_io import sha  # noqa: E402

CONFIG = ROOT / "experiments/2026-09-20-autonomous-antiloop-mid/config.json"


def sources():
    paths = [CONFIG, CONFIG.with_name("seed-audit.json")]
    paths += [
        ROOT / "scripts" / name
        for name in (
            "run_autonomous_antiloop_mid.py",
            "run_autonomous_antiloop.py",
            "analyze_autonomous_antiloop.py",
            "audit_autonomous_antiloop_mid_seeds.py",
            "audit_autonomous_antiloop_seeds.py",
            "curriculum_io.py",
            "teacher_loop_episode.py",
            "watch_hunting_curriculum.py",
        )
    ]
    paths.append(ROOT / "training/hunting_curriculum.py")
    paths.extend((ROOT / "agent_code/DagobertDuckDQNAntiLoop").rglob("*.py"))
    return {path.relative_to(ROOT).as_posix(): sha(path) for path in paths}


def main():
    runner.CONFIG = CONFIG
    runner.sources = sources
    runner.__file__ = __file__
    runner.main()


if __name__ == "__main__":
    main()
