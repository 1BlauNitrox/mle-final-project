"""Run the opaque seed audit for autonomous anti-loop iteration 3."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_autonomous_antiloop_seeds as audit  # noqa: E402

audit.EXPERIMENT = audit.ROOT / "experiments/2026-09-20-autonomous-antiloop-persistent"
audit.OWN = "experiments/2026-09-20-autonomous-antiloop-persistent/"


if __name__ == "__main__":
    audit.main()
