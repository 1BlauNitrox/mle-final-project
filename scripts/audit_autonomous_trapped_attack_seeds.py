"""Run the opaque seed audit for the trapped-opponent attack comparison."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_autonomous_antiloop_seeds as audit  # noqa: E402

audit.EXPERIMENT = audit.ROOT / "experiments/2026-09-21-autonomous-trapped-attack"
audit.OWN = "experiments/2026-09-21-autonomous-trapped-attack/"


if __name__ == "__main__":
    audit.main()
