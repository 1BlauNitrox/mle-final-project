"""Run the opaque seed audit for the one-worker iteration-5 rerun."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import audit_autonomous_antiloop_seeds as audit  # noqa: E402

audit.EXPERIMENT = audit.ROOT / "experiments/2026-09-20-autonomous-antiloop-mid-serial"
audit.OWN = "experiments/2026-09-20-autonomous-antiloop-mid-serial/"


if __name__ == "__main__":
    audit.main()
