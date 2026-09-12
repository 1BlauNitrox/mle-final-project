"""Fixed synthetic-batch timing; no trained artifact or efficacy claim."""

import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_code.DagobertDuckDQNTask3.config import DEFAULT_CONFIG  # noqa: E402
from agent_code.DagobertDuckDQNTask3.model import DQNLearner  # noqa: E402
from agent_code.DagobertDuckDQNTask3.replay import ReplayBatch  # noqa: E402

torch.set_num_threads(1)
rng = np.random.default_rng(1509901)
batch = ReplayBatch(
    rng.normal(size=(64, 39)).astype(np.float32),
    rng.integers(0, 6, 64, dtype=np.int64),
    rng.normal(size=64).astype(np.float32),
    rng.normal(size=(64, 39)).astype(np.float32),
    np.array([False, True] * 32),
    np.ones((64, 6), dtype=bool),
)
records = {"standard": [], "double": []}
started = time.monotonic()
for replica in range(3):
    for mode in (False, True) if replica % 2 == 0 else (True, False):
        learner = DQNLearner(
            config=replace(DEFAULT_CONFIG, double_dqn=mode, action_masking=True), seed=1509901
        )
        for _ in range(25):
            learner.train_batch(batch)
        before = time.perf_counter()
        for _ in range(500):
            assert time.monotonic() - started < 120
            learner.train_batch(batch)
        records["double" if mode else "standard"].append((time.perf_counter() - before) / 500)
result = {
    "scope": "synthetic_mechanics_timing_not_efficacy",
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "threads": 1,
    "source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "updates_per_repeat": 500,
    "warmup_updates": 25,
    "seconds_per_update": records,
    "ratio_of_medians": statistics.median(records["double"])
    / statistics.median(records["standard"]),
}
print("No artifact or previous timing record is overwritten.")
print(json.dumps(result, indent=2))
