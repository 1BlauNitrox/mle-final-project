"""Run unchanged supplied Task 3 opponents with isolated, reproducible RNG streams.

This is experiment instrumentation only. Submitted learned agents never import it.
"""

from __future__ import annotations

import os
import random
import runpy
import sys

import numpy as np

SUPPORTED = {"peaceful_agent", "coin_collector_agent"}
SCHEME = "task3_per_slot_v1"


def seeds(root_seed, slot):
    words = np.random.SeedSequence([root_seed, slot, 0x5441534B]).generate_state(3)
    return int(words[0]), (int(words[1]) << 32) | int(words[2])


def wrap_process_event(original, root_seed, slots):
    """Isolate both legacy RNGs per opponent and restore caller state on every exit."""

    def process_event(self, event_name, *event_args):
        if self.code_name not in slots:
            return original(self, event_name, *event_args)
        numpy_seed, python_seed = seeds(root_seed, slots[self.code_name])
        if not hasattr(self, "_task3_rng"):
            self._task3_rng = (
                np.random.RandomState(numpy_seed).get_state(),
                random.Random(python_seed).getstate(),
            )
        previous = np.random.get_state(), random.getstate()
        numpy_seed_function, python_seed_function = np.random.seed, random.seed
        np.random.set_state(self._task3_rng[0])
        random.setstate(self._task3_rng[1])
        try:
            if event_name == "setup":
                # Supplied setup() calls seed(None); bind that entropy request.
                np.random.seed = lambda value=None: numpy_seed_function(
                    numpy_seed if value is None else value
                )
                random.seed = lambda value=None, version=2: python_seed_function(
                    python_seed if value is None else value, version=version
                )
            return original(self, event_name, *event_args)
        finally:
            self._task3_rng = np.random.get_state(), random.getstate()
            np.random.seed, random.seed = numpy_seed_function, python_seed_function
            np.random.set_state(previous[0])
            random.setstate(previous[1])

    return process_event


def main():
    import agents

    root_seed = int(os.environ["BOMBERMAN_AGENT_SEED"])
    if root_seed < 0:
        raise ValueError("Opponent root seed must be non-negative")
    start = sys.argv.index("--agents") + 1
    names = []
    for name in sys.argv[start:]:
        if name.startswith("--"):
            break
        names.append(name)
    opponents = names[1:]
    if len(set(opponents)) != len(opponents) or not set(opponents) <= SUPPORTED:
        raise ValueError("Seeded Task 3 instrumentation requires distinct supported opponents")
    slots = {name: index for index, name in enumerate(names) if index > 0}
    agents.AgentRunner.process_event = wrap_process_event(
        agents.AgentRunner.process_event, root_seed, slots
    )
    sys.argv[0] = "main.py"
    runpy.run_module("main", run_name="__main__")


if __name__ == "__main__":
    main()
