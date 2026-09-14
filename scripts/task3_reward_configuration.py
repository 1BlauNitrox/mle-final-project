"""Prospective #175 reward configuration and self-contained agent export."""

import shutil
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def configured_rewards(rewards, kill_reward):
    """Change one shared module dictionary, retaining all import aliases and restoring it."""
    if kill_reward not in (5.0, 20.0):
        raise ValueError("Unregistered elimination reward")
    before = dict(rewards)
    if "KILLED_OPPONENT" not in rewards:
        raise ValueError("Missing native attribution reward")
    try:
        rewards["KILLED_OPPONENT"] = kill_reward
        yield
    finally:
        rewards.clear()
        rewards.update(before)


def export_agent(source, checkpoint, target, kill_reward):
    """Keep original checkpoint bytes and give it a matching self-contained config."""
    import torch

    source, checkpoint, target = map(Path, (source, checkpoint, target))
    payload = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if kill_reward not in (5.0, 20.0) or payload["rewards"]["KILLED_OPPONENT"] != kill_reward:
        raise ValueError("Checkpoint reward provenance mismatch")
    config = (source / "config.py").read_text(encoding="utf-8")
    marker = '"KILLED_OPPONENT": 5.0,'
    if config.count(marker) != 1:
        raise ValueError("Unexpected source reward declaration")
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pt", "logs"))
    (target / "config.py").write_text(
        config.replace(marker, f'"KILLED_OPPONENT": {kill_reward},'), encoding="utf-8"
    )
    shutil.copyfile(checkpoint, target / "checkpoint.pt")
