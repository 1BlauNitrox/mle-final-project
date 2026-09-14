"""Read-only, post-hoc issue177 replay/Q-value audit; never trains or selects models."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

DISCOUNT_FACTOR = 0.9


def q_values(state, inputs):
    value = inputs
    for layer in (0, 2, 4):
        value = torch.nn.functional.linear(
            value, state[f"layers.{layer}.weight"], state[f"layers.{layer}.bias"]
        )
        if layer != 4:
            value = torch.relu(value)
    return value


def compare(initial, learned, replay):
    size = len(replay["states"])
    if size == 0:
        raise ValueError("Empty replay cannot diagnose drift")
    idx = torch.linspace(0, size - 1, min(512, size)).long()
    next_states = replay["next_states"][idx]
    masks = replay["next_action_masks"][idx]
    if not masks.any(dim=1).all():
        raise ValueError("No eligible next action")
    with torch.no_grad():
        before = q_values(initial, next_states)
        after = q_values(learned, next_states)
        a = before.masked_fill(~masks, -torch.inf).argmax(1)
        b = after.masked_fill(~masks, -torch.inf).argmax(1)
    present = next_states[:, 26:].abs().sum(1) > 0
    result = {"samples": len(idx), "sample_indices": idx.tolist()}
    for label, select in [
        ("all", torch.ones(len(idx), dtype=torch.bool)),
        ("opponent_present", present),
        ("opponent_absent", ~present),
    ]:
        if not select.any():
            result[label] = {"samples": 0}
            continue
        result[label] = {
            "samples": int(select.sum()),
            "greedy_changed_fraction": float((a[select] != b[select]).float().mean()),
            "mean_q_initial": float(before[select].mean()),
            "mean_q_final": float(after[select].mean()),
            "mean_abs_q_shift": float((after[select] - before[select]).abs().mean()),
            "initial_actions": torch.bincount(a[select], minlength=6).tolist(),
            "final_actions": torch.bincount(b[select], minlength=6).tolist(),
        }
    rewards = replay["rewards"][idx]
    result["rewards"] = {
        "mean": float(rewards.mean()),
        "nonzero_fraction": float((rewards != 0).float().mean()),
        "positive_fraction": float((rewards > 0).float().mean()),
    }
    return result


def td_error_diagnostics(initial_online, initial_target, learned_online,
                         learned_target, replay):
    """Compare signed Bellman TD errors on the same deterministic sample."""
    size = len(replay["states"])
    if size == 0:
        raise ValueError("Empty replay cannot diagnose TD errors")
    idx = torch.linspace(0, size - 1, min(512, size)).long()
    states = replay["states"][idx]
    actions = replay["action_indices"][idx].long()
    rewards = replay["rewards"][idx]
    next_states = replay["next_states"][idx]
    terminals = replay["terminals"][idx].bool()
    masks = replay["next_action_masks"][idx].bool()
    if not masks.any(dim=1).all():
        raise ValueError("No eligible next action")

    def errors(online, target):
        current = q_values(online, states).gather(1, actions[:, None]).squeeze(1)
        next_values = q_values(target, next_states).masked_fill(~masks, -torch.inf)
        targets = rewards + DISCOUNT_FACTOR * (~terminals).float() * next_values.max(1).values
        return targets - current

    with torch.no_grad():
        before = errors(initial_online, initial_target)
        after = errors(learned_online, learned_target)
    sign_before = torch.sign(before)
    sign_after = torch.sign(after)
    return {
        "samples": len(idx),
        "sample_indices": idx.tolist(),
        "mean_signed_td_initial": float(before.mean()),
        "mean_signed_td_final": float(after.mean()),
        "mean_abs_td_initial": float(before.abs().mean()),
        "mean_abs_td_final": float(after.abs().mean()),
        "td_sign_changed_fraction": float((sign_before != sign_after).float().mean()),
        "td_sign_initial": torch.bincount((sign_before + 1).long(), minlength=3).tolist(),
        "td_sign_final": torch.bincount((sign_after + 1).long(), minlength=3).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing diagnostic output")
    import psutil

    started, cpu = time.monotonic(), time.process_time()
    torch.set_num_threads(1)
    root = args.root.resolve()
    initial = torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
    state = json.loads((root / "training-state.json").read_text())
    if state["status"] != "completed" or len(state["completed"]) != 10:
        raise ValueError("Need complete issue175 training")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    result = {
        "scope": "post_hoc_diagnostic_no_performance_selection",
        "initial_sha256": sha(root / "initial.pt"),
        "models": {},
    }
    for name, relative in state["completed"].items():
        if time.monotonic() - started > 300 or psutil.Process().memory_info().rss > 1024**3:
            raise RuntimeError("Diagnostic resource ceiling reached")
        path = root / relative / "checkpoint.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        digest = sha(path)
        if digest != json.loads(path.with_name("result.json").read_text())["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
        initial_online = initial["learner_state"]["online_network"]
        initial_target = initial["learner_state"]["target_network"]
        learned_online = payload["learner_state"]["online_network"]
        learned_target = payload["learner_state"]["target_network"]
        row = compare(
            initial_online,
            learned_online,
            payload["replay_state"],
        )
        row["td_errors"] = td_error_diagnostics(
            initial_online,
            initial_target,
            learned_online,
            learned_target,
            payload["replay_state"],
        )
        row["checkpoint_sha256"] = digest
        row["optimizer_updates"] = payload["learner_state"]["update_steps"]
        result["models"][name] = row
    result["resources"] = {
        "cpu_seconds": time.process_time() - cpu,
        "wall_seconds": time.monotonic() - started,
        "rss_bytes": psutil.Process().memory_info().rss,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {p: v[p] for p in ["opponent_present", "opponent_absent"]}
                for k, v in result["models"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
