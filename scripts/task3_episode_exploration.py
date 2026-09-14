"""Prospective phase-C temporal exploration matrix and selection-free readiness rule."""

from statistics import mean

from scripts.probe_task3_exploration import INPUT, SOURCE

SEEDS = list(range(1682101, 1682121))
RANDOM_EPISODES = [1682105, 1682111, 1682115, 1682120]


def validate(config):
    import json

    from scripts.probe_task3_exploration import CONFIG
    from scripts.probe_task3_exploration import validate as validate_a

    original = json.loads(CONFIG.read_text())
    validate_a(original)
    for field in (
        "runtime_source",
        "input_sha256",
        "input_size_bytes",
        "parent_sha256",
        "limits",
        "scenario",
        "opponents",
        "maximum_steps",
        "training_episodes",
        "agent_seed_offset",
    ):
        if config[field] != original[field]:
            raise ValueError("Phase-C shared control changed: " + field)
    if (
        config["runtime_source"] != SOURCE
        or config["input_sha256"] != INPUT
        or config["phase"] != "C"
        or config["world_seeds"] != SEEDS
        or config["random_episode_world_seeds"] != RANDOM_EPISODES
        or config["episodes"] != 40
        or config["nominal_exploration"] != 0.2
        or config["arms"] != ["stepwise", "episode_mixture"]
    ):
        raise ValueError("Unregistered temporal exploration matrix")
    if config["readiness_filter"] != {
        "mean_survival_ratio_min": 2.0,
        "strictly_longer_pairs_min": 14,
        "coin_difference_min": 0.0,
        "attack_episode_count_min": 2,
        "self_kill_rate_max": 0.5,
    }:
        raise ValueError("Unregistered temporal filter")


def decide(jobs):
    expected = {(seed, e) for seed in SEEDS for e in (0.2, 1.0 if seed in RANDOM_EPISODES else 0.0)}
    if len(jobs) != 40 or {(j["world_seed"], j["epsilon"]) for j in jobs} != expected:
        raise ValueError("Incomplete/duplicate temporal matrix")
    if any(j["status"] != "completed" or not j["weights_unchanged"] for j in jobs):
        raise ValueError("Failed/mutable temporal probe")
    control = sorted((j for j in jobs if j["epsilon"] == 0.2), key=lambda j: j["world_seed"])
    mixture = sorted((j for j in jobs if j["epsilon"] != 0.2), key=lambda j: j["world_seed"])
    means = {
        "stepwise": mean(j["survival_steps"] for j in control),
        "episode_mixture": mean(j["survival_steps"] for j in mixture),
    }
    longer = sum(
        b["survival_steps"] > a["survival_steps"] for a, b in zip(control, mixture, strict=True)
    )
    coins = mean(j["coins"] for j in mixture) - mean(j["coins"] for j in control)
    attacks = sum(j["attack_steps"] > 0 for j in mixture)
    self_kills = mean(j["self_kills"] for j in mixture)
    gates = {
        "survival_ratio": means["episode_mixture"] >= 2 * means["stepwise"],
        "paired_survival": longer >= 14,
        "coins": coins >= 0,
        "attack_exposure": attacks >= 2,
        "self_kills": self_kills <= 0.5,
    }
    return {
        "scope": "temporal_mechanism_only_not_task3_success",
        "learning_pilot_ready": all(gates.values()),
        "gates": gates,
        "mean_survival_steps": means,
        "longer_pairs": longer,
        "mean_coin_difference": coins,
        "mixture_attack_episodes": attacks,
        "mixture_self_kill_rate": self_kills,
        "selected_checkpoint": None,
        "long_training_authorized": False,
    }
