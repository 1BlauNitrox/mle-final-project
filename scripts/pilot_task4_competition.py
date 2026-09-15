"""Registered Task 3 approach pilots, adapted from the #175 runner at c4ddfa4.

Four profiles share this runner, each varying exactly one factor.
`approach-shaping` compares three opponent-approach potential scales;
`update-cadence` re-runs the #178 update interval contrast at higher power;
`compensated-kill-reward` restores a strong elimination signal under the
approach pressure that raised exposure most; and
`opponent-shrinkage` sweeps an L2 coefficient on the opponent columns, which
dials continuously between training them freely and leaving them at the
reference. All four keep the #175 frozen-inheritance restriction, the #168
episode-mixture exploration schedule and the unchanged reference as a
no-training incumbent.

Training and evaluation jobs run concurrently: the handout permits
multiprocessing during training, and evaluation is read-only replay of fixed
checkpoints on fixed seeds. That replay was described here as deterministic
however many run at once, and the dose-response comparison showed it is not.
Four of its 3,040 repeated evaluations disagreed, three of them on the
*unchanged* reference, each losing or gaining a single action in a 400-step
episode before diverging - a step exceeding its decision-time budget while the
machine was loaded. Concurrency therefore changes throughput and, rarely,
outcome; the `behavioral_repeats` gate is what detects it, and an evaluation
stage wants an otherwise idle machine. Decision latency is for the same reason
*not* taken from the contended evaluation stage: a separate serial `latency`
stage replays a registered subset on an idle machine, and only that stage feeds
the latency gate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.task3_pilot_resources import (  # noqa: E402 - support direct CLI execution
    network_sha,
    sample_tree,
    sha,
    stop_owned,
)

INPUT = "99144d1688f66dcc6369d3efc02b2b9d5755cc8d21e6f2c1e1ffef466d0a7113"
SOURCE = "c4ddfa4efadf0b3ec6d4380a4239b9cb3a097113"
PROFILES = (
    "trainable-scope",
    "opponent-mixture",
    "finetune-dose",
    "lineup-trajectory",
    "exploration-period",
)
PROFILE = os.environ.get("TASK4_PROFILE", "trainable-scope")
CONFIG = ROOT / f"experiments/2026-09-15-task4-{PROFILE}/config.json"
PROFILE_HASHES = {
    "trainable-scope": "509ed331f231a54ce3e50e394f3b473ee611bbd3cd2df1535b45c51a7b5a2a47",
    "opponent-mixture": "37491d6d2603265b292f73ca37279ea6d5ffa6cdbf71d9911a3ebffb294041a0",
    "finetune-dose": "75403e404fe3c150ae075415d20a7a287bce3bdeca9848d58864f8c6218e321d",
    "lineup-trajectory": "93d6f937a3711aa07fb93848423278a863132dbf320d7c6bcefacb01de485589",
    "exploration-period": "f2108d212d1dcc21aa8b2edf198f7d27e5d186587101ede3b8bdb32a182c3e86",
}
ARM_FACTORS = (
    "trainable_scope",
    "training_opponents",
    "potential_scale",
    "update_every",
    "kill_reward",
    "learning_rate",
    "random_episode_period",
)
# Everything the agent may be trained against, so an unregistered opponent
# cannot reach a training game through a configuration edit alone.
REGISTERED_OPPONENTS = ("rule_based_agent", "coin_collector_agent", "peaceful_agent")
REGISTERED_SCOPES = ("opponent_columns", "all_weights")
STAGES = ("training", "evaluation", "latency")


def _durable_replace(temporary, path):
    """Rename only after the bytes are on the platter, then persist the rename.

    os.replace is atomic against a concurrent reader but says nothing about
    power loss: the directory entry can reach disk before the file content
    does. That is how a previous run came back from a blackout holding a
    zero-length training-state.json and had to restart from episode zero. A run
    measured in days cannot afford that, so callers flush and fsync the file
    before the rename and the containing directory is fsynced after it.
    """
    os.replace(temporary, path)
    try:
        handle = os.open(path.parent, os.O_RDONLY)
    except OSError:
        # Windows cannot open a directory for fsync; the file's own fsync is
        # what carries the durability guarantee there.
        return
    try:
        os.fsync(handle)
    finally:
        os.close(handle)


def _write(path, value):
    # `.gitattributes` pins *.json to LF precisely so a Windows writer cannot
    # reintroduce CRLF and break a byte comparison against a clean checkout.
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _durable_replace(temporary, path)


def write_checkpoint(payload, path):
    """Persist a torch checkpoint through the same durable rename discipline."""
    import torch

    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    _durable_replace(temporary, path)


def write(path, value):
    from scripts.task3_progress_retry import with_permission_retry

    audit = ROOT / "training_outputs/progress-write-retries.jsonl"
    audit.parent.mkdir(parents=True, exist_ok=True)
    return with_permission_retry(_write, audit)(path, value)


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _hashable(value):
    """Arm factors may be lists, such as the training opponent line-up."""
    return tuple(value) if isinstance(value, list) else value


def config():
    if PROFILE not in PROFILES:
        raise ValueError("Unknown pilot profile")
    value = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = PROFILE_HASHES[PROFILE]
    if expected is not None and canonical_sha(value) != expected:
        raise ValueError("Pilot registration changed")
    if value["profile"] != PROFILE or value["runtime_source"] != SOURCE:
        raise ValueError("Unregistered pilot profile or runtime")
    if value["input_sha256"] != INPUT:
        raise ValueError("Unregistered reference input")
    arms = value["arms"]
    if list(value["arm_settings"]) != arms or "control" not in arms:
        raise ValueError("Arm settings must cover every registered arm exactly once")
    varying = [
        factor
        for factor in ARM_FACTORS
        if len({_hashable(value["arm_settings"][arm].get(factor)) for arm in arms}) > 1
    ]
    if len(varying) != 1:
        raise ValueError("A registered comparison varies exactly one factor")
    for arm in arms:
        setting = value["arm_settings"][arm]
        if setting["trainable_scope"] not in REGISTERED_SCOPES:
            raise ValueError("Unregistered trainable scope")
        if not setting["training_opponents"] or any(
            name not in REGISTERED_OPPONENTS for name in setting["training_opponents"]
        ):
            raise ValueError("Unregistered training opponent")
    if value["training_episodes"] != (
        len(arms) * value["replicas"] * value["episodes_per_replica_arm"]
    ):
        raise ValueError("Declared training episode total is inconsistent")
    suites = value["evaluation_suites"]
    per_artifact = value["evaluation_repeats"] * sum(
        len(setting["world_seeds"]) for setting in suites.values()
    )
    if value["evaluation_episodes"] != per_artifact * (len(arms) * value["replicas"] + 1):
        raise ValueError("Declared evaluation episode total is inconsistent")
    return value


def arm_of(artifact):
    """Return the registered arm an evaluation artifact was trained under."""
    if artifact == "reference":
        return "control"
    arm = artifact.rsplit("-r", 1)[0]
    if arm not in config()["arms"]:
        raise ValueError("Unknown evaluation artifact")
    return arm


def episode_epsilon(arm, agent_seed, episode):
    """Return the #168 episode mixture: one fully random episode per period.

    The mixture is binary by design - an episode is explored or exploited, never
    partly both - which is why the `initial_epsilon`, `epsilon_decay` and
    `epsilon_floor` fields registered alongside it do not reach the agent. The
    exploration knob that does is how often a random episode comes round, and a
    period of five means a fifth of a long run is spent on fully random play.

    The default period of five reproduces the registered schedule exactly,
    including its seed stream, so configurations that do not set it are
    unaffected.
    """
    cfg = config()
    if arm not in cfg["arms"] or not 0 <= episode < cfg["episodes_per_replica_arm"]:
        raise ValueError("Unknown arm or episode")
    period = cfg["arm_settings"][arm].get("random_episode_period", 5)
    if not isinstance(period, int) or period < 2:
        raise ValueError("Unregistered random episode period")
    chosen = random.Random(agent_seed + 168000000 + episode // period).randrange(period)
    return 1.0 if episode % period == chosen else 0.0


def zip_json(path, data):
    # Durable for the same reason every other progress write is: a run measured
    # in days must not lose its audit trail to a truncated write.
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as file:
            file.write(json.dumps(data, sort_keys=True, separators=(",", ":")).encode())
        stream.flush()
        os.fsync(stream.fileno())
    _durable_replace(temporary, path)


def read_zip_json(path):
    with gzip.open(path, "rb") as file:
        return json.loads(file.read().decode("utf-8"))


def resume_state(resume_dir):
    """Where a restarted training job continues, and the trail it continues from.

    The agent persists its checkpoint at the end of each episode and this module
    writes the audit trail afterwards, so a crash can leave the checkpoint one
    episode ahead of the trail it must stay consistent with. The rolling resume
    copy is taken by this module after the trail is on disk, so that pair is
    always consistent with each other; the trail is truncated to the copy's
    episode count and every episode the copy did not cover is simply replayed.

    Returns (completed_episodes, rows). A job with nothing to resume gets
    (0, []) and starts from its registered initialization.
    """
    checkpoint = Path(resume_dir) / "resume.pt"
    trail = Path(resume_dir) / "episodes.json.gz"
    if not checkpoint.exists() or not trail.exists():
        return 0, []
    import torch

    completed = torch.load(checkpoint, map_location="cpu", weights_only=True)[
        "completed_episodes"
    ]
    rows = read_zip_json(trail)
    if len(rows) < completed:
        raise ValueError(
            "Resume trail is behind its checkpoint; the audit chain cannot be "
            "reconstructed, so this replica must restart rather than resume"
        )
    return completed, rows[:completed]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "host": platform.node(),
        "processor": platform.processor(),
        "packages": {
            p: importlib.metadata.version(p) for p in ("numpy", "torch", "pygame", "psutil")
        },
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
    }


def code_hashes():
    names = [
        "scripts/pilot_task4_competition.py",
        "scripts/task4_interventions.py",
        "scripts/analyze_task4_competition.py",
        "scripts/audit_task4_seeds.py",
        "scripts/fetch_task4_inputs.py",
        "scripts/task3_pilot_resources.py",
        "scripts/frozen_opponent_inputs.py",
        "scripts/task3_reward_configuration.py",
        "scripts/task3_progress_retry.py",
        "scripts/run_task4_competition.ps1",
    ]
    return {
        name: hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for name in names
    }


def verify(root):
    binding = read_json(root / "binding.json")
    if canonical_sha(read_json(root / "config.json")) != canonical_sha(config()):
        raise ValueError("Prepared protocol copy changed")
    if binding["tool_hashes"] != code_hashes() or binding["config_sha256"] != canonical_sha(
        config()
    ):
        raise ValueError("Prepared tool/config source changed")
    if (
        sha(root / "reference.pt") != INPUT
        or sha(root / "initial.pt") != config()["initial_sha256"]
        or binding["initial_sha256"] != config()["initial_sha256"]
    ):
        raise ValueError("Prepared input changed")
    if any(
        sha(root / "source" / name) != digest
        for name, digest in read_json(root / "source-manifest.json").items()
    ):
        raise ValueError("Scientific runtime changed")
    for arm, record in read_json(root / "initialization-verification.json")["arms"].items():
        if sha(root / f"initial-{arm}.pt") != record["sha256"]:
            raise ValueError("Arm initialization changed")
    return binding


def prepare(root, parent, initial):
    cfg = config()
    if not read_json(CONFIG.with_name("seed-audit.json"))["passed"]:
        raise ValueError("Need passing registered seed audit")
    if sha(parent) != INPUT or sha(initial) != cfg["initial_sha256"]:
        raise ValueError("Wrong parent or fresh initialization")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit technical source before prepare")
    root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(parent, root / "reference.pt")
    shutil.copyfile(initial, root / "initial.pt")
    shutil.copyfile(CONFIG, root / "config.json")
    subprocess.run(
        ["git", "archive", "--format=tar", f"--output={root / 'runtime-source.tar'}", SOURCE],
        cwd=ROOT,
        check=True,
    )
    (root / "source").mkdir()
    with tarfile.open(root / "runtime-source.tar") as archive:
        archive.extractall(root / "source", filter="data")
    write(
        root / "source-manifest.json",
        {
            p.relative_to(root / "source").as_posix(): sha(p)
            for p in (root / "source").rglob("*")
            if p.is_file()
        },
    )
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "_bind", "--root", str(root)],
        check=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    write(
        root / "binding.json",
        {
            "profile": PROFILE,
            "input_sha256": INPUT,
            "initial_sha256": sha(root / "initial.pt"),
            "runtime_source": SOURCE,
            "tool_source": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "tool_hashes": code_hashes(),
            "config_sha256": canonical_sha(cfg),
            "prepared_environment": environment(),
            "seed_audit_sha256": sha(CONFIG.with_name("seed-audit.json")),
        },
    )
    verify(root)
    print(
        json.dumps(
            {
                "prepared": str(root),
                "profile": PROFILE,
                "training_episodes": cfg["training_episodes"],
                "evaluation_episodes": cfg["evaluation_episodes"],
                "scientific_execution_authorized": False,
            },
            indent=2,
        )
    )


def payload_equal(a, b):
    """Compare the entire restricted payload, including tensors and RNG/replay state."""
    import torch

    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(payload_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(payload_equal(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def arm_payload(initial, arm):
    from copy import deepcopy

    setting = config()["arm_settings"][arm]
    result = deepcopy(initial)
    result["rewards"]["KILLED_OPPONENT"] = setting["kill_reward"]
    rate = setting["learning_rate"]
    result["config"]["learning_rate"] = rate
    for group in result["learner_state"]["optimizer"]["param_groups"]:
        group["lr"] = rate
    return result


def bind(root):
    import torch

    torch.set_num_threads(1)
    cfg = config()
    initial = torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
    reference = torch.load(root / "reference.pt", map_location="cpu", weights_only=True)
    if (
        initial["completed_episodes"] != 0
        or initial["learner_state"]["update_steps"] != 0
        or initial["learner_state"]["optimizer"]["state"]
        or len(initial["replay_state"]["states"]) != 0
    ):
        raise ValueError("Need fresh initialization, empty optimizer and replay")
    if not payload_equal(initial, arm_payload(initial, "control")):
        raise ValueError("Control arm does not match the #168 initialization")
    for network in ("online_network", "target_network"):
        if not payload_equal(
            initial["learner_state"][network], reference["learner_state"][network]
        ):
            raise ValueError("Reference and initialization networks differ")
    if initial["config"]["discount_factor"] != cfg["discount_factor"]:
        raise ValueError("Registered shaping discount differs from the checkpoint")
    report = {
        "initial_sha256": sha(root / "initial.pt"),
        "reference_sha256": sha(root / "reference.pt"),
        "trainable_scopes": {
            arm: cfg["arm_settings"][arm]["trainable_scope"] for arm in cfg["arms"]
        },
        "training_opponents": {
            arm: cfg["arm_settings"][arm]["training_opponents"] for arm in cfg["arms"]
        },
        "identical_online_and_target_weights": True,
        "arms": {},
    }
    for arm in cfg["arms"]:
        payload = arm_payload(initial, arm)
        path = root / f"initial-{arm}.pt"
        if payload_equal(payload, initial):
            shutil.copyfile(root / "initial.pt", path)
        else:
            torch.save(payload, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not payload_equal(loaded, payload):
            raise ValueError("Arm initialization payload mismatch")
        report["arms"][arm] = {
            "sha256": sha(path),
            **{k: cfg["arm_settings"][arm].get(k) for k in ARM_FACTORS},
        }
    write(root / "initialization-verification.json", report)


def play(
    root,
    checkpoint,
    world_seed,
    agent_seed,
    scenario,
    opponents,
    training,
    epsilon,
    target,
    trainable_scope="opponent_columns",
    kill_reward=5.0,
    potential_scale=0.0,
    update_every=1,
    proximity_gate=None,
    l2_opponent=0.0,
):
    from unittest.mock import patch

    cfg = config()
    sys.path.insert(0, str(root / "source"))
    os.chdir(root / "source")
    from agent_code.DagobertDuckDQNTask3 import callbacks, features, train
    from agent_code.DagobertDuckDQNTask3 import config as agent_config
    from agents import AgentRunner
    from environment import BombeRLeWorld, WorldArgs
    from scripts.task3_reward_configuration import configured_rewards
    from scripts.task4_interventions import (
        RegularizedOpponentInputs,
        nearest_opponent_distance,
        training_intervention,
    )
    from training.seeded_framework import configure_diagnostic_logging, wrap_process_event

    configure_diagnostic_logging()
    actions = []
    selector = callbacks.select_action

    def behavior(**kwargs):
        action = selector(**{**kwargs, "epsilon": epsilon})
        actions.append(action)
        return action

    args = WorldArgs(
        True,
        30,
        False,
        0,
        False,
        None,
        False,
        not training,
        str(target),
        str(target / "native.json"),
        "task3-approach-pilot",
        world_seed,
        False,
        scenario,
    )
    target.mkdir(parents=True, exist_ok=False)
    with (
        configured_rewards(agent_config.REWARDS, kill_reward),
        training_intervention(
            train,
            potential_scale=potential_scale if training else 0.0,
            update_every=update_every if training else 1,
            discount_factor=cfg["discount_factor"],
            proximity_gate=proximity_gate if training else None,
        ),
        patch.dict(
            os.environ,
            {
                "BOMBERMAN_AGENT_SEED": str(agent_seed),
                "BOMBERMAN_DQN_ACTION_MASKING": "framework_legal",
                "BOMBERMAN_DQN_ESCAPE_CONTINUATIONS": "on",
            },
        ),
        patch.object(callbacks, "CHECKPOINT_PATH", checkpoint),
        patch.object(train, "CHECKPOINT_PATH", checkpoint),
        patch.object(callbacks, "_evaluation_checkpoint_path", lambda: checkpoint),
        patch.object(callbacks, "select_action", behavior),
        patch.object(
            AgentRunner,
            "process_event",
            wrap_process_event(
                AgentRunner.process_event,
                world_seed + 1000000,
                {name: i + 1 for i, name in enumerate(opponents)},
            ),
        ),
    ):
        world = BombeRLeWorld(
            args, [("DagobertDuckDQNTask3", training), *[(name, False) for name in opponents]]
        )
        learner = world.agents[0]
        policy = learner.backend.runner.fake_self
        before = network_sha(policy.policy_network)
        if training and any(
            g["lr"] != policy.config.learning_rate for g in policy.learner.optimizer.param_groups
        ):
            raise ValueError("Loaded optimizer/config learning rate mismatch")
        guard = None
        if training and trainable_scope == "opponent_columns":
            import torch

            anchor = torch.load(root / "initial.pt", weights_only=True, map_location="cpu")
            guard = RegularizedOpponentInputs(
                policy.learner, anchor["learner_state"]["online_network"], l2_opponent
            )
        steps = attacks = 0
        distances = []
        world.new_round()
        while world.running:
            alive = not learner.dead
            world.do_step()
            if not alive:
                continue
            steps += 1
            # Exposure is the mechanism endpoint, so it is recorded in both
            # stages. It is read from the public state after the step, outside
            # the agent's own measured decision window, and never feeds back
            # into acting or learning.
            state = learner.last_game_state
            attacks += bool(features.state_to_features(state)[30])
            distance = nearest_opponent_distance(state)
            if distance is not None:
                distances.append(distance)
        world.end()
        if guard is not None:
            guard.validate()
        after = network_sha(policy.policy_network)
        if not training and before != after:
            raise ValueError("Evaluation mutated network")
        native = world.round_statistics[world.round_id]
        if native["agents"][learner.name]["survival_steps"] != steps:
            raise ValueError("Native/instrumented survival mismatch")
        return {
            "world_seed": world_seed,
            "agent_seed": agent_seed,
            "behavior_epsilon": epsilon,
            "training": training,
            "opponents": list(opponents),
            "native": native,
            "attack_steps": int(attacks),
            "opponent_present_steps": len(distances),
            "opponent_distance_sum": int(sum(distances)),
            "trainable_scope": trainable_scope if training else "none",
            "kill_reward": kill_reward,
            "potential_scale": potential_scale if training else 0.0,
            "update_every": update_every if training else 1,
            "proximity_gate": proximity_gate if training else None,
            "l2_opponent": l2_opponent if training else 0.0,
            "gated_out_transitions": int(getattr(policy, "approach_gated_out", 0)),
            "greedy_actions_sha256": canonical_sha(actions),
            "online_before_sha256": before,
            "online_after_sha256": after,
            "completed_episodes": policy.completed_episodes,
            "optimizer_updates": policy.learner.update_steps if training else None,
        }


def train_job(root, output, arm, replica):
    """Train one replica of one arm, retaining what a long run has to retain.

    A checkpoint carries its replay buffer, so one copy per episode would be
    tens of gigabytes and is never read. Two cadences are kept instead, both
    registered and both cheap:

    `resume.pt` is a rolling copy taken every `resume_checkpoint_every`
    episodes so a job killed by a blackout continues from minutes ago rather
    than from episode zero. It lives beside the audit trail in a per-replica
    directory that survives the attempt it was written by, because the
    supervisor gives every retry a fresh output directory.

    `milestone-<episode>.pt` is retained at each registered milestone, so a
    trajectory can be evaluated where it was rather than only where it stopped.
    Training is sequential, so the milestone at episode N is exactly the agent a
    run of N episodes would have produced from the same seeds - which is why a
    budget question needs one long run and not one run per budget.

    Both default to off, so a registration that asks for neither behaves exactly
    as the fixed-final design did. The per-episode audit trail remains the
    chained `online_before_sha256`/`online_after_sha256` pair in
    `episodes.json.gz`, which pins the whole trajectory at negligible size.
    """
    cfg = config()
    setting = cfg["arm_settings"][arm]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.pt"
    retained = root / "training-resume" / f"{arm}-r{replica + 1}"
    retained.mkdir(parents=True, exist_ok=True)
    milestones = set(cfg.get("checkpoint_milestones", []))
    resume_every = cfg.get("resume_checkpoint_every", 0)

    completed, rows = resume_state(retained)
    if completed:
        shutil.copyfile(retained / "resume.pt", checkpoint)
        zip_json(output / "episodes.json.gz", rows)
    else:
        shutil.copyfile(root / f"initial-{arm}.pt", checkpoint)

    total = len(cfg["training_world_seeds"][replica])

    def retain(index):
        episode = index + 1
        if episode in milestones:
            shutil.copyfile(checkpoint, retained / f"milestone-{episode:06d}.pt")
        if resume_every and (episode % resume_every == 0 or episode == total):
            # The trail is durable before the copy is taken, so the pair a
            # restart reads can never disagree in the dangerous direction.
            zip_json(retained / "episodes.json.gz", rows)
            shutil.copyfile(checkpoint, retained / "resume.pt.tmp")
            _durable_replace(retained / "resume.pt.tmp", retained / "resume.pt")

    started, cpu = time.monotonic(), time.process_time()
    seed = cfg["replica_agent_seeds"][replica]
    for index, world_seed in enumerate(cfg["training_world_seeds"][replica]):
        if index < completed:
            continue
        row = play(
            root,
            checkpoint,
            world_seed,
            seed,
            "classic",
            setting["training_opponents"],
            True,
            episode_epsilon(arm, seed, index),
            output / f"episode-{index + 1:04d}",
            trainable_scope=setting["trainable_scope"],
            kill_reward=setting["kill_reward"],
            potential_scale=setting["potential_scale"],
            update_every=setting["update_every"],
            proximity_gate=setting.get("proximity_gate"),
            l2_opponent=setting.get("l2_opponent", 0.0),
        )
        if row["completed_episodes"] != index + 1:
            raise ValueError("Checkpoint episode counter mismatch")
        rows.append(row)
        zip_json(output / "episodes.json.gz", rows)
        retain(index)
    write(
        output / "result.json",
        {
            "arm": arm,
            "replica": replica,
            "episodes": len(rows),
            "checkpoint_sha256": sha(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "optimizer_updates": rows[-1]["optimizer_updates"],
            "initial_online_sha256": rows[0]["online_before_sha256"],
            "final_online_sha256": rows[-1]["online_after_sha256"],
            "episodes_sha256": sha(output / "episodes.json.gz"),
            "environment": environment(),
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.monotonic() - started,
        },
    )


def artifact_names(cfg=None):
    cfg = cfg or config()
    return ["reference"] + [
        f"{arm}-r{replica + 1}" for arm in cfg["arms"] for replica in range(cfg["replicas"])
    ]


def artifacts(root):
    cfg = config()
    state = read_json(root / "training-state.json")
    expected = set(artifact_names(cfg)) - {"reference"}
    if state["status"] != "completed" or set(state["completed"]) != expected:
        raise ValueError("Training incomplete")
    result = {"reference": root / "reference.pt"}
    for key, relative in state["completed"].items():
        directory = root / relative
        metadata = read_json(directory / "result.json")
        if (
            key != f"{metadata['arm']}-r{metadata['replica'] + 1}"
            or metadata["episodes"] != cfg["episodes_per_replica_arm"]
            or sha(directory / "checkpoint.pt") != metadata["checkpoint_sha256"]
        ):
            raise ValueError("Training artifact mismatch")
        result[key] = directory / "checkpoint.pt"
    return result


def evaluate_job(root, output, artifact, suite, latency=False):
    cfg = config()
    checkpoint = artifacts(root)[artifact]
    original_sha = sha(checkpoint)
    setting = cfg["evaluation_suites"][suite]
    seeds = setting["world_seeds"]
    repeats = 1 if latency else cfg["evaluation_repeats"]
    if latency:
        seeds = seeds[: cfg["latency_worlds"]]
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    started, cpu = time.monotonic(), time.process_time()
    for repeat in range(repeats):
        for seed in seeds:
            row = play(
                root,
                checkpoint,
                seed,
                seed + 1000000,
                setting["scenario"],
                setting["opponents"],
                False,
                0.0,
                output / f"{seed}-repeat{repeat}",
                kill_reward=cfg["arm_settings"][arm_of(artifact)]["kill_reward"],
            )
            rows.append({**row, "artifact": artifact, "suite": suite, "repeat": repeat})
            zip_json(output / "episodes.json.gz", rows)
    if sha(checkpoint) != original_sha:
        raise ValueError("Evaluation mutated checkpoint")
    write(
        output / "result.json",
        {
            "artifact": artifact,
            "suite": suite,
            "stage": "latency" if latency else "evaluation",
            "episodes": len(rows),
            "checkpoint_sha256": original_sha,
            "episodes_sha256": sha(output / "episodes.json.gz"),
            "environment": environment(),
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.monotonic() - started,
        },
    )


def selftest_job(output):
    """Minimal worker used only by the supervisor's concurrency tests.

    It plays no games and touches no scientific artifact; it exists so the
    worker-pool bookkeeping can be exercised without an authorized campaign.
    """
    output.mkdir(parents=True, exist_ok=False)
    started, cpu = time.monotonic(), time.process_time()
    deadline = started + 1.0
    while time.monotonic() < deadline:
        hashlib.sha256(b"selftest" * 4096).hexdigest()
    if os.environ.get("TASK4_SELFTEST_FAIL") == "yes":
        raise SystemExit("Deliberate self-test worker failure")
    write(
        output / "result.json",
        {
            "scope": "supervisor_selftest_not_a_scientific_job",
            "started_unix": time.time() - (time.monotonic() - started),
            "finished_unix": time.time(),
            "cpu_seconds": time.process_time() - cpu,
            "wall_seconds": time.monotonic() - started,
        },
    )


def stage_jobs(root, stage, cfg):
    if stage == "training":
        return [
            (f"{arm}-r{replica + 1}", ["_train", "--arm", arm, "--replica", str(replica)])
            for replica in range(cfg["replicas"])
            for arm in cfg["arms"]
        ]
    if stage == "evaluation":
        return [
            (f"{artifact}-{suite}", ["_evaluate", "--artifact", artifact, "--suite", suite])
            for artifact in artifacts(root)
            for suite in cfg["evaluation_suites"]
        ]
    return [
        (f"{artifact}-{cfg['latency_suite']}", [
            "_evaluate",
            "--artifact",
            artifact,
            "--suite",
            cfg["latency_suite"],
            "--latency",
        ])
        for artifact in artifacts(root)
    ]


def supervise(root, stage, resume=False):
    """Run one stage's jobs, up to `workers` at a time, under registered ceilings.

    The latency stage is registered with one worker so its recorded decision
    times describe an uncontended machine; it is the only stage that feeds the
    latency gate.
    """
    import psutil

    verify(root)
    cfg = config()
    if os.environ.get("TASK4_AUTHORIZED") != "yes":
        raise ValueError(
            "Explicit owner authorization required; "
            "set TASK4_AUTHORIZED=yes only after approval"
        )
    if (
        psutil.virtual_memory().available < cfg["minimum_free_memory_bytes"]
        or shutil.disk_usage(root).free < cfg["minimum_free_disk_bytes"]
    ):
        raise ValueError("Registered free memory and disk minimums are not met")
    workers = cfg["stage_workers"][stage]
    stage_environment = {
        k: v for k, v in environment().items() if k not in {"omp_threads", "mkl_threads"}
    }
    path = root / f"{stage}-state.json"
    if path.exists():
        if not resume:
            raise ValueError("Existing stage; inspect state and use --resume explicitly")
        state = read_json(path)
        if state.get("environment") != stage_environment:
            raise ValueError("Stage environment changed; do not move a partial stage between hosts")
        if state["status"] == "completed":
            raise ValueError("Stage already complete")
        for active in state.get("active") or []:
            try:
                if psutil.Process(active["pid"]).create_time() == active["created"]:
                    raise ValueError("Owned worker is still active")
            except psutil.NoSuchProcess:
                continue
        if state.get("active"):
            raise ValueError(
                "Abrupt supervisor exit: resource accounting needs audit before resume"
            )
    else:
        state = {
            "status": "running",
            "environment": stage_environment,
            "workers": workers,
            "completed": {},
            "attempts": [],
            "cpu_seconds": 0.0,
            "wall_seconds": 0.0,
            "peak_memory_bytes": 0,
            "active": [],
        }
    limits = cfg[f"{stage}_limits"]
    if (
        state["cpu_seconds"] >= limits["cpu_seconds"]
        or state["wall_seconds"] >= limits["wall_seconds"]
    ):
        raise ValueError("Existing allocation already exhausted")
    lock = root / f"{stage}.lock"
    if lock.exists():
        owner = read_json(lock)
        try:
            existing = psutil.Process(owner["pid"])
            if existing.create_time() == owner["created"]:
                raise ValueError("Another supervisor owns this stage")
        except psutil.NoSuchProcess:
            pass
        if not resume:
            raise ValueError("Stale lock requires explicit resume")
        lock.rename(root / f"{stage}-stale-lock-{time.time_ns()}.json")
    with lock.open("x") as file:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, file)

    pending = [job for job in stage_jobs(root, stage, cfg) if job[0] not in state["completed"]]
    running = []
    cpu_base = state["cpu_seconds"]
    finished_cpu = 0.0
    wall_base = state["wall_seconds"]
    started = time.monotonic()
    state["status"] = "running"

    def snapshot():
        state["active"] = [
            {"pid": job["child"].pid, "created": job["created"]} for job in running
        ]
        state["cpu_seconds"] = cpu_base + finished_cpu + sum(job["sampled"] for job in running)
        state["wall_seconds"] = wall_base + time.monotonic() - started

    def launch(key, arguments):
        attempt = f"{stage}/{key}/attempt-{len(state['attempts']) + 1:03d}"
        target = root / attempt
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {"job": key, "output": attempt, "status": "running"}
        state["attempts"].append(record)
        log = (target.parent / (target.name + ".log")).open("w")
        child = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *arguments,
                "--root",
                str(root),
                "--output",
                str(target),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "BOMBERMAN_COMPACT_LOGS": "1",
            },
        )
        return {
            "key": key,
            "record": record,
            "child": child,
            "log": log,
            "target": target,
            "process": psutil.Process(child.pid),
            "created": psutil.Process(child.pid).create_time(),
            "ledger": {},
            "sampled": 0.0,
        }

    try:
        while pending or running:
            while pending and len(running) < workers:
                running.append(launch(*pending.pop(0)))
                snapshot()
                write(path, state)
            memory = 0
            for job in list(running):
                try:
                    job["sampled"], job_memory = sample_tree(job["process"], job["ledger"])
                    memory += job_memory
                except psutil.NoSuchProcess:
                    pass
                if job["child"].poll() is None:
                    continue
                running.remove(job)
                job["log"].close()
                if job["child"].wait() != 0:
                    job["record"]["status"] = "failed"
                    raise RuntimeError("Pilot worker failed: " + job["key"])
                result = read_json(job["target"] / "result.json")
                finished_cpu += max(job["sampled"], result["cpu_seconds"])
                job["record"]["status"] = "completed"
                state["completed"][job["key"]] = job["record"]["output"]
            state["peak_memory_bytes"] = max(state["peak_memory_bytes"], memory)
            snapshot()
            write(path, state)
            if (
                state["cpu_seconds"] >= limits["cpu_seconds"]
                or state["wall_seconds"] >= limits["wall_seconds"]
                or memory > limits["memory_bytes"]
            ):
                raise RuntimeError("Registered allocation exhausted")
            time.sleep(0.1)
        state["status"] = "completed"
        verify(root)
    except BaseException as error:
        for job in running:
            if job["child"].poll() is None:
                stop_owned(job["child"])
            finished_cpu += job["sampled"]
            job["record"]["status"] = "failed"
            job["log"].close()
        running.clear()
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["active"] = []
        state["cpu_seconds"] = cpu_base + finished_cpu
        state["wall_seconds"] = wall_base + time.monotonic() - started
        write(path, state)
        lock.unlink()
    print(
        json.dumps(
            {"stage": stage, "status": state["status"], "completed_jobs": len(state["completed"])},
            indent=2,
        )
    )


def bundle(root, output, results=False):
    cfg = config()
    verify(root)
    artifacts(root)
    names = [
        "binding.json",
        "config.json",
        "source-manifest.json",
        "runtime-source.tar",
        "reference.pt",
        "initial.pt",
        "initialization-verification.json",
        "training-state.json",
        *[f"initial-{arm}.pt" for arm in cfg["arms"]],
    ]
    paths = [(root / name, name) for name in names]
    paths.extend((ROOT / name, "tools/" + name) for name in code_hashes())
    paths.append((CONFIG.with_name("seed-audit.json"), "seed-audit.json"))
    for path in (root / "training").rglob("*"):
        if path.is_file() and (
            (path.suffix == ".json" or path.name == "checkpoint.pt")
            or path.name.endswith(".json.gz")
        ):
            paths.append((path, path.relative_to(root).as_posix()))
    if results:
        from scripts.analyze_task4_competition import analyze

        write(root / "analysis.json", analyze(root))
        paths.extend(
            (root / name, name)
            for name in ("evaluation-state.json", "latency-state.json", "analysis.json")
        )
        for stage in ("evaluation", "latency"):
            for path in (root / stage).rglob("*"):
                if path.is_file() and (path.suffix == ".json" or path.name.endswith(".json.gz")):
                    paths.append((path, path.relative_to(root).as_posix()))
    with tarfile.open(output, "x:gz") as archive:
        for path, name in paths:
            archive.add(path, arcname=name, recursive=False)
    write(
        output.with_suffix(output.suffix + ".manifest.json"),
        {
            "sha256": sha(output),
            "size_bytes": output.stat().st_size,
            "files": len(paths),
            "tool_source": read_json(root / "binding.json")["tool_source"],
        },
    )
    print(
        json.dumps(
            {"bundle": str(output), "sha256": sha(output), "size_bytes": output.stat().st_size},
            indent=2,
        )
    )


def import_bundle(root, path, digest):
    if sha(path) != digest:
        raise ValueError("Transfer checksum mismatch")
    root.mkdir(parents=True, exist_ok=False)
    with tarfile.open(path) as archive:
        archive.extractall(root, filter="data")
    (root / "source").mkdir()
    with tarfile.open(root / "runtime-source.tar") as archive:
        archive.extractall(root / "source", filter="data")
    verify(root)
    artifacts(root)


def smoke_worker(root, output):
    cfg = config()
    verify(root)
    output.mkdir(parents=True, exist_ok=False)
    arm = cfg["smoke"]["arm"]
    arm_setting = cfg["arm_settings"][arm]
    checkpoint = output / "smoke-only.pt"
    shutil.copyfile(root / f"initial-{arm}.pt", checkpoint)
    training, training_seconds = [], []
    # Bounded mechanical warm-up; short deaths can leave two games below 500 transitions.
    for episode in range(len(cfg["smoke"]["world_seeds"])):
        started = time.monotonic()
        training.append(
            play(
                root,
                checkpoint,
                cfg["smoke"]["world_seeds"][episode],
                cfg["smoke"]["agent_seed"],
                "classic",
                arm_setting["training_opponents"],
                True,
                0.0,
                output / f"training-{episode}",
                trainable_scope=arm_setting["trainable_scope"],
                kill_reward=arm_setting["kill_reward"],
                potential_scale=arm_setting["potential_scale"],
                update_every=arm_setting["update_every"],
                proximity_gate=arm_setting.get("proximity_gate"),
                l2_opponent=arm_setting.get("l2_opponent", 0.0),
            )
        )
        training_seconds.append(time.monotonic() - started)
        if training[-1]["optimizer_updates"] > 0:
            break
    if (
        training[-1]["completed_episodes"] != len(training)
        or training[-1]["optimizer_updates"] <= 0
    ):
        raise ValueError("Smoke did not exercise genuine checkpointed learning")
    evaluation, evaluation_seconds = [], []
    for index, (suite, setting) in enumerate(cfg["evaluation_suites"].items()):
        started = time.monotonic()
        evaluation.append(
            play(
                root,
                checkpoint,
                cfg["smoke"]["world_seeds"][index + 2],
                cfg["smoke"]["agent_seed"],
                setting["scenario"],
                setting["opponents"],
                False,
                0.0,
                output / suite,
                kill_reward=arm_setting["kill_reward"],
            )
        )
        evaluation_seconds.append(time.monotonic() - started)
    from scripts.analyze_task4_competition import behavioral

    opponent_free_unchanged = []

    for index, (suite, setting) in enumerate(cfg["evaluation_suites"].items()):
        if setting["opponents"]:
            continue
        reference = play(
            root,
            root / "reference.pt",
            cfg["smoke"]["world_seeds"][index + 2],
            cfg["smoke"]["agent_seed"],
            setting["scenario"],
            [],
            False,
            0.0,
            output / (suite + "-reference"),
            kill_reward=cfg["arm_settings"]["control"]["kill_reward"],
        )
        # Only the frozen scope promises opponent-free behaviour is untouched.
        # Full fine-tuning is expected to change it, and the retention suites in
        # the real evaluation are what gate how far it may drift.
        unchanged = behavioral(reference["native"]) == behavioral(
            evaluation[index]["native"]
        ) and reference["greedy_actions_sha256"] == evaluation[index]["greedy_actions_sha256"]
        if arm_setting["trainable_scope"] == "opponent_columns" and not unchanged:
            raise ValueError("Opponent-free smoke behavior changed under a frozen scope")
        opponent_free_unchanged.append(unchanged)
    verify(root)
    zip_json(
        output / "smoke-observations.json.gz", {"training": training, "evaluation": evaluation}
    )
    write(
        output / "result.json",
        {
            "scope": "mechanical_smoke_excluded_from_pilot",
            "profile": PROFILE,
            "training_episodes": len(training),
            "evaluation_episodes": len(evaluation),
            "optimizer_updates": training[-1]["optimizer_updates"],
            "checkpoint_sha256": sha(checkpoint),
            # Throughput only, on one core and a warm cache. Used to size the
            # registered budget before launch; never a scientific measurement.
            "training_seconds_per_episode": training_seconds,
            "training_survival_steps": [row["native"]["steps"] for row in training],
            "evaluation_seconds_per_episode": evaluation_seconds,
            "trainable_scope": arm_setting["trainable_scope"],
            "training_opponents": arm_setting["training_opponents"],
            "opponent_free_behaviour_unchanged": opponent_free_unchanged,
            "evaluation_survival_steps": [row["native"]["steps"] for row in evaluation],
        },
    )


def smoke(root, output):
    import psutil

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_smoke",
        "--root",
        str(root),
        "--output",
        str(output),
    ]
    log_path = output.with_suffix(".log")
    started = time.monotonic()
    ledger = {}
    cpu = peak = 0
    with log_path.open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "BOMBERMAN_COMPACT_LOGS": "1",
            },
        )
        process = psutil.Process(child.pid)
        try:
            while child.poll() is None:
                try:
                    cpu, memory = sample_tree(process, ledger)
                except psutil.NoSuchProcess:
                    break
                peak = max(peak, memory)
                if time.monotonic() - started > 300 or cpu > 300 or memory > 1073741824:
                    raise RuntimeError("Mechanical smoke allocation exhausted")
                time.sleep(0.1)
            if child.wait() != 0:
                raise RuntimeError("Mechanical smoke failed; inspect retained log")
        except BaseException:
            if child.poll() is None:
                stop_owned(child)
            raise
        finally:
            write(
                output.with_suffix(".resources.json"),
                {
                    "cpu_seconds": cpu,
                    "wall_seconds": time.monotonic() - started,
                    "peak_memory_bytes": peak,
                },
            )
    print(json.dumps(read_json(output / "result.json"), indent=2))


def chain(root):
    """Complete the authorized allocation; stop only owned children on deadline."""
    verify(root)
    path = root / "chain-state.json"
    if path.exists():
        raise ValueError("Existing chain: inspect retained state; no automatic retries")
    if os.environ.get("TASK4_AUTHORIZED") != "yes":
        raise ValueError("Explicit authorization required")
    ceiling = config()["machine_allocation"]["overall_wall_seconds"] - 1800
    started = time.monotonic()
    state = {
        "status": "running",
        "stage": None,
        "profile": PROFILE,
        "started_unix": time.time(),
        "deadline_unix": time.time() + ceiling,
        "completed": [],
    }
    write(path, state)
    try:
        for mode in ("train", "evaluate", "latency", "results"):
            state["stage"] = mode
            write(path, state)
            command = [sys.executable, str(Path(__file__).resolve()), mode, "--root", str(root)]
            if mode == "results":
                command += ["--output", str(root / "task4-competition-evidence.tar.gz")]
            with (root / f"chain-{mode}.log").open("x") as log:
                child = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
                )
                state["active_pid"] = child.pid
                write(path, state)
                try:
                    while child.poll() is None:
                        if time.monotonic() - started >= ceiling:
                            raise RuntimeError("Registered chain allocation exhausted")
                        time.sleep(0.2)
                    if child.wait() != 0:
                        raise RuntimeError(
                            f"{mode} failed; inspect retained stage log and attempts"
                        )
                except BaseException:
                    if child.poll() is None:
                        stop_owned(child)
                    raise
            state["completed"].append(mode)
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        state["active_pid"] = None
        state["wall_seconds"] = time.monotonic() - started
        write(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "chain",
            "prepare",
            "dry-run",
            "train",
            "evaluate",
            "latency",
            "bundle",
            "results",
            "analyze",
            "import",
            "smoke",
            "_smoke",
            "_bind",
            "_train",
            "_evaluate",
            "_selftest",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--arm")
    parser.add_argument("--replica", type=int)
    parser.add_argument("--artifact")
    parser.add_argument("--suite")
    parser.add_argument("--latency", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "chain":
        chain(root)
    elif args.mode == "prepare":
        prepare(root, args.parent, args.initial)
    elif args.mode == "dry-run":
        cfg = config()
        verify(root)
        print(
            json.dumps(
                {
                    "profile": PROFILE,
                    "arms": cfg["arms"],
                    "training_jobs": len(cfg["arms"]) * cfg["replicas"],
                    "training_episodes": cfg["training_episodes"],
                    "stage_workers": cfg["stage_workers"],
                    "evaluation_jobs": len(artifact_names(cfg)) * len(cfg["evaluation_suites"]),
                    "evaluation_episodes": cfg["evaluation_episodes"],
                    "latency_jobs": len(artifact_names(cfg)),
                    "latency_episodes": cfg["latency_episodes"],
                    "budgets": {k: v for k, v in cfg.items() if "limits" in k},
                },
                indent=2,
            )
        )
    elif args.mode in {"train", "evaluate", "latency"}:
        supervise(
            root,
            {"train": "training", "evaluate": "evaluation", "latency": "latency"}[args.mode],
            args.resume,
        )
    elif args.mode in {"bundle", "results"}:
        bundle(root, args.output.resolve(), args.mode == "results")
    elif args.mode == "analyze":
        from scripts.analyze_task4_competition import analyze

        write(args.output, analyze(root))
    elif args.mode == "import":
        import_bundle(root, args.archive, args.sha256)
    elif args.mode == "smoke":
        smoke(root, args.output.resolve())
    elif args.mode == "_smoke":
        smoke_worker(root, args.output.resolve())
    elif args.mode == "_selftest":
        selftest_job(args.output.resolve())
    elif args.mode == "_bind":
        bind(root)
    elif args.mode == "_train":
        train_job(root, args.output.resolve(), args.arm, args.replica)
    else:
        evaluate_job(root, args.output.resolve(), args.artifact, args.suite, args.latency)


if __name__ == "__main__":
    main()
