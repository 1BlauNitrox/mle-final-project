"""Independently verify the portable, completed Issue 91 server evidence."""

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import fmean

import yaml

from training import analyze_issue107_task2_factorial as stats
from training.aggregate import read_episodes_csv
from training.run_issue91 import LIMITS, plan_spec
from training.run_plan import Replica, _expand_jobs, _parse_stage, _parse_suite


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(root):
    root = Path(root).resolve()
    manifest = read(root / "analysis/evidence-files.json")
    missing = []
    for name, record in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Evidence check failed: path.is_relative_to(root)")
        if not path.exists():
            missing.append(name)
            continue
        if not (sha(path) == record["sha256"] and path.stat().st_size == record["size_bytes"]):
            raise ValueError(name)
    if missing:
        raise ValueError(missing)


def verify(root):
    root = Path(root).resolve()
    verify_manifest(root)
    missing = []
    protocol = read(root / "protocol.json")
    auth = read(root / "authorization.json")
    resource = read(root / "resources.json")
    reported = read(root / "analysis/result.json")
    if (
        not protocol["config_sha256"]
        == "dee088546fb2cb5fee9519f775ddfee0359da9924a4e7ed891032ddeb96a9ce2"
    ):
        raise ValueError("Evidence check failed: protocol['config_sha256'] == 'dee088546fb2cb5f...")
    if not auth["protocol_sha256"] == sha(root / "protocol.json"):
        raise ValueError("Evidence check failed: auth['protocol_sha256'] == sha(root / 'protoco...")
    if (
        not auth["reviewed_commit"]
        == protocol["prepared_commit"]
        == "cbd52be8392f5003a91c5600fda4efd544b48ec5"
    ):
        raise ValueError("Evidence check failed: auth['reviewed_commit'] == protocol['prepared_...")
    if not (
        resource["limits"] == vars(LIMITS)
        and resource["limit_reached"] is None
        and (not resource["active_root_pids"])
    ):
        raise ValueError("Evidence check failed: resource['limits'] == vars(LIMITS) and resourc...")
    for key, limit in [
        ("cpu_seconds_consumed", LIMITS.cpu_seconds),
        ("wall_seconds_elapsed", LIMITS.wall_seconds),
        ("peak_memory_bytes", LIMITS.memory_bytes),
    ]:
        if not 0 <= resource[key] <= limit:
            raise ValueError("Evidence check failed: 0 <= resource[key] <= limit")
    import numpy as np
    import torch

    from agent_code.DagobertDuckDQNTask2.persistence import (
        load_evaluation_checkpoint,
        load_training_checkpoint,
    )
    from training.analyze_issue91 import decision, q_probe

    largest_q_difference = 0.0
    for diag in reported["training_diagnostics"]:
        path = (
            root
            / "run-plans"
            / ("issue91-" + diag["cell"].lower())
            / "artifacts"
            / diag["replica"]
            / diag["stage"]
            / "checkpoint.pt"
        )
        probe = q_probe(path)
        if not probe["gamma"] == diag["gamma"] == (0.9 if diag["cell"] == "A" else 0.97):
            raise ValueError(
                "Evidence check failed: probe['gamma'] == diag['gamma'] == (0.9 if dia..."
            )
        largest_q_difference = max(
            largest_q_difference,
            float(np.max(np.abs(np.array(probe["q_values"]) - np.array(diag["q_values"])))),
        )
        if not np.allclose(probe["q_values"], diag["q_values"], rtol=1e-05, atol=1e-05):
            raise ValueError(
                "Evidence check failed: np.allclose(probe['q_values'], diag['q_values'..."
            )
        if (
            diag["stage"] == "classic"
            and load_evaluation_checkpoint(path).completed_episodes != 10000
        ):
            raise ValueError("Final checkpoint does not contain the exact 10000-episode budget")
    initial = [load_training_checkpoint(root / f"initial-{cell}.pt") for cell in "AB"]
    for loaded, cell, gamma in zip(initial, "AB", [0.9, 0.97], strict=True):
        if not (
            loaded.config.discount_factor == gamma and loaded.config.escape_continuation_features
        ):
            raise ValueError(
                "Evidence check failed: loaded.config.discount_factor == gamma and loa..."
            )
        if not (loaded.completed_episodes == 0 and len(loaded.replay_buffer) == 0):
            raise ValueError(
                "Evidence check failed: loaded.completed_episodes == 0 and len(loaded...."
            )
        if not set(protocol["plans"][cell]["parents"].values()) == {
            sha(root / f"initial-{cell}.pt")
        }:
            raise ValueError(
                "Evidence check failed: set(protocol['plans'][cell]['parents'].values(..."
            )
    configs = [asdict(model.config) for model in initial]
    for config in configs:
        config.pop("discount_factor")
    if not configs[0] == configs[1]:
        raise ValueError("Evidence check failed: configs[0] == configs[1]")
    for attr in ["online_network", "target_network"]:
        first = getattr(initial[0].learner, attr).state_dict()
        second = getattr(initial[1].learner, attr).state_dict()
        if not all(torch.equal(first[key], second[key]) for key in first):
            raise ValueError(
                "Evidence check failed: all((torch.equal(first[key], second[key]) for ..."
            )
    rows = []
    observations = {}
    training_count = 0
    metadata_count = 0
    versions = set()
    train_diags = []
    for cell, item in protocol["plans"].items():
        path = root / item["path"]
        if not sha(path) == item["sha256"]:
            raise ValueError("Evidence check failed: sha(path) == item['sha256']")
        raw = yaml.safe_load(path.read_text())
        expected = plan_spec(cell, Path(raw["replicas"][0]["parent_artifact"]))
        for data in (raw, expected):
            for r in data["replicas"]:
                r.pop("parent_artifact", None)
        if not raw == expected:
            raise ValueError((cell, "plan"))
        directory = root / "run-plans" / raw["plan_id"]
        resolved = read(directory / "resolved_plan.json")
        status = read(directory / "status.json")
        if not resolved["fingerprints"] == item["fingerprints"]:
            raise ValueError(
                "Evidence check failed: resolved['fingerprints'] == item['fingerprints']"
            )
        replicas = tuple(Replica(**r) for r in resolved["replicas"])
        jobs = _expand_jobs(
            replicas,
            [_parse_stage(s) for s in raw["training_stages"]],
            [_parse_suite(s) for s in raw["evaluation_suites"]],
        )
        if not json.loads(json.dumps([asdict(j) for j in jobs])) == resolved["jobs"]:
            raise ValueError(
                "Evidence check failed: json.loads(json.dumps([asdict(j) for j in jobs..."
            )
        if not (
            status["status"] == "completed" and set(status["jobs"]) == {j.run_id for j in jobs}
        ):
            raise ValueError(
                "Evidence check failed: status['status'] == 'completed' and set(status..."
            )
        for key in (
            "agent",
            "action_masking",
            "escape_continuations",
            "replay_treatment",
            "reward_variant",
        ):
            if not resolved[key] == raw[key]:
                raise ValueError("Evidence check failed: resolved[key] == raw[key]")
        final = {
            j.replica: status["jobs"][j.run_id]["artifact"]["sha256"]
            for j in jobs
            if j.kind == "training" and j.stage_or_suite == "classic"
        }
        for job in jobs:
            record = status["jobs"][job.run_id]
            if not record["status"] == "completed":
                raise ValueError("Evidence check failed: record['status'] == 'completed'")
            attempt = record["attempts"][-1]
            if not attempt["status"] == "completed":
                raise ValueError("Evidence check failed: attempt['status'] == 'completed'")
            run = directory / attempt["output"]
            meta = read(run / "metadata.json")
            metadata_count += 1
            versions.add(meta["python_version"])
            stats._validate_job_campaign(meta, auth, job.run_id)
            if not (
                meta["status"] == "completed"
                and meta["return_code"] == 0
                and (meta["git_dirty"] is False)
            ):
                raise ValueError(
                    "Evidence check failed: meta['status'] == 'completed' and meta['return..."
                )
            for k in ("world_seed", "agent_seed", "scenario", "rounds"):
                if not meta[k] == getattr(job, k):
                    raise ValueError("Evidence check failed: meta[k] == getattr(job, k)")
            if not (meta["mode"] == job.kind and meta["opponents"] == []):
                raise ValueError(
                    "Evidence check failed: meta['mode'] == job.kind and meta['opponents']..."
                )
            detail = meta["run_plan"]
            if not detail["fingerprints"] == resolved["fingerprints"]:
                raise ValueError(
                    "Evidence check failed: detail['fingerprints'] == resolved['fingerprin..."
                )
            for k in (
                "action_masking",
                "escape_continuations",
                "replay_treatment",
                "reward_variant",
            ):
                if not detail[k] == resolved[k]:
                    raise ValueError("Evidence check failed: detail[k] == resolved[k]")
            if not (
                detail["processes"] == 1 and detail["artifact_writable"] == (job.kind == "training")
            ):
                raise ValueError(
                    "Evidence check failed: detail['processes'] == 1 and detail['artifact_..."
                )
            episodes = read_episodes_csv(run / "episodes.csv")
            if not len(episodes) == job.rounds:
                raise ValueError("Evidence check failed: len(episodes) == job.rounds")
            if job.kind == "training":
                training_count += len(episodes)
                diag = next(
                    d
                    for d in reported["training_diagnostics"]
                    if (d["cell"], d["replica"], d["stage"])
                    == (cell, job.replica, job.stage_or_suite)
                )
                for metric in ("mean_loss", "mean_abs_td_error"):
                    vals = [r[metric] for r in episodes if r.get(metric) is not None]
                    if not (vals and all(math.isfinite(v) for v in vals)):
                        raise ValueError(
                            "Evidence check failed: vals and all((math.isfinite(v) for v in vals))"
                        )
                    if not fmean(vals) == diag[metric]:
                        raise ValueError("Evidence check failed: fmean(vals) == diag[metric]")
                train_diags.append(diag)
            else:
                expected_hash = (
                    final[job.replica] if cell in ("A", "B") else item["parents"][job.replica]
                )
                if not record["artifact"]["sha256"] == expected_hash:
                    raise ValueError(
                        "Evidence check failed: record['artifact']['sha256'] == expected_hash"
                    )
                row = episodes[0]
                if not all(row[k] is not None for k in stats.DETERMINISTIC_COLUMNS):
                    raise ValueError(
                        "Evidence check failed: all((row[k] is not None for k in stats.DETERMI..."
                    )
                if not row["initially_available_coins"] > 0:
                    raise ValueError("Evidence check failed: row['initially_available_coins'] > 0")
                if not stats._latency_ok(row):
                    raise ValueError("Evidence check failed: stats._latency_ok(row)")
                observations[cell, job.replica, job.stage_or_suite, job.world_seed] = row
                if job.stage_or_suite.endswith("-primary"):
                    rows.append(
                        dict(
                            row,
                            cell=cell,
                            model=job.replica,
                            scenario=job.scenario,
                            world_seed=job.world_seed,
                            agent_seed=job.agent_seed,
                            collection_fraction=row["coins_collected"]
                            / row["initially_available_coins"],
                        )
                    )
    if not (training_count == 100000 and len(observations) == 2720 and (len(rows) == 1360)):
        raise ValueError("Evidence check failed: training_count == 100000 and len(observations)...")
    for (cell, replica, suite, seed), row in observations.items():
        if suite.endswith("-primary"):
            repeat = observations[cell, replica, suite.replace("-primary", "-repeat"), seed]
            if not all(row[k] == repeat[k] for k in stats.DETERMINISTIC_COLUMNS):
                raise ValueError(
                    "Evidence check failed: all((row[k] == repeat[k] for k in stats.DETERM..."
                )
    summaries = stats._summarize(rows)
    for item in summaries:
        item["collected_per_revealed_coin_proxy"] = (
            item["mean_coins"] * item["episodes"] / item["coins_found"]
            if item["coins_found"]
            else None
        )
    if not summaries == reported["summaries"]:
        raise ValueError("Evidence check failed: summaries == reported['summaries']")
    values = stats._metric_values(rows)
    primary = stats._contrast(values, "B", "A", "classic", "collection", 9101, 1.0)
    if not primary == reported["primary"]:
        raise ValueError("Evidence check failed: primary == reported['primary']")
    for scenario in ("classic", "coin-heaven", "loot-crate"):
        for metric in ("collection", "survival", "self_kill"):
            name = scenario + "_" + metric
            contrast = stats._contrast(
                values,
                "B",
                "A",
                scenario,
                metric,
                stats._stable_seed("issue91_" + name),
                -1.0 if metric == "self_kill" else 1.0,
            )
            if not contrast == reported["guards"][name]:
                raise ValueError("Evidence check failed: contrast == reported['guards'][name]")
    contrasts = {}
    for cell in ("A", "B"):
        contrasts[f"{cell.lower()}_classic_minus_untrained"] = stats._contrast(
            values, cell, "untrained", "classic", "collection", 9110 + ord(cell), 1.0
        )
        contrasts[f"{cell.lower()}_coin_heaven_minus_task1"] = stats._contrast(
            values, cell, "frozen_task1", "coin-heaven", "collection", 9210 + ord(cell), 1.0
        )
        if (
            not stats._absolute_gates(cell, rows, summaries, contrasts, True)
            == reported["absolute_gates"][cell]
        ):
            raise ValueError(
                "Evidence check failed: stats._absolute_gates(cell, rows, summaries, c..."
            )
    if not decision(primary, reported["guards"], True, True) == reported["decision"]:
        raise ValueError("Evidence check failed: decision(primary, reported['guards'], True, Tr...")
    if (
        not stats._select_representative(reported["selection"]["cell"], summaries)
        == reported["selection"]["replica"]
    ):
        raise ValueError("Evidence check failed: stats._select_representative(reported['selecti...")
    if (
        not reported["selection"]["task2_complete"]
        == reported["absolute_gates"][reported["selection"]["cell"]]["overall_passed"]
    ):
        raise ValueError("Evidence check failed: reported['selection']['task2_complete'] == rep...")
    return dict(
        episode_metrics_verified=True,
        metadata_jobs_verified=metadata_count,
        training_episodes=training_count,
        evaluation_episodes=len(observations),
        primary_episodes=len(rows),
        determinism_verified=True,
        all_episode_latency_checks_passed=True,
        reported_statistics_reproduced=True,
        python_versions=sorted(versions),
        missing_checkpoint_files=missing,
        artifact_bytes_verified=True,
        initial_networks_identical=True,
        all_stage_gammas_verified=True,
        q_probe_rtol=1e-05,
        q_probe_atol=1e-05,
        maximum_q_probe_absolute_difference=largest_q_difference,
        primary=primary,
        decision=reported["decision"],
        selection=reported["selection"],
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.campaign_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
