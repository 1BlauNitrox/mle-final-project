"""The proposed treatment cannot overwrite parents or select on an invalid-only gain."""

import gzip
import json
import tarfile

import numpy as np
import pytest
import torch

from tests.test_task3_campaign import passing_rows
from training import task3_mask_campaign as mask


def rows():
    data = passing_rows()
    result = [dict(r, arm="masked" if r["arm"] == "candidate" else "reference") for r in data]
    result += [
        dict(r, arm="control", opponents_eliminated=0) for r in data if r["arm"] == "candidate"
    ]
    return result


def test_matrix_is_paired_disjoint_and_unchanged_retention_contract():
    config, plans, report = mask.validate()
    original = mask.campaign.validate_protocol()[0]
    assert config["gates"] == original["gates"]
    assert report["training_episodes"] == 100000 and report["evaluation_episodes"] == 3520
    assert not report["compute_authorized"]
    assert plans["masked"].action_masking == "framework_legal"
    assert plans["control"].action_masking == "none"


def test_paired_bootstrap_keeps_correlated_model_and_seed_noise():
    control = np.arange(200).reshape(5, 40)
    assert mask.paired_interval(control + 0.25, control) == {
        "mean": 0.25,
        "lower": 0.25,
        "upper": 0.25,
    }
    with pytest.raises(ValueError):
        mask.paired_interval(control, control[:4])


def test_selects_median_only_with_benefit_and_all_original_gates():
    config = mask.validate()[0]
    assert mask.decide(rows(), config)["selected_replica"] == "r3"
    for mutation in (
        {"opponents_eliminated": 0},
        {"coins_collected": 0},
        {"invalid_actions": 5},
        {"self_kills": 1},
    ):
        evidence = rows()
        for row in evidence:
            if row["arm"] == "masked":
                row.update(mutation)
        assert mask.decide(evidence, config)["selected_replica"] is None
    evidence = rows()
    for row in evidence:
        if row["arm"] == "control":
            row["opponents_eliminated"] = 1
    assert mask.decide(evidence, config)["selected_replica"] is None


def test_masked_copy_changes_only_flag_and_rejects_trained_or_tampered_state(tmp_path):
    source = mask.ROOT / "agent_code/DagobertDuckDQNTask3/checkpoint.pt"
    original_hash = mask.sha256(source)
    target = tmp_path / "masked.pt"
    mask.masked_initialization(source, target)
    assert mask.sha256(source) == original_hash
    with pytest.raises(ValueError, match="existing"):
        mask.masked_initialization(source, target)
    payload = torch.load(target, weights_only=True)
    payload["learner_state"]["online_network"]["layers.0.weight"][0, 0] += 1
    torch.save(payload, target)
    with pytest.raises(ValueError, match="changed weights"):
        mask.verify_masked_initialization(source, target)
    payload = torch.load(source, weights_only=True)
    payload["completed_episodes"] = 1
    trained = tmp_path / "trained.pt"
    torch.save(payload, trained)
    with pytest.raises(ValueError, match="fresh"):
        mask.masked_initialization(trained, tmp_path / "another.pt")


def test_compact_export_retains_reference_replica_and_failure_bytes(tmp_path):
    root, binding, analysis = (tmp_path / name for name in ("runs", "binding", "analysis"))
    for path in (root, binding, analysis):
        path.mkdir()
    required = root / "plans/reference/replicas/frozen/agent/checkpoint.pt"
    required.parent.mkdir(parents=True)
    required.write_bytes(b"reference")
    failed = root / "plans/control/jobs/job/attempt-001/metadata.json"
    failed.parent.mkdir(parents=True)
    failed.write_text('{"status":"failed"}')
    (failed.parent / "verbose.log").write_text("unnecessary")
    (binding / "binding.json").write_text("{}")
    config = mask.validate()[0]
    result = mask.decide(rows(), config)
    with gzip.open(analysis / "observations.json.gz", "wt") as file:
        json.dump({"evaluation": rows()}, file)
    result.update(
        protocol_sha256=mask.sha256(mask.CONFIG),
        observations_sha256=mask.sha256(analysis / "observations.json.gz"),
    )
    mask.write_json(analysis / "result.json", result)
    mask.write_json(analysis / "source-manifest.json", {})
    output = tmp_path / "evidence.tar.gz"
    mask.export(root, binding, analysis, output)
    with tarfile.open(output) as archive:
        names = archive.getnames()
    assert "campaign/" + required.relative_to(root).as_posix() in names
    assert "campaign/" + failed.relative_to(root).as_posix() in names
    assert not any(name.endswith(".log") for name in names)
    with pytest.raises(ValueError, match="previous export"):
        mask.export(root, binding, analysis, output)
