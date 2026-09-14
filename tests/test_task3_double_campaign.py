"""The new study preserves every gate and requires benefit, identity and immutable export."""

import gzip
import json
import tarfile

import pytest

from tests.test_task3_mask_campaign import rows as mask_rows
from training import task3_double_campaign as campaign


def rows():
    return [dict(r, arm="double" if r["arm"] == "masked" else r["arm"]) for r in mask_rows()]


def test_matrix_modes_budget_and_seeds():
    config, plans, report = campaign.validate()
    assert report["training_episodes"] == 100000
    assert report["evaluation_episodes"] == 3520
    assert not report["compute_authorized"]
    assert config["bootstrap"]["seed"] == 150
    assert plans["control"].action_masking == plans["double"].action_masking == "framework_legal"
    assert plans["control"].replicas[0].world_seed == 150001


def test_selection_requires_treatment_benefit_and_every_guard():
    config = campaign.validate()[0]
    assert campaign.decide(rows(), config)["selected_arm"] == "double"
    assert campaign.decide(rows(), config)["selected_replica"] == "r3"
    for mutation in (
        {"coins_collected": 0},
        {"self_kills": 1},
        {"opponents_eliminated": 0},
        {"invalid_actions": 5},
    ):
        sample = rows()
        for row in sample:
            if row["arm"] == "double":
                row.update(mutation)
        assert campaign.decide(sample, config)["selected_arm"] is None
    sample = rows()
    for row in sample:
        if row["arm"] == "control":
            row["opponents_eliminated"] = 1
    assert campaign.decide(sample, config)["selected_replica"] is None


def test_export_keeps_failures_models_and_checked_observations(tmp_path):
    root, binding, analysis = [tmp_path / n for n in ("runs", "binding", "analysis")]
    for path in (root, binding, analysis):
        path.mkdir()
    model = root / "plans/control/replicas/r1/agent/checkpoint.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"immutable-model")
    failure = root / "plans/control/jobs/j/attempt-001/metadata.json"
    failure.parent.mkdir(parents=True)
    failure.write_text('{"status":"failed"}')
    with gzip.open(analysis / "observations.json.gz", "wt") as file:
        json.dump({"evaluation": rows()}, file)
    result = campaign.decide(rows(), campaign.validate()[0])
    result.update(
        protocol_sha256=campaign.sha256(campaign.CONFIG),
        observations_sha256=campaign.sha256(analysis / "observations.json.gz"),
    )
    campaign.write_json(analysis / "result.json", result)
    campaign.write_json(analysis / "source-manifest.json", {})
    archive = tmp_path / "evidence.tar.gz"
    campaign.export(root, binding, analysis, archive)
    with tarfile.open(archive) as tar:
        assert "campaign/" + model.relative_to(root).as_posix() in tar.getnames()
        assert "campaign/" + failure.relative_to(root).as_posix() in tar.getnames()
    with pytest.raises(ValueError, match="previous export"):
        campaign.export(root, binding, analysis, archive)
    result["selected_arm"] = "control"
    campaign.write_json(analysis / "result.json", result)
    with pytest.raises(ValueError, match="Decision mismatch"):
        campaign.verify(analysis)


def test_incomplete_export_preserves_failure_without_claiming_analysis(tmp_path):
    root, binding = tmp_path / "runs", tmp_path / "binding"
    root.mkdir()
    binding.mkdir()
    failure = root / "jobs/failed/attempt-001/metadata.json"
    failure.parent.mkdir(parents=True)
    failure.write_text('{"status":"failed"}')
    (tmp_path / "supervisor.log").write_text("Recorded failure trace")
    archive = tmp_path / "partial.tar.gz"
    campaign.export_files(root, binding, None, archive)
    manifest = campaign.read_json(archive.with_suffix(".gz.manifest.json"))
    assert manifest["evidence_scope"] == "partial_unanalyzed"
    with tarfile.open(archive) as tar:
        assert "campaign/jobs/failed/attempt-001/metadata.json" in tar.getnames()
        assert "handoff/supervisor.log" in tar.getnames()
        assert not any(n.startswith("analysis/") for n in tar.getnames())
    second = tmp_path / "second.tar.gz"
    old_manifest = second.with_suffix(".gz.manifest.json")
    old_manifest.write_text("Retained previous manifest")
    with pytest.raises(ValueError, match="previous export"):
        campaign.export_files(root, binding, None, second)
    assert old_manifest.read_text() == "Retained previous manifest"
    lock = root / ".task3-campaign.lock"
    lock.write_text("Owned by running process")
    with pytest.raises(ValueError, match="still active"):
        campaign.export_files(root, binding, None, tmp_path / "active.tar.gz")
