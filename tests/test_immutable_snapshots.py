"""Immutable snapshots must never share storage with a mutable live model."""

import os
import stat

import pytest

from training import issue163_storage
from training.immutable_snapshots import ImmutableSnapshots, probe_links


def test_copy_then_link_preserves_inputs_and_never_links_live_checkpoint(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    model = source / "checkpoint.pt"
    model.write_bytes(b"model")
    (source / "callbacks.py").write_text("code")
    store = ImmutableSnapshots(tmp_path / "objects")
    a, b = tmp_path / "a", tmp_path / "b"
    store.snapshot(source, a)
    store.snapshot(source, b)
    assert os.path.samefile(a / "checkpoint.pt", b / "checkpoint.pt")
    assert not os.path.samefile(model, a / "checkpoint.pt")
    model.write_bytes(b"updated")
    assert (a / "checkpoint.pt").read_bytes() == b"model"
    assert not ((a / "checkpoint.pt").stat().st_mode & stat.S_IWUSR)
    with pytest.raises(FileExistsError):
        store.snapshot(source, a)
    assert len(list((tmp_path / "objects").iterdir())) == 2


def test_changed_object_fails_closed(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "checkpoint.pt").write_bytes(b"model")
    store = ImmutableSnapshots(tmp_path / "objects")
    store.snapshot(source, tmp_path / "a")
    obj = next(store.root.iterdir())
    obj.chmod(stat.S_IWRITE)
    obj.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        store.snapshot(source, tmp_path / "b")
    assert (tmp_path / "b.tmp").is_dir()


def test_store_source_overlap_rejected_and_probe_is_isolated(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match="separate"):
        ImmutableSnapshots(source / "objects").snapshot(source, tmp_path / "snapshot")
    probe_links(tmp_path)
    assert list(tmp_path.iterdir()) == [source]


def test_budget_keeps_records_export_and_reserve():
    from training.task3_attack_campaign import validate

    plans = validate()[1]
    report = issue163_storage.storage_budget(plans)
    assert report["required_free_bytes"] == 35 * 1024**3
    assert report["training_jobs"] == 10 and report["evaluation_jobs"] == 1760
    assert report["free_reserve_bytes"] == 4 * 1024**3
