"""Synthetic checks, separate from the full external ZIP experiment."""
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "experiment", Path(__file__).resolve().parents[1] / "tools/experiment_crowd_keypoints.py")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def document(confidence=1.0):
    people = {"person_id": -1}
    for name, size in experiment.load_source().GROUPS:
        people[name] = [value for _ in range(size) for value in (512, 1024, confidence)]
        people[name.replace("_2d", "_3d")] = []
    return dict(version=1.3, people=people)


def archive(tmp_path, documents):
    path = tmp_path / "sample.zip"
    with zipfile.ZipFile(path, "w") as stream:
        for index, doc in documents:
            base = "NIA_SL_FS0001_CROWD18_F"
            name = f"keypoint/18/{base}/{base}_{index:012d}_keypoints.json"
            stream.writestr(name, json.dumps(doc))
    return path


def test_coordinate_bins_and_positive_confidence():
    stats = experiment.Coordinates(3)
    stats.add(np.array([[0, 1, 0], [1, 256, 1], [4096, -1, 1.2]]))
    result = stats.result()
    assert result["min_xy"] == [0, -1]
    assert result["max_xy"] == [4096, 256]
    assert result["positive_confidence_joints"][0]["mean_xy"] == [None, None]
    assert result["positive_confidence_joints"][2]["mean_xy"] == [4096, -1]
    assert result["invalid_confidence_values"] == 1
    assert stats.histogram.sum(axis=1).tolist() == [3, 3]
    assert stats.histogram[0, -1] == 1


def test_empty_statistics_json_is_finite():
    stats = experiment.Coordinates(21)
    stats.add(np.empty((0, 3)))
    assert stats.result()["min_xy"] == [None, None]
    json.dumps(stats.result(), allow_nan=False)


def test_real_parser_reuse_gap_confidence_and_immutability(tmp_path):
    path = archive(tmp_path, [(0, document()), (2, document(1.2))])
    before = path.read_bytes()
    result = experiment.experiment(path)
    assert result["frames"] == 2
    assert result["strict_groups"] == 0
    assert result["errors"] == {"confidence": 1}
    assert result["missing_indices"] == 1
    assert result["converted_ctc_inputs"] == 0
    assert result["fps"] == "unverified"
    assert result["sha256_before"] == result["sha256_after"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10**400])
def test_extreme_values_stay_rejected(tmp_path, value):
    doc = document()
    doc["people"]["pose_keypoints_2d"][0] = value
    result = experiment.experiment(archive(tmp_path, [(0, doc)]))
    assert sum(result["errors"].values()) == 1
    assert result["strict_groups"] == 0
    assert result["coordinates"]["pose_keypoints_2d"]["min_xy"] == [None, None]


def test_source_adapter_still_requires_evidence():
    experiment.load_source()
    import sys
    adapter = sys.modules["_crowd_experiment_source.crowd_keypoint_adapter"]
    assert adapter.conversion_blockers(None) == ("source_spec_missing",)
    with pytest.raises(ValueError, match="source_spec_missing"):
        adapter.adapt_crowd_sequence([], [], video_id="NIA_SL_FS0001_CROWD18_F.mp4")


@pytest.mark.parametrize("duration", [0, -1, True, 10**400, float("inf"), None])
def test_label_metadata_invalid_duration_preserves_unknown_fps(tmp_path, duration):
    keypoints = archive(tmp_path, [(0, document())])
    labels = tmp_path / "labels.zip"
    with zipfile.ZipFile(labels, "w") as stream:
        stream.writestr("label.json", json.dumps(dict(metaData=dict(
            name="NIA_SL_FS0001_CROWD18_F.mp4", duration=duration))))
    result = experiment.label_metadata(labels, keypoints)
    assert result["errors"] == {"duration": 1}
    assert result["exact_keypoint_matches"] == 1
    assert result["fps"] == "unverified"
    assert result["sha256_before"] == result["sha256_after"]


def test_label_metadata_exact_names_and_duration(tmp_path):
    keypoints = archive(tmp_path, [(0, document())])
    labels = tmp_path / "labels.zip"
    with zipfile.ZipFile(labels, "w") as stream:
        for i, name in enumerate(("NIA_SL_FS0001_CROWD18_F.mp4", "NIA_SL_FS0001_CROWD19_F.mp4")):
            stream.writestr(f"label{i}.json", json.dumps(dict(metaData=dict(name=name, duration=7.967))))
    result = experiment.label_metadata(labels, keypoints)
    assert result["duration_range_seconds"] == [7.967, 7.967]
    assert result["label_without_keypoint"] == 1
    assert result["keypoint_without_label"] == 0
    assert result["errors"] == {}
