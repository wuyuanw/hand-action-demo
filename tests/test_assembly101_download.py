import csv
import os
from pathlib import Path

from tools.download_assembly101 import (
    FIXED_VIEWS,
    HF_CN_ENDPOINT,
    configure_hf_endpoint,
    hf_recording_patterns,
    lowres_filename,
    recordings_from_annotations,
    select_covering_recordings,
    split_videos,
)


def write_split(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["video", "verb_cls"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_v8_pattern_for_selected_recording():
    patterns = hf_recording_patterns(["rec_a"], "v8")
    assert patterns == [f"recordings/rec_a/{FIXED_VIEWS['v8']}"]


def test_all_recordings_v8_uses_wildcard():
    patterns = hf_recording_patterns(["all"], "v8")
    assert patterns == [f"recordings/*/{FIXED_VIEWS['v8']}"]


def test_720p_filename_mapping():
    assert lowres_filename("C10404_rgb.mp4") == "C10404_rgb_low.mp4"
    assert lowres_filename("HMC_21176875_mono10bit.mp4") == "HMC_21176875_mono10bit_low.mp4"


def test_recording_selection_filters_by_verb(tmp_path: Path):
    rows = [
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "screw"},
        {"video": "rec_a/C10404_rgb.mp4", "verb_cls": "pick up"},
        {"video": "rec_c/C10404_rgb.mp4", "verb_cls": "unscrew"},
    ]
    for split in ("train.csv", "validation.csv", "test.csv"):
        write_split(tmp_path / split, rows)

    selected = recordings_from_annotations(tmp_path, {"screw", "unscrew"})
    assert selected == ["rec_b", "rec_c"]


def test_recording_selection_respects_available_720p_set(tmp_path: Path):
    rows = [
        {"video": "rec_a/C10404_rgb.mp4", "verb_cls": "pick up"},
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "screw"},
    ]
    for split in ("train.csv", "validation.csv", "test.csv"):
        write_split(tmp_path / split, rows)

    selected = recordings_from_annotations(tmp_path, None, {"rec_b"})
    assert selected == ["rec_b"]


def test_single_recording_prefers_maximum_target_verb_coverage(tmp_path: Path):
    rows = [
        {"video": "rec_a/C10404_rgb.mp4", "verb_cls": "pick up"},
        {"video": "rec_a/C10404_rgb.mp4", "verb_cls": "put down"},
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "pick up"},
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "put down"},
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "position"},
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "screw"},
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "unscrew"},
    ]
    write_split(tmp_path / "train.csv", rows)

    labels = {"pick up", "put down", "position", "screw", "unscrew"}
    selected, covered, _ = select_covering_recordings(tmp_path, "train.csv", labels, 1)
    assert selected == ["rec_b"]
    assert covered == labels


def test_default_selection_stops_when_all_labels_are_covered(tmp_path: Path):
    labels = {"pick up", "put down"}
    rows = [
        {"video": "rec_a/C10404_rgb.mp4", "verb_cls": "pick up"},
        {"video": "rec_a/C10404_rgb.mp4", "verb_cls": "put down"},
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "pick up"},
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "put down"},
        {"video": "rec_c/C10404_rgb.mp4", "verb_cls": "pick up"},
        {"video": "rec_c/C10404_rgb.mp4", "verb_cls": "put down"},
    ]
    write_split(tmp_path / "train.csv", rows)

    selected, covered, _ = select_covering_recordings(tmp_path, "train.csv", labels, 3)
    assert len(selected) == 1
    assert covered == labels


def test_fill_to_limit_keeps_selecting_recordings_for_training(tmp_path: Path):
    labels = {"pick up", "put down"}
    rows = [
        {"video": "rec_a/C10404_rgb.mp4", "verb_cls": "pick up"},
        {"video": "rec_a/C10404_rgb.mp4", "verb_cls": "put down"},
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "pick up"},
        {"video": "rec_b/C10404_rgb.mp4", "verb_cls": "put down"},
        {"video": "rec_c/C10404_rgb.mp4", "verb_cls": "pick up"},
        {"video": "rec_c/C10404_rgb.mp4", "verb_cls": "put down"},
    ]
    write_split(tmp_path / "train.csv", rows)

    selected, covered, _ = select_covering_recordings(
        tmp_path,
        "train.csv",
        labels,
        3,
        fill_to_limit=True,
    )
    assert len(selected) == 3
    assert set(selected) == {"rec_a", "rec_b", "rec_c"}
    assert covered == labels


def test_cn_mirror_sets_endpoint_and_disables_xet(monkeypatch):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    endpoint = configure_hf_endpoint("cn")
    assert endpoint == HF_CN_ENDPOINT
    assert os.environ["HF_ENDPOINT"] == HF_CN_ENDPOINT
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_split_videos():
    assert split_videos("all") == ["all"]
    assert split_videos("rec_a, rec_b") == ["rec_a", "rec_b"]
