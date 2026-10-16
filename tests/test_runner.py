import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision import CLASSES, STAGES, Config
from vision.charts import plot_all
from vision.runner import process_video, save_metrics


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A tiny synthetic clip: one bright 'person' walking across a static background."""
    path = tmp_path_factory.mktemp("clip") / "walk.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for i in range(50):
        f = np.full((240, 320, 3), 90, np.uint8)
        f[100:150, 20 + 5 * i:40 + 5 * i] = 230
        writer.write(f)
    writer.release()
    return path


CFG = Config(bg_history=30, min_area=100, warmup_frames=10, track_min_hits=3)


def test_process_video_rows_and_summary(video):
    result = process_video(str(video), CFG)
    rows, s = result["rows"], result["summary"]
    assert len(rows) == s["frames"] == 50
    assert set(f"{st}_ms" for st in STAGES) <= set(rows[0])
    assert s["pipeline_fps_mean"] > 0 and s["end_to_end_fps"] > 0
    assert s["pipeline_ms"]["min"] <= s["pipeline_ms"]["median"] <= s["pipeline_ms"]["p95"] <= s["pipeline_ms"]["max"]
    assert sum(s["stage_share"].values()) == pytest.approx(1.0)
    assert s["unique_objects"] == 1


def test_max_frames_and_scale(video):
    result = process_video(str(video), CFG, max_frames=12, scale=0.5)
    assert result["summary"]["frames"] == 12
    assert result["summary"]["frame_size"] == [160, 120]


def test_outputs_written(video, tmp_path):
    result = process_video(str(video), CFG, out_dir=tmp_path, sample_every=20)
    save_metrics(result, tmp_path)
    plot_all(result, tmp_path)
    for name in ("annotated.mp4", "mask.mp4", "frame_metrics.csv", "summary.json",
                 "chart_stage_timing.png", "chart_fps_over_time.png",
                 "chart_latency_histogram.png", "chart_counts_over_time.png"):
        assert (tmp_path / name).stat().st_size > 0, name
    assert len(list((tmp_path / "frames").glob("frame_*.png"))) == 2
    with open(tmp_path / "frame_metrics.csv", newline="") as f:
        assert len(list(csv.DictReader(f))) == 50
    assert json.loads((tmp_path / "summary.json").read_text())["summary"]["frames"] == 50
    cap = cv2.VideoCapture(str(tmp_path / "annotated.mp4"))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 50


def test_bad_source_raises():
    with pytest.raises(RuntimeError):
        process_video("does_not_exist.avi", CFG)
