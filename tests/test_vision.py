import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision import STAGES, CentroidTracker, Config, Pipeline


# ---------- tracker ----------
def test_tracker_keeps_id_for_a_moving_object():
    tr = CentroidTracker(max_distance=30)
    first = tr.update([(10, 10)])
    second = tr.update([(18, 12)])
    assert first == second == [1]


def test_tracker_new_object_gets_new_id():
    tr = CentroidTracker(max_distance=30)
    tr.update([(10, 10)])
    assert tr.update([(12, 10), (200, 200)]) == [1, 2]
    assert tr.total_ids_created == 2


def test_tracker_matches_closest_pairs_without_swapping():
    tr = CentroidTracker(max_distance=100)
    tr.update([(0, 0), (50, 0)])
    assert tr.update([(52, 0), (2, 0)]) == [2, 1]


def test_tracker_drops_lost_tracks_then_reissues_new_id():
    tr = CentroidTracker(max_distance=30, max_missing=2)
    tr.update([(10, 10)])
    for _ in range(3):
        tr.update([])
    assert tr.tracks == {}
    assert tr.update([(10, 10)]) == [2]


# ---------- pipeline ----------
def synthetic_frames(n=40, size=(240, 320)):
    """Static noisy-free background with one bright 20x50 'person' walking left to right."""
    frames = []
    for i in range(n):
        f = np.full((*size, 3), 90, np.uint8)
        x = 20 + 6 * i
        f[100:150, x:x + 20] = 230
        frames.append(f)
    return frames


def run(frames, cfg=None):
    pipe = Pipeline(cfg or Config(bg_history=30, min_area=100))
    return pipe, [pipe.process(f) for f in frames]


def test_pipeline_detects_and_tracks_one_moving_object():
    pipe, results = run(synthetic_frames())
    last = results[-1]
    assert len(last.detections) == 1
    assert last.detections[0].label == "person-like"
    assert pipe.tracker.total_ids_created == 1          # one persistent ID, no flicker


def test_pipeline_outputs_are_well_formed():
    frames = synthetic_frames(10)
    _, results = run(frames)
    r = results[-1]
    assert r.annotated.shape == frames[0].shape
    assert r.mask.shape == frames[0].shape[:2] and set(np.unique(r.mask)) <= {0, 255}
    assert set(STAGES) | {"total"} == set(r.timings_ms)
    assert r.timings_ms["total"] >= sum(r.timings_ms[s] for s in STAGES) * 0.99
    assert sum(r.counts.values()) == len(r.detections)


def test_empty_scene_has_no_detections():
    static = [np.full((120, 160, 3), 100, np.uint8)] * 15
    _, results = run(static)
    assert all(len(r.detections) == 0 for r in results)


def test_classification_rules():
    pipe = Pipeline(Config())
    assert pipe.classify(20, 60) == "person-like"
    assert pipe.classify(40, 30) == "small object"
    assert pipe.classify(100, 120) == "large blob"


def test_tracker_min_hits_ignores_flicker():
    tr = CentroidTracker(max_distance=30, max_missing=1, min_hits=3)
    tr.update([(10, 10)])                  # seen once, then gone
    tr.update([]); tr.update([])
    assert tr.total_ids_created == 0
    for _ in range(3):
        tr.update([(100, 100)])            # persists -> counts
    assert tr.total_ids_created == 1


def test_shape_filter_rejects_thin_streaks_and_slivers():
    pipe = Pipeline(Config())
    assert pipe.keep_shape(20, 60)          # person-shaped
    assert not pipe.keep_shape(60, 12)      # thin horizontal streak
    assert not pipe.keep_shape(5, 80)       # sliver


def test_warmup_suppresses_early_detections():
    pipe, results = run(synthetic_frames(30), Config(bg_history=30, min_area=100, warmup_frames=25, track_min_hits=1))
    assert all(len(r.detections) == 0 for r in results[:25])
