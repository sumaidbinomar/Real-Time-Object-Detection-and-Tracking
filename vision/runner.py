"""Run the pipeline over a video (file or webcam) and collect frame-by-frame metrics."""
import csv
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np

from .pipeline import CLASSES, STAGES, Config, Pipeline


def _open_writer(path: Path, fps: float, size) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    return writer


def process_video(source: Union[str, int], config: Config = None, out_dir: Optional[Path] = None,
                  max_frames: Optional[int] = None, scale: float = 1.0, show: bool = False,
                  sample_every: int = 0) -> Dict:
    """Process `source` frame by frame.

    source        path to a video file, or a webcam index (e.g. 0)
    out_dir       if given, write annotated.mp4, mask.mp4 and sampled frames there
    scale         resize factor applied to every frame before processing
    show          display a live window (press q to quit)
    sample_every  save every n-th annotated frame + mask as PNG (0 = off)

    Returns {"rows": per-frame metrics, "summary": aggregate metrics, "config": ...}.
    """
    config = config or Config()
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video source {source!r}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    pipe = Pipeline(config)

    writer_ann = writer_mask = None
    frame_size = None
    rows: List[Dict] = []
    wall_start = time.perf_counter()
    index = 0
    while max_frames is None or index < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frame_size = [frame.shape[1], frame.shape[0]]
        result = pipe.process(frame)
        index += 1

        row = {"frame": index, "time_s": round((index - 1) / src_fps, 3),
               "detections": len(result.detections), **{f"count_{c}": result.counts[c] for c in CLASSES},
               "unique_objects": pipe.tracker.total_ids_created,
               **{f"{s}_ms": round(result.timings_ms[s], 4) for s in STAGES},
               "total_ms": round(result.timings_ms["total"], 4),
               "fps": round(1000 / result.timings_ms["total"], 2)}
        rows.append(row)

        if out_dir is not None:
            h, w = frame.shape[:2]
            if writer_ann is None:
                writer_ann = _open_writer(out_dir / "annotated.mp4", src_fps, (w, h))
                writer_mask = _open_writer(out_dir / "mask.mp4", src_fps, (w, h))
            writer_ann.write(result.annotated)
            writer_mask.write(cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR))
            if sample_every and index % sample_every == 0:
                frames_dir = out_dir / "frames"
                frames_dir.mkdir(exist_ok=True)
                cv2.imwrite(str(frames_dir / f"frame_{index:04d}.png"), result.annotated)
                cv2.imwrite(str(frames_dir / f"mask_{index:04d}.png"), result.mask)
        if show:
            cv2.imshow("Real-time detection (q to quit)", result.annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    wall = time.perf_counter() - wall_start
    cap.release()
    for w in (writer_ann, writer_mask):
        if w is not None:
            w.release()
    if show:
        cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError(f"no frames could be read from {source!r}")
    summary = summarize(rows, wall, pipe.tracker.total_ids_created)
    summary["frame_size"] = frame_size
    summary["scale"] = scale
    return {"rows": rows, "summary": summary, "config": asdict(config)}


def summarize(rows: List[Dict], wall_s: float, unique_objects: int) -> Dict:
    """Aggregate per-frame metrics. Warm-up frames are included in timings but excluded from detection stats."""
    totals = [r["total_ms"] for r in rows]
    ordered = sorted(totals)
    p95 = ordered[int(0.95 * (len(ordered) - 1))]
    mean_total = statistics.mean(totals)
    stage_mean = {s: statistics.mean(r[f"{s}_ms"] for r in rows) for s in STAGES}
    return {
        "frames": len(rows),
        "pipeline_ms": {"mean": mean_total, "median": statistics.median(totals), "p95": p95,
                        "min": min(totals), "max": max(totals), "stdev": statistics.pstdev(totals)},
        "pipeline_fps_mean": 1000 / mean_total,
        "end_to_end_fps": len(rows) / wall_s,   # includes video decoding and encoding
        "stage_ms_mean": stage_mean,
        "stage_share": {s: stage_mean[s] / sum(stage_mean.values()) for s in STAGES},
        "detections_per_frame": {"mean": statistics.mean(r["detections"] for r in rows),
                                 "max": max(r["detections"] for r in rows)},
        "mean_count_per_class": {c: statistics.mean(r[f"count_{c}"] for r in rows) for c in CLASSES},
        "unique_objects": unique_objects,
    }


def save_metrics(result: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "frame_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    payload = {"summary": result["summary"], "config": result["config"]}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
