"""Parameter sensitivity study: vary one setting at a time and record speed and counts.

    python run_experiments.py            # first 400 frames of the sample video, ~2-3 minutes
Writes results/experiments.csv and results/chart_parameter_sweeps.png.
"""
import csv
from dataclasses import replace
from pathlib import Path

from vision import Config
from vision.charts import plot_sweeps
from vision.runner import process_video

ROOT = Path(__file__).parent
SOURCE = ROOT / "data" / "vtest.avi"
RESULTS = ROOT / "results"
FRAMES = 400     # first 40 s of the clip, to keep the study quick
REPEATS = 2      # timing is noisy; keep the faster of two runs (counts are deterministic)

DEFAULT = Config()
GRID = {  # name -> (how to apply a value, values to try, default value)
    "blur_kernel": ("config", [1, 3, 5, 9, 15], DEFAULT.blur_kernel),
    "adaptive_block": ("config", [11, 21, 41, 81], DEFAULT.adaptive_block),
    "track_min_hits": ("config", [1, 3, 5, 8], DEFAULT.track_min_hits),
    "frame_scale": ("scale", [0.5, 0.75, 1.0], 1.0),
}


def evaluate(config: Config, scale: float) -> dict:
    best = None
    for _ in range(REPEATS):
        s = process_video(str(SOURCE), config, max_frames=FRAMES, scale=scale)["summary"]
        if best is None or s["pipeline_ms"]["mean"] < best["pipeline_ms"]["mean"]:
            best = s
    return {"mean_ms": round(best["pipeline_ms"]["mean"], 3), "fps": round(best["pipeline_fps_mean"], 1),
            "mean_detections": round(best["detections_per_frame"]["mean"], 3), "unique_objects": best["unique_objects"]}


def main() -> None:
    sweeps, flat = {}, []
    for name, (kind, values, default) in GRID.items():
        sweeps[name] = []
        for value in values:
            cfg, scale = (replace(DEFAULT, **{name: value}), 1.0) if kind == "config" else (DEFAULT, value)
            row = {"parameter": name, "value": value, **evaluate(cfg, scale)}
            sweeps[name].append(row)
            flat.append(row)
            print(f"{name}={value}: {row['mean_ms']} ms, {row['mean_detections']} det/frame, {row['unique_objects']} unique")
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "experiments.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    plot_sweeps(sweeps, {n: g[2] for n, g in GRID.items()}, RESULTS / "chart_parameter_sweeps.png")
    print("Saved experiments.csv and chart_parameter_sweeps.png")


if __name__ == "__main__":
    main()
