"""Run the detection and tracking pipeline on a video and export results to results/.

    python main.py                          # sample video -> annotated video, masks, metrics, charts
    python main.py --source path/to/clip.mp4
    python main.py --source 0 --show        # live webcam window (press q to quit)
    python main.py --max-frames 100         # quick run
"""
import argparse
from pathlib import Path

from vision import Config
from vision.charts import plot_all
from vision.demo import make_demo_gif
from vision.runner import process_video, save_metrics

ROOT = Path(__file__).parent
DEFAULT_SOURCE = ROOT / "data" / "vtest.avi"
RESULTS = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time object detection and tracking pipeline.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="video file path, or webcam index such as 0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--show", action="store_true", help="show a live window while processing")
    parser.add_argument("--out", default=str(RESULTS), help="output directory")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    if isinstance(source, str) and not Path(source).exists():
        raise SystemExit(f"{source} not found. Run `python download_video.py` first, or pass --source.")

    out = Path(args.out)
    print(f"Processing {source} ...")
    result = process_video(source, Config(), out_dir=out, max_frames=args.max_frames, show=args.show, sample_every=80)
    save_metrics(result, out)
    plot_all(result, out)
    if result["summary"]["frames"] > 310:  # enough footage for a short demo clip
        make_demo_gif(out / "annotated.mp4", out / "demo.gif")

    s = result["summary"]
    print(f"{s['frames']} frames | pipeline {s['pipeline_ms']['mean']:.1f} ms/frame "
          f"({s['pipeline_fps_mean']:.0f} FPS), end-to-end {s['end_to_end_fps']:.0f} FPS")
    print(f"mean detections/frame {s['detections_per_frame']['mean']:.2f}, unique objects {s['unique_objects']}")
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
