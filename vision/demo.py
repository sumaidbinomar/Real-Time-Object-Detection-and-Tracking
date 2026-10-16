"""Turn a slice of the annotated video into a small GIF (for the README)."""
from pathlib import Path

import cv2
from PIL import Image


def make_demo_gif(video: Path, out: Path, start_frame: int = 250, frames: int = 50,
                  scale: float = 0.4, step: int = 1, fps: int = 10) -> None:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    images = []
    for i in range(frames):
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not images:
        raise RuntimeError(f"no frames read from {video}")
    images[0].save(out, save_all=True, append_images=images[1:], duration=int(1000 / fps), loop=0, optimize=True)
