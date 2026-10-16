"""Download the sample input video (pedestrians walking, from the OpenCV samples repository).

Output: data/vtest.avi
"""
import urllib.request
from pathlib import Path

import cv2

VIDEO_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"
VIDEO_PATH = Path(__file__).parent / "data" / "vtest.avi"


def download(url: str = VIDEO_URL, path: Path = VIDEO_PATH) -> Path:
    path.parent.mkdir(exist_ok=True)
    if not path.exists():
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path}")
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"{path.name}: {width}x{height}, {frames} frames at {fps:.1f} fps ({frames / fps:.1f} s)")
    return path


if __name__ == "__main__":
    download()
