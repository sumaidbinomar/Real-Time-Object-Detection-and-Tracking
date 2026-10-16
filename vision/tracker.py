"""Centroid tracker: gives each detected object a persistent ID across frames.

Each frame, existing tracks are matched to new detections by nearest centroid
(closest pairs first, within `max_distance` pixels). Unmatched detections start new
tracks; tracks that stay unmatched for more than `max_missing` frames are dropped.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

Point = Tuple[float, float]


@dataclass
class Track:
    track_id: int
    centroid: Point
    missing: int = 0   # consecutive frames without a match
    age: int = 1       # frames this track has been alive


class CentroidTracker:
    def __init__(self, max_distance: float = 60.0, max_missing: int = 5, min_hits: int = 1):
        self.max_distance = max_distance
        self.max_missing = max_missing
        self.min_hits = min_hits          # frames a track must be matched before it counts as a real object
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.confirmed_ids = set()

    @property
    def total_ids_created(self) -> int:
        """How many distinct objects have been confirmed so far (flickers shorter than min_hits don't count)."""
        return len(self.confirmed_ids)

    def is_confirmed(self, track_id: int) -> bool:
        return track_id in self.confirmed_ids

    def update(self, centroids: List[Point]) -> List[int]:
        """Match detections to tracks. Returns one track ID per input centroid (same order)."""
        assigned: List[int] = [0] * len(centroids)

        # All (distance, track, detection) pairs within range, closest first (greedy matching).
        pairs = sorted(
            (math.dist(t.centroid, c), tid, i)
            for tid, t in self.tracks.items() for i, c in enumerate(centroids)
            if math.dist(t.centroid, c) <= self.max_distance
        )
        used_tracks, used_dets = set(), set()
        for _, tid, i in pairs:
            if tid in used_tracks or i in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(i)
            track = self.tracks[tid]
            track.centroid, track.missing, track.age = centroids[i], 0, track.age + 1
            if track.age >= self.min_hits:
                self.confirmed_ids.add(tid)
            assigned[i] = tid

        for tid in list(self.tracks):  # age out tracks that found no match
            if tid not in used_tracks:
                self.tracks[tid].missing += 1
                if self.tracks[tid].missing > self.max_missing:
                    del self.tracks[tid]

        for i, c in enumerate(centroids):  # unmatched detections become new objects
            if i not in used_dets:
                self.tracks[self.next_id] = Track(self.next_id, c)
                if self.min_hits <= 1:
                    self.confirmed_ids.add(self.next_id)
                assigned[i] = self.next_id
                self.next_id += 1
        return assigned
