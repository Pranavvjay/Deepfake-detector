"""Run yarPPG on every UBFC video, as a second opinion for ubfc_eval.py.

yarPPG needs its own environment (.venv-yarppg), so run it with that Python:
    .venv-yarppg\\Scripts\\python yarppg_run.py C:/data/ubfc/DATASET_2

Writes yarppg_ubfc.csv: one heart rate per video for yarPPG's green and CHROM
methods. yarPPG's heart rates are "frames per beat"; we convert them to bpm.
"""

import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yarppg

OUT_PATH = "yarppg_ubfc.csv"
METHODS = ["green", "chrom"]
WINDOW_SECONDS = 10  # yarPPG gives no heart rate until it has this much video


def run_one(video):
    fps = yarppg.get_video_fps(str(video))
    detector = yarppg.FaceMeshDetector()
    processors = {"green": yarppg.Processor(), "chrom": yarppg.ChromProcessor()}
    calcs = {m: yarppg.PeakBasedHrCalculator(fs=fps, window_seconds=WINDOW_SECONDS) for m in METHODS}
    bpms = {m: [] for m in METHODS}

    # One face detection per frame, shared by both methods.
    for frame in yarppg.frames_from_video(str(video)):
        roi = detector.detect(frame)
        for m in METHODS:
            frames_per_beat = calcs[m].update(processors[m].process(roi).value)
            bpms[m].append(yarppg.bpm_from_frames_per_beat(frames_per_beat, fps))

    # yarPPG gives a heart rate every few frames; the video's answer is the median.
    return video.parent.name, {m: float(np.nanmedian(bpms[m])) for m in METHODS}


def main():
    videos = sorted(Path(sys.argv[1]).glob("*/vid.avi"))
    print(f"{len(videos)} videos")
    start = time.monotonic()
    with open(OUT_PATH, "w", newline="") as f, Pool(4) as pool:
        writer = csv.writer(f)
        writer.writerow(["subject"] + [f"{m}_bpm" for m in METHODS])
        for i, (subject, bpm) in enumerate(pool.imap_unordered(run_one, videos), 1):
            writer.writerow([subject] + [f"{bpm[m]:.1f}" for m in METHODS])
            left = (time.monotonic() - start) / i * (len(videos) - i)
            print(f"[{i}/{len(videos)}] {subject}: green {bpm['green']:.0f}, "
                  f"chrom {bpm['chrom']:.0f} bpm  (about {left / 60:.0f} min left)")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
