"""Step 4, properly: how accurate is our heart rate? Tested on UBFC-rPPG.

UBFC-rPPG DATASET_2 has one folder per person, each with:
    vid.avi            the face video
    ground_truth.txt   a finger pulse sensor recorded at the same time
                       (line 1: pulse wave, line 2: heart rate, line 3: time in s)

Usage:
    python ubfc_eval.py --ubfc C:/data/ubfc/DATASET_2

1. Records every video with main.py's face tracking (saved in signals_ubfc/,
   so it only happens once).
2. Works out each video's heart rate with green, CHROM and POS.
3. Works out the true heart rate from the pulse sensor, the same way (FFT).
4. Prints the error for each method. If yarppg_ubfc.csv exists (from
   yarppg_run.py), yarPPG is compared too.
"""

import argparse
import csv
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import main
from analyze import FS, METHODS, bandpass, heart_rate, load, resample

YARPPG_RESULTS = "yarppg_ubfc.csv"


def record_one(job):
    video, out_path = job
    main.save_signal(main.record(str(video), show=False), out_path)
    return video.parent.name


def record_all(videos, out_dir, workers):
    out_dir.mkdir(exist_ok=True)
    jobs = [(v, out_dir / f"{v.parent.name}.csv") for v in videos
            if not (out_dir / f"{v.parent.name}.csv").exists()]
    if not jobs:
        return
    print(f"Recording {len(jobs)} videos with {workers} workers...")
    start = time.monotonic()
    with Pool(workers) as pool:
        for i, subject in enumerate(pool.imap_unordered(record_one, jobs), 1):
            left = (time.monotonic() - start) / i * (len(jobs) - i)
            print(f"  [{i}/{len(jobs)}] {subject}  (about {left / 60:.0f} min left)")


def true_bpm(ground_truth_path):
    """Heart rate from the finger sensor's pulse wave, using the same FFT method.
    nan if the file is missing (e.g. only part of the dataset was downloaded)."""
    if not Path(ground_truth_path).exists():
        return float("nan")
    rows = np.loadtxt(ground_truth_path)
    wave, times = rows[0], rows[2]
    _, wave = resample(times, wave, FS)
    return heart_rate(bandpass(wave, FS), FS)[0]


def our_bpm(csv_path, method):
    """Heart rate from all three patches added together."""
    t, patches = load(csv_path)
    waves = []
    for r, g, b in patches.values():
        r, g, b = (resample(t, c, FS)[1] for c in (r, g, b))
        wave = METHODS[method](r, g, b, FS)
        waves.append(wave / wave.std())
    return heart_rate(sum(waves), FS)[0]


def patch_bpms(csv_path, method="pos"):
    """Each patch's own heart rate, to see whether the patches agree."""
    t, patches = load(csv_path)
    bpms = []
    for r, g, b in patches.values():
        r, g, b = (resample(t, c, FS)[1] for c in (r, g, b))
        bpms.append(heart_rate(METHODS[method](r, g, b, FS), FS)[0])
    return "/".join(f"{b:.0f}" for b in bpms)


def no_truth_summary(all_guesses, columns):
    """Without ground truth: how far apart are the methods from each other?"""
    print("\nNo ground_truth.txt found, so no error against the real pulse.")
    print("Instead: how far each method is from our POS answer (bpm):")
    print(f"{'method':14s} {'mean gap':>9s} {'within 5 bpm':>13s}")
    for c in columns:
        if c == "pos":
            continue
        gap = np.array([g[c] - g["pos"] for g in all_guesses])
        gap = gap[~np.isnan(gap)]
        if len(gap):
            print(f"{c:14s} {np.abs(gap).mean():9.1f} {(np.abs(gap) <= 5).mean():13.0%}")


def load_yarppg():
    if not Path(YARPPG_RESULTS).exists():
        return {}
    with open(YARPPG_RESULTS) as f:
        return {row["subject"]: row for row in csv.DictReader(f)}


def main_():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ubfc", required=True, help="folder with subject1, subject2, ...")
    parser.add_argument("--out", default="signals_ubfc")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    videos = sorted(Path(args.ubfc).glob("*/vid.avi"),
                    key=lambda v: int("".join(filter(str.isdigit, v.parent.name)) or 0))
    if not videos:
        raise SystemExit(f"No */vid.avi found in {args.ubfc}")
    out_dir = Path(args.out)
    record_all(videos, out_dir, args.workers)

    yarppg = load_yarppg()
    columns = list(METHODS) + (["yarppg_green", "yarppg_chrom"] if yarppg else [])
    errors = {c: [] for c in columns}
    all_guesses = []

    print(f"\n{'subject':10s} {'true':>5s}  " + "  ".join(f"{c:>12s}" for c in columns)
          + "   POS per patch (forehead/left/right)")
    for video in videos:
        subject = video.parent.name
        truth = true_bpm(video.parent / "ground_truth.txt")
        guesses = {m: our_bpm(out_dir / f"{subject}.csv", m) for m in METHODS}
        if yarppg and subject in yarppg:
            guesses["yarppg_green"] = float(yarppg[subject]["green_bpm"])
            guesses["yarppg_chrom"] = float(yarppg[subject]["chrom_bpm"])
        guesses = {c: guesses.get(c, float("nan")) for c in columns}
        for c in columns:
            errors[c].append(guesses[c] - truth)
        all_guesses.append(guesses)
        print(f"{subject:10s} {truth:5.0f}  " + "  ".join(f"{guesses[c]:12.0f}" for c in columns)
              + "   " + patch_bpms(out_dir / f"{subject}.csv"))

    if all(np.isnan(e).all() for e in errors.values()):
        no_truth_summary(all_guesses, columns)
        return

    print(f"\nOver {len(videos)} videos (bpm):")
    print(f"{'method':14s} {'MAE':>6s} {'RMSE':>6s} {'within 5 bpm':>13s} {'videos':>7s}")
    for c in columns:
        e = np.array(errors[c])
        e = e[~np.isnan(e)]  # yarPPG gives nan when it lost the face
        print(f"{c:14s} {np.abs(e).mean():6.1f} {np.sqrt((e ** 2).mean()):6.1f} "
              f"{(np.abs(e) <= 5).mean():13.0%} {len(e):7d}")
    print("\nMAE = average error in bpm. RMSE punishes big misses more.")
    print("'videos' = how many videos the method gave an answer for.")


if __name__ == "__main__":
    main_()
