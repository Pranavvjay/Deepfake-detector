"""Steps 1-2: find skin patches (forehead + both cheeks) and record their colour.

Usage:
    python main.py                          # use the webcam, saves signal.csv
    python main.py video.mp4                # use a video file
    python main.py --batch C:/data/dfdc/train_sample_videos --out signals
                                            # every video in a folder, no window,
                                            # one csv per video in signals/

Keys (window mode):
    n  toggle landmark numbers (use this to pick patch landmarks)
    q  quit (saves the signal)
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from multiprocessing import Pool
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# Model files live next to this script, so it works from any folder.
MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATH = str(MODELS_DIR / "face_landmarker.task")

# MediaPipe is built for close-up faces and misses small, far-away ones (common
# in DFDC). For video files, YuNet first finds the face, and MediaPipe then works
# on a crop around it where the face is big.
YUNET_PATH = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
DETECT_WIDTH = 640  # YuNet looks at a smaller copy of the frame, for speed
CROP_SCALE = 2.0  # crop side = this x the face's size, so the whole head fits
CROP_SIZE = 256  # crops are resized to this before MediaPipe sees them
REDETECT_FRAMES = 10  # run YuNet again every this many frames
SIGNAL_PATH = "signal.csv"
PLOT_SECONDS = 10  # how much signal the live graph shows
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Skin regions. More skin pixels averaged = less noise, so instead of small boxes
# we take all the skin inside the face outline, minus eyes, eyebrows and mouth,
# and split it into three regions:
#   forehead     = above the eyebrow line
#   left cheek   = below the eyes, on the person's own left of the nose
#   right cheek  = below the eyes, on the person's own right
# Landmark sets come from MediaPipe (its face mesh has 478 points).
def _ids(connections):
    return sorted({i for c in connections for i in (c.start, c.end)})


_C = vision.FaceLandmarksConnections
FACE_OVAL = [c.start for c in _C.FACE_LANDMARKS_FACE_OVAL]  # in order around the face
EXCLUDE = [  # areas cut out of the skin: each eye with its eyebrow, and the lips
    _ids(_C.FACE_LANDMARKS_LEFT_EYE + _C.FACE_LANDMARKS_LEFT_EYEBROW),
    _ids(_C.FACE_LANDMARKS_RIGHT_EYE + _C.FACE_LANDMARKS_RIGHT_EYEBROW),
    _ids(_C.FACE_LANDMARKS_LIPS),
]
OVAL_SHRINK = 0.9  # pull the outline in a little, to stay off hair, ears and background
EXCLUDE_GROW = 1.2  # push the cut-outs out a little, to clear eyelashes and lip edges
BROW_LINE = (105, 334)  # middle of right and left eyebrow
EYE_LINE = (145, 374)  # bottom of right and left eye
MIDLINE = (168, 2)  # between the eyes, and just under the nose
LEFT_SIDE_POINT = 263  # left eye's outer corner: tells us which side is "left"

PATCHES = ["forehead", "left_cheek", "right_cheek"]
PATCH_COLOURS = {  # BGR, for drawing
    "forehead": (0, 0, 255),
    "left_cheek": (255, 0, 255),
    "right_cheek": (255, 255, 0),
}


def create_landmarker():
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def find_landmarks(landmarker, frame_bgr, timestamp_ms):
    """Return a (478, 2) array of pixel coordinates, or None if no face."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, timestamp_ms)
    if not result.face_landmarks:
        return None

    # MediaPipe gives x, y as fractions of the image size (0 to 1).
    h, w = frame_bgr.shape[:2]
    points = [(lm.x * w, lm.y * h) for lm in result.face_landmarks[0]]
    return np.array(points)


class FaceCropper:
    """Finds the face with YuNet and keeps a steady square crop around it."""

    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(YUNET_PATH, "", (DETECT_WIDTH, DETECT_WIDTH), 0.6)
        self.crop = None  # (centre x, centre y, side) in full-frame pixels
        self.frames_since_detect = 0

    def detect(self, frame):
        h, w = frame.shape[:2]
        scale = DETECT_WIDTH / w
        small = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)))
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(small)
        if faces is None:
            return None
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])[:4] / scale  # biggest face
        return x + fw / 2, y + fh / 2, max(fw, fh) * CROP_SCALE

    def update(self, frame):
        """Return the current crop, or None if there's no face."""
        self.frames_since_detect += 1
        if self.crop is not None and self.frames_since_detect < REDETECT_FRAMES:
            return self.crop
        self.frames_since_detect = 0
        found = self.detect(frame)
        if found is None:
            self.crop = None
        elif self.crop is None:
            self.crop = found
        else:
            # Only move the crop if the face really moved, so it doesn't jitter.
            cx, cy, side = self.crop
            moved = np.hypot(found[0] - cx, found[1] - cy) > 0.1 * side
            resized = abs(found[2] - side) > 0.15 * side
            if moved or resized:
                self.crop = found
        return self.crop


def find_landmarks_cropped(landmarker, cropper, frame_bgr, timestamp_ms):
    """Like find_landmarks, but on a crop around the face. Points come back
    in full-frame pixels, so patch colours are measured on the original frame."""
    crop = cropper.update(frame_bgr)
    if crop is None:
        return None
    cx, cy, side = crop
    # getRectSubPix fills any part of the crop that falls off the frame edge.
    patch = cv2.getRectSubPix(frame_bgr, (int(side), int(side)), (cx, cy))
    patch = cv2.resize(patch, (CROP_SIZE, CROP_SIZE))
    points = find_landmarks(landmarker, patch, timestamp_ms)
    if points is None:
        return None
    return points * (side / CROP_SIZE) + (cx - side / 2, cy - side / 2)


def scaled(pts, factor):
    """Shrink (factor < 1) or grow (factor > 1) a set of points about their centre."""
    centre = pts.mean(axis=0)
    return centre + (pts - centre) * factor


def side_of_line(xs, ys, p1, p2):
    """For each pixel, which side of the line p1 -> p2 it is on (+ or -)."""
    return (p2[0] - p1[0]) * (ys - p1[1]) - (p2[1] - p1[1]) * (xs - p1[0])


def patch_masks(points, frame_shape):
    """Return (x0, y0, {patch: mask}). Each mask covers the face's bounding box,
    whose top-left corner in the frame is (x0, y0); True = skin in that patch."""
    h, w = frame_shape[:2]
    oval = scaled(points[FACE_OVAL], OVAL_SHRINK)
    x0, y0 = np.floor(oval.min(axis=0)).astype(int).clip(0)
    x1, y1 = np.ceil(oval.max(axis=0)).astype(int)
    x1, y1 = min(x1, w), min(y1, h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return x0, y0, None

    # Draw the skin area: the face outline, minus the cut-outs.
    skin = np.zeros((y1 - y0, x1 - x0), np.uint8)
    offset = np.array([x0, y0])
    cv2.fillPoly(skin, [np.round(oval - offset).astype(np.int32)], 1)
    for ids in EXCLUDE:
        hull = cv2.convexHull(np.round(scaled(points[ids], EXCLUDE_GROW) - offset).astype(np.int32))
        cv2.fillPoly(skin, [hull], 0)
    skin = skin.astype(bool)

    # Split it into regions using lines through landmarks. Lines follow the
    # face, so the split still works when the head is tilted.
    ys, xs = np.mgrid[y0:y1, x0:x1]
    nose_tip = points[MIDLINE[1]]
    brow = side_of_line(xs, ys, points[BROW_LINE[0]], points[BROW_LINE[1]])
    eyes = side_of_line(xs, ys, points[EYE_LINE[0]], points[EYE_LINE[1]])
    mid = side_of_line(xs, ys, points[MIDLINE[0]], points[MIDLINE[1]])
    # The nose tip is below both lines, and the left eye corner is on the left
    # of the midline; comparing signs with them avoids caring which way is "up".
    below_brow = np.sign(brow) == np.sign(side_of_line(*nose_tip, *points[list(BROW_LINE)]))
    below_eyes = np.sign(eyes) == np.sign(side_of_line(*nose_tip, *points[list(EYE_LINE)]))
    left_side = np.sign(mid) == np.sign(side_of_line(*points[LEFT_SIDE_POINT], *points[list(MIDLINE)]))

    return x0, y0, {
        "forehead": skin & ~below_brow,
        "left_cheek": skin & below_eyes & left_side,
        "right_cheek": skin & below_eyes & ~left_side,
    }


def patch_mean_rgb(frame_bgr, x0, y0, mask):
    """Average red, green and blue of the pixels where the mask is True."""
    if mask is None or mask.sum() < 10:
        return None
    region = frame_bgr[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
    b, g, r = region[mask].mean(axis=0)
    return r, g, b


def draw_signals(frame, recent, height=120):
    """Draw each patch's recent green values as a line along the bottom."""
    h, w = frame.shape[:2]
    top = h - height
    cv2.rectangle(frame, (0, top), (w, h), (0, 0, 0), -1)

    for name, values in recent.items():
        if len(values) < 2:
            continue
        # The pulse is a tiny wobble, so stretch each line to fill the panel.
        v = np.array(values)
        span = max(v.max() - v.min(), 1e-6)
        ys = top + height - 10 - (v - v.min()) / span * (height - 20)
        xs = np.linspace(0, w - 1, len(v))
        line = np.column_stack([xs, ys]).astype(np.int32)
        cv2.polylines(frame, [line], False, PATCH_COLOURS[name], 1)


def save_signal(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["time_s"]
        for name in PATCHES:
            header += [f"{name}_r", f"{name}_g", f"{name}_b"]
        writer.writerow(header)
        writer.writerows(rows)


def draw(frame, points, masks, show_numbers):
    # Tint each patch's skin in its colour.
    x0, y0, patch = masks
    if patch is not None:
        region = frame[y0:y0 + next(iter(patch.values())).shape[0],
                       x0:x0 + next(iter(patch.values())).shape[1]]
        for name, mask in patch.items():
            region[mask] = 0.6 * region[mask] + 0.4 * np.array(PATCH_COLOURS[name])

    for i, (x, y) in enumerate(points):
        if show_numbers:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (int(x), int(y)),
                        cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 255), 1)


def record(source, show=True):
    """Read a webcam (source=0) or video file and return one row per frame:
    time_s, then r, g, b for each patch."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise OSError(f"Could not open {source!r}")

    # Every beats-per-minute number later depends on this.
    fps = cap.get(cv2.CAP_PROP_FPS)
    is_file = source != 0
    if show:
        print(f"Reported FPS: {fps}")

    landmarker = create_landmarker()
    cropper = FaceCropper() if is_file else None
    show_numbers = False
    start = time.monotonic()
    frame_count = 0
    rows = []  # time_s, then r, g, b for each patch, for every frame with a face
    recent = {name: deque(maxlen=int(PLOT_SECONDS * (fps or 30))) for name in PATCHES}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # A file's frames are evenly spaced at its FPS, however slowly we process
        # them. A webcam's are not, so there we use the clock.
        if is_file:
            timestamp_ms = int(frame_count * 1000 / fps)
        else:
            timestamp_ms = int((time.monotonic() - start) * 1000)
        frame_count += 1
        if cropper is not None:
            points = find_landmarks_cropped(landmarker, cropper, frame, timestamp_ms)
        else:
            points = find_landmarks(landmarker, frame, timestamp_ms)

        if points is not None:
            # Measure before drawing, so the tint doesn't change the colour.
            masks = patch_masks(points, frame.shape)
            x0, y0, patch = masks
            colours = {name: patch_mean_rgb(frame, x0, y0, patch and patch[name])
                       for name in PATCHES}
            # Only keep frames where every patch is on screen, so the columns line up.
            if all(c is not None for c in colours.values()):
                row = [timestamp_ms / 1000]
                for name in PATCHES:
                    row += colours[name]
                    recent[name].append(colours[name][1])
                rows.append(row)

        if not show:
            continue

        if points is None:
            cv2.putText(frame, "No face", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            draw(frame, points, masks, show_numbers)

        draw_signals(frame, recent)
        measured_fps = frame_count / max(time.monotonic() - start, 1e-6)
        cv2.putText(frame, f"{measured_fps:.1f} fps", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Step 3 needs at least 20-30 s; turns green when there's enough.
        recorded = rows[-1][0] - rows[0][0] if len(rows) > 1 else 0
        colour = (0, 255, 0) if recorded >= 30 else (0, 165, 255)
        cv2.putText(frame, f"recorded {recorded:.0f}s / 30s", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)

        cv2.imshow("Step 2: patch signals", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n"):
            show_numbers = not show_numbers

    cap.release()
    if show:
        cv2.destroyAllWindows()
    landmarker.close()
    return rows


def process_one(job):
    """Batch worker: record one video with no window and save its csv."""
    video, out_path = job
    try:
        rows = record(str(video), show=False)
    except OSError as e:
        return f"{video.name}: {e}"
    save_signal(rows, out_path)
    return f"{video.name}: {len(rows)} frames with a face"


def run_batch(folder, out_dir, workers):
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    videos = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    # Skip videos already done, so a stopped batch can carry on where it left off.
    jobs = [(v, out_dir / f"{v.stem}.csv") for v in videos if not (out_dir / f"{v.stem}.csv").exists()]
    print(f"{len(videos)} videos, {len(videos) - len(jobs)} already done, "
          f"{len(jobs)} to go, using {workers} workers")

    start = time.monotonic()
    with Pool(workers) as pool:
        for i, message in enumerate(pool.imap_unordered(process_one, jobs), 1):
            elapsed = time.monotonic() - start
            left = elapsed / i * (len(jobs) - i)
            print(f"[{i}/{len(jobs)}] {message}  (about {left / 60:.0f} min left)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", help="video file (default: webcam)")
    parser.add_argument("--batch", help="folder of videos to process with no window")
    parser.add_argument("--out", default="signals", help="where batch csv files go")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch, args.out, args.workers)
        return

    try:
        rows = record(args.source or 0)
    except OSError as e:
        sys.exit(str(e))
    save_signal(rows, SIGNAL_PATH)
    print(f"Saved {len(rows)} frames to {SIGNAL_PATH}")


if __name__ == "__main__":
    main()
