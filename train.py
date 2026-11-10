"""Steps 6-7: train a real/fake classifier on window features and test it fairly.

Usage:
    python train.py --metadata C:/data/dfdc/train_sample_videos/metadata.json
    python train.py --metadata ... --signals signals --model svm

metadata.json (from DFDC) says, for each video, whether it is REAL or FAKE and,
for fakes, which real video it was made from ("original").

Fair testing: a fake and the real video it was made from show the same person,
so they are kept in the same group, and GroupKFold never puts one group on both
sides of the train/test line. Every window votes; a video's answer is the
average of its windows' votes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import video_features


def load_dataset(signals_dir, metadata_path):
    """Return window features X, labels y (1 = fake), groups, and each window's video."""
    metadata = json.loads(Path(metadata_path).read_text())
    X, y, groups, videos = [], [], [], []
    names, skipped = None, 0

    for csv_path in sorted(Path(signals_dir).glob("*.csv")):
        info = metadata.get(f"{csv_path.stem}.mp4")
        if info is None:
            continue
        rows, row_names = video_features(csv_path)
        if len(rows) == 0:
            skipped += 1
            continue
        names = row_names

        is_fake = info["label"] == "FAKE"
        # A real video is its own group; a fake joins the group of its original.
        # (FaceForensics++ sets "group" itself: a fake mixes two people.)
        group = info.get("group") or (info.get("original") if is_fake else f"{csv_path.stem}.mp4")
        for row in rows:
            X.append(row)
            y.append(int(is_fake))
            groups.append(group)
            videos.append(csv_path.stem)

    print(f"Skipped {skipped} videos where the face wasn't found for long enough")
    return np.array(X), np.array(y), np.array(groups), np.array(videos), names


def make_model(kind):
    # class_weight="balanced": DFDC has far more fakes than reals, and without
    # this the model can score well just by always saying "fake".
    if kind == "svm":
        # An SVM only says which side of its line a point is on; the calibration
        # wrapper turns that into a probability so windows can be averaged.
        clf = CalibratedClassifierCV(SVC(class_weight="balanced"), ensemble=False)
    else:
        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    # StandardScaler puts every feature on the same scale before the classifier sees it.
    return make_pipeline(StandardScaler(), clf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="DFDC metadata.json")
    parser.add_argument("--signals", default="signals", help="folder of csv files from main.py --batch")
    parser.add_argument("--model", choices=["logreg", "svm"], default="logreg")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    X, y, groups, videos, names = load_dataset(args.signals, args.metadata)
    if len(X) == 0:
        raise SystemExit("No usable videos found. Check --signals and --metadata.")
    n_videos = len(set(videos))
    n_fake = len(set(videos[y == 1]))
    print(f"{n_videos} videos ({n_fake} fake, {n_videos - n_fake} real), "
          f"{len(X)} windows, {len(set(groups))} groups, {X.shape[1]} features\n")

    # Each window gets a fake-probability from a model that never saw its group.
    window_prob = np.zeros(len(X))
    for train_idx, test_idx in GroupKFold(n_splits=args.folds).split(X, y, groups):
        model = make_model(args.model).fit(X[train_idx], y[train_idx])
        window_prob[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    # Windows vote: a video's score is the average of its windows' scores.
    video_ids = sorted(set(videos))
    video_prob = np.array([window_prob[videos == v].mean() for v in video_ids])
    video_true = np.array([y[videos == v][0] for v in video_ids])
    video_pred = (video_prob > 0.5).astype(int)

    print("Results on people the model never trained on:")
    print(f"  windows: balanced accuracy {balanced_accuracy_score(y, window_prob > 0.5):.1%}")
    print(f"  videos:  balanced accuracy {balanced_accuracy_score(video_true, video_pred):.1%}")
    print(f"           AUC {roc_auc_score(video_true, video_prob):.2f}")
    print(f"           plain accuracy {np.mean(video_pred == video_true):.1%} "
          f"(always saying 'fake' would get {video_true.mean():.1%})")

    tn, fp, fn, tp = confusion_matrix(video_true, video_pred).ravel()
    print("\n  videos        called real  called fake")
    print(f"  really real   {tn:11d}  {fp:11d}")
    print(f"  really fake   {fn:11d}  {tp:11d}")
    print("\n  Balanced accuracy: average of 'reals called real' and 'fakes called fake'.")
    print("  50% = coin flip.  AUC: 0.5 = coin flip, 1.0 = perfect.")

    if args.model == "logreg":
        # Fit once on everything just to see which features the model leans on.
        coefs = make_model("logreg").fit(X, y)[-1].coef_[0]
        print("\nFeatures, biggest influence first (+ pushes towards fake):")
        for i in np.argsort(-np.abs(coefs)):
            print(f"  {coefs[i]:+.2f}  {names[i]}")


if __name__ == "__main__":
    main()
