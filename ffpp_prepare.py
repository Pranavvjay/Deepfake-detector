"""Get downloaded FaceForensics++ videos ready for main.py --batch and train.py.

Usage:
    python ffpp_prepare.py C:/data/ffpp                      raw, Deepfakes only
    python ffpp_prepare.py C:/data/ffpp --compression c23 \\
        --methods Deepfakes Face2Face FaceSwap NeuralTextures

FaceForensics++ keeps real and fake videos in separate folders:
    original_sequences/youtube/<compression>/videos/000.mp4        real person 000
    manipulated_sequences/<method>/<compression>/videos/000_003.mp4  video 000,
                                                     faked using person 003
This script:
1. Links all of them into one folder, C:/data/ffpp/all_<compression>/ (hard
   links: no extra disk space). Fakes are renamed "<method>_000_003.mp4" so
   names can't clash. (The first raw run named them "fake_000_003.mp4" in all/.)
2. Writes metadata.json (every video) plus metadata_<method>.json (the real
   videos and one method's fakes), in the same format as DFDC, so train.py
   works as is and each method can be tested on its own.

Grouping: a fake like 000_003 involves two people (000 and 003). So 000, 003
and all their fakes go in one group. GroupKFold then never trains on any of
them while testing on another, so the model can't "recognise" people.
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def link(src, dst):
    """Same file under a second name; copy only if linking isn't possible."""
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--compression", default="raw", choices=["raw", "c23", "c40"])
    parser.add_argument("--methods", nargs="+", default=["Deepfakes"])
    args = parser.parse_args()

    root = Path(args.root)
    real_dir = root / f"original_sequences/youtube/{args.compression}/videos"
    fake_dirs = {m: root / f"manipulated_sequences/{m}/{args.compression}/videos"
                 for m in args.methods}
    out = root / ("all" if args.compression == "raw" else f"all_{args.compression}")
    out.mkdir(exist_ok=True)
    prefix = {m: "fake" if args.compression == "raw" else m for m in args.methods}

    # Fakes come in pairs (000_003 and 003_000), so they define the groups.
    group_of = {}
    for fake_dir in fake_dirs.values():
        for fake in fake_dir.glob("*.mp4"):
            a, b = fake.stem.split("_")
            group_of.setdefault(a, f"pair_{min(a, b)}_{max(a, b)}")
            group_of.setdefault(b, f"pair_{min(a, b)}_{max(a, b)}")

    reals = {}
    for real in sorted(real_dir.glob("*.mp4")):
        link(real, out / real.name)
        reals[real.name] = {"label": "REAL", "group": group_of.get(real.stem, real.stem)}

    everything = dict(reals)
    for method, fake_dir in fake_dirs.items():
        fakes = {}
        for fake in sorted(fake_dir.glob("*.mp4")):
            name = f"{prefix[method]}_{fake.name}"
            link(fake, out / name)
            target = fake.stem.split("_")[0]
            fakes[name] = {"label": "FAKE", "method": method,
                           "original": f"{target}.mp4", "group": group_of[target]}
        (out / f"metadata_{method}.json").write_text(json.dumps({**reals, **fakes}, indent=1))
        everything.update(fakes)
        print(f"{method}: {len(fakes)} fakes")

    (out / "metadata.json").write_text(json.dumps(everything, indent=1))
    print(f"{len(reals)} real, {len(everything) - len(reals)} fake, "
          f"{len(set(m['group'] for m in everything.values()))} groups -> {out}")


if __name__ == "__main__":
    main()
