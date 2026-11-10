# Pulse-based deepfake detection

Can you spot a deepfake by checking whether the face has a heartbeat?

Every heartbeat slightly changes the colour of your skin. A normal webcam can pick
this up (this is called **remote photoplethysmography, rPPG**). In a real video,
the forehead and both cheeks should all show the *same* pulse. Intel's
[FakeCatcher](https://arxiv.org/abs/1901.02212) builds on this idea: a fake face
shouldn't have a consistent pulse.

This project builds that pipeline from scratch and tests it honestly on four
datasets. It follows [Hari's (@harixganesh)](https://github.com/harixganesh)
step-by-step guide.

## How it works

```
video → find the face (MediaPipe, plus YuNet for small faces)
      → 3 skin regions: forehead, left cheek, right cheek (eyes, brows, lips cut out)
      → average colour of each region, every frame
      → turn colour into a pulse wave (green / CHROM / POS)
      → band-pass 0.7–3 Hz → FFT → heart rate in bpm
      → compare the regions (correlation, coherence, bpm spread) → 13 features
      → logistic regression / SVM → real or fake
```

Testing uses **GroupKFold**, so the model is always tested on people it never
saw during training. Without this, a model can score well just by recognising faces.

## Results

### 1. Heart rate from a webcam ✅

Measured against my own pulse, counted by hand:

| Method | Webcam | Counted |
|---|---|---|
| Green | 63 bpm | 68 bpm |
| POS | 70 | 68 |
| POS | 77 | 80 |
| POS, all three regions | 60 | 66 |

POS (Wang et al., 2017) was the most reliable. It handles uneven lighting much better than the green channel.

### 2. Clean lab video: UBFC-rPPG ✅

Uncompressed video of 6 subjects. The ground-truth files weren't in the part I downloaded, so these
show **agreement between methods**, not error against a finger sensor.

| | Result |
|---|---|
| All 3 regions give the same bpm | 5 of 6 subjects (6th within 9 bpm) |
| [yarPPG](https://github.com/SamProell/yarppg) green within 5 bpm of my POS | 5 of 6 |

On clean video, the pulse is clearly there.

### 3. Compressed web video: DFDC sample ❌

| | Result |
|---|---|
| Real videos where the 3 regions agree (within 12 bpm) | 16% |
| Detector (whole-face masks) | 55.8% balanced accuracy, AUC 0.59 |

DFDC clips are short (10 s), heavily compressed and show small faces. The pulse doesn't survive,
so a detector can't use a "missing pulse".

### 4. FaceForensics++ ⚠️

50 real videos (25 groups of people) plus 50 fakes per method:

| Fakes | Quality | Balanced accuracy | AUC |
|---|---|---|---|
| Deepfakes | raw | **69%** | 0.78 |
| Deepfakes | c23 | 63% | 0.69 |
| Face2Face | c23 | **70%** | 0.74 |
| FaceSwap | c23 | 55% | 0.56 |
| NeuralTextures | c23 | 51% | 0.54 |
| All four together | c23 | 53% | 0.55 |

50% = coin flip. With only 25 groups, each number is uncertain by roughly ±10%.

## What I learned

- **The pulse is real, but fragile.** On clean video every face region agrees on the heart rate. On
  compressed internet video (DFDC, and FF++, which comes from YouTube) they agree only about
  16–22% of the time.
- **Compression hurts directly.** On the same Deepfakes videos, accuracy fell from 69% (raw) to 63% (c23).
- **What the FF++ detector actually learned is face consistency, not heartbeats.** Its strongest features:
  the two cheeks moving together (→ fake) and the forehead moving with the cheeks (→ real). Face swaps
  regenerate the middle of the face but usually keep the original forehead, so the detector spots
  that seam.
- **So it only catches fakes that regenerate a large part of the face.** NeuralTextures only changes
  the mouth, and the detector is at chance. One model for all four methods doesn't work with
  13 simple features.

## Files

| File | What it does |
|---|---|
| `main.py` | Face tracking and skin regions; records colour signals from a webcam, a video file, or a folder (`--batch`) |
| `analyze.py` | Turns a signal into heart rate (green / CHROM / POS), with plots and region comparison |
| `features.py` | The 13 region-agreement features per 5-second window |
| `train.py` | Trains and tests the detector with GroupKFold |
| `ubfc_eval.py` | Heart-rate test on UBFC-rPPG (against ground truth if present, otherwise between methods) |
| `yarppg_run.py` | Runs yarPPG on the same videos as a baseline (needs its own environment; yarPPG 1.0 needed small import fixes to run with MediaPipe 1.2) |
| `ffpp_prepare.py` | Organises FaceForensics++ downloads and writes labels with fair person-groups |
| `models/` | MediaPipe face landmarker (Apache 2.0) and OpenCV YuNet face detector (MIT) |

## Running it

Python 3.14 was used.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

python main.py                    # webcam: sit still for 30 s, press q
python analyze.py signal.csv      # heart rate + plots (analysis.png)

# deepfake detector on a labelled folder (DFDC-style metadata.json)
python main.py --batch path/to/videos --out signals
python train.py --metadata path/to/videos/metadata.json --signals signals
```

**Datasets aren't included.** Each needs its own access or licence:
[UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg),
[DFDC](https://www.kaggle.com/c/deepfake-detection-challenge),
[FaceForensics++](https://github.com/ondyari/FaceForensics).

## References

- U. Ciftci, I. Demir, L. Yin. *FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals.* TPAMI 2020.
- G. de Haan, V. Jeanne. *Robust Pulse Rate From Chrominance-Based rPPG* (CHROM). IEEE TBME 2013.
- W. Wang et al. *Algorithmic Principles of Remote PPG* (POS). IEEE TBME 2017.
- A. Rössler et al. *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV 2019.
- S. Bobbia et al. *Unsupervised skin tissue segmentation for remote photoplethysmography* (UBFC-rPPG). Pattern Recognition Letters 2019.
- [yarPPG](https://github.com/SamProell/yarppg) by Samuel Pröll.
