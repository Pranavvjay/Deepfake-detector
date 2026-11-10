"""Step 6a: turn one video's signal csv into a few numbers per window.

Each window becomes one row of features. On a real face the patches share one
heartbeat, so they should agree; on a fake they often don't.

Per pair of patches (3 pairs):  peak coherence, mean coherence, correlation
Per patch (3 patches):          pulse strength (how much of the band's power is
                                in the strongest spike)
Across patches:                 spread of the three heart rates (bpm)
"""

import itertools

import numpy as np
from scipy import signal

from analyze import FS, HIGH_HZ, LOW_HZ, heart_rate, load, pulse_pos, resample

WINDOW_SECONDS = 5  # DFDC clips are about 10 s, so this gives 2-3 overlapping windows
STEP_SECONDS = 2.5
MIN_SECONDS = 6  # skip videos where the face was found for less than this


def feature_names(patches):
    names = []
    for a, b in itertools.combinations(patches, 2):
        names += [f"peak_coh {a}-{b}", f"mean_coh {a}-{b}", f"corr {a}-{b}"]
    names += [f"strength {p}" for p in patches]
    names.append("bpm_spread")
    return names


def pulse_strength(wave, fs):
    """Share of the pulse band's power that sits in its strongest frequency.
    A clean pulse puts most power in one spike; noise spreads it out."""
    bpm, freqs, power = heart_rate(wave, fs)
    in_band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    return power[in_band].max() / power[in_band].sum(), bpm


def window_features(waves, fs):
    """waves: {patch: filtered pulse wave for one window}. Returns a list of numbers."""
    row = []
    for a, b in itertools.combinations(waves, 2):
        # 64 samples = about 2 s per piece, so a 5 s window still has several pieces to average.
        freqs, coh = signal.coherence(waves[a], waves[b], fs=fs, nperseg=64)
        in_band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
        row += [coh[in_band].max(), coh[in_band].mean(), np.corrcoef(waves[a], waves[b])[0, 1]]

    bpms = []
    for wave in waves.values():
        strength, bpm = pulse_strength(wave, fs)
        row.append(strength)
        bpms.append(bpm)
    row.append(np.std(bpms))
    return row


def video_features(csv_path):
    """Return (feature rows, feature names), one row per window.
    Returns no rows if the face wasn't found for long enough."""
    try:
        t, patches = load(csv_path)
    except (ValueError, IndexError):  # empty or unreadable csv
        return np.empty((0, 0)), []
    if t.size < 2 or t[-1] - t[0] < MIN_SECONDS:
        return np.empty((0, 0)), []

    # Filter the whole clip once, then cut windows, so each window doesn't
    # suffer from the filter's start-up wobble at its edges.
    waves = {}
    for name, (r, g, b) in patches.items():
        r, g, b = (resample(t, c, FS)[1] for c in (r, g, b))
        waves[name] = pulse_pos(r, g, b, FS)

    size, step = int(WINDOW_SECONDS * FS), int(STEP_SECONDS * FS)
    n = len(next(iter(waves.values())))
    rows = [
        window_features({p: w[i:i + size] for p, w in waves.items()}, FS)
        for i in range(0, n - size + 1, step)
    ]
    return np.array(rows), feature_names(list(patches))
