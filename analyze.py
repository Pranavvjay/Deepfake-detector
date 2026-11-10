"""Steps 3 and 5: heart rate for each patch, and how well the patches agree.

Usage:
    python analyze.py                        # reads signal.csv, uses POS
    python analyze.py other.csv
    python analyze.py --method green         # or chrom / pos

Per patch: even out the timing -> turn r, g, b into one pulse signal (green, CHROM
or POS) -> band-pass filter -> FFT -> strongest peak.
Between patches: coherence (do they share the same rhythm?) and correlation
(do their filtered waves rise and fall together?).
Saves a picture to analysis.png.
"""

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

FS = 30.0  # rate we resample to, in samples per second
LOW_HZ, HIGH_HZ = 0.7, 3.0  # 42 to 180 beats a minute
MIN_SECONDS = 20
WINDOW_SECONDS = 10  # length of each window for the per-window correlation
STEP_SECONDS = 5

# Matches the box colours in main.py.
PLOT_COLOURS = {"forehead": "red", "left_cheek": "magenta", "right_cheek": "c"}


def load(path):
    """Return the times and a {patch name: (r, g, b)} dict."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    names = data.dtype.names
    if "g" in names:  # older recordings had only the forehead
        return data["time_s"], {"forehead": (data["r"], data["g"], data["b"])}
    patches = [n[:-2] for n in names if n.endswith("_g")]
    return data["time_s"], {p: tuple(data[f"{p}_{c}"] for c in "rgb") for p in patches}


def resample(t, values, fs):
    """Webcam frames don't arrive at perfectly even times, but the filter and
    FFT assume they do. So we read the signal off at exactly 1/fs steps."""
    even_t = np.arange(t[0], t[-1], 1 / fs)
    return even_t, np.interp(even_t, t, values)


def bandpass(values, fs):
    sos = signal.butter(3, [LOW_HZ, HIGH_HZ], btype="bandpass", fs=fs, output="sos")
    # sosfiltfilt runs the filter forwards then backwards, so the wave isn't shifted in time.
    return signal.sosfiltfilt(sos, values)


def pulse_green(r, g, b, fs):
    return bandpass(signal.detrend(g), fs)


def pulse_chrom(r, g, b, fs):
    """CHROM (de Haan & Jeanne, 2013). Brightness changes scale r, g and b
    together; the pulse doesn't. Two colour mixes are built so that the
    brightness part can be subtracted away."""
    rn, gn, bn = r / r.mean(), g / g.mean(), b / b.mean()
    x = bandpass(3 * rn - 2 * gn, fs)
    y = bandpass(1.5 * rn + gn - 1.5 * bn, fs)
    return x - (x.std() / y.std()) * y


def pulse_pos(r, g, b, fs):
    """POS (Wang et al., 2017). Same idea as CHROM, but worked out over short
    1.6 s windows that are stitched together, so it adapts as the light changes."""
    rgb = np.vstack([r, g, b])
    n = rgb.shape[1]
    win = int(1.6 * fs)
    out = np.zeros(n)
    for i in range(n - win + 1):
        c = rgb[:, i:i + win]
        c = c / c.mean(axis=1, keepdims=True)  # each colour relative to its own average
        s1 = c[1] - c[2]
        s2 = -2 * c[0] + c[1] + c[2]
        h = s1 + (s1.std() / max(s2.std(), 1e-9)) * s2
        out[i:i + win] += h - h.mean()
    return bandpass(out, fs)


METHODS = {"green": pulse_green, "chrom": pulse_chrom, "pos": pulse_pos}


def heart_rate(values, fs):
    """Return (bpm, freqs, power) from the strongest frequency in the pulse band."""
    power = np.abs(np.fft.rfft(values)) ** 2
    freqs = np.fft.rfftfreq(len(values), d=1 / fs)
    in_band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    peak = freqs[in_band][np.argmax(power[in_band])]
    return peak * 60, freqs, power


def band_coherence(a, b, fs):
    """How strongly two signals share the same frequencies, 0 (not at all) to 1.

    Returns (freqs, coherence, peak and average coherence inside the pulse band).
    A pulse sits at one frequency, so the peak matters most; the average also
    counts all the band's noise-only frequencies, so it is lower even when real."""
    nperseg = min(256, len(a) // 4)  # 256 samples = about 8.5 s per piece
    freqs, coh = signal.coherence(a, b, fs=fs, nperseg=nperseg)
    in_band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    return freqs, coh, coh[in_band].max(), coh[in_band].mean()


def window_correlations(a, b, fs):
    """Correlation of two filtered waves in each window: 1 = move together,
    0 = unrelated, -1 = opposite."""
    size, step = int(WINDOW_SECONDS * fs), int(STEP_SECONDS * fs)
    return np.array([
        np.corrcoef(a[i:i + size], b[i:i + size])[0, 1]
        for i in range(0, len(a) - size + 1, step)
    ])


def short_name(name):
    return name.replace("_cheek", " cheek")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="signal.csv")
    parser.add_argument("--method", choices=METHODS, default="pos")
    args = parser.parse_args()

    t, patches = load(args.path)
    duration = t[-1] - t[0]
    measured_fps = (len(t) - 1) / duration
    print(f"{len(t)} frames over {duration:.1f} s ({measured_fps:.1f} fps), method: {args.method}")
    if duration < MIN_SECONDS:
        print(f"Warning: under {MIN_SECONDS} s, so the result is rough.")
    # With N seconds of signal, FFT bins are 1/N Hz apart = 60/N bpm apart.
    print(f"Heart-rate bins are {60 / duration:.1f} bpm apart\n")

    filtered, spectra = {}, {}
    for name, (r, g, b) in patches.items():
        r, g, b = (resample(t, c, FS)[1] for c in (r, g, b))
        if max(r.mean(), g.mean(), b.mean()) > 220:
            print(f"Warning: {short_name(name)} is very bright (near 255), so it may be overexposed.")
        filtered[name] = METHODS[args.method](r, g, b, FS)
        bpm, freqs, power = heart_rate(filtered[name], FS)
        spectra[name] = (bpm, freqs, power)
        print(f"{short_name(name):12s} heart rate: {bpm:.0f} bpm")

    if len(filtered) > 1:
        # Noise differs between patches but the pulse is shared, so adding them
        # (each scaled to the same size) strengthens the pulse.
        combined = sum(w / w.std() for w in filtered.values())
        print(f"{'combined':12s} heart rate: {heart_rate(combined, FS)[0]:.0f} bpm")

    pairs = list(itertools.combinations(filtered, 2))
    coherences = {}
    if pairs:
        print(f"\nAgreement between patches (pulse band {LOW_HZ}-{HIGH_HZ} Hz):")
        print(f"{'pair':28s} {'peak coh':>8s} {'mean coh':>8s} {'corr (mean of windows)':>23s}")
    for a, b in pairs:
        freqs, coh, peak_coh, mean_coh = band_coherence(filtered[a], filtered[b], FS)
        coherences[(a, b)] = (freqs, coh)
        corr = window_correlations(filtered[a], filtered[b], FS)
        label = f"{short_name(a)} vs {short_name(b)}"
        print(f"{label:28s} {peak_coh:8.2f} {mean_coh:8.2f} {corr.mean():23.2f}")

    rows = 3 if pairs else 2
    fig, axes = plt.subplots(rows, 1, figsize=(10, 3 * rows))
    seconds = np.arange(len(next(iter(filtered.values())))) / FS

    for name, wave in filtered.items():
        # Scale each wave to the same size so their timing is easy to compare.
        axes[0].plot(seconds, wave / wave.std(), color=PLOT_COLOURS[name],
                     label=short_name(name), linewidth=1)
    axes[0].set_title(f"Filtered waves, {args.method} (on a real face these should rise and fall together)")
    axes[0].set_xlabel("seconds")
    axes[0].legend(loc="upper right")

    for name, (bpm, freqs, power) in spectra.items():
        band = (freqs >= 0.5) & (freqs <= 4)
        axes[1].plot(freqs[band] * 60, power[band] / power[band].max(),
                     color=PLOT_COLOURS[name], label=f"{short_name(name)}: {bpm:.0f} bpm")
    axes[1].set_title("Spectrum of each patch (tallest spike = heart rate)")
    axes[1].set_xlabel("beats per minute")
    axes[1].legend(loc="upper right")

    if pairs:
        for (a, b), (freqs, coh) in coherences.items():
            band = (freqs >= 0.5) & (freqs <= 4)
            axes[2].plot(freqs[band] * 60, coh[band], label=f"{short_name(a)} vs {short_name(b)}")
        axes[2].axvspan(LOW_HZ * 60, HIGH_HZ * 60, color="grey", alpha=0.1)
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Coherence between patches (1 = same rhythm, 0 = unrelated)")
        axes[2].set_xlabel("beats per minute")
        axes[2].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig("analysis.png", dpi=100)
    print("\nSaved analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
