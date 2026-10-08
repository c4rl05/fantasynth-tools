"""Per-stem continuous features at 100 fps -> out/<slug>/curves.npz.

Keys are "<stem>_<feature>" with stem in {mix, vocals, bass, drums, guitar, piano, other,
kick, snare, toms, hh, ride, crash} (drum sub-stems lose their "drums_" prefix so the key
splits cleanly on the first underscore). Features: rms_db for everything; centroid and
onset for the full-band stems.

These keys are DrumSep's own stem-file names. The events file uses the app's lane names
instead, so assemble.CURVE_STEM_RENAME exports "hh" as "hats" when it reads this file.
"""
import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # music-events/, for workspace.py
from workspace import AUDIO, OUT, all_slugs  # noqa: E402

SR = 44100
FPS = 100
HOP = SR // FPS
FULL_BAND = {"mix", "vocals", "bass", "drums", "guitar", "piano", "other"}


def load_mono(path):
    y, sr = sf.read(str(path), always_2d=True)
    assert sr == SR, (path, sr)
    return y.mean(axis=1).astype(np.float32)


def features(y, full):
    out = {}
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP, center=True)[0]
    out["rms_db"] = librosa.amplitude_to_db(rms, ref=1.0, top_db=None).astype(np.float32)
    if full:
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP, center=True))
        cent = librosa.feature.spectral_centroid(S=S, sr=SR)[0]
        # Silence has an undefined centroid; hold it at the track median so quiet frames
        # don't spike the curve.
        quiet = rms < 10 ** (-60 / 20)
        cent[quiet] = np.median(cent[~quiet]) if (~quiet).any() else 0
        out["centroid"] = np.log2(np.maximum(cent, 20)).astype(np.float32)
        out["onset"] = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S, ref=np.max), sr=SR,
                                                    hop_length=HOP).astype(np.float32)
    return out


def run(slug):
    d = OUT / slug
    sources = {"mix": AUDIO / f"{slug}.wav"}
    for p in sorted((d / "stems").glob("*.wav")):
        name = p.stem.replace("drums_", "") if p.stem.startswith("drums_") else p.stem
        sources[name] = p
    arrays, n = {}, None
    for name, path in sources.items():
        for feat, v in features(load_mono(path), name in FULL_BAND).items():
            arrays[f"{name}_{feat}"] = v
            n = len(v) if n is None else min(n, len(v))
    arrays = {k: v[:n] for k, v in arrays.items()}
    np.savez_compressed(d / "curves.npz", fps=np.float32(FPS), **arrays)
    print(f"{slug}: {len(sources)} sources, {len(arrays)} curves x {n} frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    for s in (all_slugs() if args.all else [args.slug]):
        run(s)
