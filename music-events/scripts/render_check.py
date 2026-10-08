"""Render listening checks from out/<slug>/events.json (the assembled, snapped data).

check_grid.mp3        original + metronome (high beep on downbeats) + chime at section starts
check_drums_split.mp3 LEFT = original, RIGHT = drums synthesised from the event list
check_drums_mix.mp3   original ducked + synthesised drums, centred
check_bass_split.mp3  LEFT = original, RIGHT = bass notes synthesised (if notes.bass exists)
check_lyrics_split.mp3 LEFT = vocals stem, RIGHT = a tick at every aligned word start (2 kHz; 3 kHz
                      and louder at line starts; 700 Hz for words with confidence < 0.6; soft
                      ticks at syllable starts inside a spelled word) (if lyrics exist)
check_vocal_onsets_split.mp3 LEFT = vocals stem, RIGHT = vocal onsets (1.2 kHz flux, 600 Hz
                      pitch change) (if vocalOnsets exist)

Split files make timing errors obvious on headphones: a flam between ears is a misplaced hit.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # music-events/, for workspace.py
from workspace import AUDIO, OUT, all_slugs  # noqa: E402

SR = 44100
rng = np.random.default_rng(7)


def env(n, tau_s):
    return np.exp(-np.arange(n) / (tau_s * SR))


def noise(n, highpass=False):
    x = rng.standard_normal(n)
    return np.diff(x, prepend=0.0) * 0.5 if highpass else x


def tone(freq, n, tau_s):
    return np.sin(2 * np.pi * freq * np.arange(n) / SR) * env(n, tau_s)


def sweep(f0, f1, n, tau_s):
    f = f1 + (f0 - f1) * env(n, 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, tau_s)


VOICES = {
    "kick": lambda: sweep(120, 45, int(0.25 * SR), 0.07) * 1.0,
    "snare": lambda: (noise(int(0.15 * SR)) * env(int(0.15 * SR), 0.04) * 0.5
                      + tone(190, int(0.15 * SR), 0.03) * 0.4),
    "toms": lambda: sweep(160, 90, int(0.3 * SR), 0.1) * 0.7,
    "hatClosed": lambda: noise(int(0.04 * SR), True) * env(int(0.04 * SR), 0.008) * 0.5,
    "hatOpen": lambda: noise(int(0.25 * SR), True) * env(int(0.25 * SR), 0.07) * 0.4,
    "crash": lambda: noise(int(1.2 * SR), True) * env(int(1.2 * SR), 0.35) * 0.35,
    "ride": lambda: (noise(int(0.4 * SR), True) * env(int(0.4 * SR), 0.12) * 0.2
                     + tone(5200, int(0.4 * SR), 0.15) * 0.08),
}


def place(buf, t, sound, gain=1.0):
    a = int(round(t * SR))
    if a < 0 or a >= len(buf):
        return
    z = min(len(buf), a + len(sound))
    buf[a:z] += sound[: z - a] * gain


def beat_time(ev, b):
    return ev["gridMarker"] + np.asarray(b, dtype=float) * 60.0 / ev["tempo"][0]["bpm"]


def to_mp3(stereo, path):
    peak = np.abs(stereo).max()
    if peak > 0.99:
        stereo = stereo * (0.99 / peak)
    wav = path.with_suffix(".wav")
    sf.write(str(wav), stereo.astype(np.float32), SR)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
                    "-b:a", "192k", str(path)], check=True)
    wav.unlink()


def render(slug):
    d = OUT / slug
    ev = json.loads((d / "events.json").read_text(encoding="utf-8"))
    y, sr = sf.read(str(AUDIO / f"{slug}.wav"), always_2d=True)
    assert sr == SR
    mono = y.mean(axis=1)
    n = len(mono)

    # Grid: 1.5 kHz on downbeats, 1 kHz on other beats, a two-tone chime at section starts.
    click = np.zeros(n)
    period = 60.0 / ev["tempo"][0]["bpm"]
    n_beats = int((n / SR - ev["gridMarker"]) / period) + 1
    for i in range(max(0, int(np.ceil(-ev["gridMarker"] / period))), n_beats):
        down = i % ev["meter"] == 0
        place(click, beat_time(ev, i), tone(1500 if down else 1000, int(0.05 * SR), 0.012), 0.5 if down else 0.25)
    for s in ev.get("sections", []):
        t = beat_time(ev, s["startBeat"])
        place(click, t, tone(660, int(0.5 * SR), 0.2) + tone(990, int(0.5 * SR), 0.2), 0.35)
    to_mp3(np.stack([mono * 0.6 + click, mono * 0.6 + click], axis=1), d / "check_grid.mp3")

    # Drums from the event list.
    drums = np.zeros(n)
    for cls, lane in (ev.get("drums") or {}).items():
        voice = VOICES.get(cls)
        if voice is None:
            continue
        sound = voice()
        for b, v in zip(lane["b"], lane["v"]):
            place(drums, beat_time(ev, b), sound, 0.3 + 0.7 * v)
    to_mp3(np.stack([mono, drums * 0.8], axis=1), d / "check_drums_split.mp3")
    to_mp3(np.stack([mono * 0.35 + drums * 0.6, mono * 0.35 + drums * 0.6], axis=1), d / "check_drums_mix.mp3")

    # Bass notes, if transcribed.
    bass = (ev.get("notes") or {}).get("bass")
    if bass:
        synth = np.zeros(n)
        for b, ln, p, v in zip(bass["b"], bass["len"], bass["pitch"], bass["v"]):
            f = 440.0 * 2 ** ((p - 69) / 12)
            m = int(ln * period * SR)
            tt = np.arange(m) / SR
            wave = sum(np.sin(2 * np.pi * f * k * tt) / k for k in (1, 2, 3, 4))
            fade = np.minimum(1, np.minimum(tt / 0.005, (m / SR - tt) / 0.02))
            place(synth, beat_time(ev, b), wave * fade * 0.3, 0.3 + 0.7 * v)
        to_mp3(np.stack([mono, synth], axis=1), d / "check_bass_split.mp3")
    # Lyrics and vocal onsets, against the vocals stem (on headphones a tick early or late
    # against the word's first consonant is a timing error).
    lyr, von = ev.get("lyrics"), ev.get("vocalOnsets")
    if lyr or von:
        vy, vsr = sf.read(str(d / "stems" / "vocals.wav"), always_2d=True)
        assert vsr == SR
        voc = np.zeros(n)
        voc[:min(n, len(vy))] = vy.mean(axis=1)[:n]
    if lyr:
        ticks = np.zeros(n)
        tick = lambda f, g: tone(f, int(0.04 * SR), 0.01) * g  # noqa: E731
        line_starts = {round(b, 3) for b in lyr["lines"]["b"]}
        w = lyr["words"]
        for i, b in enumerate(w["b"]):
            if round(b, 3) in line_starts:
                place(ticks, beat_time(ev, b), tick(3000, 0.7))
            elif w.get("v") and w["v"][i] < 0.6:  # hand-timed words have no v
                place(ticks, beat_time(ev, b), tick(700, 0.5))
            else:
                place(ticks, beat_time(ev, b), tick(2000, 0.5))
            for sb, _ in (w["syl"][i] or [])[1:]:
                place(ticks, beat_time(ev, sb), tick(2000, 0.2))
        to_mp3(np.stack([voc, ticks], axis=1), d / "check_lyrics_split.mp3")
    if von:
        ticks = np.zeros(n)
        for b, v, k in zip(von["b"], von["v"], von["kind"]):
            place(ticks, beat_time(ev, b), tone(600 if k else 1200, int(0.04 * SR), 0.01), 0.25 + 0.35 * v)
        to_mp3(np.stack([voc, ticks], axis=1), d / "check_vocal_onsets_split.mp3")
    print(f"{slug}: rendered {', '.join(p.name for p in sorted(d.glob('check_*.mp3')))}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    for s in (all_slugs() if args.all else [args.slug]):
        render(s)
