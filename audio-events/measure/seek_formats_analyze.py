"""Analyse seek_formats.mjs runs: media-element currentTime vs the audio it outputs, per format x seek.

    <workspace>\\venv-main\\Scripts\\python.exe audio-events\\measure\\seek_formats_analyze.py seektest\\runs\\track-a_chromium_paused [...more run dirs]

A relative run dir that does not exist from the current directory is looked up under the
workspace, so `seektest\\runs\\...` works from the repo root.

Method: the capture of the element's output (seek_formats.mjs) is cross-correlated against
ffmpeg's decode of the SAME served file (so Opus pre-skip / AAC edit list are applied the
ffmpeg way), resampled to the context rate. That gives "context frame f carried file-PCM
sample f - L". Each logged (ctx.currentTime, audio.currentTime) pair then gives
    err = audio.currentTime - (file-PCM time carried in the graph at that moment)
POSITIVE = currentTime runs AHEAD of the audio.
err contains Chromium's element -> Web Audio pipeline latency, so it is also reported minus a PCM
control: the 44.1 kHz WAV runs (for 44.1 kHz formats) and the 48 kHz WAV runs (for Opus).

Also: alignment of each file's ffmpeg decode against the ORIGINAL MP3's ffmpeg decode (the timeline
the events were extracted from). align > 0 = the format's audio is LATE vs the MP3 decode.
So the error against the event timeline = err_corrected + align.
Writes <run dir>/analysis.json and prints tables.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags, resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # audio-events/, for workspace.py
from workspace import WORKSPACE  # noqa: E402

SEEKDIR = WORKSPACE / "seektest"
CACHE = SEEKDIR / "cache"
_mem = {}


def decode_ch0(file):
    """ffmpeg's decode of seektest/<file>, channel 0, native rate -> (sr, float64 array). Cached."""
    if file in _mem:
        return _mem[file]
    CACHE.mkdir(exist_ok=True)
    npy = CACHE / f"{file}.ch0.npy"
    p = SEEKDIR / file
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels",
                        "-of", "json", str(p)], capture_output=True, text=True, check=True)
    s = json.loads(r.stdout)["streams"][0]
    sr, ch = int(s["sample_rate"]), int(s["channels"])
    if npy.exists() and npy.stat().st_mtime > p.stat().st_mtime:
        x = np.load(npy)
    else:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-i", str(p), "-map", "0:a:0", "-f", "f32le", "-"],
                           capture_output=True, check=True)
        x = np.frombuffer(r.stdout, dtype="<f4").reshape(-1, ch)[:, 0].copy()
        np.save(npy, x)
    _mem[file] = (sr, x.astype(np.float64))
    return _mem[file]


def at_rate(file, R):
    k = (file, R)
    if k not in _mem:
        sr, x = decode_ch0(file)
        g = np.gcd(R, sr)
        _mem[k] = x if sr == R else resample_poly(x, R // g, sr // g)
    return _mem[k]


def lag_of(a, b, maxlag=None, center=0):
    """L such that a[n] ~ b[n - L], searched within center +- maxlag; plus parabolic sub-sample L and NCC."""
    c = correlate(a, b, mode="full", method="fft")
    lags = correlation_lags(len(a), len(b), mode="full")
    if maxlag is not None:
        m = np.abs(lags - center) <= maxlag
        c, lags = c[m], lags[m]
    k = int(np.argmax(c))
    L = int(lags[k])
    frac = 0.0
    if 0 < k < len(c) - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        d = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / d if d != 0 else 0.0
    aa, bb = overlap(a, b, L)
    ncc = float(np.dot(aa, bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))
    return L, L + frac, ncc


def overlap(a, b, L):
    if L >= 0:
        n = min(len(a) - L, len(b))
        return a[L:L + n], b[:n]
    n = min(len(a), len(b) + L)
    return a[:n], b[-L:-L + n]


def analyse_run(d, key, run, secs):
    R = run["sampleRate"]
    x = at_rate(run["file"], R)
    y = np.fromfile(d / f"{key}.f32", dtype="<f4").astype(np.float64)
    pairs = np.array(json.loads((d / f"{key}.pairs.json").read_text()))
    ctxF = pairs[:, 0] * R - run["f0"]
    mediaT = pairs[:, 1]
    s0 = run["seek"] if run["seek"] is not None else 0.0
    ok = (mediaT > s0 + 1.0) & (ctxF > 0) & (ctxF < len(y)) & (mediaT < mediaT.max() - 0.2)
    if run.get("seekedAtCtx") is not None:           # playing-mode seek: only after the seek settles
        cut = int((run["seekedAtCtx"] + 0.5) * R - run["f0"])
        ok &= ctxF > cut
        y = y.copy(); y[:max(cut, 0)] = 0.0
    L0 = int(np.median(ctxF[ok] - mediaT[ok] * R))
    # correlate against a slice of the reference around where the capture should sit
    a = max(0, int(-L0) - R)
    b = min(len(x), int(-L0) + len(y) + R)
    Ls, Lsf, ncc = lag_of(y, x[a:b], maxlag=R // 4, center=L0 + a)
    L = Ls - a
    # stability: local lag per 1 s window, +-50 ms around L
    local = []
    start = max(int(ctxF[ok].min()), 0)
    for w0 in range(start + R // 2, len(y) - R, R):
        seg = y[w0:w0 + R]
        if np.max(np.abs(seg)) < 1e-3:
            continue
        lo = w0 - L - R // 20
        if lo < 0 or lo + R + R // 10 > len(x):
            continue
        xs = x[lo:lo + R + R // 10]
        c = correlate(xs, seg, mode="valid", method="fft")   # index j: seg ~ xs[j:j+len(seg)]
        local.append(w0 - (lo + int(np.argmax(c))))
    pcmT = (ctxF - L) / R
    err = (mediaT - pcmT)[ok] * 1000
    return {"fmt": run["fmt"], "seek": run["seek"], "rep": run["rep"], "ctxRate": R, "ncc": round(ncc, 4),
            "lagFrac": round(Lsf - Ls, 3), "localLagMinMax": [min(local) - L, max(local) - L] if local else None,
            "pairs": int(ok.sum()), "errMedian": float(np.median(err)), "errP5": float(np.percentile(err, 5)),
            "errP95": float(np.percentile(err, 95)), "duration": run["duration"]}


def alignment(slug, fmt):
    """Lag of ffmpeg's decode of <slug>.<fmt> vs ffmpeg's decode of <slug>.mp3, at several positions (ms)."""
    sr, x = decode_ch0(f"{slug}.{fmt}")
    ref = at_rate(f"{slug}.mp3", sr)
    out = {}
    for t in [2.0, 30.0, 61.3, 120.0, 200.0]:
        n0 = int(t * sr)
        seg = x[n0:n0 + sr]
        if len(seg) < sr or np.max(np.abs(seg)) < 1e-3:
            continue
        m = sr // 20
        rs = ref[n0 - m:n0 + sr + m]
        # seg[n] = x[n0+n] ~ rs[n-L] = ref[n0-m+n-L] -> x[k] ~ ref[k-(L+m)]; aligned <=> L = -m
        L, Lf, ncc = lag_of(seg, rs, maxlag=m, center=-m)
        out[t] = {"lagSamples": round(Lf + m, 3), "ms": round((Lf + m) / sr * 1000, 3), "ncc": round(ncc, 4)}
    return out


def main():
    dirs = [Path(p) if Path(p).exists() or Path(p).is_absolute() else WORKSPACE / p for p in sys.argv[1:]]
    for d in dirs:
        meta = json.loads((d / "meta.json").read_text())
        slug = meta["slug"]
        rows = []
        for key, run in meta["runs"].items():
            if "error" in run:
                rows.append({"fmt": run["fmt"], "seek": run["seek"], "rep": run["rep"], "error": run["error"]})
                continue
            rows.append(analyse_run(d, key, run, meta["secs"]))
        fmts = list(dict.fromkeys(r["fmt"] for r in rows))
        seeks = list(dict.fromkeys(r["seek"] for r in rows))
        good = [r for r in rows if "error" not in r]
        ctl44 = float(np.median([r["errMedian"] for r in good if r["fmt"] == "wav"]))
        ctl48 = float(np.median([r["errMedian"] for r in good if r["fmt"] == "w48.wav"])) if "w48.wav" in fmts else ctl44
        align = {f: alignment(slug, f) for f in fmts if f != "mp3"}
        summary = {}
        for f in fmts:
            ctl = ctl48 if f in ("opus", "webm", "w48.wav") else ctl44
            summary[f] = {}
            for s in seeks:
                rs = [r for r in good if r["fmt"] == f and r["seek"] == s]
                if not rs:
                    continue
                meds = [r["errMedian"] for r in rs]
                summary[f][str(s)] = {"rawMs": round(float(np.median(meds)), 2),
                                      "corrMs": round(float(np.median(meds)) - ctl, 2),
                                      "repRangeMs": round(max(meds) - min(meds), 2),
                                      "pairP5P95Ms": [round(min(r["errP5"] for r in rs) - ctl, 2), round(max(r["errP95"] for r in rs) - ctl, 2)],
                                      "minNcc": min(r["ncc"] for r in rs),
                                      "localLag": [r["localLagMinMax"] for r in rs]}
        res = {"browser": meta["browser"], "slug": slug, "seekMode": meta["seekMode"], "control44Ms": round(ctl44, 2),
               "control48Ms": round(ctl48, 2), "alignVsMp3": align, "summary": summary, "runs": rows}
        (d / "analysis.json").write_text(json.dumps(res, indent=1))
        print(f"\n=== {slug} | {meta['browser']} | seek mode {meta['seekMode']} | control44 {ctl44:.2f} ms, control48 {ctl48:.2f} ms")
        print("currentTime - output, ms, minus PCM control (positive = currentTime AHEAD). cell = median [rep range] {min ncc}")
        print(f"{'fmt':9}" + "".join(f"{str(s):>22}" for s in seeks))
        for f in fmts:
            cells = []
            for s in seeks:
                c = summary[f].get(str(s))
                cells.append(f"{c['corrMs']:+7.2f} [{c['repRangeMs']:.2f}] {{{c['minNcc']:.3f}}}" if c else "-")
            print(f"{f:9}" + "".join(f"{x:>22}" for x in cells))
        errs = [r for r in rows if "error" in r]
        if errs:
            print("errors:", errs)
        print("decode alignment vs MP3 decode (ms, + = late):",
              json.dumps({f: {t: v["ms"] for t, v in a.items()} for f, a in align.items()}))


if __name__ == "__main__":
    main()
