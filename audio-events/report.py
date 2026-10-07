"""Per-track report card for the audio-events pipeline.

Measures out/<slug>/events.json against the audio and the separated stems
(out/<slug>/stems/*.wav) with detectors of its own. It never reads raw_*.json, grid.json
or curves.npz, so an upstream bug cannot vouch for itself.

    python report.py --slug track-a
    python report.py --all
    python report.py --slug track-a --events <dir>/events.json   # a variant; report.json lands beside it

Prints a readable report and writes report.json beside the events file.

A consumer fires every event at gridMarker + b * 60 / bpm, with b already swung, so that is
the time measured here, never b + r.

Methods, distilled from the independent verification rounds (<workspace>/verify/v1..v5, r2_*, r3_*):

1. GRID. Kick attacks from stems/drums_kick.wav: 5 ms centred RMS (1 ms hop, dBFS); onsets
   are peaks of the rise over the preceding 20 ms (>= 15 dB, >= 150 ms apart) that reach within
   20 dB of the stem's p99 in the next 50 ms. Each is refined on the 1 ms-smoothed Hilbert
   envelope h in [o-20, o+30] ms: attack = first sample with h >= max/2. Offset = attack minus
   the nearest grid beat, for kicks within 0.15 beat of a beat.
   DOWNBEAT sanity: the beat%4 phase of kick re-entries (an on-beat kick after >= 4 kick-free
   beats) and of bass entries (per-beat bass level back within 15 dB of its p95 after >= 4 beats
   more than 30 dB below it), and whether every section startBeat is a multiple of 4.
2. DRUM LANES (kick; snare; hats = hatClosed + hatOpen) against the matching DrumSep stem, same
   5 ms envelope, rise = level minus the minimum over the preceding 20 ms.
   Precision: a lane hit is confirmed when the rise within +-30 ms reaches 6 dB and the level
   within [-30, +60] ms comes within 30 dB of the stem's p99.
   Recall: conservative onsets of my own (rise >= 12 dB, within 15 dB of p99, no other >= 6 dB
   rise in the 50 ms before and no second 12 dB onset in the 50 ms after) count as caught when
   a lane hit lies within +-30 ms. Low-confidence hits (c < 0.5, one detector) get their own
   precision. False hits and misses are clustered into bar ranges.
3. HATS open vs closed: ring = time for drums_hh (10 ms RMS) to fall 12 dB below the peak found
   in [-20, +40] ms of the hit (never past the neighbouring hat hits), measured up to the next
   hat hit (censored there) and capped at 500 ms. Separation = AUC(open rings longer than closed).
4. CURVES: curves.stems.{drums,bass}.rms_db decoded and compared with my own RMS (2048-sample
   rectangular window centred on beat k/samplesPerBeat, p5..p95 normalised like the encoder):
   Pearson r at lag 0, best integer lag within +-24 samples, best time offset within +-60 ms
   (both positive when the curve is late against the audio).
5. SECTIONS: per-bar level (mean power, dB, floored 40 dB below the stem's p95 bar) of drums,
   bass and the mix, and kick fill (share of a bar's 4 beats with an on-beat kick from method 1).
   At each boundary: after minus before over up to 4 bars each side, not crossing the
   neighbouring boundaries. Then every bar line is scanned (2 bars each side, isolated 1-bar
   dips such as fills and pre-drop gaps removed first) for strong changes with no boundary
   within +-1 bar.
6. LYRICS (only when events.json has `lyrics`) against stems/vocals.wav. Evidence of my own,
   never the pipeline's vocalOnsets (the aligner snaps word starts to those, so they would
   vouch for themselves): 10 ms centred RMS (5 ms hop); the voice is ACTIVE where its 55 ms
   median is within 25 dB of the stem's p99 (none when p99 < -40 dBFS, notes.py's silence
   gate). Two onset kinds, because a word start shows up in one of two ways:
   - a LEVEL RISE (>= 6 dB over the preceding 100 ms, peaks >= 80 ms apart), dated at the
     half-rise point like the kick attack: an entry from a breath, a stop or a nasal;
   - a SPECTRAL-FLUX peak (48 log bands 80 Hz-8 kHz, 2048-sample window, 5 ms hop, lag 2,
     above 1.5x its 0.4 s mean and 10% of its p99 over active time): a legato word that
     tiles onto the previous one has no level rise, only a change of vowel or pitch, and
     judging it on level alone would fail a correct alignment of legato singing.
   Word-start precision = share of word starts within +-60 ms of either kind. Evidence this
   dense is near most instants, so the CHANCE rate (share of active time within +-60 ms of
   evidence) is reported with it and the verdict uses kappa = (P - chance) / (1 - chance),
   0 for starts placed at random in the voice, 1 for all confirmed. ENTRIES (a word >= 250 ms
   after the previous word's end) are also scored against level rises alone, where chance is
   low and the evidence unambiguous. Coverage: share of active time inside a word (padded
   100 ms each side: an edge measured to 60 ms plus the voice's decay is not an uncovered
   vocal) or an extra, and inside a word alone; uncovered runs >= 0.5 s are listed. Sanity:
   words ordered and non-overlapping (0.002-beat rounding tolerance), len > 0, durations,
   count of words with v < 0.6.
7. VOCAL ONSETS (only when events.json has `vocalOnsets`), informational, events alone: counts
   by kind (0 flux, 1 pitch jump); for pitch jumps, the share with no Basic Pitch vocal note
   start within +-50 ms (information Basic Pitch lacks); for those note starts, the share
   with a vocal onset of either kind within +-50 ms. Note starts are taken at b + r (Basic
   Pitch's detection time): the 16th snap moves a note by up to 1/6 beat, more than the window.

Verdicts (solid / mostly / check) use the thresholds in VERDICT_RULES below; vocal onsets
are 'info' (a measurement, not a pass/fail).
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.fft import next_fast_len
from scipy.ndimage import median_filter, minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks, hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workspace import AUDIO, OUT  # noqa: E402

SR = 44100
WIN = 0.030  # lane matching tolerance, seconds

# ---------------------------------------------------------------- verdict thresholds
GRID_SOLID = (5.0, 10.0)    # |median attack offset| ms, p90 of |offset| ms
GRID_MOSTLY = (10.0, 20.0)
DOWNBEAT_SOLID = 0.80       # share of downbeat evidence on beat%4 == 0 (chance is 0.25)
DOWNBEAT_MOSTLY = 0.60
LANE_SOLID = (0.95, 0.95)   # precision, recall
LANE_MOSTLY = (0.80, 0.80)
LANE_TIMING_MS = 10.0       # |median(lane - onset)| over caught onsets; above it a lane is at best 'mostly'
HATS_SOLID = 0.95           # AUC(open rings longer than closed)
HATS_MOSTLY = 0.85
HATS_MIN_N = 3              # fewer hits in either lane: separation not measurable
# Ring is read on a 10 ms RMS, the window assemble's open/closed rule uses. A 5 ms RMS peaks on
# the stick transient, so a clicky open hat whose wash sits > 12 dB under the click reads as
# short: on one test track (A) AUC is 0.82 at 5 ms, 0.85 at 10 ms, 0.95 at 20 ms, while
# two others (B, C) stay >= 0.99 at every window. That spread is itself a finding.
HATS_RMS_MS = 10.0
CURVE_SOLID = 0.99          # r at lag 0, and best lag must be 0
CURVE_MOSTLY = 0.95         # r at lag 0, and |best lag| <= 1
SEC_CHANGE = (3.0, 3.0, 0.25)   # |drums dB|, |bass dB|, |kick fill|: any one = the boundary has a change
SEC_STRONG = (6.0, 6.0, 0.75)   # same, for a change strong enough to deserve a boundary
SEC_MOSTLY_ISSUES = 2       # relabelling boundaries with no change + unmarked strong changes
VOX_SILENT_DBFS = -40.0     # vocals p99 below this: separation residue, no voice (notes.py's gate)
VOX_GATE_DB = 25.0          # active voice: within this of the stem's p99
LYR_WIN = 0.060             # word start vs onset evidence, seconds
LYR_RISE = (0.10, 6.0)      # level rise: look-back s, dB
LYR_FLUX = (0.4, 1.5, 0.1)  # flux peak: mean window s, factor over the mean, floor as a share of p99
LYR_ENTRY_GAP = 0.25        # a word this long after the previous word's end is an entry, seconds
LYR_COVER_PAD = 0.10        # word edges padded by this for coverage, seconds
LYR_LOW_V = 0.6             # word confidence below this counts as low
LYR_TOL_BEATS = 0.002       # rounding tolerance of 3-decimal beats (b, len) when testing order/overlap
# Precision, kappa and coverage by words + extras. PROVISIONAL: set before any alignment
# existed, with Basic Pitch vocal note starts standing in for word starts (they include
# pitch changes inside a word, so they are a harder set than word starts): P 0.84 /
# kappa 0.59 on test track C (chance 0.61), 0.79 / 0.44 on track B (chance 0.62), both 'mostly';
# shifting the same starts 30 ms late already reads 'check' (kappa 0.27 / 0.17), and shifts of
# 60-150 ms either way give kappa <= 0.01. Revisit on the first real alignment.
# A 60 ms window on evidence this dense cannot be as strict as the drum lanes'.
LYR_SOLID = (0.85, 0.50, 0.90)
LYR_MOSTLY = (0.70, 0.30, 0.75)
VON_WIN = 0.050             # vocal onset vs Basic Pitch note start, seconds

VERDICT_RULES = {
    "grid": f"solid: |median| <= {GRID_SOLID[0]} ms and p90|off| <= {GRID_SOLID[1]} ms; "
            f"mostly: <= {GRID_MOSTLY[0]} / {GRID_MOSTLY[1]} ms",
    "downbeat": f"solid: >= {DOWNBEAT_SOLID:.0%} of kick re-entries + bass entries on beat%4 == 0 (>= 3 of them) "
                f"and every section start a multiple of 4; mostly: >= {DOWNBEAT_MOSTLY:.0%}",
    "lanes": f"solid: P >= {LANE_SOLID[0]} and R >= {LANE_SOLID[1]} and |median timing| <= {LANE_TIMING_MS} ms; "
             f"mostly: P >= {LANE_MOSTLY[0]} and R >= {LANE_MOSTLY[1]}",
    "hats": f"solid: AUC >= {HATS_SOLID}; mostly: AUC >= {HATS_MOSTLY}; n/a below {HATS_MIN_N} hits in either lane",
    "curves": f"solid: r@0 >= {CURVE_SOLID} and best lag 0 on drums and bass; mostly: r@0 >= {CURVE_MOSTLY} and |lag| <= 1",
    "sections": "solid: no relabelling boundary without a change, no strong change without a boundary, "
                f"no empty section; mostly: <= {SEC_MOSTLY_ISSUES} such issues",
    "lyrics": f"solid: word-start P >= {LYR_SOLID[0]} (+-{LYR_WIN * 1000:g} ms of a level rise or flux peak), "
              f"kappa over chance >= {LYR_SOLID[1]}, active voice covered by words + extras >= {LYR_SOLID[2]:.0%}; "
              f"mostly: >= {LYR_MOSTLY[0]} / {LYR_MOSTLY[1]} / {LYR_MOSTLY[2]:.0%}; check on any unordered, "
              "overlapping or zero-length word or a silent vocals stem; n/a without lyrics",
    "vocalOnsets": f"info: counts by kind, pitch jumps with no Basic Pitch vocal note start within +-{VON_WIN * 1000:g} ms, "
                   "note starts with a vocal onset in the same window; n/a without vocalOnsets",
}


# ---------------------------------------------------------------- loading
class Grid:
    """The grid as a consumer of events.json sees it."""

    def __init__(self, e):
        self.t0 = float(e["gridMarker"])
        self.bpm = float(e["tempo"][0]["bpm"])
        self.p = 60.0 / self.bpm

    def time(self, b):
        return self.t0 + np.asarray(b, float) * self.p

    def beat(self, t):
        return (np.asarray(t, float) - self.t0) / self.p


def load_mono(path):
    y, sr = sf.read(str(path), always_2d=True, dtype="float32")
    if sr != SR:
        raise ValueError(f"{path}: {sr} Hz, expected {SR}")
    return y.mean(axis=1).astype(np.float64)


def lane(e, g, classes):
    """Fire times, beats and confidences of one or more drum lanes, sorted by time."""
    b, c = [], []
    for k in classes:
        L = e["drums"].get(k, {"b": []})
        b += list(L["b"])
        cc = list(L.get("c") or [])
        c += cc if len(cc) == len(L["b"]) else [1.0] * len(L["b"])  # missing or ragged c: treat as agreed
    b, c = np.array(b, float), np.array(c, float)
    o = np.argsort(b, kind="stable")
    return g.time(b[o]), b[o], c[o]


# ---------------------------------------------------------------- envelopes and detectors
def rms_env_db(y, win_ms=5.0, hop_ms=1.0):
    """Centred RMS envelope in dBFS: value i is the RMS of the window centred on i*hop."""
    hop = int(round(SR * hop_ms / 1000))
    win = int(round(SR * win_ms / 1000))
    c = np.concatenate([[0.0], np.cumsum(y * y)])
    centers = np.arange(0, len(y), hop)
    a = np.clip(centers - win // 2, 0, len(y))
    z = np.clip(centers + win - win // 2, 0, len(y))
    ms = (c[z] - c[a]) / np.maximum(z - a, 1)
    return 10 * np.log10(ms + 1e-12), hop / SR


def rise_db(env_db, look):
    """Level minus the minimum over the preceding `look` frames, now included ([i-look, i])."""
    return env_db - minimum_filter1d(env_db, size=look + 1, origin=look // 2, mode="nearest")


def clear_onsets(env_db, dt, rise_min=12.0, loud_below_p99=15.0, iso_ms=50, look_ms=20):
    """Conservative onsets: a rise of >= rise_min dB over the preceding look_ms, a peak in the
    next 50 ms within loud_below_p99 dB of the stem's p99, no >= 6 dB rise peak (above the
    p99-30 dB noise floor) in the iso_ms before and no second rise_min onset in the iso_ms after.
    Smaller rises after the onset are the body of the same hit."""
    r = rise_db(env_db, int(look_ms / 1000 / dt))
    pk, _ = find_peaks(r, height=rise_min, distance=int(iso_ms / 1000 / dt))
    ref = np.percentile(env_db, 99)
    w = int(0.05 / dt)
    keep = np.array([i for i in pk if env_db[i: i + w].max() >= ref - loud_below_p99], int)
    pk6, _ = find_peaks(r, height=6.0)
    pk6 = pk6[env_db[pk6] >= ref - 30]
    iso = int(iso_ms / 1000 / dt)
    out = [i for i in keep
           if not ((pk6 < i) & (pk6 >= i - iso)).any() and not ((pk > i) & (pk <= i + iso)).any()]
    return np.array(out, int) * dt


def confirm(env, r, dt, times, rise_min=6.0, below=30.0):
    """Lane hit confirmed: rise >= rise_min within +-30 ms and level within `below` dB of p99 in [-30, +60] ms."""
    ref = np.percentile(env, 99)
    ok = np.zeros(len(times), bool)
    for k, t in enumerate(times):
        i = int(round(t / dt))
        a, z = max(i - int(WIN / dt), 0), i + int(WIN / dt) + 1
        if a >= len(r) or z <= 0:
            continue
        ok[k] = r[a:z].max() >= rise_min and env[a: i + int(0.06 / dt) + 1].max() >= ref - below
    return ok


def nearest_dist(a, b):
    """For each a, signed distance to the nearest b (b - a); inf when b is empty."""
    a = np.asarray(a, float)
    b = np.sort(np.asarray(b, float))
    if b.size == 0:
        return np.full(a.size, np.inf)
    i = np.searchsorted(b, a)
    lo = np.clip(i - 1, 0, b.size - 1)
    hi = np.clip(i, 0, b.size - 1)
    dlo, dhi = b[lo] - a, b[hi] - a
    return np.where(np.abs(dlo) <= np.abs(dhi), dlo, dhi)


def kick_attacks(y):
    """Attack (half-maximum of the Hilbert envelope) of every kick in the kick stem, seconds."""
    env, dt = rms_env_db(y, 5, 1)
    ref = np.percentile(env, 99)
    r = rise_db(env, int(0.02 / dt))
    pk, _ = find_peaks(r, height=15, distance=int(0.15 / dt))
    h = np.abs(hilbert(y, N=next_fast_len(len(y))))[: len(y)]
    h = uniform_filter1d(h, max(1, int(SR * 0.001)))
    out = []
    for i in pk:
        if env[i: i + int(0.05 / dt)].max() < ref - 20:
            continue
        c = int(round(i * dt * SR))
        a, z = max(c - int(0.02 * SR), 0), min(c + int(0.03 * SR), len(h))
        seg = h[a:z]
        out.append((a + np.argmax(seg >= 0.5 * seg.max())) / SR)
    return np.array(out)


def span_db(csum, t_edges):
    """Mean power in dB between consecutive edge times, from a cumulative sum of y**2."""
    s = np.clip(np.round(np.asarray(t_edges) * SR).astype(int), 0, len(csum) - 1)
    ms = (csum[s[1:]] - csum[s[:-1]]) / np.maximum(s[1:] - s[:-1], 1)
    return 10 * np.log10(ms + 1e-12)


def floored(db):
    """Clamp silence to 40 dB below the p95 level, so an element entering from silence is a
    bounded step instead of +80 dB."""
    return np.maximum(db, np.percentile(db, 95) - 40.0)


def decode_curve(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.uint8).astype(float)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------------------------------------------------------------- helpers for the text
def clusters(bars, bar_label, gap=4):
    """Group bar indices into runs (a new run after a gap of more than `gap` bars), biggest first."""
    bars = np.sort(np.asarray(bars, int))
    if bars.size == 0:
        return []
    runs = np.split(bars, np.nonzero(np.diff(bars) > gap)[0] + 1)
    out = []
    for run in runs:
        labels = sorted({bar_label(x) for x in run})
        out.append({"bars": [int(run[0]), int(run[-1])], "n": int(run.size), "sections": labels})
    return sorted(out, key=lambda d: -d["n"])


def fmt_clusters(cl, top=4):
    if not cl:
        return "-"
    big = [c for c in cl if c["n"] >= 2][:top]
    rest = sum(c["n"] for c in cl) - sum(c["n"] for c in big)
    parts = [f"bars {c['bars'][0]}-{c['bars'][1]} ({'/'.join(c['sections'])}) x{c['n']}" for c in big]
    if rest:
        parts.append(f"+{rest} scattered")
    return ", ".join(parts)


def rnd(x, n=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), n)


# ---------------------------------------------------------------- 1. grid + downbeat
def grid_check(g, attacks):
    b = g.beat(attacks)
    near = np.abs(b - np.round(b)) < 0.15
    off = (attacks[near] - g.time(np.round(b[near]))) * 1000
    if off.size == 0:
        return {"verdict": "check", "n": 0, "note": "no on-beat kicks found"}
    med, p90 = float(np.median(off)), float(np.percentile(np.abs(off), 90))
    quarters = [rnd(np.median(q), 1) for q in np.array_split(off, 4) if q.size]
    if abs(med) <= GRID_SOLID[0] and p90 <= GRID_SOLID[1]:
        v = "solid"
    elif abs(med) <= GRID_MOSTLY[0] and p90 <= GRID_MOSTLY[1]:
        v = "mostly"
    else:
        v = "check"
    return {"verdict": v, "n": int(near.sum()), "medianMs": rnd(med, 1), "p90AbsMs": rnd(p90, 1),
            "p10Ms": rnd(np.percentile(off, 10), 1), "p90Ms": rnd(np.percentile(off, 90), 1),
            "driftByQuarterMs": quarters, "offBeatKicks": int((~near).sum())}


def onbeat_kick_beats(g, attacks):
    b = g.beat(attacks)
    return np.unique(np.round(b[np.abs(b - np.round(b)) < 0.15]).astype(int))


def bass_entries(bass_beat_db):
    """Beats where the bass comes in from silence: level >= p95 - 15 dB after >= 4 beats below p95 - 30 dB."""
    ref = np.percentile(bass_beat_db, 95)
    quiet = bass_beat_db < ref - 30
    return np.array([b for b in range(4, len(bass_beat_db))
                     if bass_beat_db[b] >= ref - 15 and quiet[b - 4:b].all()], int)


def downbeat_check(e, kb, bass_in):
    """kb: on-beat kick beats; bass_in: bass entry beats. Drum-stem entries are not used as
    evidence: fills pick up a beat early (on one test track the drums enter on beats 15 and 31)."""
    first = kb[:1][kb[:1] >= 4]  # the track's first kick counts when beats 0..3 have none
    re = np.concatenate([first, kb[1:][np.diff(kb) >= 5]]).astype(int)  # >= 4 kick-free beats before it
    re_ph = np.mod(re, 4)
    ba_ph = np.mod(bass_in, 4)
    ev = np.concatenate([re_ph, ba_ph])
    off_bar = [s["startBeat"] for s in e.get("sections") or [] if float(s["startBeat"]) % 4 != 0]
    share = float(np.mean(ev == 0)) if ev.size else float("nan")
    if off_bar:
        v = "check"
    elif ev.size < 3:
        v = "n/a"
    elif share >= DOWNBEAT_SOLID:
        v = "solid"
    elif share >= DOWNBEAT_MOSTLY:
        v = "mostly"
    else:
        v = "check"
    return {"verdict": v, "shareOnDownbeat": rnd(share, 2),
            "kickReentryPhase": np.bincount(re_ph, minlength=4).tolist(), "kickReentryBeats": re.tolist(),
            "bassEntryPhase": np.bincount(ba_ph, minlength=4).tolist(), "bassEntryBeats": bass_in.tolist(),
            "sectionStartsOffBar": off_bar}


# ---------------------------------------------------------------- 2. lanes
def lane_check(name, g, times, beats, conf, env, dt, bar_label):
    r = rise_db(env, int(0.02 / dt))
    ok = confirm(env, r, dt, times)
    on = clear_onsets(env, dt, rise_min=12, loud_below_p99=15, iso_ms=50)
    dist = nearest_dist(on, times)  # lane minus onset
    hit = np.abs(dist) <= WIN
    P = float(ok.mean()) if ok.size else float("nan")
    R = float(hit.mean()) if hit.size else float("nan")
    low = conf < 0.5
    timing = float(np.median(dist[hit]) * 1000) if hit.any() else float("nan")
    fb = np.floor(beats[~ok] / 4).astype(int)
    mb = np.floor(g.beat(on[~hit]) / 4).astype(int)
    note = None
    if not ok.size and not on.size:
        v = "n/a"
    elif not on.size:  # hits in the lane, no clear onset in the stem: recall cannot be measured
        v = "mostly" if P >= LANE_MOSTLY[0] else "check"
        note = "no clear onsets in the stem; judged on precision alone"
    elif P >= LANE_SOLID[0] and R >= LANE_SOLID[1] and abs(timing) <= LANE_TIMING_MS:
        v = "solid"
    elif P >= LANE_MOSTLY[0] and R >= LANE_MOSTLY[1]:
        v = "mostly"
    else:
        v = "check"  # includes an empty lane over an audible stem (P undefined, R 0)
    return {"verdict": v, "note": note, "n": int(ok.size), "precision": rnd(P), "onsets": int(on.size), "recall": rnd(R),
            "timingMedianMs": rnd(timing, 1), "timingMedianAbsMs": rnd(np.median(np.abs(dist[hit])) * 1000, 1) if hit.any() else None,
            "lowConf": {"n": int(low.sum()), "precision": rnd(ok[low].mean()) if low.any() else None},
            "highConfPrecision": rnd(ok[~low].mean()) if (~low).any() else None,
            "falseHits": {"n": int((~ok).sum()), "clusters": clusters(fb, bar_label), "beats": [rnd(x, 2) for x in beats[~ok]]},
            "misses": {"n": int((~hit).sum()), "clusters": clusters(mb, bar_label), "beats": [rnd(x, 2) for x in g.beat(on[~hit])]}}


# ---------------------------------------------------------------- 3. open vs closed hats
def hats_check(g, e, env, dt):
    hb = {k: e["drums"].get(k, {"b": []})["b"] for k in ("hatClosed", "hatOpen")}
    allh = np.sort(g.time(hb["hatClosed"] + hb["hatOpen"]))
    res = {}
    for k in ("hatClosed", "hatOpen"):
        ring, cens = [], []
        for t in g.time(hb[k]):
            # the peak search stays between the neighbouring hat hits (same-time duplicates
            # across the two lanes excluded), or it can land on the next hit's peak
            prv, nxt = allh[allh < t - 0.001], allh[allh > t + 0.001]
            i = int(round(t / dt))
            a = max(i - int(0.02 / dt), int(prv[-1] / dt) + 1 if prv.size else 0, 0)
            z = min(i + int(0.04 / dt) + 1, int(nxt[0] / dt) if nxt.size else len(env), len(env))
            if a >= z:
                continue
            p = a + int(np.argmax(env[a:z]))
            end = min(p + int(0.5 / dt), len(env))
            below = np.nonzero(env[p:end] < env[p] - 12)[0]
            fall = (below[0] if below.size else end - p) * dt * 1000  # 500 ms cap, or the file's end
            gap = (nxt[0] - p * dt) * 1000 if nxt.size else np.inf
            ring.append(min(fall, gap))
            cens.append(bool(gap < fall or (not below.size and end - p < int(0.5 / dt))))
        res[k] = (np.array(ring), np.array(cens, bool))
    c, o = res["hatClosed"][0], res["hatOpen"][0]
    out = {k: {"n": int(v[0].size), "ringMedianMs": rnd(np.median(v[0]), 0) if v[0].size else None,
               "ringP10P90Ms": [rnd(np.percentile(v[0], 10), 0), rnd(np.percentile(v[0], 90), 0)] if v[0].size else None,
               "cutByNextHit": rnd(v[1].mean(), 2) if v[1].size else None,
               "ringAtLeast150ms": rnd(np.mean(v[0] >= 150), 2) if v[0].size else None}
           for k, v in res.items()}
    if c.size < HATS_MIN_N or o.size < HATS_MIN_N:
        out.update(verdict="n/a", auc=None, note=f"closed {c.size}, open {o.size}: too few to separate")
        return out
    auc = float(np.mean(o[:, None] > c[None, :]) + 0.5 * np.mean(o[:, None] == c[None, :]))
    v = "solid" if auc >= HATS_SOLID else "mostly" if auc >= HATS_MOSTLY else "check"
    out.update(verdict=v, auc=rnd(auc))
    if min(c.size, o.size) < 10:
        out["note"] = f"only {min(c.size, o.size)} hits in the smaller lane"
    return out


# ---------------------------------------------------------------- 4. curves
def curve_check(g, e, name, csum, dur):
    C = e.get("curves") or {}
    s = (C.get("stems") or {}).get(name) or {}
    if not s.get("rms_db") or not C.get("samplesPerBeat"):
        return {"verdict": "n/a", "note": f"no curves.stems.{name}.rms_db"}
    spb = C["samplesPerBeat"]
    dec = decode_curve(s["rms_db"])
    t = g.time(np.arange(len(dec)) / spb)

    def mine_at(tt):
        s = np.round(tt * SR).astype(int)
        a = np.clip(s - 1024, 0, len(csum) - 1)
        z = np.clip(s + 1024, 0, len(csum) - 1)
        db = 10 * np.log10((csum[z] - csum[a]) / 2048 + 1e-10)  # zero-padded like librosa center=True
        lo, hi = np.percentile(db, 5), np.percentile(db, 95)
        return np.clip((db - lo) / max(hi - lo, 1e-9), 0, 1) * 255

    mine = mine_at(t)
    idx = np.nonzero((t > 0.05) & (t < dur - 0.05))[0]
    r0 = corr(dec[idx], mine[idx])
    lags = np.arange(-24, 25)
    rl = []
    for L in lags:  # dec[i + L] against mine[i]: a positive lag = the decoded curve is late
        j = idx + L
        m = (j >= 0) & (j < len(dec))
        rl.append(corr(dec[j[m]], mine[idx[m]]))
    best = int(lags[int(np.argmax(rl))])
    # my RMS read at t - d matches the curve best: the curve describes the audio d ms earlier,
    # i.e. it is d ms late. Same sign convention as the lag.
    deltas = np.arange(-60, 61)
    rd = [corr(dec[idx], mine_at(t - d / 1000)[idx]) for d in deltas]
    bd = int(deltas[int(np.argmax(rd))])
    if r0 >= CURVE_SOLID and best == 0:
        v = "solid"
    elif r0 >= CURVE_MOSTLY and abs(best) <= 1:
        v = "mostly"
    else:
        v = "check"
    return {"verdict": v, "n": int(len(dec)), "r0": rnd(r0, 4), "bestLag": best, "rBestLag": rnd(max(rl), 4),
            "lagMs": rnd(best * g.p / spb * 1000, 1), "bestOffsetMs": bd,
            "medianAbsDiff": rnd(np.median(np.abs(dec[idx] - mine[idx])), 1),
            "curveBeats": rnd(len(dec) / spb, 2)}


# ---------------------------------------------------------------- 5. sections
def remove_dips(x, step, back):
    """Replace isolated 1-bar dips/spikes (a bar >= step away from both neighbours in the same
    direction, the neighbours within `back` of each other) by the mean of the neighbours."""
    y = x.copy()
    dips = []
    for i in range(1, len(x) - 1):
        a, b = x[i] - x[i - 1], x[i] - x[i + 1]
        if abs(a) >= step and abs(b) >= step and np.sign(a) == np.sign(b) and abs(x[i + 1] - x[i - 1]) < back:
            y[i] = 0.5 * (x[i - 1] + x[i + 1])
            dips.append(i)
    return y, dips


def sections_check(e, F, nbars):
    """F: per-bar features {'drums','bass','mix' (dB), 'kick' (fill 0..1)}."""
    secs = e.get("sections") or []
    if not secs:
        return {"verdict": "check", "note": "no sections", "boundaries": [], "noChange": [], "relabelNoChange": [],
                "unmarkedStrong": [], "oneBarDips": {}, "emptySections": [], "audioBars": nbars}
    bounds = [int(s["startBeat"] // 4) for s in secs]
    ends = bounds[1:] + [int(np.ceil((secs[-1]["startBeat"] + secs[-1]["beats"]) / 4))]
    empty = [bounds[i] for i in range(len(secs)) if ends[i] <= bounds[i]]  # shorter than a bar: a defect of the file
    rows = []
    for i in range(1, len(secs)):
        B = bounds[i]
        row = {"bar": B, "from": secs[i - 1]["label"], "to": secs[i]["label"], "change": False}
        if B <= bounds[i - 1] or ends[i] <= B:
            row["skip"] = "next to an empty section"
            rows.append(row)
            continue
        n = min(4, B - bounds[i - 1], ends[i] - B, nbars - B, B)
        if n <= 0:
            row["skip"] = "beyond the audio" if B >= nbars else "before the audio"
            rows.append(row)
            continue
        d = {k: float(F[k][B:B + n].mean() - F[k][B - n:B].mean()) for k in F}
        change = abs(d["drums"]) >= SEC_CHANGE[0] or abs(d["bass"]) >= SEC_CHANGE[1] or abs(d["kick"]) >= SEC_CHANGE[2]
        row.update({k: rnd(v, 2 if k == "kick" else 1) for k, v in d.items()}, span=n, change=bool(change))
        rows.append(row)
    # strong changes anywhere, with isolated 1-bar dips taken out first
    G, dips = {}, {}
    for k, step, back in (("drums", SEC_STRONG[0], SEC_CHANGE[0]), ("bass", SEC_STRONG[1], SEC_CHANGE[1]),
                          ("kick", 0.5, SEC_CHANGE[2])):
        G[k], dips[k] = remove_dips(F[k][:nbars], step, back)
    s = np.zeros(nbars)
    D = {}
    for B in range(2, nbars - 1):
        d = {k: G[k][B:B + 2].mean() - G[k][B - 2:B].mean() for k in G}
        D[B] = d
        s[B] = max(abs(d["drums"]) / SEC_STRONG[0], abs(d["bass"]) / SEC_STRONG[1], abs(d["kick"]) / SEC_STRONG[2])
    unmarked = []
    for B in range(2, nbars - 1):
        if s[B] >= 1 and s[B] >= s[B - 1] and s[B] >= s[B + 1] and min(abs(B - b) for b in bounds) > 1:
            unmarked.append({"bar": B, **{k: rnd(v, 2 if k == "kick" else 1) for k, v in D[B].items()}})
    measured = [r for r in rows if "skip" not in r]
    relabel_no_change = [r for r in measured if not r["change"] and r["from"] != r["to"]]
    issues = len(relabel_no_change) + len(unmarked) + len(empty)
    v = "solid" if issues == 0 else "mostly" if issues <= SEC_MOSTLY_ISSUES else "check"
    return {"verdict": v, "boundaries": rows, "noChange": [r["bar"] for r in measured if not r["change"]],
            "relabelNoChange": [r["bar"] for r in relabel_no_change], "unmarkedStrong": unmarked,
            "oneBarDips": {k: v for k, v in dips.items() if v}, "emptySections": empty,
            "sectionsEndBar": ends[-1], "audioBars": nbars}


# ---------------------------------------------------------------- 6. lyrics
def band_flux(y, nfft=2048, hop=220, lag=2, nb=48):
    """Log-band spectral flux: mean positive dB change over `lag` frames in nb log bands
    80 Hz-8 kHz. Returns (flux, time of each value, seconds)."""
    n = 1 + (len(y) - nfft) // hop if len(y) >= nfft else 0
    fr = np.fft.rfftfreq(nfft, 1 / SR)
    band = np.digitize(fr, np.geomspace(80, 8000, nb + 1)) - 1
    ok = (band >= 0) & (band < nb)
    M = np.zeros((nb, fr.size))
    M[band[ok], np.nonzero(ok)[0]] = 1
    M /= np.maximum(M.sum(1, keepdims=True), 1)
    w = np.hanning(nfft)
    D = np.zeros((n, nb))
    for s in range(0, n, 4096):  # chunked: a whole track of frames at once is gigabytes
        idx = np.arange(s, min(n, s + 4096))[:, None] * hop + np.arange(nfft)
        D[s:s + len(idx)] = 10 * np.log10(np.abs(np.fft.rfft(y[idx] * w, axis=1)) ** 2 @ M.T + 1e-10)
    if n:
        D = np.maximum(D, D.max() - 80)
    fl = np.zeros(n)
    if n > lag:
        fl[lag:] = np.maximum(D[lag:] - D[:-lag], 0).mean(1)
    # frame i is centred on i*hop + nfft/2; the lag-2 difference sits between frames i-2 and i
    return fl, np.arange(n) * hop / SR + nfft / 2 / SR - lag * hop / 2 / SR


def vocal_evidence(y):
    """Independent onset evidence and activity of the vocals stem (method 6)."""
    env, dt = rms_env_db(y, 10, 5)
    ref = float(np.percentile(env, 99)) if env.size else -200.0
    silent = ref < VOX_SILENT_DBFS
    active = np.zeros(env.size, bool) if silent else median_filter(env, 11, mode="nearest") >= ref - VOX_GATE_DB
    lead = int(round(0.03 / dt))

    def is_active(i):
        return active[np.clip(np.asarray(i, int), 0, max(active.size - 1, 0))] if active.size else np.zeros(0, bool)

    look = int(round(LYR_RISE[0] / dt))
    r = rise_db(env, look)
    pk, _ = find_peaks(r, height=LYR_RISE[1], distance=int(0.08 / dt))
    pk = pk[is_active(pk + lead)] if pk.size else pk
    rise = []
    for i in pk:  # date a rise at its half-rise point after the minimum, like the kick attack
        m = max(i - look, 0) + int(np.argmin(env[max(i - look, 0):i + 1]))
        rise.append((m + int(np.argmax(env[m:i + 1] >= env[m] + r[i] / 2))) * dt)
    fl, ft = band_flux(y)
    fa = is_active(np.round(ft / dt))
    flux = np.zeros(0)
    if fa.any():
        hop = ft[1] - ft[0]
        thr = np.maximum(uniform_filter1d(fl, max(1, int(LYR_FLUX[0] / hop))) * LYR_FLUX[1],
                         LYR_FLUX[2] * np.percentile(fl[fa], 99))
        fp, _ = find_peaks(fl, height=1e-9, distance=max(1, int(0.08 / hop)))
        fp = fp[(fl[fp] > thr[fp]) & fa[fp]]
        flux = ft[fp]
    return {"env": env, "dt": dt, "active": active, "refDb": ref, "silent": bool(silent),
            "rise": np.array(rise), "flux": flux, "both": np.sort(np.concatenate([rise, flux]))}


def near_share(times, ev, win):
    return float(np.mean(np.abs(nearest_dist(times, ev)) <= win)) if len(times) else float("nan")


def runs_of(mask):
    """(start, end_exclusive) index runs where mask is True."""
    m = np.diff(np.concatenate([[0], np.asarray(mask, np.int8), [0]]))
    return list(zip(np.nonzero(m == 1)[0], np.nonzero(m == -1)[0]))


def lyrics_check(e, g, ev):
    """e: events; ev: vocal_evidence() of stems/vocals.wav, or None when the stem is missing."""
    L = e.get("lyrics")
    if not L:
        return {"verdict": "n/a", "note": "no lyrics in events.json"}
    W = L.get("words") or {}
    b = np.asarray(W.get("b") or [], float)
    ln = np.asarray(W.get("len") or [], float)
    v = np.asarray(W.get("v") or [], float)  # no v: hand-timed words (handTimed in config.json), confidence n/a
    t, d = g.time(b), ln * g.p
    xb = np.asarray((L.get("extras") or {}).get("b") or [], float)
    xl = np.asarray((L.get("extras") or {}).get("len") or [], float)
    unordered = int((np.diff(b) < -LYR_TOL_BEATS).sum())
    overlaps = int((b[:-1] + ln[:-1] > b[1:] + LYR_TOL_BEATS).sum())
    nonpos = int((ln <= 0).sum())
    out = {"words": int(b.size), "lines": len((L.get("lines") or {}).get("b") or []), "extras": int(xb.size),
           "extrasSeconds": rnd(xl.sum() * g.p, 1), "syllabified": sum(1 for s in W.get("syl") or [] if s),
           "unordered": unordered, "overlaps": overlaps, "nonPositive": nonpos,
           "durationMs": {"median": rnd(np.median(d) * 1000, 0), "min": rnd(d.min() * 1000, 0)} if d.size else None,
           "lowConf": int((v < LYR_LOW_V).sum()) if v.size else None, "medianConf": rnd(np.median(v), 2) if v.size else None,
           "source": L.get("source")}
    if ev is None or ev["silent"] or not b.size:
        out.update(verdict="check", note="no vocals stem" if ev is None else
                   "vocals stem silent" if ev["silent"] else "lyrics with no words")
        return out
    dt, act = ev["dt"], ev["active"]
    ft = np.arange(act.size) * dt
    ta = ft[act]
    # precision against both kinds, and the rate a start placed at random in the voice would get
    dist = nearest_dist(t, ev["both"])
    ok = np.abs(dist) <= LYR_WIN
    P = float(ok.mean())
    chance = near_share(ta, ev["both"], LYR_WIN)
    kappa = (P - chance) / (1 - chance) if chance < 1 else float("nan")
    gap = np.concatenate([[np.inf], t[1:] - (t[:-1] + d[:-1])])
    entry = gap >= LYR_ENTRY_GAP
    # coverage of active voice
    cw = np.zeros(act.size, bool)
    for a, z in zip(t - LYR_COVER_PAD, t + d + LYR_COVER_PAD):
        cw[max(int(a / dt), 0):max(int(np.ceil(z / dt)), 0)] = True
    cu = cw.copy()
    for a, z in zip(g.time(xb), g.time(xb + xl)):
        cu[max(int(a / dt), 0):max(int(np.ceil(z / dt)), 0)] = True
    n_act = int(act.sum())
    cov_w = float((act & cw).sum() / n_act) if n_act else float("nan")
    cov_u = float((act & cu).sum() / n_act) if n_act else float("nan")
    gaps = [(a, z) for a, z in runs_of(act & ~cu) if (z - a) * dt >= 0.5]
    gaps.sort(key=lambda r: r[0] - r[1])
    out.update(precision=rnd(P), chance=rnd(chance), kappa=rnd(kappa),
               byRise=rnd(near_share(t, ev["rise"], LYR_WIN)), byFlux=rnd(near_share(t, ev["flux"], LYR_WIN)),
               timingMedianMs=rnd(np.median(dist[ok]) * 1000, 1) if ok.any() else None,
               entries={"n": int(entry.sum()), "precision": rnd(near_share(t[entry], ev["rise"], LYR_WIN)),
                        "chance": rnd(near_share(ta, ev["rise"], LYR_WIN))},
               unconfirmedBeats=[rnd(x, 2) for x in b[~ok]],
               activeSeconds=rnd(n_act * dt, 1), coverage=rnd(cov_u), coverageWords=rnd(cov_w),
               uncovered={"n": len(gaps), "seconds": rnd(sum(z - a for a, z in gaps) * dt, 1),
                          "longest": [[rnd(g.beat(a * dt), 2), rnd((z - a) * dt / g.p, 2)] for a, z in gaps[:8]]})
    if unordered or overlaps or nonpos:
        out["verdict"] = "check"
    elif P >= LYR_SOLID[0] and kappa >= LYR_SOLID[1] and cov_u >= LYR_SOLID[2]:
        out["verdict"] = "solid"
    elif P >= LYR_MOSTLY[0] and kappa >= LYR_MOSTLY[1] and cov_u >= LYR_MOSTLY[2]:
        out["verdict"] = "mostly"
    else:
        out["verdict"] = "check"
    return out


def vocal_onsets_check(e, g):
    """Do pitch-jump onsets add anything Basic Pitch's vocal notes do not already have?"""
    O = e.get("vocalOnsets")
    if not O:
        return {"verdict": "n/a", "note": "no vocalOnsets in events.json"}
    t = g.time(O.get("b") or [])
    kind = np.asarray(O.get("kind") or [0] * t.size, int)
    N = (e.get("notes") or {}).get("vocals") or {}
    nb = np.asarray(N.get("b") or [], float)
    nr = np.asarray(N.get("r") or [0.0] * nb.size, float)
    nt = g.time(nb) + (nr / 1000 if nr.size == nb.size else 0.0)  # Basic Pitch's own time: b + r
    pitch = t[kind == 1]
    out = {"verdict": "info", "n": int(t.size), "flux": int((kind == 0).sum()), "pitch": int(pitch.size),
           "notes": int(nt.size)}
    if not nt.size:
        out.update(pitchNew=None, notesMatched=None, note="no notes.vocals to compare with")
        return out
    out.update(pitchNew=rnd(1 - near_share(pitch, nt, VON_WIN)) if pitch.size else None,
               pitchNewN=int((np.abs(nearest_dist(pitch, nt)) > VON_WIN).sum()),
               notesMatched=rnd(near_share(nt, t, VON_WIN)),
               notesMatchedFlux=rnd(near_share(nt, t[kind == 0], VON_WIN)),
               notesMatchedPitch=rnd(near_share(nt, pitch, VON_WIN)))
    return out


# ---------------------------------------------------------------- driver
def build(slug, events_path):
    t_start = time.perf_counter()
    e = json.loads(Path(events_path).read_text(encoding="utf-8"))
    g = Grid(e)
    stems = OUT / slug / "stems"
    mix_path = AUDIO / f"{slug}.wav"
    dur = sf.info(str(mix_path)).duration
    audio_beats = float(g.beat(dur))
    nbars = int(audio_beats // 4)
    bar_sec = np.full(max(nbars, 1) + 64, "", dtype=object)
    for s in e["sections"]:
        a = int(s["startBeat"] // 4)
        bar_sec[a: int(np.ceil((s["startBeat"] + s["beats"]) / 4))] = s["label"]

    def bar_label(b):
        return bar_sec[b] if 0 <= b < len(bar_sec) and bar_sec[b] else "outside"

    rep = {"slug": slug, "events": str(events_path), "bpm": g.bpm, "gridMarker": g.t0,
           "lengthBeats": e.get("lengthBeats"), "audioBeats": rnd(audio_beats, 2)}
    lanes = {}

    y = load_mono(stems / "drums_kick.wav")
    attacks = kick_attacks(y)
    env, dt = rms_env_db(y, 5, 1)
    del y
    lanes["kick"] = lane_check("kick", g, *lane(e, g, ["kick"]), env, dt, bar_label)
    rep["grid"] = grid_check(g, attacks)
    kb = onbeat_kick_beats(g, attacks)

    y = load_mono(stems / "drums_snare.wav")
    env, dt = rms_env_db(y, 5, 1)
    del y
    lanes["snare"] = lane_check("snare", g, *lane(e, g, ["snare"]), env, dt, bar_label)

    y = load_mono(stems / "drums_hh.wav")
    env, dt = rms_env_db(y, 5, 1)
    env_ring, dt_ring = rms_env_db(y, HATS_RMS_MS, 1)
    del y
    lanes["hats"] = lane_check("hats", g, *lane(e, g, ["hatClosed", "hatOpen"]), env, dt, bar_label)
    rep["hats"] = hats_check(g, e, env_ring, dt_ring)
    rep["lanes"] = lanes

    bar_edges = g.time(np.arange(nbars + 1) * 4)
    beat_edges = g.time(np.arange(int(audio_beats) + 1))
    F, curves = {}, {}
    for name, path in (("drums", stems / "drums.wav"), ("bass", stems / "bass.wav"), ("mix", mix_path)):
        y = load_mono(path)
        csum = np.concatenate([[0.0], np.cumsum(y * y)])
        del y
        F[name] = floored(span_db(csum, bar_edges))
        if name != "mix":
            curves[name] = curve_check(g, e, name, csum, dur)
        if name == "bass":
            bass_in = bass_entries(span_db(csum, beat_edges))
        del csum
    F["kick"] = np.bincount(kb[(kb >= 0) & (kb < nbars * 4)] // 4, minlength=nbars)[:nbars] / 4.0
    cv = [c["verdict"] for c in curves.values()]
    curves["verdict"] = next((x for x in ("check", "mostly", "solid") if x in cv), "n/a")
    rep["curves"] = curves
    rep["sections"] = sections_check(e, F, nbars)
    rep["downbeat"] = downbeat_check(e, kb, bass_in)
    ev = None
    if e.get("lyrics") and (stems / "vocals.wav").exists():  # the stem is only read when there is something to judge
        ev = vocal_evidence(load_mono(stems / "vocals.wav"))
    rep["lyrics"] = lyrics_check(e, g, ev)
    rep["vocalOnsets"] = vocal_onsets_check(e, g)
    rep["verdicts"] = {"grid": rep["grid"]["verdict"], "downbeat": rep["downbeat"]["verdict"],
                       **{k: v["verdict"] for k, v in lanes.items()}, "hatsOpenClosed": rep["hats"]["verdict"],
                       "curves": curves["verdict"], "sections": rep["sections"]["verdict"],
                       "lyrics": rep["lyrics"]["verdict"], "vocalOnsets": rep["vocalOnsets"]["verdict"]}
    rep["rules"] = VERDICT_RULES
    rep["seconds"] = round(time.perf_counter() - t_start, 1)
    return rep


def render(rep):
    L = []
    w = L.append
    w(f"=== {rep['slug']}  {rep['bpm']:g} bpm  gridMarker {rep['gridMarker']:.4f} s  "
      f"lengthBeats {rep['lengthBeats']} (audio {rep['audioBeats']})  [{rep['events']}]")
    v = rep["verdicts"]
    gr = rep["grid"]
    w(f"  GRID      {v['grid']:6}  kick attack - beat: median {gr.get('medianMs')} ms, p90 |off| {gr.get('p90AbsMs')} ms"
      f" (n {gr['n']}; by quarter {gr.get('driftByQuarterMs')}; {gr.get('offBeatKicks')} off-beat kicks)")
    db = rep["downbeat"]
    w(f"  DOWNBEAT  {v['downbeat']:6}  on beat%4==0: {db['shareOnDownbeat']} | kick re-entries by phase {db['kickReentryPhase']}"
      f" | bass entries by phase {db['bassEntryPhase']} | section starts off the bar: {db['sectionStartsOffBar'] or 'none'}")
    for k, ln in rep["lanes"].items():
        lc = ln["lowConf"]
        low = f"c<0.5: n {lc['n']} P {lc['precision']}" if lc["n"] else "no c<0.5 hits"
        w(f"  {k.upper():9} {ln['verdict']:6}  P {ln['precision']} (n {ln['n']}; {low})"
          f"  R {ln['recall']} (of {ln['onsets']} clear onsets)  lane-onset median {ln['timingMedianMs']} ms")
        w(f"              false hits {ln['falseHits']['n']}: {fmt_clusters(ln['falseHits']['clusters'])}")
        w(f"              misses     {ln['misses']['n']}: {fmt_clusters(ln['misses']['clusters'])}")
        if ln.get("note"):
            w(f"              note: {ln['note']}")
    h = rep["hats"]
    hc, ho = h["hatClosed"], h["hatOpen"]
    w(f"  OPEN/CLOSED {h['verdict']:6} AUC {h['auc']} | open n {ho['n']} ring med {ho['ringMedianMs']} ms, >=150 ms {ho['ringAtLeast150ms']}"
      f" | closed n {hc['n']} ring med {hc['ringMedianMs']} ms, >=150 ms {hc['ringAtLeast150ms']}" + (f" | {h['note']}" if h.get("note") else ""))
    c = rep["curves"]
    w(f"  CURVES    {c['verdict']:6}  " + "; ".join(
        f"{k} r@0 {c[k]['r0']} best lag {c[k]['bestLag']:+d} ({c[k]['lagMs']:+.1f} ms) best offset {c[k]['bestOffsetMs']:+d} ms"
        if "r0" in c[k] else f"{k}: {c[k].get('note')}" for k in ("drums", "bass"))
      + "  (+ = curve late)")
    s = rep["sections"]
    w(f"  SECTIONS  {s['verdict']:6}  {len(s['boundaries'])} boundaries; no change at bars {s['noChange'] or '-'}"
      f" (relabelled with no change: {s['relabelNoChange'] or 'none'}); strong changes with no boundary: "
      + (", ".join(f"bar {u['bar']}" for u in s["unmarkedStrong"]) or "none")
      + (f"; empty sections at bars {s['emptySections']}" if s.get("emptySections") else "")
      + (f"; {s['note']}" if s.get("note") else ""))
    w("    boundary              drums   bass   mix   kickFill")
    for r in s["boundaries"]:
        if "skip" in r:
            w(f"    bar {r['bar']:3d} {r['from']:>9}->{r['to']:<9} {r['skip']}")
            continue
        w(f"    bar {r['bar']:3d} {r['from']:>9}->{r['to']:<9} {r['drums']:+6.1f} {r['bass']:+6.1f} {r['mix']:+5.1f}  {r['kick']:+5.2f}"
          + ("" if r["change"] else "   <- no change" + (" (relabelled)" if r["from"] != r["to"] else "")))
    for u in s["unmarkedStrong"]:
        w(f"    unmarked bar {u['bar']:3d}: drums {u['drums']:+.1f} bass {u['bass']:+.1f} kickFill {u['kick']:+.2f}")
    if s["oneBarDips"]:
        w(f"    isolated 1-bar dips (fills / pre-drop gaps, not counted): {s['oneBarDips']}")
    ly = rep["lyrics"]
    if "precision" in ly:
        w(f"  LYRICS    {ly['verdict']:6}  word starts P {ly['precision']} (chance {ly['chance']}, kappa {ly['kappa']};"
          f" rise {ly['byRise']} flux {ly['byFlux']}; median {ly['timingMedianMs']} ms)"
          f" | entries n {ly['entries']['n']} P {ly['entries']['precision']} vs rises (chance {ly['entries']['chance']})")
        u = ly["uncovered"]
        w(f"              {ly['words']} words, {ly['lines']} lines, {ly['extras']} extras ({ly['extrasSeconds']} s)"
          f" | active voice {ly['activeSeconds']} s covered: words + extras {ly['coverage']}, words {ly['coverageWords']}"
          f"; {u['n']} uncovered runs >= 0.5 s ({u['seconds']} s)"
          + (f", longest at beats {', '.join(str(x[0]) for x in u['longest'][:4])}" if u["longest"] else ""))
        dm = ly["durationMs"]
        conf = (f"v < {LYR_LOW_V}: {ly['lowConf']} (median v {ly['medianConf']})" if ly["medianConf"] is not None
                else "confidence n/a (no v)")
        w(f"              duration median {dm['median']} ms, min {dm['min']} ms | {conf}"
          f" | unordered {ly['unordered']}, overlapping {ly['overlaps']}, len <= 0 {ly['nonPositive']}")
    else:
        w(f"  LYRICS    {ly['verdict']:6}  {ly.get('note')}")
    vo = rep["vocalOnsets"]
    if vo["verdict"] == "n/a":
        w(f"  VOX ONSETS {vo['verdict']:5}  {vo.get('note')}")
    else:
        w(f"  VOX ONSETS {vo['verdict']:5}  {vo['n']} onsets: flux {vo['flux']}, pitch {vo['pitch']}"
          + (f" | pitch jumps with no Basic Pitch start +-{VON_WIN * 1000:g} ms: {vo['pitchNew']} ({vo['pitchNewN']})"
             f" | of {vo['notes']} note starts, with an onset: {vo['notesMatched']} (flux {vo['notesMatchedFlux']},"
             f" pitch {vo['notesMatchedPitch']})" if vo["notesMatched"] is not None else f" | {vo.get('note')}"))
    w(f"  ({rep['seconds']} s)")
    return "\n".join(L)


def slugs_all():
    return sorted(p.parent.name for p in OUT.glob("*/events.json") if (p.parent / "stems").is_dir())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--slug")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--events", help="events.json to measure instead of out/<slug>/events.json; report.json is written beside it")
    a = ap.parse_args(argv)
    if bool(a.slug) == bool(a.all) or (a.events and not a.slug):
        ap.error("give --slug <slug> (optionally with --events) or --all")
    todo = [(a.slug, Path(a.events) if a.events else OUT / a.slug / "events.json")] if a.slug else \
        [(s, OUT / s / "events.json") for s in slugs_all()]
    for slug, ev in todo:
        rep = build(slug, ev)
        print(render(rep), flush=True)
        (ev.parent / "report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
