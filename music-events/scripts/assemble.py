"""Merge the raw stage outputs into one Fantasynth events file per track.

Reads out/<slug>/{grid.json, raw_adtof.json, raw_drumsep_onsets.json, raw_notes_*.json,
curves.npz, stems/drums_*.wav} (whatever exists) and writes out/<slug>/events.json.

Time base: beats on the fitted rigid grid (beat 0 is a downbeat). Raw seconds survive
only as snap residuals, for QA.
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # music-events/, for workspace.py
from workspace import AUDIO, OUT, all_slugs  # noqa: E402

STEP = 0.25            # 1/16 note in beats
SAMPLES_PER_BEAT = 24  # curve resolution: 16ths and triplets both land on samples
MATCH_MS = 30          # detections within this window count as the same hit
SOLO_DOM_DB = 10       # a DrumSep-only hit is kept if its sub-stem is this much louder than every other
MIN_SWING_MS = 25      # odd 16ths less than this late (vs the even ones) count as straight.
                       # Detector offsets alone read as 5-18 ms "swing" on straight lanes
                       # (test track B); test track A's real hat swing is 35-39 ms.


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


class Grid:
    def __init__(self, g):
        self.t0 = g["t0"]
        self.period = g["beatPeriod"]
        self.meter = g.get("meter", 4)

    def beat(self, t):
        return (np.asarray(t, dtype=float) - self.t0) / self.period

    def time(self, b):
        return self.t0 + np.asarray(b, dtype=float) * self.period


def snap(grid, times, swing=0.0, step=STEP):
    """Snap times to the 16th grid. With `swing`, the 2nd and 4th 16th of every beat offer
    BOTH a straight and a swung slot (test track A mixes the two in one hat lane; with only the
    swung slot a straight hit was pulled 35 ms late and still counted as on-grid).
    Returns (snapped beats, residual ms, on-grid mask)."""
    b = grid.beat(times)
    straight = np.round(b / step) * step
    snapped = straight
    if swing:
        swung = np.round((b - step - swing) / (2 * step)) * 2 * step + step + swing
        snapped = np.where(np.abs(b - swung) < np.abs(b - straight), swung, straight)
    resid_ms = (b - snapped) * grid.period * 1000.0
    on_grid = np.abs(resid_ms) <= (step * grid.period * 1000.0) / 3.0
    return snapped, resid_ms, on_grid


def estimate_swing(grid, times):
    """Swing of one lane, in beats: the median position of its hits on the 2nd/4th 16th
    (0 if straight). Measured per lane, since swing isn't global: test track A's hats land +33 ms
    on those 16ths while its snares sit on the straight ones.
    The GATE uses odd-minus-even lateness, so a detector's constant timing bias (ADTOF runs
    ~12 ms early) can't by itself read as swing. The RETURNED value is the raw odd median,
    because that's where the hits are: on track A's hats (timed by DrumSep) it gives 35 ms
    against 30-37 ms measured from the audio, where odd-minus-even would give 44 ms."""
    b = grid.beat(np.asarray(times, dtype=float))
    idx = np.round(b / STEP).astype(int)
    resid_ms = (b - idx * STEP) * grid.period * 1000
    odd = idx % 2 == 1
    if odd.sum() < 32 or (~odd).sum() < 8:
        return 0.0, {"oddHits": int(odd.sum())}
    even_ms = float(np.median(resid_ms[~odd]))
    # A lane can mix straight and swung odd 16ths (track A's hats: +7 ms and +35 ms); snap()
    # offers both slots, so look for a distinct LATE peak: the densest 20 ms window of odd
    # hits lying at least MIN_SWING_MS after the even hits. Jitter around a straight 16th
    # (test track C, +/-5 ms) has no such peak.
    odd_ms = np.sort(resid_ms[odd])
    late = odd_ms[odd_ms - even_ms >= MIN_SWING_MS]
    info = {"oddHits": int(odd.sum()), "evenMedianMs": round(even_ms, 1)}
    if late.size == 0:
        return 0.0, info
    counts = [np.sum(np.abs(late - c) <= 10) for c in late]
    peak = late[int(np.argmax(counts))]
    cluster = late[np.abs(late - peak) <= 10]
    info.update({"swungHits": int(cluster.size), "swingMs": round(float(np.median(cluster)), 1)})
    if cluster.size < max(24, 0.2 * odd.sum()):
        return 0.0, info
    return float(np.median(cluster)) / (grid.period * 1000), info


def lane(grid, times, vels, source, conf=None, swing=None):
    """One drum lane in columnar form. Off-grid hits keep their unsnapped beat and r=0
    (a residual to a step they weren't snapped to would double-count). `swing` defaults to
    the lane's own estimate."""
    times = np.asarray(times, dtype=float)
    vels = np.asarray(vels, dtype=float)
    conf = np.ones_like(vels) if conf is None else np.asarray(conf, dtype=float)
    if swing is None:
        swing, _ = estimate_swing(grid, times) if times.size else (0.0, None)
    if times.size == 0:
        return {"b": [], "v": [], "c": [], "r": [], "swing": 0.0, "offGrid": 0, "source": source}
    snapped, resid, on_grid = snap(grid, times, swing)
    beats = np.where(on_grid, snapped, grid.beat(times))
    resid = np.where(on_grid, resid, 0.0)
    # Two detections snapping to the same 16th collapse into one; keep the louder. Keyed by
    # the 16th index, so a straight and a swung slot of the same 16th can't both keep one
    # strike (seen on test track B: 150.25/150.288).
    order = np.lexsort((-vels, beats))
    keep, seen = [], set()
    for i in order:
        key = int(np.round(beats[i] / STEP)) if on_grid[i] else round(float(beats[i]), 4)
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    keep = np.array(sorted(keep, key=lambda i: beats[i]), dtype=int)
    return {
        "b": [round(float(x), 4) for x in beats[keep]],
        "v": [round(float(x), 3) for x in vels[keep]],
        "c": [round(float(x), 2) for x in conf[keep]],
        "r": [round(float(x), 1) for x in resid[keep]],
        "swing": round(float(swing), 4),
        "offGrid": int((~on_grid[keep]).sum()),
        "source": source,
    }


def stem_envelope(path, fps=100):
    """Linear RMS envelope of a stem at `fps`, or None if the stem is missing."""
    if not path.exists():
        return None
    import soundfile as sf
    y, sr = sf.read(str(path), always_2d=True)
    y = y.mean(axis=1)
    hop = sr // fps
    n = len(y) // hop
    frames = y[: n * hop].reshape(n, hop)
    return np.sqrt((frames ** 2).mean(axis=1) + 1e-12)


OPEN_HAT_DECAY_MS = 150


def split_hats(times, vels, hh_env, fps=100):
    """A hat is open if the DrumSep hh stem takes >= OPEN_HAT_DECAY_MS to fall 12 dB below
    its peak, measured only up to the next hat. Riley & Dixon's rule (stays above 75% of
    peak) never fired on the test tracks. Measured decay to -12 dB: track B's off-beat hats
    ring ~170-220 ms (open), track C's 16ths ~30 ms (closed). A hit that never falls 12 dB
    before the next one is sustained wash (track C's phrase-start cymbals), not an open hat.
    Returns boolean mask 'open'."""
    times = np.asarray(times, dtype=float)
    if hh_env is None or times.size == 0:
        return np.zeros(times.size, dtype=bool)
    nxt = np.append(times[1:], np.inf)
    is_open = np.zeros(times.size, dtype=bool)
    for i, t in enumerate(times):
        a = int(t * fps)
        span = int(min(nxt[i] - t, 0.4) * fps)
        seg = hh_env[a: a + span]
        if seg.size < 5:
            continue
        p = int(np.argmax(seg[:5]))
        below = np.nonzero(seg[p:] < seg[p] * 10 ** (-12 / 20))[0]
        # Still ringing at the end of the window counts as ringing for the whole window:
        # a long open hat before a far next hit is open; wash under a 16th pattern
        # (window < 150 ms) stays closed.
        rang_ms = (below[0] if below.size else seg.size - p) * 1000 / fps
        is_open[i] = rang_ms >= OPEN_HAT_DECAY_MS
    return is_open


def split_cymbals(times, crash_env, ride_env, fps=100):
    """Crash vs ride by which DrumSep stem is louder in the 60 ms after the hit.
    Simpler than Riley & Dixon's refractory rule; returns boolean mask 'crash'."""
    times = np.asarray(times, dtype=float)
    if crash_env is None or ride_env is None or times.size == 0:
        return np.ones(times.size, dtype=bool)
    out = np.zeros(times.size, dtype=bool)
    for i, t in enumerate(times):
        a, z = int(t * fps), int(t * fps) + 6
        out[i] = crash_env[a:z].mean() >= ride_env[a:z].mean()
    return out


def adtof_hits(raw, name):
    return [(h["t"], {"vel": h["vel"]}) for h in (raw or {}).get(name, [])]


def sep_hits(seps, *parts):
    return sorted(((h["t"], {"db": h.get("db", -60.0), "dom": h.get("dom", -99.0)})
                   for p in parts for h in seps.get(p, [])), key=lambda x: x[0])


def nearest(sorted_t, t):
    """Index of the value in a sorted array closest to t."""
    j = int(np.searchsorted(sorted_t, t))
    if j == 0:
        return 0
    if j == len(sorted_t):
        return len(sorted_t) - 1
    return j if sorted_t[j] - t < t - sorted_t[j - 1] else j - 1


def vote(detectors, window_ms=MATCH_MS):
    """Group hits from several detectors into clusters, at most one hit per detector each.
    `detectors` = {name: [(t, payload), ...]} (each list sorted by t). Non-maximum
    suppression: every hit proposes a cluster made of the nearest hit of each detector
    within the window; proposals are taken best-first (most detectors, then tightest) and
    may not reuse a hit. Hits left over become single-detector clusters.
    Returns [{name: (t, payload)}]."""
    w = window_ms / 1000.0
    data = {n: (np.array([h[0] for h in hits], dtype=float), [h[1] for h in hits])
            for n, hits in detectors.items()}
    proposals = []
    for n, (ts, _) in data.items():
        for t in ts:
            members = {}
            for m, (tm, _) in data.items():
                if tm.size:
                    j = nearest(tm, t)
                    if abs(tm[j] - t) <= w:
                        members[m] = j
            spread = sum(abs(data[m][0][j] - t) for m, j in members.items())
            proposals.append((-len(members), spread, members))
    proposals.sort(key=lambda p: (p[0], p[1]))
    used = {n: set() for n in data}
    clusters = []
    for _, _, members in proposals:
        if any(j in used[m] for m, j in members.items()):
            continue
        for m, j in members.items():
            used[m].add(j)
        clusters.append({m: (data[m][0][j], data[m][1][j]) for m, j in members.items()})
    # A proposal blocked by a reused hit can strand its anchor; keep those as singles.
    for m, (tm, payloads) in data.items():
        for j in range(len(tm)):
            if j not in used[m]:
                clusters.append({m: (tm[j], payloads[j])})
    return clusters


def pick(clusters, keep, time_order, n_detectors):
    """Keep clusters for which `keep(cluster)` is true. Time from the first detector in
    `time_order` that saw it; velocity = loudest ADTOF activation, or for a DrumSep-only
    hit its level mapped from -40..-10 dBFS; confidence = share of detectors that agree."""
    times, vels, confs = [], [], []
    for c in clusters:
        if not keep(c):
            continue
        times.append(next(c[n][0] for n in time_order if n in c))
        adtof = [c[n][1]["vel"] for n in ("stem", "mix") if n in c]
        vels.append(max(adtof) if adtof else float(np.clip((c["sep"][1]["db"] + 40) / 30, 0.1, 1.0)))
        confs.append(len(c) / n_detectors)
    order = np.argsort(times)
    return np.array(times, dtype=float)[order], np.array(vels, dtype=float)[order], np.array(confs, dtype=float)[order]


def two_votes(c):
    return len(c) >= 2


def two_votes_or_solo_sep(c):
    """Two detectors agree, or DrumSep alone with that sub-stem clearly the loudest drum.
    Track B's build rolls are DrumSep-only: ADTOF found 2-6 snares where the snare stem
    has 112-124 onsets, up to a median 25 dB over every other drum sub-stem."""
    return len(c) >= 2 or (set(c) == {"sep"} and c["sep"][1]["dom"] >= SOLO_DOM_DB)


HAT_SOLO_DOM_DB = -8  # DrumSep-only hat away from any kick/snare


def hat_rule(kick_t, snare_t, seps):
    """Hats need extra rules, all measured on one test track (A):
    - ADTOF (stem AND mix) reads kicks and backbeat claps as hats: 110 on-beat 'hats' where
      the hh stem sits a median 18 dB under the other drums, and 46 on snare-lane beats.
      A hat that coincides with a kick or snare therefore needs the DrumSep hh vote.
    - its swung 16ths (+33 ms on the 2nd/4th 16th) are heard only by DrumSep, at a
      median -4 dB dominance: ADTOF catches 14 of ~370. A DrumSep-only hat is accepted at
      HAT_SOLO_DOM_DB when nothing else hits within the window. 'Anything else' includes
      raw DrumSep kick/snare onsets, not just the voted lanes: its swung ghost snares
      (snare stem +35 dB) are not in the snare lane and passed as 129 false hats."""
    hitters = np.sort(np.concatenate([kick_t, snare_t]))
    # Raw DrumSep kick/snare onsets veto only where they dominate: track A's ghost snares sit
    # at +35 dB, while quiet raw snare onsets (dom -12) vetoed 10 real hats in track B's builds.
    loud = [t for t, p in sep_hits(seps, "kick", "snare") if p["dom"] >= 0]
    others = np.sort(np.concatenate([hitters, loud]))
    w = MATCH_MS / 1000.0

    def near(ts, t):
        return ts.size > 0 and abs(ts[nearest(ts, t)] - t) <= w

    def keep(c):
        t = next(iter(c.values()))[0]
        if len(c) >= 2:
            return "sep" in c or not near(hitters, t)
        if set(c) == {"sep"}:
            dom = c["sep"][1]["dom"]
            return dom >= SOLO_DOM_DB or (dom >= HAT_SOLO_DOM_DB and not near(others, t))
        return False
    return keep


def build_drums(grid, d):
    """Three detectors per class: ADTOF on the drum stem, ADTOF on the full mix, and onsets
    on the matching DrumSep sub-stem. Measured on the test tracks: a single ADTOF pass alone
    adds junk (on track C, 53 off-beat stem-only "kicks" were bass notes; the stem pass
    misses most hats on track B), so a hit needs two votes, except isolated DrumSep hits."""
    stem = (load_json(d / "raw_adtof.json") or {}).get("classes")
    if not stem:
        return None, {}, grid
    mix = (load_json(d / "raw_adtof_mix.json") or {}).get("classes", {})
    seps = load_json(d / "raw_drumsep_onsets.json") or {}
    stats = {}

    def three(cls, parts):
        return {"stem": adtof_hits(stem, cls), "mix": adtof_hits(mix, cls), "sep": sep_hits(seps, *parts)}

    # Kick: DrumSep timing is closest to the attack and tightest on the grid (measured).
    kick_t, kick_v, kick_c = pick(vote(three("kick", ["kick"])), two_votes, ("sep", "stem", "mix"), 3)
    snare_t, snare_v, snare_c = pick(vote(three("snare", ["snare"])), two_votes_or_solo_sep,
                                     ("stem", "mix", "sep"), 3)
    hat_t, hat_v, hat_c = pick(vote(three("hihat", ["hh"])), hat_rule(kick_t, snare_t, seps),
                               ("sep", "stem", "mix"), 3)
    # Toms: DrumSep's tom stem is junk on all three test tracks and stem-ADTOF toms were kick
    # tails, so require both ADTOF passes and no kick in the same window.
    tom_t, tom_v, tom_c = pick(vote({"stem": adtof_hits(stem, "tom"), "mix": adtof_hits(mix, "tom")}),
                               lambda c: {"stem", "mix"} <= set(c), ("stem", "mix"), 2)
    if tom_t.size and kick_t.size:
        far = np.min(np.abs(tom_t[:, None] - kick_t[None, :]), axis=1) > MATCH_MS / 1000.0
        tom_t, tom_v, tom_c = tom_t[far], tom_v[far], tom_c[far]
    # Cymbals: the empty DrumSep ride/crash stems and the mix pass both produce junk on
    # their own, so the drum-stem pass must be one of the votes.
    cym_t, cym_v, cym_c = pick(vote(three("cymbal", ["ride", "crash"])),
                               lambda c: "stem" in c and len(c) >= 2, ("stem", "mix", "sep"), 3)

    # Align the grid to the confirmed on-beat kicks: on four-to-the-floor tracks the kick
    # attack IS the beat, and the beat trackers land ~20 ms late (measured).
    b = grid.beat(kick_t)
    on_beat = np.abs(b - np.round(b)) < 0.12
    shift = float(np.median((b[on_beat] - np.round(b[on_beat])) * grid.period)) if on_beat.sum() >= 16 else 0.0
    grid.t0 += shift
    stats["gridShiftToKicksMs"] = round(shift * 1000, 1)
    stats["onBeatKicks"] = int(on_beat.sum())

    # Open and closed hats are one instrument: one swing estimate for both.
    hat_swing, stats["hatSwing"] = estimate_swing(grid, hat_t) if hat_t.size else (0.0, None)

    drums = {
        "kick": lane(grid, kick_t, kick_v, "2-of-3 vote adtof-stem/adtof-mix/drumsep; drumsep timing", kick_c),
        "snare": lane(grid, snare_t, snare_v, "2-of-3 vote or isolated drumsep; claps land here (no clap class)",
                      snare_c),
        # "toms" (not ADTOF's singular "tom" class): the events file uses the app's lane
        # vocabulary, which matches the DrumSep stem name drums_toms.wav.
        "toms": lane(grid, tom_t, tom_v, "adtof stem AND mix, not coincident with a kick", tom_c),
    }
    is_open = split_hats(hat_t, hat_v, stem_envelope(d / "stems" / "drums_hh.wav"))
    drums["hatClosed"] = lane(grid, hat_t[~is_open], hat_v[~is_open],
                              "2-of-3 vote or isolated drumsep; closed by decay", hat_c[~is_open], hat_swing)
    drums["hatOpen"] = lane(grid, hat_t[is_open], hat_v[is_open],
                            "2-of-3 vote or isolated drumsep; open = decay >= 150 ms before next hit",
                            hat_c[is_open], hat_swing)
    is_crash = split_cymbals(cym_t, stem_envelope(d / "stems" / "drums_crash.wav"),
                             stem_envelope(d / "stems" / "drums_ride.wav"))
    drums["crash"] = lane(grid, cym_t[is_crash], cym_v[is_crash], "adtof-stem + 1 vote; crash by DrumSep loudness",
                          cym_c[is_crash])
    drums["ride"] = lane(grid, cym_t[~is_crash], cym_v[~is_crash], "adtof-stem + 1 vote; ride by DrumSep loudness",
                         cym_c[~is_crash])

    stats["hits"] = {k: len(v["b"]) for k, v in drums.items()}
    stats["soloDrumSep"] = {k: sum(1 for c in v["c"] if c < 0.5) for k, v in drums.items() if k != "toms"}
    stats["offGrid"] = {k: v["offGrid"] for k, v in drums.items()}
    return drums, stats, grid


def build_notes(grid, d):
    notes = {}
    for path in sorted(d.glob("raw_notes_*.json")):
        stem = path.stem.replace("raw_notes_", "")
        raw = load_json(path)["notes"]
        if not raw:
            continue
        t = np.array([n["t"] for n in raw])
        swing, _ = estimate_swing(grid, t)
        snapped, resid, on_grid = snap(grid, t, swing)
        b = np.where(on_grid, snapped, grid.beat(t))
        resid = np.where(on_grid, resid, 0.0)  # off-grid notes keep their raw beat; no residual
        ends = grid.beat(t + np.array([n["dur"] for n in raw]))
        ln = np.maximum(ends - b, STEP / 2)
        order = np.argsort(b)
        notes[stem] = {
            "b": [round(float(b[i]), 4) for i in order],
            "len": [round(float(ln[i]), 3) for i in order],
            "pitch": [int(raw[i]["pitch"]) for i in order],
            "v": [round(float(raw[i]["vel"]), 3) for i in order],
            "r": [round(float(resid[i]), 1) for i in order],
            "swing": round(float(swing), 4),
        }
    return notes


LYRICS_SOURCE = "forced alignment (MMS_FA + wav2vec2 LV60K, CTC, mono/L/R mixture)"
ONSET_KINDS = {"flux": 0, "pitch": 1}


def build_lyrics(grid, d):
    """raw_lyrics.json (seconds) -> the `lyrics` and `vocalOnsets` blocks (beats on the
    SHIFTED grid). Not snapped: vocals sit off the grid on purpose. Either block is None when
    it would be empty (no lines / no onsets, or no raw_lyrics.json at all).

    Hand-timed words (handTimed in config.json) carry no `conf`: then `words.v` is omitted and the
    app's words lane hits at 1. `source` is the raw file's own when it names one."""
    raw = load_json(d / "raw_lyrics.json")
    if not raw:
        return None, None

    def span(t, dur):
        b = float(grid.beat(t))
        # round the END and derive len from the rounded ends, so legato words that tile
        # in seconds still tile exactly in beats (rounding b and len separately does not)
        b0, b1 = round(b, 3), round(float(grid.beat(t + dur)), 3)
        return b0, round(b1 - b0, 3)

    lines = {"b": [], "len": [], "text": []}
    words = {"b": [], "len": [], "v": [], "w": [], "line": [], "syl": []}
    for ln in raw.get("lines") or []:
        ws = ln.get("words") or []
        if not ws:
            continue
        i = len(lines["b"])
        b, n = span(ws[0]["t"], ws[-1]["t"] + ws[-1]["d"] - ws[0]["t"])
        lines["b"].append(b)
        lines["len"].append(n)
        lines["text"].append(ln["text"])
        for w in ws:
            b, n = span(w["t"], w["d"])
            words["b"].append(b)
            words["len"].append(n)
            words["v"].append(round(float(w["conf"]), 3) if "conf" in w else None)
            words["w"].append(w["w"])
            words["line"].append(i)
            syl = w.get("syl")
            words["syl"].append([list(span(t, dur)) for t, dur in syl] if syl else None)
    has_v = [v is not None for v in words["v"]]
    if not any(has_v):
        del words["v"]
    elif not all(has_v):
        raise ValueError(f"raw_lyrics.json: {has_v.count(False)} of {len(has_v)} words have no conf; "
                         f"a track is either aligned or hand-timed, not both")
    lyrics = None
    if lines["b"]:
        extras = {"b": [], "len": []}
        for x in raw.get("extras") or []:
            b, n = span(x["t"], x["d"])
            extras["b"].append(b)
            extras["len"].append(n)
        lyrics = {"lines": lines, "words": words, "extras": extras, "source": raw.get("source") or LYRICS_SOURCE}
    onsets = raw.get("onsets") or []
    vocal_onsets = {
        "b": [round(float(grid.beat(o["t"])), 3) for o in onsets],
        "v": [round(float(o["s"]), 3) for o in onsets],
        "kind": [ONSET_KINDS[o["kind"]] for o in onsets],
    } if onsets else None
    return lyrics, vocal_onsets


VOCAL_ENTRY_GAP = 4  # beats of silence before a line that make its first word a vocal entry


def lyric_entries(words):
    """Beats where the voice comes (back) in: the first word of a line starting at least
    VOCAL_ENTRY_GAP beats after the previous word ends, and the first line. `words` is the
    `lyrics.words` block (b, len, line)."""
    entries, prev_end, prev_line = [], None, None
    for b, n, line in zip(words["b"], words["len"], words["line"]):
        if line != prev_line and (prev_end is None or b - prev_end >= VOCAL_ENTRY_GAP):
            entries.append(b)
        prev_end, prev_line = b + n, line
    return entries


MIN_SECTION_BARS = 2


def entry_bar(e):
    """Pickup rule (mexicat/pdoom-video analysis/analyze.py, MIT; see THIRD_PARTY_NOTICES.md): an entry in the first half of bar k starts the section
    at bar k (a long pickup belongs to its own bar); from beat 2 on it is a short pickup
    of 2 beats or less leading into the next downbeat, bar k + 1."""
    k = int(np.floor(e / 4))
    return k if e - 4 * k < 2 else k + 1


def apply_vocal_entries(sections, entries):
    """Pull an inner boundary exactly one bar from a lyric entry's bar onto it, unless that
    leaves a section under MIN_SECTION_BARS. The first section's start and the last one's end
    never move; a boundary already on an entry bar, or moved once, stays put. Sections are
    {startBar, endBar, ...}; a moved section gets `lyricsBar`. Returns (sections, moved)."""
    sections = [dict(s) for s in sections]
    if len(sections) < 2:
        return sections, 0
    targets = sorted({entry_bar(e) for e in entries})
    locked = {i for i in range(1, len(sections)) if sections[i]["startBar"] in targets}
    moved = 0
    for t in targets:
        if any(sections[i]["startBar"] == t for i in range(1, len(sections))):
            continue
        for i in range(1, len(sections)):
            s, prev = sections[i], sections[i - 1]
            if i in locked or abs(s["startBar"] - t) != 1:
                continue
            if t - prev["startBar"] < MIN_SECTION_BARS or s["endBar"] - t < MIN_SECTION_BARS:
                continue
            s["startBar"] = prev["endBar"] = t
            s["lyricsBar"] = t
            locked.add(i)
            moved += 1
            break
    return sections, moved


def encode_curve(values):
    lo, hi = np.percentile(values, 5), np.percentile(values, 95)
    norm = np.clip((values - lo) / max(hi - lo, 1e-9), 0, 1)
    return base64.b64encode(np.round(norm * 255).astype(np.uint8).tobytes()).decode("ascii")


# curves.npz keys follow the stem FILE names (drums_hh.wav -> "hh"), which are DrumSep's
# vocabulary and must not be renamed (renaming them would force a re-separation). The events
# file uses the app's lane vocabulary instead, so the one disagreement is renamed here, at
# the boundary. Idempotent: a key already spelled the app's way passes through.
CURVE_STEM_RENAME = {"hh": "hats"}


def build_curves(grid, d, n_beats):
    path = d / "curves.npz"
    if not path.exists():
        return None, None
    z = np.load(path)
    fps = float(z["fps"])
    k = np.arange(int(n_beats * SAMPLES_PER_BEAT))
    t = grid.time(k / SAMPLES_PER_BEAT)
    stems, per_beat = {}, {}
    for key in z.files:
        if key == "fps":
            continue
        stem, feat = key.split("_", 1)
        stem = CURVE_STEM_RENAME.get(stem, stem)
        src = z[key]
        frames = np.clip(t * fps, 0, len(src) - 1)
        vals = np.interp(frames, np.arange(len(src)), src)
        stems.setdefault(stem, {})[feat] = encode_curve(vals)
        if feat == "rms_db":
            per_beat[stem] = vals
    return {"samplesPerBeat": SAMPLES_PER_BEAT, "encoding": "u8-base64-p5p95", "stems": stems}, per_beat


PHRASE_BARS = 8


def snap_to_phrases(sections):
    """allin1 boundaries drift by up to about a bar (one test track came out 7,9,8,...,7,9 bars).
    EDM phrases are 8 bars, so find the phrase offset most boundaries agree on and pull
    any boundary within one bar of that phrase grid onto it. Boundaries further away
    are left alone (a genuine 4-bar pickup stays)."""
    if len(sections) < 2:
        return sections, None
    bounds = [s["startBar"] for s in sections[1:]]
    votes = np.bincount(np.array(bounds) % PHRASE_BARS, minlength=PHRASE_BARS)
    offset = int(np.argmax(votes))
    moved = 0
    for s in sections[1:]:
        k = round((s["startBar"] - offset) / PHRASE_BARS)
        target = offset + k * PHRASE_BARS
        if target != s["startBar"] and abs(target - s["startBar"]) <= 1:
            s["startBar"] = target
            moved += 1
    for a, b in zip(sections, sections[1:]):
        a["endBar"] = b["startBar"]
    sections = [s for s in sections if s["endBar"] > s["startBar"]]
    return sections, {"offset": offset, "votes": votes.tolist(), "moved": moved}


SPLIT_KICK_FILL = 0.75  # kick fill change across a 4-bar mark that forces a split
SPLIT_DRUMS_DB = 6      # drums-stem level step (4 bars either side) that forces a split
SPLIT_BASS_DB = 9


LEVEL_FLOOR_DB = -50  # quieter than this counts as silence when comparing sections


def bar_profile(n_bars, kick_beats, per_beat_rms):
    """Per bar: fraction of beats with an on-beat kick, and the mean linear amplitude of the
    drums and bass stems."""
    fill = np.zeros(n_bars)
    on = kick_beats[np.abs(kick_beats - np.round(kick_beats)) < 1e-6]
    for bar in (np.floor(on / 4).astype(int)):
        if 0 <= bar < n_bars:
            fill[bar] += 0.25

    def amp(stem):
        v = (per_beat_rms or {}).get(stem)
        if v is None:
            return np.zeros(n_bars)
        spb = 4 * SAMPLES_PER_BEAT
        return np.array([np.mean(10 ** (v[i * spb:(i + 1) * spb] / 20)) if v[i * spb:(i + 1) * spb].size else 0.0
                         for i in range(n_bars)])
    return fill, amp("drums"), amp("bass")


def level_db(amps):
    """dB of the MEAN amplitude, floored. Averaging dB let silence dominate: two silent
    halves at -95 vs -85 dBFS, or one silent bar in a flat 8, each forced a split."""
    return max(20 * np.log10(np.mean(amps) + 1e-12), LEVEL_FLOOR_DB)


def split_sections(sections, fill, drums_amp, bass_amp):
    """allin1 misses some boundaries (track A bar 68: kicks 4/bar -> 0 and bass -42 dB with
    no boundary; track C bar 26: drums +9 dB mid-section). Split at a 4-bar mark inside a
    section when the 4 bars either side differ clearly in kick fill or stem level."""
    out = []
    for s in sections:
        a, z = s["startBar"], s["endBar"]
        cuts = []
        for c in range(a + 4, z - 3, 4):
            before, after = slice(max(c - 4, a), c), slice(c, min(c + 4, z))
            if after.stop <= after.start:
                continue
            d_fill = abs(fill[after].mean() - fill[before].mean())
            d_drums = abs(level_db(drums_amp[after]) - level_db(drums_amp[before]))
            d_bass = abs(level_db(bass_amp[after]) - level_db(bass_amp[before]))
            if d_fill >= SPLIT_KICK_FILL or d_drums >= SPLIT_DRUMS_DB or d_bass >= SPLIT_BASS_DB:
                cuts.append(c)
        edges = [a] + cuts + [z]
        for i, (x, y) in enumerate(zip(edges, edges[1:])):
            out.append({**s, "startBar": x, "endBar": y, "split": i > 0})
    return out


def relabel_sections(sections, drums, per_beat_rms, audio_beats, vocal_entries=None):
    """EDM labels from energy. Pure heuristics; `labelSource` records the evidence so a
    manual override can replace any of it. `vocal_entries` (beats, from lyric_entries)
    refine the phrase-snapped boundaries before the energy splits."""
    if not sections:
        return []
    sections, phrase = snap_to_phrases([dict(s) for s in sections])
    print(f"    phrase grid: offset {phrase['offset']} bars, boundary votes {phrase['votes']}, "
          f"{phrase['moved']} boundaries moved" if phrase else "    phrase grid: n/a")
    if vocal_entries is not None:
        sections, moved = apply_vocal_entries(sections, vocal_entries)
        print(f"    lyric entries: {len(vocal_entries)}, {moved} boundaries moved "
              f"{[s['startBar'] for s in sections if 'lyricsBar' in s]}")
    kick_beats = np.array(drums["kick"]["b"]) if drums else np.array([])
    n_bars = int(np.ceil(audio_beats / 4))
    fill, drums_amp, bass_amp = bar_profile(n_bars, kick_beats, per_beat_rms)
    sections = split_sections(sections, fill, drums_amp, bass_amp)
    print(f"    energy splits: {[s['startBar'] for s in sections if s['split']]}")
    rows = []
    for s in sections:
        a, z = s["startBar"] * 4, s["endBar"] * 4
        bars = max(s["endBar"] - s["startBar"], 1)
        kicks_on_beat = ((kick_beats >= a) & (kick_beats < z) & (np.abs(kick_beats - np.round(kick_beats)) < 1e-6)).sum()
        kick_fill = kicks_on_beat / (bars * 4)
        energy = {}
        for stem in ("drums", "bass"):
            v = per_beat_rms.get(stem) if per_beat_rms else None
            if v is not None:
                seg = v[a * SAMPLES_PER_BEAT: z * SAMPLES_PER_BEAT]
                energy[stem] = float(np.mean(10 ** (seg / 20))) if seg.size else 0.0
        rows.append((s, kick_fill, energy))

    def norm(stem):
        vals = [r[2].get(stem, 0.0) for r in rows]
        top = max(vals) or 1.0
        return [v / top for v in vals]

    drums_n, bass_n = norm("drums"), norm("bass")
    out = []
    for i, (s, fill, _) in enumerate(rows):
        level = 0.5 * drums_n[i] + 0.5 * bass_n[i]
        if fill >= 0.75 and level >= 0.8:
            label = "drop"
        elif fill < 0.25 and level < 0.5:
            label = "breakdown"
        else:
            label = "groove"
        out.append({
            "startBeat": s["startBar"] * 4,
            "beats": (s["endBar"] - s["startBar"]) * 4,
            "label": label,
            "allin1": s["label"],
            "kickFill": round(float(fill), 2),
            "energy": round(float(level), 2),
            "split": s["split"],
            # Only the fragment that starts on the moved boundary carries the lyric evidence.
            "lyricsBar": None if s["split"] else s.get("lyricsBar"),
        })
    # The grid can extend past the audio; the last section ends with the audio. Trim
    # BEFORE labelling so the outro can't be assigned to a section that then gets dropped.
    end = int(np.floor(audio_beats))
    out = [s for s in out if s["startBeat"] < end]
    if out:
        out[-1]["beats"] = min(out[-1]["beats"], end - out[-1]["startBeat"])
    # Build: the lower-energy run-up into a drop, with the kick mostly out (track A bars 40-47
    # has 3.5 kicks/bar at near-drop energy and is not a build).
    for i, s in enumerate(out):
        nxt = out[i + 1] if i + 1 < len(out) else None
        if (s["label"] != "drop" and nxt and nxt["label"] == "drop" and s["energy"] < nxt["energy"]
                and s["kickFill"] < 0.75):
            s["label"] = "build"
    # Intro: everything before the kick first comes in, except the build into the first
    # drop (track A bars 8-23 and track C bars 10-17 came out "breakdown" before any drop).
    first_kick = next((i for i, s in enumerate(out) if s["kickFill"] >= 0.75), len(out))
    for s in out[:first_kick]:
        if s["label"] != "build":
            s["label"] = "intro"
    if len(out) > 1 and out[-1]["label"] != "drop":
        out[-1]["label"] = "outro"
    # Energy-split fragments are kept even when they repeat the neighbour's label: merging
    # them back threw away real changes (track C bar 26 drums +17 dB, track B bar 92).
    for s in out:
        split, lyrics_bar = s.pop("split"), s.pop("lyricsBar")
        s["labelSource"] = (f"rule(kickFill={s.pop('kickFill')}, energy={s['energy']}); allin1={s.pop('allin1')}"
                            + ("; energy split" if split else "")
                            + (f"; lyrics entry bar {lyrics_bar}" if lyrics_bar is not None else ""))
    return out


def assemble(slug):
    d = OUT / slug
    g = load_json(d / "grid.json")
    if not g:
        print(f"{slug}: no grid.json, skipping")
        return
    import soundfile as sf
    duration = sf.info(str(AUDIO / f"{slug}.wav")).duration
    grid = Grid(g)
    drums, drum_stats, grid = build_drums(grid, d)  # shifts t0 onto the kicks, sets swing
    audio_beats = grid.beat(duration).item()
    notes = build_notes(grid, d)
    curves, per_beat = build_curves(grid, d, int(np.ceil(audio_beats)))
    lyrics, vocal_onsets = build_lyrics(grid, d)
    entries = lyric_entries(lyrics["words"]) if lyrics else None
    tools = {
        "beats": "beat_this final0 + all-in-one-infer 3.1.0, rigid grid fit",
        "stems": "BS-Roformer-SW + MDX23C DrumSep (audio-separator 0.47.0)",
        "drums": "adtof-pytorch 85c192e",
        "notes": "basic-pitch 0.4.0 per stem",
    }
    if lyrics:
        tools["lyrics"] = lyrics["source"]
    events = {
        # 2: drums.tom -> drums.toms, curves.stems.hh -> curves.stems.hats (the app's lanes).
        "version": 2,
        "source": {"audio": f"audio/{slug}.wav", "tools": tools},
        "tempo": [{"beat": 0, "bpm": g["bpm"]}],
        "gridMarker": round(grid.t0, 4),
        "meter": grid.meter,
        "lengthBeats": round(audio_beats, 2),
        "gridFit": {**(g.get("fit") or {}), "shiftToKicksMs": drum_stats.get("gridShiftToKicksMs"),
                    "onBeatKicks": drum_stats.get("onBeatKicks"), "hatSwing": drum_stats.get("hatSwing")},
        "sections": relabel_sections(g.get("sections", []), drums, per_beat, audio_beats, entries),
        "drums": drums,
        "notes": notes,
        # Additive keys (version stays 2): present only when the lyrics stage found something.
        **({"lyrics": lyrics} if lyrics else {}),
        **({"vocalOnsets": vocal_onsets} if vocal_onsets else {}),
        "curves": curves,
    }
    (d / "events.json").write_text(json.dumps(events, separators=(",", ":")))
    size = (d / "events.json").stat().st_size / 1024
    print(f"{slug}: bpm {g['bpm']}, t0 {g['t0']:.4f}s -> {grid.t0:.4f}s "
          f"(shift {drum_stats.get('gridShiftToKicksMs')} ms from {drum_stats.get('onBeatKicks')} on-beat kicks), "
          f"swing ms {{{', '.join(f'{k}: {round(v['swing'] * grid.period * 1000, 1)}' for k, v in (drums or {}).items() if v['swing'])}}}, "
          f"{len(events['sections'])} sections,\n    drums {drum_stats.get('hits')},"
          f"\n    DrumSep-only {drum_stats.get('soloDrumSep')}, offGrid {drum_stats.get('offGrid')}, "
          f"notes {{{', '.join(f'{k}: {len(v['b'])}' for k, v in notes.items())}}}, "
          f"curves {'yes' if curves else 'no'}, {size:.0f} KB")
    if lyrics or vocal_onsets:
        moved = sum("lyrics entry" in s["labelSource"] for s in events["sections"])
        print(f"    lyrics: {len(lyrics['lines']['b']) if lyrics else 0} lines, "
              f"{len(lyrics['words']['b']) if lyrics else 0} words, "
              f"{len(lyrics['extras']['b']) if lyrics else 0} extras, "
              f"{len(vocal_onsets['b']) if vocal_onsets else 0} vocal onsets, "
              f"{moved} sections moved by lyrics")
    for s in events["sections"]:
        print(f"    bar {s['startBeat'] // 4:>4} +{s['beats'] // 4:<3} {s['label']:<10} {s['labelSource']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    for s in (all_slugs() if args.all else [args.slug]):
        assemble(s)
