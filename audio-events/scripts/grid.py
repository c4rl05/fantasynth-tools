"""GRID stage: beat trackers -> one rigid 4/4 grid per track.

For each track:
  1. Beat This! (final0, cuda, minimal/non-DBN postprocessing) -> out/<slug>/raw_beatthis.json
  2. all-in-one-infer (harmonix-all, cuda)                      -> out/<slug>/raw_allin1.json
  3. rigid grid fit t = t0 + i*P on Beat This! beats (iterative 35 ms outlier rejection,
     seeded from a beat-index chain and from a phase-coherence scan, see fit_free),
     integer-bpm snap when it costs <= 1 ms median residual, allin1 fitted too for agreement
  4. downbeat phase by vote of both trackers' downbeats (index mod 4); t0 moved onto a downbeat
  5. allin1 segments snapped to bar lines                       -> grid.json "sections"
  6. out/<slug>/grid.json (contract fields only) + out/<slug>/grid_diag.json (measurements)

Usage:  python grid.py --slug track-a | --all   [--reuse-raw] [--int-rule median|median+comb]
  --reuse-raw  skip the models and refit from existing raw_*.json (fit iteration only)
  --int-rule   integer-bpm acceptance; "median" = contract's literal rule, default adds the
               audio-comb test (see INT_RULE)

Definitions used in grid.json (the contract leaves these open):
  - downbeatPhase: beat-in-bar (0..3) of the first detected beat (the earliest
    grid-aligned beat of either tracker). 0 = the music starts on a downbeat.
  - t0 = firstBeat - downbeatPhase * beatPeriod: the downbeat at or before the first beat
    (negative if the music starts mid-bar close to t=0).
  - downbeatVotes[tracker][j]: that tracker's downbeats landing on beat-in-bar j of the
    FINAL grid (j=0 is the chosen downbeat), so votes[0]/sum is the tracker's agreement.
  - nBeats: grid beats i >= 0 with t0 + i*beatPeriod < WAV duration.
  - fit.inliers/outliers: Beat This! detections within/outside 35 ms of the final grid
    (inliers + outliers = sources.beatthis; a second detection on an occupied grid beat
    counts as an outlier).
"""
import argparse
import json
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # audio-events/, for workspace.py
from workspace import AUDIO, OUT, all_slugs, track_conf  # noqa: E402

THR = 0.035           # outlier threshold, seconds
METER = 4
INT_SNAP_TOL_MS = 1.0
# "median": the contract's literal rule (integer iff median residual worse by <= 1 ms).
# "median+comb" (default): ALSO accept the integer when the audio comb (below) says the
# integer grid is at least as sharp as the 0.01-snapped one. Can only add integer
# acceptances, never remove one. See grid_diag.json intSnap for both verdicts.
INT_RULE = "median+comb"


def ground_truth(slug):
    """A hand-measured grid for the slug, {"bpm", "gridMarker"} from its entry in
    <workspace>/config.json, or None. Used ONLY for the diagnostic comparison in
    grid_diag.json, never for fitting."""
    t = track_conf(slug)
    if t.get("bpm") is None or t.get("gridMarker") is None:
        return None
    return {"bpm": float(t["bpm"]), "gridMarker": float(t["gridMarker"])}


# ----------------------------------------------------------------------------- models

_BT_MODEL = None


def run_beatthis(wav: Path):
    """Returns (beats, downbeats, info)."""
    global _BT_MODEL
    import torch
    from beat_this.inference import File2Beats

    info = {}
    t = time.perf_counter()
    if _BT_MODEL is None:
        _BT_MODEL = File2Beats(checkpoint_path="final0", device="cuda", dbn=False)
        info["loadSec"] = round(time.perf_counter() - t, 2)
    else:
        info["loadSec"] = 0.0
    info["modelDevice"] = str(next(_BT_MODEL.model.parameters()).device)
    info["postprocessing"] = _BT_MODEL.frames2beats.type
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    beats, downbeats = _BT_MODEL(str(wav))
    torch.cuda.synchronize()
    info["inferSec"] = round(time.perf_counter() - t, 2)
    info["cudaPeakMB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    return np.asarray(beats, float), np.asarray(downbeats, float), info


def run_allin1(wav: Path, tmp: Path):
    """Returns (result_dict, info). Demucs stems stay in memory; spectrogram temp files
    go under out/<slug>/_allin1_tmp, which is removed afterwards."""
    import torch
    import allin1_infer

    info = {}
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    res = allin1_infer.analyze(
        str(wav),
        out_dir=None,
        model="harmonix-all",
        device="cuda",
        demix_dir=str(tmp / "demix"),
        spec_dir=str(tmp / "spec"),
        keep_byproducts=False,
        multiprocess=False,
    )
    torch.cuda.synchronize()
    info["totalSec"] = round(time.perf_counter() - t, 2)
    info["cudaPeakMB"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    shutil.rmtree(tmp, ignore_errors=True)
    info["tmpRemoved"] = not tmp.exists()
    out = {
        "bpm": res.bpm,
        "beats": [round(float(x), 5) for x in res.beats],
        "downbeats": [round(float(x), 5) for x in res.downbeats],
        "beat_positions": [int(x) for x in res.beat_positions],
        "segments": [
            {"start": round(float(s.start), 5), "end": round(float(s.end), 5), "label": s.label}
            for s in res.segments
        ],
    }
    return out, info


# ----------------------------------------------------------------------------- fitting

def _lstsq(i, t):
    A = np.stack([np.ones_like(i, dtype=float), i.astype(float)], axis=1)
    (t0, P), *_ = np.linalg.lstsq(A, t, rcond=None)
    return float(t0), float(P)


def _initial_period(b):
    ibi = np.diff(b)
    med = float(np.median(ibi))
    sel = ibi[np.abs(ibi / med - 1.0) < 0.08]
    return float(sel.mean()), med


def _sequential_index(b, P0):
    """Integer beat index per detection, chained from the middle of the longest run of
    regular IBIs so one spurious early detection can't poison the chain. -1 = unindexed."""
    n = len(b)
    ibi = np.diff(b)
    ok = np.abs(ibi / P0 - 1.0) < 0.08
    best_len, best_start, cur_len, cur_start = 0, 0, 0, 0
    for k, v in enumerate(ok):
        if v:
            if cur_len == 0:
                cur_start = k
            cur_len += 1
            if cur_len > best_len:
                best_len, best_start = cur_len, cur_start
        else:
            cur_len = 0
    anchor = best_start + best_len // 2
    idx = np.zeros(n, dtype=np.int64)
    ok_idx = np.zeros(n, dtype=bool)
    idx[anchor] = 0
    ok_idx[anchor] = True
    for direction in (1, -1):
        last_t, last_i = b[anchor], 0
        k = anchor + direction
        while 0 <= k < n:
            r = (b[k] - last_t) / P0 * direction
            m = int(round(r))
            if m >= 1 and abs(r - m) < 0.2:
                idx[k] = last_i + direction * m
                ok_idx[k] = True
                last_t, last_i = b[k], idx[k]
            k += direction
    return idx, ok_idx


def _assign(b, t0, P, thr):
    """Global indexing against a grid. Returns (idx, resid, inlier_mask) with one inlier
    per grid index at most (the closest)."""
    idx = np.round((b - t0) / P).astype(np.int64)
    resid = b - (t0 + idx * P)
    inl = np.abs(resid) <= thr
    # dedupe: several detections on one grid beat -> keep the closest
    order = np.argsort(np.abs(resid))
    seen = set()
    for k in order:
        if not inl[k]:
            continue
        if idx[k] in seen:
            inl[k] = False
        else:
            seen.add(idx[k])
    return idx, resid, inl


def _refine(b, t0, P, thr, max_iter):
    """Iterative global re-indexing + least squares from a (t0, P) seed."""
    inl_prev = None
    for it in range(max_iter):
        idx, resid, inl = _assign(b, t0, P, thr)
        if inl_prev is not None and np.array_equal(inl, inl_prev):
            break
        t0, P = _lstsq(idx[inl], b[inl])
        inl_prev = inl
    idx, resid, inl = _assign(b, t0, P, thr)
    return t0, P, idx, resid, inl, it + 1


COHERENCE_RANGE = 0.03  # coherence seed scans P0 * (1 +- this)
COHERENCE_STEP_S = 1e-5  # period step: ~4 ms of drift over a 400-beat track


def _coherence_seed(b, P0):
    """(t0, P) maximising the phase coherence |mean exp(2 pi i b / P)| of ALL detections.
    Detections off the true grid (a drumless intro or breakdown, where Beat This! wanders
    to half/double time) are incoherent and average out, so they cannot drag the period
    the way they drag a sequential index chain."""
    Ps = np.arange(P0 * (1 - COHERENCE_RANGE), P0 * (1 + COHERENCE_RANGE), COHERENCE_STEP_S)
    z = np.exp(2j * np.pi * b[None, :] / Ps[:, None]).mean(axis=1)
    k = int(np.argmax(np.abs(z)))
    P = float(Ps[k])
    return float(np.angle(z[k]) / (2 * np.pi) * P), P


def fit_free(b, thr=THR, max_iter=50):
    """Free (t0, P) least-squares fit with iterative outlier rejection, seeded twice: from a
    sequential index chain, and from a phase-coherence period scan. The coherence fit is
    taken only when it ends with strictly more inliers. One test track needed it: its
    frame-quantised IBIs seeded 120.1 bpm, the chain miscounted through the drumless intro
    and breakdown, and the fit settled at 119.65 bpm with 40 of 439 inliers; the true grid
    is 121 bpm. A track whose chain fit already wins keeps that fit exactly."""
    P0, med = _initial_period(b)
    seq, m = _sequential_index(b, P0)
    t0, P = _lstsq(seq[m], b[m])
    # express t0 relative to index 0 = anchor; re-index globally from here on
    t0, P, idx, resid, inl, iters = _refine(b, t0, P, thr, max_iter)
    seed = "sequential"
    alt = _refine(b, *_coherence_seed(b, P0), thr, max_iter)
    if alt[4].sum() > inl.sum():
        t0, P, idx, resid, inl, iters = alt
        seed = "coherence"
    return {"t0": t0, "P": P, "idx": idx, "resid": resid, "inl": inl, "P0": P0,
            "ibiMedian": med, "iterations": iters, "seqIndexed": int(m.sum()), "seed": seed}


def fit_fixed_period(b, P, t0_init, thr=THR, max_iter=50):
    """Refit t0 only for a fixed period, with iterative outlier rejection."""
    t0 = t0_init
    inl_prev = None
    for _ in range(max_iter):
        idx, resid, inl = _assign(b, t0, P, thr)
        if inl_prev is not None and np.array_equal(inl, inl_prev):
            break
        t0 = float(np.mean(b[inl] - idx[inl] * P))
        inl_prev = inl
    idx, resid, inl = _assign(b, t0, P, thr)
    return {"t0": t0, "P": P, "idx": idx, "resid": resid, "inl": inl}


# ----------------------------------------------------------------------------- audio comb
# Independent of both trackers: fold a high-band (3-10 kHz) energy envelope of the mix over
# a candidate grid. A correct fixed tempo stacks every beat's transient at one phase, so the
# folded peak is sharp; a tempo error of 0.01 bpm smears it by ~20-35 ms over a track.

COMB_STEP_S = 0.0005  # phase resolution of the fold
COMB_PH = np.arange(-0.2, 0.2, COMB_STEP_S)


def hf_envelope(wav):
    import soundfile as sf
    import scipy.signal as ss
    x, sr = sf.read(str(wav), dtype="float32")
    x = x.mean(axis=1) if x.ndim == 2 else x
    y = ss.sosfiltfilt(ss.butter(4, [3000, 10000], "band", fs=sr, output="sos"), x)  # zero phase
    e = np.convolve(y.astype(np.float64) ** 2, np.ones(int(0.001 * sr)) / int(0.001 * sr), mode="same")
    step = int(round(sr * COMB_STEP_S))
    return e[::step], sr / step  # (envelope, its true sample rate)


def comb_peak(env, t0, P, i_lo, i_hi):
    """(peak height, peak phase ms) of the beat-folded envelope, each beat mean-normalised."""
    env, fs = env
    i = np.arange(i_lo, i_hi)
    idx = np.round((t0 + i[:, None] * P + COMB_PH[None, :]) * fs).astype(np.int64)
    ok = (idx[:, 0] >= 0) & (idx[:, -1] < len(env))
    seg = env[idx[ok]]
    m = seg.mean(axis=1, keepdims=True)
    seg = seg[m[:, 0] > 0] / m[m[:, 0] > 0]
    acc = seg.mean(axis=0)
    k = int(np.argmax(acc))
    return float(acc[k]), float(COMB_PH[k] * 1000)


def ms_stats(resid, inl):
    a = np.abs(resid[inl]) * 1000
    if len(a) == 0:
        return {"median": None, "p95": None, "mean": None, "signedMean": None}
    return {"median": round(float(np.median(a)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "mean": round(float(a.mean()), 2),
            "signedMean": round(float(np.mean(resid[inl]) * 1000), 2)}


def wrap(x, period):
    return (x + period / 2) % period - period / 2


def clusters(times, gap):
    """Group sorted times into runs separated by more than `gap` seconds."""
    times = sorted(times)
    out = []
    for t in times:
        if out and t - out[-1][1] <= gap:
            out[-1][1] = t
            out[-1][2] += 1
        else:
            out.append([t, t, 1])
    return out


def label_at(t, segments):
    for s in segments:
        if s["start"] <= t < s["end"]:
            return s["label"]
    return None


# ----------------------------------------------------------------------------- per track

def wav_info(path):
    """(duration s, start of the first 5 ms window above -50 dBFS RMS) of a 16-bit WAV."""
    with wave.open(str(path), "rb") as w:
        sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
        head = np.frombuffer(w.readframes(min(n, sr * 30)), dtype="<i2").reshape(-1, ch)
    x = head.astype(np.float64).mean(axis=1) / 32768.0
    hop = int(0.005 * sr)
    rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode="valid"))
    loud = np.nonzero(rms > 10 ** (-50 / 20))[0]
    return n / sr, (float(loud[0]) / sr if len(loud) else None)


def process(slug, reuse_raw=False):
    wav = AUDIO / f"{slug}.wav"
    od = OUT / slug
    od.mkdir(parents=True, exist_ok=True)
    diag = {"slug": slug, "runtime": {}}
    T0 = time.perf_counter()

    # 1-2. models ------------------------------------------------------------------------
    if reuse_raw:
        bt_raw = json.loads((od / "raw_beatthis.json").read_text())
        a1 = json.loads((od / "raw_allin1.json").read_text())
        prev = od / "grid_diag.json"
        if prev.exists():
            diag["runtime"] = json.loads(prev.read_text()).get("runtime", {})
            diag["runtime"]["reusedRaw"] = True
    else:
        import torch
        diag["runtime"]["cuda"] = torch.cuda.is_available()
        diag["runtime"]["gpu"] = torch.cuda.get_device_name(0)
        diag["runtime"]["torch"] = torch.__version__
        bt_b, bt_d, bt_info = run_beatthis(wav)
        bt_raw = {"beats": [round(float(x), 5) for x in bt_b],
                  "downbeats": [round(float(x), 5) for x in bt_d]}
        (od / "raw_beatthis.json").write_text(json.dumps(bt_raw))
        diag["runtime"]["beatthis"] = bt_info
        a1, a1_info = run_allin1(wav, od / "_allin1_tmp")
        (od / "raw_allin1.json").write_text(json.dumps(a1))
        diag["runtime"]["allin1"] = a1_info

    t_fit = time.perf_counter()
    dur, first_audible = wav_info(wav)
    b = np.asarray(bt_raw["beats"], float)
    bd = np.asarray(bt_raw["downbeats"], float)
    ab = np.asarray(a1["beats"], float)
    ad = np.asarray(a1["downbeats"], float)
    segs = a1["segments"]

    # 3. grid fit --------------------------------------------------------------------------
    raw = fit_free(b)
    P_raw, t0_raw = raw["P"], raw["t0"]
    bpm_raw = 60.0 / P_raw
    raw_stats = ms_stats(raw["resid"], raw["inl"])

    # half/double tempo guard: compare with allin1's own free fit
    a1fit = fit_free(ab)
    ratio = a1fit["P"] / P_raw
    tempo_flag = None
    if abs(ratio - 2) < 0.1 or abs(ratio - 0.5) < 0.05:
        tempo_flag = f"beatthis/allin1 period ratio {ratio:.3f} (half/double tempo)"
    if not 70 <= bpm_raw <= 200:
        tempo_flag = (tempo_flag or "") + f" beatthis bpmRaw {bpm_raw:.2f} out of 70-200"

    # snap: integer if the median residual is not worse by > 1 ms. Compared on the SAME
    # detections (the free fit's inliers, same indices), refitting t0 only.
    def t0_for(Pfix):
        m_ = raw["inl"]
        t0_ = float(np.mean(b[m_] - raw["idx"][m_] * Pfix))
        return t0_, b - (t0_ + raw["idx"] * Pfix)

    bpm_int = float(round(bpm_raw))
    t0_int, res_int = t0_for(60.0 / bpm_int)
    int_stats = ms_stats(res_int, raw["inl"])
    median_rule = int_stats["median"] <= raw_stats["median"] + INT_SNAP_TOL_MS
    # audio comb: is the integer grid at least as sharp as the 0.01-snapped raw grid?
    env = hf_envelope(wav)
    span = (int(raw["idx"][raw["inl"]].min()), int(raw["idx"][raw["inl"]].max()) + 1)
    bpm_001 = round(bpm_raw, 2)
    comb_int = comb_peak(env, t0_int, 60.0 / bpm_int, *span)
    comb_001 = comb_peak(env, t0_for(60.0 / bpm_001)[0], 60.0 / bpm_001, *span)
    comb_rule = comb_int[0] >= comb_001[0]
    scan = []
    for cand in np.round(np.arange(bpm_raw - 0.05, bpm_raw + 0.0505, 0.001), 3):
        scan.append((comb_peak(env, t0_for(60.0 / cand)[0], 60.0 / cand, *span)[0], float(cand)))
    bpm_comb = max(scan)[1]
    use_int = median_rule or (INT_RULE == "median+comb" and comb_rule)
    bpm = bpm_int if use_int else bpm_001
    P = 60.0 / bpm
    t0_start, _ = t0_for(P)
    fin = fit_fixed_period(b, P, t0_start)  # t0-only refit with re-selection of inliers
    fin_stats = ms_stats(fin["resid"], fin["inl"])
    t0_fit = fin["t0"]  # time of fit-index 0 (arbitrary phase for now)
    comb_final = comb_peak(env, t0_fit, P, *span)  # phase = HF transients vs the final grid

    # sensitivity of bpmRaw to the outlier threshold
    sens = {}
    for thr in (0.020, 0.035, 0.050):
        f = fit_free(b, thr=thr)
        sens[f"{int(thr*1000)}ms"] = {"bpmRaw": round(60 / f["P"], 4), "inliers": int(f["inl"].sum())}

    # tempo stability: free fits on each half of the inliers
    halves = {}
    inl_t = b[raw["inl"]]
    mid = np.median(inl_t)
    for name, mask in (("firstHalf", raw["inl"] & (b < mid)), ("secondHalf", raw["inl"] & (b >= mid))):
        t0h, Ph = _lstsq(raw["idx"][mask], b[mask])
        halves[name] = round(60 / Ph, 4)

    # allin1 against the final Beat This! grid
    a_idx, a_res, a_inl = _assign(ab, t0_fit, P, THR)
    a1_stats_on_bt = ms_stats(a_res, a_inl)
    mid_i = (fin["idx"][fin["inl"]].min() + fin["idx"][fin["inl"]].max()) // 2
    mid_bt = t0_fit + mid_i * P
    # allin1's own grid index nearest the track middle
    k_a = round((mid_bt - a1fit["t0"]) / a1fit["P"])
    mid_a = a1fit["t0"] + k_a * a1fit["P"]
    agreement = {
        "allin1BpmRaw": round(60 / a1fit["P"], 4),
        "allin1ReportedBpm": a1["bpm"],
        "bpmDiff": round(bpm_raw - 60 / a1fit["P"], 4),
        "phaseOffsetMsMidTrack": round(float(wrap(mid_a - mid_bt, P)) * 1000, 2),
        "allin1OwnFitResidualMs": ms_stats(a1fit["resid"], a1fit["inl"]),
        "allin1OwnFitInliers": int(a1fit["inl"].sum()),
        "allin1OnBeatthisGrid": {"inliers": int(a_inl.sum()), "outliers": int((~a_inl).sum()),
                                 "residualMs": a1_stats_on_bt},
        "periodRatio": round(ratio, 5),
        "tempoFlag": tempo_flag,
    }

    # 4. downbeat phase -----------------------------------------------------------------------
    def votes_fit(times):
        k = np.round((times - t0_fit) / P).astype(np.int64)
        r = times - (t0_fit + k * P)
        v = np.bincount(np.mod(k, METER), minlength=METER)
        return v, k, r

    v_bt, k_bt, r_bt = votes_fit(bd)
    v_a1, k_a1, r_a1 = votes_fit(ad)
    phi = int(np.argmax(v_bt + v_a1))

    # first musical beat: earliest grid-aligned detection of either tracker
    first_candidates = list(fin["idx"][fin["inl"]]) + list(a_idx[a_inl])
    f = int(min(first_candidates))
    m = f - ((f - phi) % METER)
    downbeat_phase = int((f - phi) % METER)  # beat-in-bar of the first beat
    t0 = t0_fit + m * P

    def reorder(v):
        return [int(v[(phi + j) % METER]) for j in range(METER)]

    votes = {"beatthis": reorder(v_bt), "allin1": reorder(v_a1)}

    # final indices relative to t0
    idx_final = fin["idx"] - m
    inl = fin["inl"]
    outl_times = b[~inl]

    # minority downbeats (where do the trackers disagree with the chosen phase?)
    def minority(times, k):
        bib = np.mod(k - m, METER)
        return [(round(float(t), 2), int(x)) for t, x in zip(times, bib) if x != 0]

    db_detail = {
        "beatthisOffGrid": int(np.sum(np.abs(r_bt) > THR)),
        "allin1OffGrid": int(np.sum(np.abs(r_a1) > THR)),
        "beatthisMinority": minority(bd, k_bt),
        "allin1Minority": minority(ad, k_a1),
    }

    # outliers / gaps: where are they?
    bar_len = METER * P

    def describe_clusters(times):
        out = []
        for s, e, n in clusters(times, gap=2 * bar_len):
            out.append({"start": round(s, 2), "end": round(e, 2), "count": n,
                        "bars": [int(np.floor((s - t0) / bar_len)), int(np.floor((e - t0) / bar_len))],
                        "labels": sorted({label_at(s, segs), label_at(e, segs)} - {None})})
        return out

    occupied = set(idx_final[inl].tolist())
    first_i, last_i = min(occupied), max(occupied)
    missing = [t0 + i * P for i in range(first_i, last_i + 1) if i not in occupied]

    n_beats = int(np.floor((dur - t0) / P)) + 1 if dur > t0 else 0

    # 5. sections -----------------------------------------------------------------------------
    def snap_bar(t):
        bi = int(round((t - t0) / bar_len))
        return bi, abs(t - (t0 + bi * bar_len)) * 1000

    sections, dropped = [], []
    for s in segs:
        sb, ss = snap_bar(s["start"])
        eb, es = snap_bar(s["end"])
        rec = {"startBar": sb, "endBar": eb, "label": s["label"],
               "startRaw": round(s["start"], 3), "endRaw": round(s["end"], 3),
               "snapMs": round(max(ss, es), 1)}
        if eb <= sb:
            dropped.append(rec)
        else:
            sections.append(rec)
    lengths = [s["endBar"] - s["startBar"] for s in sections]

    # 6. write ----------------------------------------------------------------------------------
    grid = {
        "bpm": bpm,
        "bpmRaw": round(bpm_raw, 4),
        "t0": round(t0, 6),
        "beatPeriod": round(P, 9),
        "meter": METER,
        "downbeatPhase": downbeat_phase,
        "nBeats": n_beats,
        "fit": {"residualMsMedian": fin_stats["median"], "residualMsP95": fin_stats["p95"],
                "inliers": int(inl.sum()), "outliers": int((~inl).sum())},
        "downbeatVotes": votes,
        "sources": {"beatthis": int(len(b)), "allin1": int(len(ab))},
        "sections": sections,
    }
    (od / "grid.json").write_text(json.dumps(grid, indent=2))

    # ground truth comparison (diagnostic only)
    gt = None
    g = ground_truth(slug)
    if g:
        gm = g["gridMarker"]
        beat_off = float(wrap(gm - t0, P))
        pos = (gm - t0) / P
        k = int(round(pos))
        # detections against the GT grid, for comparison with ours
        gP = 60.0 / g["bpm"]
        _, g_res, g_inl = _assign(b, gm, gP, THR)
        gt = {
            "bpm": g["bpm"], "gridMarker": gm,
            "beatOffsetMs": round(beat_off * 1000, 2),
            "markerBeatInBar": int(k % METER),
            "markerOnDownbeat": bool(k % METER == 0),
            "barOffsetMs": round(float(wrap(gm - t0, bar_len)) * 1000, 2),
            "beatthisOnGtGrid": {"inliers": int(g_inl.sum()), "residualMs": ms_stats(g_res, g_inl)},
        }

    diag.update({
        "durationSec": round(dur, 3),
        "firstAudibleSec": None if first_audible is None else round(first_audible, 4),
        "fitRaw": {"bpmRaw": round(bpm_raw, 5), "t0": round(t0_raw, 6), "residualMs": raw_stats,
                   "inliers": int(raw["inl"].sum()), "iterations": raw["iterations"],
                   "initialPeriod": round(raw["P0"], 6), "ibiMedian": round(raw["ibiMedian"], 5),
                   "seqIndexed": raw["seqIndexed"], "seed": raw["seed"]},
        "intSnap": {"bpmInt": bpm_int, "residualMsSameBeats": int_stats,
                    "medianRule": bool(median_rule), "combPeakInt": round(comb_int[0], 3),
                    "combPeak001": round(comb_001[0], 3), "combRule": bool(comb_rule),
                    "rule": INT_RULE, "chosen": bool(use_int)},
        "comb": {"bpmBestScan": bpm_comb, "scanRange": [round(bpm_raw - 0.05, 3), round(bpm_raw + 0.05, 3)],
                 "finalGridPeak": round(comb_final[0], 3),
                 "hfTransientOffsetMs": round(comb_final[1], 1)},
        "fitFinal": {"bpm": bpm, "residualMs": fin_stats, "inliers": int(inl.sum()),
                     "outliers": int((~inl).sum()),
                     "outlierFraction": round(float((~inl).mean()), 4)},
        "thresholdSensitivity": sens,
        "tempoByHalf": halves,
        "agreement": agreement,
        "downbeat": {"phiFitIndex": phi, "firstBeatFitIndex": f, "downbeatPhase": downbeat_phase,
                     "votesFinal": votes,
                     "decisiveness": {k2: round(v[0] / max(1, sum(v)), 3) for k2, v in votes.items()},
                     **db_detail},
        "outlierClusters": describe_clusters(outl_times.tolist()),
        "missingGridBeats": {"count": len(missing), "span": [first_i, last_i],
                             "clusters": describe_clusters(missing)},
        "sections": {"count": len(sections), "dropped": dropped, "barLengths": lengths,
                     "mult4": sum(1 for L in lengths if L % 4 == 0),
                     "mult8": sum(1 for L in lengths if L % 8 == 0),
                     "mult16": sum(1 for L in lengths if L % 16 == 0),
                     "maxSnapMs": max((s["snapMs"] for s in sections), default=None)},
        "groundTruth": gt,
    })
    diag["runtime"]["fitSec"] = round(time.perf_counter() - t_fit, 3)
    if not reuse_raw:
        diag["runtime"]["trackTotalSec"] = round(time.perf_counter() - T0, 2)
    (od / "grid_diag.json").write_text(json.dumps(diag, indent=2))
    summarize(slug, grid, diag)
    return grid, diag


def summarize(slug, grid, diag):
    fr, ff, ag, db = diag["fitRaw"], diag["fitFinal"], diag["agreement"], diag["downbeat"]
    print(f"\n=== {slug}  ({diag['durationSec']} s)")
    print(f"  bpmRaw {grid['bpmRaw']}  -> bpm {grid['bpm']}  (int snap: raw med {fr['residualMs']['median']} ms,"
          f" int med {diag['intSnap']['residualMsSameBeats']['median']} ms)  intSnap {diag['intSnap']}")
    print(f"  comb {diag['comb']}  firstAudible {diag['firstAudibleSec']}")
    print(f"  t0 {grid['t0']}  P {grid['beatPeriod']}  downbeatPhase {grid['downbeatPhase']}  nBeats {grid['nBeats']}")
    print(f"  residual med {ff['residualMs']['median']} ms  p95 {ff['residualMs']['p95']} ms"
          f"  inliers {ff['inliers']}  outliers {ff['outliers']} ({ff['outlierFraction']*100:.1f}%)")
    print(f"  tempo by half {diag['tempoByHalf']}  thr sensitivity {diag['thresholdSensitivity']}")
    print(f"  allin1: bpmRaw {ag['allin1BpmRaw']} (reported {ag['allin1ReportedBpm']})  bpmDiff {ag['bpmDiff']}"
          f"  phase {ag['phaseOffsetMsMidTrack']} ms  on-BT-grid {ag['allin1OnBeatthisGrid']}")
    print(f"  votes {db['votesFinal']}  decisiveness {db['decisiveness']}")
    print(f"  outlier clusters {diag['outlierClusters']}")
    print(f"  missing grid beats {diag['missingGridBeats']['count']}: {diag['missingGridBeats']['clusters']}")
    for s in grid["sections"]:
        print(f"    bars {s['startBar']:4d}-{s['endBar']:4d} ({s['endBar']-s['startBar']:3d})  {s['label']:8s}"
              f"  raw {s['startRaw']:.2f}-{s['endRaw']:.2f}  snap {s['snapMs']} ms")
    if diag["sections"]["dropped"]:
        print(f"  dropped zero-length sections: {diag['sections']['dropped']}")
    if diag["groundTruth"]:
        print(f"  ground truth: {diag['groundTruth']}")
    print(f"  runtime: {diag['runtime']}")


def main():
    global INT_RULE
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--reuse-raw", action="store_true")
    ap.add_argument("--int-rule", choices=["median", "median+comb"], default=INT_RULE)
    args = ap.parse_args()
    slugs = all_slugs() if args.all else [args.slug]
    INT_RULE = args.int_rule
    for s in slugs:
        process(s, reuse_raw=args.reuse_raw)


if __name__ == "__main__":
    sys.exit(main())
