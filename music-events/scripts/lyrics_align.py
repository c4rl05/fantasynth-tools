"""Forced alignment of KNOWN lyric text to a vocal stem, plus vocal-signal features.

Pure numpy/scipy/numba/librosa: no torch at import, so the maths is unit-testable
(tests/test_lyrics_align.py). The stage script `lyrics.py` computes the CTC
emissions with torchaudio and calls into this module.

Method (ported from https://github.com/mexicat/pdoom-video, `analysis/ctcalign.py` +
`analysis/align.py`, with the vocal-signal features after `analysis/vocal_feats.py`;
MIT, Copyright (c) 2026 Giacomo Magnanini; see THIRD_PARTY_NOTICES.md):

* Emissions: two character-level CTC models (torchaudio MMS_FA and
  WAV2VEC2_ASR_LARGE_LV60K_960H), each mapped onto one alphabet (blank, a-z, ') and
  run on the vocal stem as mono, left and right (double-tracked choruses are often
  panned, so one channel is closer to a single voice). The six are fused as a
  probability MIXTURE, so one model's confident "no" cannot veto another's "yes".
* ONE Viterbi pass over the whole song. A garbage token (STAR) sits between lines and
  at both ends; its per-frame score is the frame's best symbol minus `margin`, so it
  absorbs ad-libs, backing vocals and anything else not in the text, while lyric
  letters win wherever they really match. No line timings are needed as input.
* Signal refinement of each unit start (CTC has known biases): rest-onset, onset-snap
  (rejecting onsets followed by frication, which belong to the previous word), and a
  fricative walk-back; ends follow the voice (15 dB drop for 60 ms), legato words tile.
* Confidence per word from agreement between independent alignments + posterior.
"""
from __future__ import annotations

import re

import numpy as np

FRAME = 0.02  # s per CTC emission frame (hop 320 @ 16 kHz)
ALPHA = ["-"] + list("abcdefghijklmnopqrstuvwxyz'")
AIDX = {c: i for i, c in enumerate(ALPHA)}
STAR = len(ALPHA)  # garbage token id (an extra emission column)

# ----------------------------------------------------------------------------- text

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen " \
        "fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> list[str]:
    """0..9999 spelled out the way it is usually sung ("1998" -> nineteen ninety eight)."""
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        t, o = divmod(n, 10)
        return [_TENS[t]] + ([_ONES[o]] if o else [])
    if n < 1000:
        h, r = divmod(n, 100)
        return [_ONES[h], "hundred"] + (number_words(r) if r else [])
    if 1100 <= n < 10000 and n % 100 and n // 100 % 10:  # years: nineteen ninety eight
        a, b = divmod(n, 100)
        return number_words(a) + (["oh"] + number_words(b) if b < 10 else number_words(b))
    th, r = divmod(n, 1000)
    return number_words(th) + ["thousand"] + (number_words(r) if r else [])


def pron(token: str, table: dict | None = None) -> list[str]:
    """Display token -> pronunciation sub-words (letters and internal apostrophes).

    `table` maps a display token (exact, or case-folded without edge punctuation) to a
    spelling such as "ay gee i". Sub-words become syllable spans, so an acronym spelled
    letter by letter gets one span per letter. Digits are spelled out. Returns [] for a
    token with nothing pronounceable (it is then dropped from the alignment)."""
    table = table or {}
    key = token.strip()
    bare = re.sub(r"^[^\w']+|[^\w']+$", "", key.lower().replace("’", "'"))
    for k in (key, bare):
        if k in table:
            return table[k].split()
    w = key.lower().replace("’", "'")
    parts = []
    for chunk in re.split(r"(\d+)", w):
        if chunk.isdigit():
            parts += number_words(int(chunk)) if len(chunk) <= 4 else [_ONES[int(c)] for c in chunk]
        else:
            chunk = re.sub(r"[^a-z' ]", " ", chunk)
            parts += [p.strip("'") for p in chunk.split() if p.strip("'")]
    return parts


# ------------------------------------------------------------------------ emissions

def to_common(em: np.ndarray, labels: list[str]) -> np.ndarray:
    """Map one model's log-probs [T, V] onto ALPHA, renormalised per frame. The
    word separator '|' (lv60k) is folded into blank; other symbols are dropped."""
    em = em.astype(np.float64)
    out = np.full((em.shape[0], len(ALPHA)), -1e4)
    blank_cols = [labels.index("-")] + ([labels.index("|")] if "|" in labels else [])
    out[:, 0] = np.logaddexp.reduce(em[:, blank_cols], axis=1)
    for i, c in enumerate(labels):
        k = c.lower()
        if k in AIDX and k != "-":
            out[:, AIDX[k]] = em[:, i]
    out -= np.logaddexp.reduce(out, axis=1, keepdims=True)
    return out


def fuse(ems: list[np.ndarray]) -> np.ndarray:
    """Probability mixture (mean of probabilities) of log-prob arrays of equal shape."""
    return np.logaddexp.reduce(np.stack(ems), axis=0) - np.log(len(ems))


def pad_to(em: np.ndarray, n: int) -> np.ndarray:
    if len(em) >= n:
        return em[:n]
    return np.concatenate([em, np.repeat(em[-1:], n - len(em), 0)])


# -------------------------------------------------------------------------- viterbi

def _viterbi_py(E, tgt, lo, hi):
    T, L = E.shape[0], len(tgt)
    S = 2 * L + 1
    NEG = -1e18
    prev = np.full(S, NEG)
    cur = np.full(S, NEG)
    bp = np.zeros((T, S), np.int8)  # 0 stay, 1 from s-1, 2 from s-2
    prev[0] = E[0, 0]
    if lo[0] <= 0 <= hi[0]:
        prev[1] = E[0, tgt[0]]
    for t in range(1, T):
        for s in range(S):
            if s % 2 == 1:
                j = (s - 1) // 2
                if t < lo[j] or t > hi[j]:
                    cur[s] = NEG
                    continue
                e = E[t, tgt[j]]
            else:
                e = E[t, 0]
            best = prev[s]
            arg = 0
            if s >= 1 and prev[s - 1] > best:
                best = prev[s - 1]
                arg = 1
            if s >= 2 and s % 2 == 1:
                j = (s - 1) // 2
                if tgt[j] != tgt[j - 1] and prev[s - 2] > best:
                    best = prev[s - 2]
                    arg = 2
            cur[s] = best + e
            bp[t, s] = arg
        for s in range(S):
            prev[s] = cur[s]
    s = S - 1 if prev[S - 1] >= prev[S - 2] else S - 2
    score = prev[s]
    path = np.zeros(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= bp[t, s]
    return path, score


try:  # numba makes the O(T*S) loop take seconds instead of minutes
    import numba
    _viterbi = numba.njit(cache=True)(_viterbi_py)
except ImportError:  # pragma: no cover
    _viterbi = _viterbi_py


def build_targets(lines_tokens: list[list[str]], pron_fn):
    """Target ids with STAR between lines and at both ends.
    index rows: (line, token, sub, pos_a, pos_b) = the char span of each sub-word."""
    tgt, index = [STAR], []
    for li, toks in enumerate(lines_tokens):
        for ti, tok in enumerate(toks):
            for si, sw in enumerate(pron_fn(tok)):
                a = len(tgt)
                tgt.extend(AIDX[c] for c in sw)
                index.append((li, ti, si, a, len(tgt)))
        tgt.append(STAR)
    return tgt, index


NO_PATH = -1e17  # a Viterbi score below this means the constraints left no valid path


def align(E: np.ndarray, lines_tokens: list[list[str]], pron_fn=pron, margin: float = 1.5,
          line_windows: dict | None = None):
    """Global constrained alignment of every line in one pass.

    line_windows: {line_index: (t_lo_s, t_hi_s)}: every letter of that line must lie inside
    the window. A PRIOR from approximate line times (an LRC, an earlier transcript): lyrics
    repeat, and when a hook is sung more often than the text lists it, the unconstrained
    path may give a listed line to a different repeat. Returns score <= NO_PATH when the
    windows cannot be satisfied (the caller falls back to no windows).

    Returns (spans, score). spans[(li, ti)] = [(start_s, end_s, meanprob), ...], one per
    pronunciation sub-word; tokens with no pronounceable letters are absent."""
    T = E.shape[0]
    Ex = np.concatenate([E, E.max(axis=1, keepdims=True) - margin], axis=1)
    tgt, index = build_targets(lines_tokens, pron_fn)
    tgt = np.asarray(tgt, np.int64)
    if len(tgt) > T:  # more symbols than frames: no path can exist
        raise ValueError(f"text too long for the audio ({len(tgt)} symbols, {T} frames)")
    lo = np.zeros(len(tgt), np.int64)
    hi = np.full(len(tgt), T - 1, np.int64)
    for (li, ti, si, a, b) in index:
        if line_windows and li in line_windows:
            wl, wh = line_windows[li]
            lo[a:b] = np.maximum(lo[a:b], max(0, int(wl / FRAME)))
            hi[a:b] = np.minimum(hi[a:b], min(T - 1, int(wh / FRAME)))
    path, score = _viterbi(Ex, tgt, lo, hi)
    if score <= NO_PATH:
        return {}, float(score)
    tokpos = np.where(path % 2 == 1, (path - 1) // 2, -1)
    P = np.exp(Ex[np.arange(T), np.where(tokpos >= 0, tgt[np.maximum(tokpos, 0)], 0)])
    spans: dict = {}
    for (li, ti, si, a, b) in index:
        fr = np.where((tokpos >= a) & (tokpos < b))[0]
        if len(fr) == 0:  # cannot happen with a valid path; guard anyway
            continue
        spans.setdefault((li, ti), []).append(
            (fr.min() * FRAME, (fr.max() + 1) * FRAME, float(P[fr].mean())))
    return spans, float(score)


def word_table(spans, lines_tokens):
    out = []
    for li, toks in enumerate(lines_tokens):
        for ti, tok in enumerate(toks):
            sp = spans.get((li, ti))
            if not sp:
                continue
            out.append(dict(li=li, ti=ti, w=tok, start=sp[0][0], end=sp[-1][1],
                            conf=float(np.mean([s[2] for s in sp])), subs=[(s[0], s[1]) for s in sp]))
    return out


# ------------------------------------------------------------------ vocal features

FEAT_SR = 22050
FEAT_HOP = 110  # ~5 ms


def vocal_features(y: np.ndarray, sr: int = FEAT_SR) -> dict:
    """5 ms features of a mono vocal stem: RMS dB, band energies, sibilance ratio
    (dB 4-10.5 kHz minus dB 90-1500 Hz), log-mel flux onset strength, pYIN f0."""
    import librosa
    hop = FEAT_HOP
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop, center=True)[0]
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop, center=True)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    sib = S[(freqs > 4000) & (freqs < 10500)].sum(0)
    low = S[(freqs > 90) & (freqs < 1500)].sum(0)
    mel = librosa.feature.melspectrogram(S=S, sr=sr, n_mels=96, fmax=8000)
    # n_fft must match the STFT above: onset_strength pads its output by lag + n_fft // (2 hop)
    # frames assuming its own n_fft (default 2048), which put every onset ~31 ms late
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(mel, ref=np.max), sr=sr,
                                         hop_length=hop, n_fft=1024, lag=2, max_size=3)
    f0, voiced, vprob = librosa.pyin(y, fmin=70, fmax=1000, sr=sr, frame_length=2048,
                                     hop_length=hop, center=True)
    n = min(len(rms), len(f0), S.shape[1], len(onset))
    return dict(hop=hop / sr,
                rms_db=librosa.amplitude_to_db(rms[:n], ref=1.0),
                sib_ratio=librosa.power_to_db(sib[:n] + 1e-10) - librosa.power_to_db(low[:n] + 1e-10),
                onset=onset[:n], f0=f0[:n], voiced=voiced[:n].astype(bool), vprob=vprob[:n])


def activity(f, rel_db=22.0, abs_db=-48.0, win_s=1.5):
    """Vocal activity: RMS above an absolute floor AND within rel_db of the local max
    (a relative gate, so quiet verses and loud choruses are both handled)."""
    from scipy.ndimage import maximum_filter1d, median_filter
    r = median_filter(f["rms_db"], 5)
    loc = maximum_filter1d(r, max(1, int(win_s / f["hop"])))
    return (r > abs_db) & (r > loc - rel_db), r


def runs(mask):
    """(start, end_exclusive) index runs where mask is True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


REST_MAX = 0.5  # s: the furthest rest-onset may move a start back (pdoom's was unbounded)
END_TAIL = 1.0  # s: with no 15 dB drop, a word ends at most this long after its CTC end
FRIC_START = re.compile(r"(s|sh|ch|z|f|th|j|c[eiy]|x|h)")
VOICED_TH = {"the", "there", "there's", "that", "that's", "they", "this", "then", "than", "those", "these", "though"}
FRIC_END = re.compile(r"(s|z|f|x|ce|se|ze|sh|ch)$")


def refine(words, f, pron_fn=pron):
    """Signal refinement of CTC boundaries, per sub-word unit. Sets start, end, subs,
    start_rule and keeps ctc_start/ctc_end. See the module docstring for the rules."""
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    hop = f["hop"]
    act, r = activity(f, rel_db=27.0)
    n = len(r)
    sil = ~act
    sib = uniform_filter1d(f["sib_ratio"], 3)
    on = f["onset"]
    pk_idx, _ = find_peaks(on, height=0.3 * np.percentile(on, 99), distance=max(1, int(0.04 / hop)))
    units = []
    for k, w in enumerate(words):
        w["ctc_start"], w["ctc_end"] = w["start"], w["end"]
        texts = pron_fn(w["w"])
        for j, (a, b) in enumerate(w["subs"]):
            units.append(dict(k=k, j=j, text=texts[j] if j < len(texts) else "", cs=a, ce=b, s=a, rule="ctc"))
    for u_i, u in enumerate(units):
        pu = units[u_i - 1] if u_i else None
        prev_ce = pu["ce"] if pu else 0.0
        prev_cs = pu["cs"] if pu else 0.0
        s = u["cs"]
        # 1. rest-onset: CTC fires late on held vowels after a rest
        i0, i1 = int(prev_ce / hop), int(s / hop)
        done = False
        if i1 - i0 > int(0.05 / hop):
            rs = [(a, b) for a, b in runs(sil[i0:i1]) if (b - a) * hop >= 0.05]
            if rs:
                onset = (i0 + rs[-1][1]) * hop
                # a re-entry more than REST_MAX before the CTC start is some other voice
                # (a backing vocal, an ad-lib, a reverb tail): leave it to onset-snap
                if 0.04 < s - onset <= REST_MAX:
                    u["s"], u["rule"] = onset, "rest-onset"
                    done = True
                elif s - onset <= 0.04:
                    done = True
        # 2. onset snap (ignore onsets followed by frication: previous word's coda)
        if not done and u["text"]:
            vowel_init = u["text"][0] in "aeiou"
            lo = max(prev_ce - 0.06, prev_cs + 0.08, s - (0.25 if vowel_init else 0.12))
            hi = s + 0.04
            best, bsc = None, 0.0
            for p in pk_idx:
                if not (lo <= p * hop <= hi) or np.median(sib[p:p + max(1, int(0.06 / hop))]) >= -5:
                    continue
                dt = s - p * hop
                wgt = 1.0 if dt < 0.06 else max(0.4, 1 - (dt - 0.06) / 0.4)
                if on[p] * wgt > bsc:
                    best, bsc = p, on[p] * wgt
            if best is not None:
                t = best * hop - 0.01
                if abs(t - s) > 0.02:
                    u["s"], u["rule"] = t, "onset-snap"
        # 3. fricative: CTC places the consonant at the END of the frication
        if u["text"] and FRIC_START.match(u["text"]) and u["text"] not in VOICED_TH:
            prev_fric_end = pu is not None and FRIC_END.search(pu["text"] or "") is not None
            lo_t = prev_ce + 0.02 if prev_fric_end else prev_ce - 0.10
            if pu is not None:
                lo_t = max(lo_t, pu["s"] + 0.10)
            lo = int(max(lo_t, s - 0.30, 0) / hop)
            a, b = max(int((s - 0.15) / hop), lo), int((s + 0.06) / hop)
            if b > a:
                pk = a + int(np.argmax(sib[a:b]))
                base = np.percentile(sib[max(0, lo - int(0.4 / hop)):lo + 1], 25) if lo > 0 else -40
                if sib[pk] >= base + 8:
                    thr = 0.5 * (base + sib[pk])
                    j = pk
                    while j - 1 >= lo and sib[j - 1] > thr:
                        j -= 1
                    if j * hop < u["s"] - 0.02:
                        u["s"], u["rule"] = j * hop, "fricative"
    for u_i in range(1, len(units)):  # monotonic unit starts
        units[u_i]["s"] = max(units[u_i]["s"], units[u_i - 1]["s"] + 0.03)
    by_word: dict = {}
    for u in units:
        by_word.setdefault(u["k"], []).append(u)
    for k, w in enumerate(words):
        us = by_word[k]
        w["start"], w["start_rule"] = us[0]["s"], us[0]["rule"]
        w["unit_starts"] = [u["s"] for u in us]
    need = max(1, int(0.06 / hop))
    for k, w in enumerate(words):
        nxt = words[k + 1]["start"] if k + 1 < len(words) else n * hop
        e = max(w["ctc_end"], w["start"] + 0.08)
        j0, j1, jn = int(w["start"] / hop), int(e / hop), int(nxt / hop)
        level = np.percentile(r[j0:max(j1, j0 + 1)], 90)
        low = r < level - 15.0
        q = None
        for j in range(j1, min(jn, n - need)):
            if low[j] and low[j:j + need].all():
                q = j * hop
                break
        # no drop found: the voice goes on (backing vocal, pad, reverb); do not let the word
        # run to the next word, which may be half a minute later
        end = min(q, nxt) if q is not None else min(nxt, e + END_TAIL)
        end = max(end, w["start"] + 0.04)
        if nxt - end < 0.03:  # legato: tile with the next word
            end = nxt
        w["end"] = end
        us = w["unit_starts"]
        w["subs"] = [(us[i], us[i + 1] if i + 1 < len(us) else end) for i in range(len(us))]
    # global order: start >= previous start + 20 ms; the previous end clipped to it
    for k in range(1, len(words)):
        if words[k]["start"] < words[k - 1]["start"] + 0.02:
            words[k]["start"] = words[k - 1]["start"] + 0.02
        if words[k - 1]["end"] > words[k]["start"]:
            words[k - 1]["end"] = words[k]["start"]
    return words


def confidence(words, alts: dict):
    """0..1 per word: 0.4 + 0.4 * (share of independent alignments whose CTC start is
    within 60 ms) + 0.2 * (CTC posterior, saturating at 0.5). Alignments are matched
    by (line, token), since a token can be missing from one of them."""
    idx = {name: {(w["li"], w["ti"]): w for w in ws} for name, ws in alts.items()}
    for w in words:
        key = (w["li"], w["ti"])
        ds = [abs(tab[key]["start"] - w["ctc_start"]) for tab in idx.values() if key in tab]
        agree = float(np.mean([d <= 0.06 for d in ds])) if ds else 0.0
        p = min(1.0, w["conf"] / 0.5)
        w["agree"] = round(agree, 2)
        w["conf_final"] = round(float(np.clip(0.4 + 0.4 * agree + 0.2 * p, 0, 1)), 2)
    return words


def extras(words, f, abs_db=-36.0, pad=0.1, merge=0.4, min_len=0.5):
    """Item 3: vocal activity not covered by any aligned word (ad-libs, chants,
    backing pads, unlisted lines), as [(start_s, end_s)]."""
    from scipy.ndimage import median_filter
    hop = f["hop"]
    act = median_filter(f["rms_db"], 9) > abs_db
    cover = np.zeros_like(act)
    for w in words:
        cover[max(0, int((w["start"] - pad) / hop)):int((w["end"] + pad) / hop)] = True
    out: list[list[float]] = []
    for a, b in runs(act & ~cover):
        if out and a * hop - out[-1][1] < merge:
            out[-1][1] = b * hop
        else:
            out.append([a * hop, b * hop])
    return [(round(a, 3), round(b, 3)) for a, b in out if b - a >= min_len]


def vocal_onsets(f, jump_st=0.8):
    """Item 2: vocal note onsets = log-mel flux peaks (adaptive threshold, active voice
    only) PLUS pitch jumps > `jump_st` semitones while voiced ("legato note changes
    that have little spectral flux"). Returns [(t, strength, kind)], kind 0 = flux,
    1 = pitch jump (strength fixed 0.35 unless it merges with a flux onset)."""
    import librosa
    from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d
    from scipy.signal import find_peaks
    hop = f["hop"]
    on, rms = f["onset"], f["rms_db"]
    loc = maximum_filter1d(rms, max(1, int(2.0 / hop)))
    active = (rms > -45) & (rms > loc - 25)
    thr = uniform_filter1d(on, max(1, int(0.4 / hop))) * 1.5 + 0.15 * np.percentile(on, 99)
    pk, _ = find_peaks(on, height=0, distance=max(1, int(0.09 / hop)))
    lead = int(0.03 / hop)
    pk = [p for p in pk if on[p] > thr[p] and active[min(len(active) - 1, p + lead)]]
    t_flux = np.array(pk, float) * hop
    s_flux = np.array([on[p] for p in pk], float)
    midi = librosa.hz_to_midi(np.where(f["voiced"], f["f0"], np.nan))
    med = median_filter(np.nan_to_num(midi, nan=0.0), 9)
    w = max(1, int(0.04 / hop))
    a, b = med[:-2 * w], med[2 * w:]
    jump = (a > 0) & (b > 0) & (np.abs(b - a) > jump_st) & active[w:len(med) - w]
    t_pitch, last = [], -10 ** 9
    for i in np.where(jump)[0] + w:
        if i - last > int(0.1 / hop):
            t_pitch.append(i * hop)
        last = i
    norm = np.percentile(s_flux, 95) + 1e-9 if len(s_flux) else 1.0
    ev = [(float(t), float(min(1.0, max(0.1, s / norm))), 0) for t, s in zip(t_flux, s_flux)]
    for t in t_pitch:
        if len(t_flux) == 0 or np.min(np.abs(t_flux - t)) > 0.08:
            ev.append((float(t), 0.35, 1))
    ev.sort()
    return ev
