"""Lyric alignment + vocal onsets on the vocals stem -> out/<slug>/raw_lyrics.json (seconds).

    <venv-main>/Scripts/python.exe scripts/lyrics.py --slug track-a [--track my_track] [--app <path>] [--text lyrics.txt]

The lyric TEXT is known: it comes from the app's track file <app>/public/music/<track>.json
(inline `lyrics`, word- or line-level, only the words are used, never their times), or from
a plain text file (one lyric line per line) with --text. The stage finds WHEN each word is
sung; see lyrics_align.py for the method. The chunked CTC emission computation is ported
from mexicat/pdoom-video `analysis/ctc_emissions.py` (MIT, Copyright (c) 2026 Giacomo
Magnanini; see THIRD_PARTY_NOTICES.md).

<track> is --track, else the slug's "appTrack" in <workspace>/config.json. <app> is --app,
else FANTASYNTH_APP, else "app" in config.json (workspace.app_root), resolved only when a
track file is read. A named track whose file does not exist is an ERROR: the stage exits
non-zero before writing anything, rather than overwriting raw_lyrics.json with no words.
With no track named at all (and no --text), and for a track file without lyrics (an
instrumental), the stage writes vocal onsets only.

Models download on first use into <workspace>/models/torch (TORCH_HOME), about 2.4 GB for
the two aligners. Emissions are cached per stem in out/<slug>/lyrics_em/ and recomputed
when stems/vocals.wav is newer.

A track marked "handTimed" in config.json is not aligned: its inline lyrics were timed by
hand, so the words, their times and the line grouping are taken from the track file verbatim
(no confidences). Vocal onsets and extras are computed as usual.

Per-track pronunciation fixes live in config.json, under the track's slug:
    "tracks": {"<slug>": {"appTrack": "<track>", "pron": {"AGI": "ay gee i", "DJ": "dee jay"}}}
Keys match a display token exactly or case-folded without edge punctuation; a spelling
with several sub-words gives the word one syllable span per sub-word.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # audio-events/, for workspace.py
import workspace  # noqa: E402
from workspace import MODELS, OUT, app_root, hand_timed  # noqa: E402

import lyrics_align as LA  # noqa: E402

SILENT_DBFS = -40.0  # same gate as notes.py: below this the stem is separation residue
CTC_SR = 16000
HOP = 320  # 20 ms at 16 kHz
CHUNK_S, CTX_S = 20.0, 3.0
MODELS_CTC = {"mms": "MMS_FA", "lv60k": "WAV2VEC2_ASR_LARGE_LV60K_960H"}
SOURCES = ("mono", "left", "right")
MARGIN = 1.5
LOW_CONF = 0.6
PRIOR_WINDOW = 3.0  # s: a line must lie within its reference line time +- this (0 = no prior)
MOVED_S = 1.0       # a line whose free alignment starts this far from the constrained one is flagged
# The app checkout. Resolved lazily (track_file), from --app (APP_ARG, set by main()), else
# FANTASYNTH_APP, else config.json; tests set APP directly.
APP = None
APP_ARG = None


# ------------------------------------------------------------------------------ text

def track_file(track):
    """<app>/public/music/<track>.json. A missing file is an error, never "no lyrics": a run
    that went on would overwrite raw_lyrics.json with no words and exit 0."""
    global APP
    if APP is None:
        APP = app_root(APP_ARG)
    path = Path(APP) / "public" / "music" / f"{track}.json"
    if not path.is_file():
        sys.exit(f"{path} not found: the lyrics stage reads the lyric text of track {track!r} from it. "
                 f"Check --track (or the slug's appTrack in config.json) and --app; nothing was written")
    return path


def text_from_track(track):
    """(lines, times) from the track file's inline lyrics, or (None, None) when the file has
    none (an instrumental). lines = display tokens per lyric line; times = each line's
    (start, end) seconds in that file, used only as a loose prior (LRC or transcript times
    are approximate). A missing track file exits (track_file)."""
    lyr = json.loads(track_file(track).read_text(encoding="utf-8")).get("lyrics")
    if not lyr:
        return None, None
    lines, times = [], []
    for ln in lyr.get("lines", []):
        ws = [w for w in ln.get("words", []) if w.get("w", "").strip()]
        if not ws:
            continue
        lines.append(" ".join(w["w"].strip() for w in ws).split())
        times.append((float(ws[0]["t"]), float(ws[-1]["t"]) + float(ws[-1].get("d") or 0)))
    return (lines, times) if lines else (None, None)


def frozen_prior(d, track, lines_tokens, ref_times):
    """The line times used as the prior, frozen on the first run in out/<slug>/lyrics_prior.json.

    After `export_app.py --lyrics` the track file holds THIS stage's output, so reading the
    prior from it again would let a misplaced line anchor itself forever. The first run's
    times (an LRC, a transcript, hand edits) are kept and reused while the text is unchanged."""
    if not ref_times:
        return ref_times
    p = d / "lyrics_prior.json"
    text = [" ".join(t) for t in lines_tokens]
    if p.exists():
        saved = json.loads(p.read_text(encoding="utf-8"))
        if saved.get("text") == text:
            return [tuple(x) for x in saved["times"]]
        print("    lyric text changed since the prior was frozen: refreezing from the track file")
    if track_is_aligned(track):
        print("    WARNING: the track's lyrics already carry aligner confidences (c) and no frozen "
              "prior exists: the prior is this stage's own earlier output")
    p.write_text(json.dumps(dict(track=track, text=text, times=[list(t) for t in ref_times]), indent=1,
                            ensure_ascii=False), encoding="utf-8")
    return ref_times


def track_is_aligned(track):
    lyr = json.loads(track_file(track).read_text(encoding="utf-8")).get("lyrics") or {}
    return any("c" in w for ln in lyr.get("lines", []) for w in ln.get("words", []))


def hand_timed_lines(track):
    """The track file's inline lyrics as they stand, for a hand-timed track: per non-empty
    line, [{w, t, d}] with the file's words and seconds verbatim and its line grouping. Rest
    words ({"w": ""}) are dropped; a line left empty is dropped."""
    lyr = json.loads(track_file(track).read_text(encoding="utf-8")).get("lyrics") or {}
    out = []
    for ln in lyr.get("lines", []):
        ws = [dict(w=w["w"], t=float(w["t"]), d=float(w.get("d") or 0))
              for w in ln.get("words", []) if w.get("w", "").strip()]
        if ws:
            out.append(ws)
    return out


def hand_timed_doc(doc, track, reason, lines, f):
    """Fill `doc` from hand-timed lines: words carry no conf; extras (when the stem has
    features `f`) are measured against these words exactly as against aligned ones."""
    doc.update(source=f"hand-timed (public/music/{track}.json)", handTimed=reason, prior=None,
               models=None, sources=None, margin=None,  # nothing was aligned
               lines=[dict(text=" ".join(w["w"] for w in ws), words=ws) for ws in lines])
    words = [dict(start=w["t"], end=w["t"] + w["d"]) for ws in lines for w in ws]
    if f is not None:
        doc["extras"] = [dict(t=r3(a), d=r3(r3(b) - r3(a))) for a, b in LA.extras(words, f)]
    doc["stats"].update(words=len(words), lines=len(lines), extrasSeconds=r3(sum(e["d"] for e in doc["extras"])))
    return doc


def text_from_file(path):
    lines = [ln.split() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln] or None


def pron_table(track):
    """The track's pronunciation fixes (tracks.<slug>.pron in config.json), {} when none."""
    return workspace.pron_table(track) if track else {}


# ------------------------------------------------------------------------- emissions

def vocal_sources(path):
    """mono / left / right of the vocals stem at 16 kHz, peak-normalised."""
    import soxr
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    out = {}
    for name, x in (("mono", y.mean(1)), ("left", y[:, 0]), ("right", y[:, 1])):
        x = soxr.resample(np.ascontiguousarray(x), sr, CTC_SR)
        out[name] = (x / (np.abs(x).max() + 1e-9)).astype(np.float32)
    return out


def load_models():
    os.environ.setdefault("TORCH_HOME", str(MODELS / "torch"))
    import torch
    import torchaudio
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for key, name in MODELS_CTC.items():
        bundle = getattr(torchaudio.pipelines, name)
        m = bundle.get_model(with_star=False) if key == "mms" else bundle.get_model()
        labels = list(bundle.get_labels(star=None)) if key == "mms" else list(bundle.get_labels())
        models[key] = (m.to(dev).eval(), labels)
    return models, dev


def emission(model, dev, y):
    """Frame log-probs [n_frames, V] of one mono 16 kHz signal: 20 s chunks with 3 s of
    context either side, keeping only each chunk's own frames."""
    import torch
    n_frames = len(y) // HOP
    chunk, ctx = int(CHUNK_S * CTC_SR), int(CTX_S * CTC_SR)
    out = None
    for s in range(0, len(y), chunk):
        a, b = max(0, s - ctx), min(len(y), s + chunk + ctx)
        x = torch.from_numpy(y[a:b])[None].to(dev)
        with torch.inference_mode():
            em, _ = model(x)
            em = torch.log_softmax(em.float(), dim=-1)[0].cpu().numpy()
        if out is None:
            out = np.full((n_frames, em.shape[1]), np.nan, np.float32)
        f0 = a // HOP
        lo, hi = s // HOP, min(n_frames, (s + chunk) // HOP)
        seg = em[lo - f0:hi - f0]
        out[lo:lo + len(seg)] = seg
    valid = np.where(~np.isnan(out[:, 0]))[0]
    out[valid.max() + 1:] = out[valid.max()]
    return out


def all_emissions(d, vocals):
    """{'<model>_<source>': common-alphabet log-probs}, cached in lyrics_em/."""
    cache = d / "lyrics_em"
    cache.mkdir(exist_ok=True)
    stamp = vocals.stat().st_mtime
    ems, need = {}, []
    for key in MODELS_CTC:
        for src in SOURCES:
            p = cache / f"{key}_{src}.npz"
            if p.exists() and p.stat().st_mtime >= stamp:
                z = np.load(p, allow_pickle=False)
                ems[f"{key}_{src}"] = LA.to_common(z["em"], [str(x) for x in z["labels"]])
            else:
                need.append((key, src, p))
    if need:
        t = time.time()
        models, dev = load_models()
        sources = vocal_sources(vocals)
        for key, src, p in need:
            model, labels = models[key]
            em = emission(model, dev, sources[src])
            np.savez_compressed(p, em=em, labels=np.array(labels))
            ems[f"{key}_{src}"] = LA.to_common(em, labels)
        print(f"    emissions: {len(need)} computed on {dev} in {time.time() - t:.1f}s")
    n = max(len(e) for e in ems.values())
    return {k: LA.pad_to(e, n) for k, e in ems.items()}


# ---------------------------------------------------------------------------- stage

def features(vocals):
    import soxr
    y, sr = sf.read(str(vocals), dtype="float32", always_2d=True)
    y = soxr.resample(np.ascontiguousarray(y.mean(1)), sr, LA.FEAT_SR)
    return LA.vocal_features(y, LA.FEAT_SR)


def r3(x):
    return round(float(x), 3)


def run(slug, track=None, text_path=None, window=PRIOR_WINDOW):
    d = OUT / slug
    vocals = d / "stems" / "vocals.wav"
    if not vocals.exists():
        sys.exit(f"{vocals} missing: run the separate stage first")
    track = track or workspace.app_track(slug)
    ref_times = None
    reason = None if text_path else hand_timed(track)
    hand = hand_timed_lines(track) if reason else None
    if reason:
        # no prior, no alignment: the text and its times both come from the track file
        lines_tokens = None
        print(f"    {track} is hand-timed ({reason}): taking its word times verbatim, not aligning")
    elif text_path:
        lines_tokens = text_from_file(text_path)
    elif track:
        lines_tokens, ref_times = text_from_track(track)
        ref_times = frozen_prior(d, track, lines_tokens, ref_times)
    else:
        lines_tokens = None
    y, _ = sf.read(str(vocals), always_2d=True)
    level = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)
    doc = dict(version=1, track=track if (lines_tokens or hand) and not text_path else None,
               text=str(text_path) if text_path else None,
               models=list(MODELS_CTC.values()), sources=list(SOURCES), margin=MARGIN,
               levelDbfs=round(float(level), 1), lines=[], extras=[], onsets=[], stats={})
    if level < SILENT_DBFS:
        print(f"    vocals {level:.1f} dBFS < {SILENT_DBFS}: silent, nothing to align")
        if hand:
            hand_timed_doc(doc, track, reason, hand, None)
        (d / "raw_lyrics.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        return doc

    t = time.time()
    f = features(vocals)
    print(f"    vocal features in {time.time() - t:.1f}s")
    onsets = LA.vocal_onsets(f)
    doc["onsets"] = [dict(t=r3(ot), s=round(s, 2), kind="pitch" if k else "flux") for ot, s, k in onsets]
    doc["stats"]["onsets"] = {"flux": sum(1 for o in onsets if o[2] == 0), "pitch": sum(1 for o in onsets if o[2] == 1)}

    if hand:
        hand_timed_doc(doc, track, reason, hand, f)
        (d / "raw_lyrics.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        s = doc["stats"]
        print(f"    {s['words']} hand-timed words in {s['lines']} lines; {len(doc['extras'])} extras "
              f"({s['extrasSeconds']} s); onsets {s['onsets']}")
        return doc

    if not lines_tokens:
        print(f"    no lyric text for {slug} (track {track}): vocal onsets only")
        (d / "raw_lyrics.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
        return doc

    table = pron_table(track)
    pron_fn = lambda tok: LA.pron(tok, table)  # noqa: E731
    ems = all_emissions(d, vocals)
    t = time.time()
    fused6 = LA.fuse([ems[f"{m}_{s}"] for m in MODELS_CTC for s in SOURCES])
    wins = None
    if ref_times and window > 0:
        wins = {li: (a - window, b + window) for li, (a, b) in enumerate(ref_times)}
    free_spans, free_score = LA.align(fused6, lines_tokens, pron_fn, MARGIN)
    spans, score = LA.align(fused6, lines_tokens, pron_fn, MARGIN, wins) if wins else (free_spans, free_score)
    if score <= LA.NO_PATH:
        print(f"    WARNING: the line-time prior (+-{window}s) admits no alignment; aligning without it")
        wins, spans, score = None, free_spans, free_score
    doc["prior"] = f"track line times +-{window}s" if wins else None
    words = LA.word_table(spans, lines_tokens)

    def alt(em):
        sp, sc = LA.align(em, lines_tokens, pron_fn, MARGIN, wins)
        return LA.word_table(sp if sc > LA.NO_PATH else LA.align(em, lines_tokens, pron_fn, MARGIN)[0], lines_tokens)
    alts = {"mms": alt(ems["mms_mono"]), "lv60k": alt(ems["lv60k_mono"]),
            "left": alt(LA.fuse([ems["mms_left"], ems["lv60k_left"]])),
            "right": alt(LA.fuse([ems["mms_right"], ems["lv60k_right"]]))}
    free_first = {}
    for w in LA.word_table(free_spans, lines_tokens):
        free_first.setdefault(w["li"], w["start"])
    print(f"    {6 if wins else 5} alignments in {time.time() - t:.1f}s")
    words = LA.refine(words, f, pron_fn)
    words = LA.confidence(words, alts)

    lines = []
    for li, toks in enumerate(lines_tokens):
        ws = [w for w in words if w["li"] == li]
        if not ws:
            continue
        out = []
        for w in ws:
            # durations from rounded ends, so words that tile in time still tile in the file
            e = dict(w=w["w"], t=r3(w["start"]), d=r3(r3(w["end"]) - r3(w["start"])), conf=w["conf_final"],
                     agree=w["agree"], rule=w["start_rule"],
                     ctc=[r3(w["ctc_start"]), r3(r3(w["ctc_end"]) - r3(w["ctc_start"]))])
            if len(w["subs"]) > 1:
                e["syl"] = [[r3(a), r3(r3(b) - r3(a))] for a, b in w["subs"]]
            out.append(e)
        ln = dict(text=" ".join(toks), words=out)
        if ref_times:
            ln["ref"] = r3(ref_times[li][0])
        if li in free_first and abs(free_first[li] - ws[0]["ctc_start"]) > MOVED_S:
            ln["free"] = r3(free_first[li])  # where the unconstrained path put it: review
        lines.append(ln)
    doc["lines"] = lines
    doc["extras"] = [dict(t=r3(a), d=r3(r3(b) - r3(a))) for a, b in LA.extras(words, f)]
    conf = np.array([w["conf_final"] for w in words])
    rules = {}
    for w in words:
        rules[w["start_rule"]] = rules.get(w["start_rule"], 0) + 1
    doc["stats"].update(words=len(words), tokens=sum(len(t) for t in lines_tokens), lines=len(lines),
                        lowConf=int((conf < LOW_CONF).sum()), medianConf=r3(np.median(conf)),
                        meanAgree=r3(np.mean([w["agree"] for w in words])), startRules=rules,
                        scorePerFrame=r3(score / len(fused6)),
                        priorMovedLines=sum(1 for ln in lines if "free" in ln),
                        extrasSeconds=r3(sum(e["d"] for e in doc["extras"])))
    (d / "raw_lyrics.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    s = doc["stats"]
    print(f"    {s['words']}/{s['tokens']} words in {s['lines']} lines, median conf {s['medianConf']}, "
          f"{s['lowConf']} below {LOW_CONF}, agreement {s['meanAgree']}, rules {rules}; "
          f"{len(doc['extras'])} extras ({s['extrasSeconds']} s); onsets {s['onsets']}; "
          f"prior {doc['prior']}, {s['priorMovedLines']} lines moved by it")
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--track", help="app track whose file <app>/public/music/<track>.json holds the lyric text "
                                    "(default: the slug's appTrack in config.json)")
    ap.add_argument("--app", help='the Fantasynth app checkout (default: FANTASYNTH_APP, else "app" in config.json)')
    ap.add_argument("--text", help="plain text lyric file, one line per line (overrides --track)")
    ap.add_argument("--window", type=float, default=PRIOR_WINDOW,
                    help="line-time prior from the track file, +- seconds (0 = none)")
    args = ap.parse_args()
    global APP_ARG
    APP_ARG = args.app
    if args.all:
        for s in workspace.all_slugs():
            print(f"== {s}")
            run(s)
    elif args.slug:
        run(args.slug, args.track, args.text, args.window)
    else:
        ap.error("--slug or --all")


if __name__ == "__main__":
    main()
