"""Export <workspace>/out/<slug>/events.json into the Fantasynth app's public/music/ folder.

    <workspace>\\venv-main\\Scripts\\python.exe audio-events\\export_app.py --slug track-a --track my_track
    <workspace>\\venv-main\\Scripts\\python.exe audio-events\\export_app.py --all
    <workspace>\\venv-main\\Scripts\\python.exe audio-events\\export_app.py --all --flac
    <workspace>\\venv-main\\Scripts\\python.exe audio-events\\export_app.py --all --app C:\\path\\to\\the\\app

<app> is the Fantasynth app checkout: --app, else FANTASYNTH_APP, else "app" in
<workspace>/config.json (workspace.app_root). To export into an app worktree, pass --app.
The slug -> app track map, and the title/file used to create a new track file, come from
"tracks" in config.json; --all exports every slug there that has an "appTrack" and is not
marked "all": false.

For each track it
  1. writes <app>/public/music/events/<track>.events.json: compact JSON, QA-only lane
     fields dropped (`r`, `source`, `offGrid`), `exportedAt` added. When nothing but
     `exportedAt` would change, the file is left alone, so a re-run is a no-op;
  2. points <app>/public/music/<track>.json at it with MINIMAL TEXT EDITS (those files
     hold large hand-formatted lyrics, so the file is never re-serialised): top-level
     `bpm` and `gridMarker` are rewritten in place and `"events"` goes on the line after
     `gridMarker`. The result is then parsed and every other key is checked unchanged;
  3. creates the track file when it does not exist yet (from the slug's config "title" and
     "file": url = /local-music/<file>);
  4. writes the extracted sections as the app's top-level `"sections"`, appended as the
     last key and laid out like the hand-made track files, with the label mapped to a
     `sectionType` by SECTION_TYPES (the app's UI uses the same table). Only when the track
     has no `sections` key or an empty list: hand-authored sections are never replaced
     unless --force-sections is passed;
  5. with --flac: writes "<name>.flac" next to the track's "<name>.mp3" in the local
     music folder (LOCAL_MUSIC_PATH, else ~/Music/Live: the folder vite serves as
     /local-music/), checks that it decodes sample-identically to <workspace>/audio/<slug>.wav
     (the analysis input), and switches the track's `url` from .mp3 to .flac. Why: after a
     seek, Chromium's <audio> reports currentTime 25 ms ahead of the output for an MP3
     (encoder delay), while a 16-bit FLAC seeks sample-exactly (measure/seek_formats_*,
     <workspace>/seektest/). An existing file in that folder is never overwritten or deleted;
  6. adds the track file to public/music/index.json in TITLE order when it is missing
     (that index is sorted by title, not filename);
  7. with --lyrics: writes the aligned lyrics from events.json (the lyrics stage) as the
     track's inline `"lyrics"`, one entry per word ({t, d, w, c}: seconds, seconds, text,
     alignment confidence), grouped into the lines the text came in. Replaces the hand-timed
     lyrics wholesale; refuses when events.json has no lyrics block (with --all, such
     tracks are skipped and named). A track marked "handTimed" in config.json is never written:
     its hand-fitted lyrics are the source the lyrics stage read, so --lyrics skips it
     (with or without --all) and says why.

Lyrics `t`/`d` are in seconds, so moving `gridMarker` does not move them.
"""
import argparse
import bisect
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

import workspace
from workspace import AUDIO, OUT, app_root, hand_timed, music_dir


def track_meta(slug):
    """{"track", "title", "url"} for a slug from config.json "tracks" (appTrack, title,
    file). title/url only matter when the track file does not exist yet and has to be created."""
    t = workspace.track_conf(slug)
    meta = {"track": t.get("appTrack")}
    if t.get("title") and t.get("file"):
        meta.update(title=t["title"], url=LOCAL_PREFIX + t["file"])
    return meta

TOP_KEYS = ["version", "source", "tempo", "gridMarker", "meter", "lengthBeats", "gridFit",
            "sections", "drums", "notes", "curves", "lyrics", "vocalOnsets"]
LANE_KEEP = {"drums": ["b", "v", "c", "swing"], "notes": ["b", "len", "pitch", "v", "swing"]}
LANE_DROP = {"r", "source", "offGrid"}  # QA only
# The lyrics stage's blocks: every field is app-facing, so nothing is dropped, but an
# unknown one still stops the export (the same contract as the lanes above).
LYRICS_KEEP = {"lines": ["b", "len", "text"], "words": ["b", "len", "v", "w", "line", "syl"],
               "extras": ["b", "len"]}
LYRICS_META = ["source"]
VOCAL_ONSETS_KEEP = ["b", "v", "kind"]
MANAGED = ("url", "bpm", "gridMarker", "events", "sections", "lyrics")  # the only top-level keys this script may change

# events.json section label -> the app's integer sectionType. FIXED: the app's section UI
# implements the same table, so never renumber; add new labels at the end.
SECTION_TYPES = {"intro": 0, "build": 1, "drop": 2, "breakdown": 3, "groove": 4, "outro": 5}

LOCAL_PREFIX = "/local-music/"  # the URL prefix the app's dev server serves MUSIC_DIR under
WAV_FORMAT = (2, 2, 44100)      # channels, bytes per sample, rate: audio/<slug>.wav (run_all.py decode)
# ffmpeg args for the FLAC. The same decode chain as run_all.py's decode WAV (-ar/-ac/s16), and
# the same container options as the seektest FLAC that measured sample-exact.
FLAC_ENC = ["-map", "0:a:0", "-map_metadata", "-1", "-ar", "44100", "-ac", "2",
            "-c:a", "flac", "-sample_fmt", "s16"]
FLAC_STREAM = {"codec_type": "audio", "codec_name": "flac", "sample_fmt": "s16",
               "sample_rate": "44100", "channels": "2", "bits_per_raw_sample": "16"}


def fmt_num(x, decimals):
    x = round(float(x), decimals)
    return str(int(x)) if x == int(x) else repr(x)


def no_dupes(pairs):
    keys = [k for k, _ in pairs]
    dup = {k for k in keys if keys.count(k) > 1}
    if dup:
        raise ValueError(f"duplicate keys {sorted(dup)}")
    return dict(pairs)


def keep_fields(where, obj, keep, drop=()):
    """`obj` cut down to the `keep` fields, in that order; exits on a field in neither list."""
    stray = set(obj) - set(keep) - set(drop)
    if stray:
        sys.exit(f"{where} has fields this exporter does not know: {sorted(stray)}")
    return {f: obj[f] for f in keep if f in obj}


def build_export(ev):
    unknown = set(ev) - set(TOP_KEYS)
    if unknown:
        sys.exit(f"events.json has top-level keys this exporter does not know: {sorted(unknown)}")
    out = {}
    for k in TOP_KEYS:
        if k not in ev:
            continue
        if k in LANE_KEEP:
            out[k] = {name: keep_fields(f"{k}.{name}", lane, LANE_KEEP[k], LANE_DROP)
                      for name, lane in ev[k].items()}
        elif k == "lyrics":
            ly = keep_fields(k, ev[k], list(LYRICS_KEEP) + LYRICS_META)
            out[k] = {name: keep_fields(f"{k}.{name}", v, LYRICS_KEEP[name]) if name in LYRICS_KEEP else v
                      for name, v in ly.items()}
        elif k == "vocalOnsets":
            out[k] = keep_fields(k, ev[k], VOCAL_ONSETS_KEEP)
        else:
            out[k] = ev[k]
        if k == "version":
            out["exportedAt"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def top_indent(text):
    m = re.search(r"\{\r?\n([ \t]+)\"", text)
    if not m:
        sys.exit("cannot find the top-level indentation")
    return m.group(1)


def newline(text):
    return "\r\n" if "\r\n" in text else "\n"


_WS = re.compile(r"[ \t\r\n]*")


def top_members(text):
    """[(key, key_start, value_start, value_end)] for every member of the top-level object,
    as offsets into `text`. Lets an edit or a check address one key's exact bytes."""
    dec = json.JSONDecoder()
    i = _WS.match(text).end()
    if text[i:i + 1] != "{":
        sys.exit("the track file is not a JSON object")
    i = _WS.match(text, i + 1).end()
    members = []
    if text[i:i + 1] == "}":
        return members
    while True:
        ks = i
        key, i = dec.raw_decode(text, i)
        if not isinstance(key, str):
            sys.exit(f"expected a key at offset {ks}")
        i = _WS.match(text, i).end()
        if text[i:i + 1] != ":":
            sys.exit(f"expected ':' after key {key!r}")
        vs = _WS.match(text, i + 1).end()
        _, ve = dec.raw_decode(text, vs)
        members.append((key, ks, vs, ve))
        i = _WS.match(text, ve).end()
        if text[i:i + 1] == ",":
            i = _WS.match(text, i + 1).end()
            continue
        if text[i:i + 1] != "}":
            sys.exit(f"expected ',' or '}}' after the value of {key!r}")
        return members


def member(text, key):
    hits = [m for m in top_members(text) if m[0] == key]
    if len(hits) != 1:
        sys.exit(f'expected one top-level "{key}", found {len(hits)}')
    return hits[0]


def replace_value(text, key, value_text):
    """Swap the bytes of one top-level value, leaving everything else untouched."""
    _, _, vs, ve = member(text, key)
    return text[:vs] + value_text + text[ve:]


def append_member(text, key, value_text):
    """Add a top-level key after the last one, in the file's own indent and line ending."""
    last_ve = top_members(text)[-1][3]
    return text[:last_ve] + f",{newline(text)}{top_indent(text)}{json.dumps(key)}: {value_text}" + text[last_ve:]


def edit_track(text, bpm_s, gm_s, events_rel):
    """Rewrite bpm/gridMarker in place and put "events" right after gridMarker."""
    nl = newline(text)
    ind = top_indent(text)

    def line_re(key):
        # a top-level key line: exactly the top-level indent, a scalar value, optional comma
        return re.compile(rf'^{re.escape(ind)}"{key}"\s*:\s*([^\r\n,]*?)(,?)[ \t]*(?=\r?$)', re.M)

    for key, val in (("bpm", bpm_s), ("gridMarker", gm_s)):
        hits = list(line_re(key).finditer(text))
        if len(hits) != 1:
            sys.exit(f'expected one top-level "{key}" line, found {len(hits)}')
        m = hits[0]
        text = text[:m.start()] + f'{ind}"{key}": {val}{m.group(2)}' + text[m.end():]

    ev_hits = list(line_re("events").finditer(text))
    ev_val = json.dumps(events_rel)
    if len(ev_hits) > 1:
        sys.exit('more than one top-level "events" line')
    if ev_hits:
        m = ev_hits[0]
        text = text[:m.start()] + f'{ind}"events": {ev_val}{m.group(2)}' + text[m.end():]
    else:
        m = line_re("gridMarker").search(text)
        if m.group(2):   # gridMarker is followed by more keys
            text = text[:m.end()] + f'{nl}{ind}"events": {ev_val},' + text[m.end():]
        else:            # gridMarker was the last key
            text = text[:m.end()] + f',{nl}{ind}"events": {ev_val}' + text[m.end():]
    return text


def verify_track(old_text, new_text, expect):
    """`expect` = {managed key: the value it must now have}. Every other key, managed or
    not, must be unchanged: equal as JSON, in the same order, and byte-identical as text."""
    old = json.loads(old_text, object_pairs_hook=no_dupes)
    new = json.loads(new_text, object_pairs_hook=no_dupes)
    strip = lambda d: {k: v for k, v in d.items() if k not in MANAGED}
    assert strip(old) == strip(new), f"a key other than {'/'.join(MANAGED)} changed"
    assert [k for k in old if k not in MANAGED] == [k for k in new if k not in MANAGED], "key order changed"
    for k in MANAGED:
        if k in expect:
            assert new.get(k) == expect[k], f'"{k}" is {new.get(k)!r}, expected {expect[k]!r}'
        else:
            assert (k in old) == (k in new) and old.get(k) == new.get(k), f'"{k}" changed but this edit does not own it'
    if "events" in expect:
        keys = list(new)
        assert keys[keys.index("gridMarker") + 1] == "events", '"events" is not right after "gridMarker"'
    # textual: every key this edit does not own keeps its exact bytes and its place; so do the
    # text before the first key and after the last value; separators keep the file's style
    om, nm = top_members(old_text), top_members(new_text)
    kept = lambda t, ms: [(k, t[ks:ve]) for k, ks, _, ve in ms if k not in expect]
    assert kept(old_text, om) == kept(new_text, nm), "the text of a key this edit does not own changed"
    assert old_text[:om[0][1]] == new_text[:nm[0][1]], "text before the first key changed"
    assert old_text[om[-1][3]:] == new_text[nm[-1][3]:], "text after the last value changed"
    seps = lambda t, ms: {t[a[3]:b[1]] for a, b in zip(ms, ms[1:])}
    allowed = seps(old_text, om) or {f",{newline(old_text)}{top_indent(old_text)}"}
    assert seps(new_text, nm) <= allowed, f"new separator style {seps(new_text, nm) - allowed}"
    if "\r\n" in old_text:
        assert "\n" not in new_text.replace("\r\n", ""), "a bare LF went into a CRLF file"
    return [l for l in difflib.ndiff(old_text.splitlines(), new_text.splitlines()) if l[:2] in ("- ", "+ ")]


def apply_edit(tpath, old_text, new_text, expect):
    diff = verify_track(old_text, new_text, expect)
    if new_text != old_text:
        tpath.write_bytes(new_text.encode("utf-8"))
    return diff


def new_track_text(meta, bpm_s, gm_s, events_rel, nl):
    lines = ["{",
             f'  "title": {json.dumps(meta["title"], ensure_ascii=False)},',
             f'  "url": {json.dumps(meta["url"], ensure_ascii=False)},',
             f'  "bpm": {bpm_s},',
             f'  "gridMarker": {gm_s},',
             f'  "events": {json.dumps(events_rel)}',
             "}"]
    return nl.join(lines) + nl


# ---------------------------------------------------------------- sections

def as_num(x):
    x = float(x)
    return int(x) if x.is_integer() else x


def app_sections(ev):
    """events.json sections -> the app's section objects. They must tile the track: each
    starts where the previous ends, and the last ends at or before lengthBeats."""
    out = []
    for s in ev.get("sections") or []:
        label = s["label"]
        if label not in SECTION_TYPES:
            sys.exit(f"section label {label!r} has no sectionType in SECTION_TYPES")
        start, beats = as_num(s["startBeat"]), as_num(s["beats"])
        if beats <= 0:
            sys.exit(f"section at beat {start} has length {beats}")
        if out and start != out[-1]["startBeat"] + out[-1]["beatLength"]:
            sys.exit(f"sections are not contiguous: one starts at {start}, the previous ends at "
                     f"{out[-1]['startBeat'] + out[-1]['beatLength']}")
        out.append({"startBeat": start, "beatLength": beats, "sectionType": SECTION_TYPES[label],
                    "label": label, "params": {}})
    if out:
        end = out[-1]["startBeat"] + out[-1]["beatLength"]
        if out[0]["startBeat"] < 0 or end > ev["lengthBeats"]:
            sys.exit(f"sections span {out[0]['startBeat']}..{end}, outside 0..lengthBeats {ev['lengthBeats']}")
    return out


def block_text(value, ind, nl):
    """The value text, laid out like the hand-made track files (and their lyrics):
    one key per line, nested one indent deeper than the top-level keys."""
    lines = json.dumps(value, indent=ind, ensure_ascii=False).split("\n")
    return (nl + ind).join(lines)


def write_sections(tpath, ev, force):
    text = tpath.read_bytes().decode("utf-8")
    cur = json.loads(text)
    secs = app_sections(ev)
    if not secs:
        print(f"   {tpath.name}: events.json has no sections; nothing to write")
        return
    if cur.get("sections") == secs:
        print(f"   {tpath.name}: already has these {len(secs)} sections")
        return
    if "sections" in cur and cur["sections"] != [] and not force:
        n = len(cur["sections"]) if isinstance(cur["sections"], list) else cur["sections"]
        print(f"   {tpath.name}: keeps its existing sections ({n}); --force-sections replaces them")
        return
    value = block_text(secs, top_indent(text), newline(text))
    new_text = replace_value(text, "sections", value) if "sections" in cur else append_member(text, "sections", value)
    diff = apply_edit(tpath, text, new_text, {"sections": secs})
    end = secs[-1]["startBeat"] + secs[-1]["beatLength"]
    counts = {}
    for s in secs:
        counts[s["label"]] = counts.get(s["label"], 0) + 1
    print(f"   {tpath.name}: wrote {len(secs)} sections, beats {secs[0]['startBeat']}..{end} "
          f"(lengthBeats {ev['lengthBeats']}), {counts}; {len(diff)} added lines, all other keys verified unchanged")


# ---------------------------------------------------------------- lyrics

REST_S = 2.0  # s: a gap between lyric lines longer than this is written as an empty line


def app_lyrics(ev):
    """events.json `lyrics` (beats on the events grid) -> the app's inline lyrics
    {"lines": [{"words": [{t, d, w, c}]}]}, t/d in seconds of the audio. Converted with the
    EVENTS file's own grid at full precision, not the rounded bpm/gridMarker of the track."""
    tempo = ev["tempo"]
    if len(tempo) != 1:
        sys.exit(f"events.json has {len(tempo)} tempo entries; the lyrics conversion assumes one")
    spb = 60.0 / float(tempo[0]["bpm"])
    gm = float(ev["gridMarker"])
    lines, words = ev["lyrics"]["lines"], ev["lyrics"]["words"]
    out = [{"words": []} for _ in lines["b"]]
    vs = words.get("v") or [None] * len(words["b"])  # no v: hand-timed words, no confidence
    for b, n, w, v, line in zip(words["b"], words["len"], words["w"], vs, words["line"], strict=True):
        if not 0 <= line < len(out):
            sys.exit(f"lyrics word {w!r} points at line {line}, but there are {len(out)} lines")
        # d never rounds to 0: a word's progress is divided by it
        out[line]["words"].append({"t": as_num(round(gm + b * spb, 3)), "d": as_num(max(round(n * spb, 3), 0.001)),
                                   "w": w, **({"c": as_num(round(float(v), 2))} if v is not None else {})})
    lines = [ln for ln in out if ln["words"]]
    # A rest: the app shows a line until the next one starts, so a gap longer than REST_S
    # gets an empty line (the hand-made files' {"w": ""} convention), and so does the tail.
    song_end = gm + float(ev.get("lengthBeats", 0)) * spb
    with_rests = []
    for i, ln in enumerate(lines):
        with_rests.append(ln)
        last = ln["words"][-1]
        end = float(last["t"]) + float(last["d"])
        nxt = float(lines[i + 1]["words"][0]["t"]) if i + 1 < len(lines) else song_end
        if nxt - end > REST_S:
            with_rests.append({"words": [{"t": as_num(round(end, 3)), "d": as_num(round(nxt - end, 3)), "w": ""}]})
    return {"lines": with_rests}


def hand_timed_note(track):
    """The skip message when `track` is marked handTimed in config.json, else None."""
    reason = hand_timed(track)
    return reason and (f"{track}.json is hand-timed ({reason}; handTimed in {workspace.CONFIG}): "
                       f"--lyrics leaves its lyrics alone")


def write_lyrics(tpath, ev):
    note = hand_timed_note(tpath.stem)
    if note:  # also guarded in export(); this keeps any other caller honest
        print(f"   {note}")
        return
    if not ev.get("lyrics"):
        sys.exit(f"{tpath.name}: events.json has no lyrics block (run the lyrics stage, then assemble); "
                 f"--lyrics has nothing to write")
    text = tpath.read_bytes().decode("utf-8")
    cur = json.loads(text)
    lyr = app_lyrics(ev)
    nw = sum(len(ln["words"]) for ln in lyr["lines"])
    if cur.get("lyrics") == lyr:
        print(f"   {tpath.name}: already has these lyrics ({len(lyr['lines'])} lines, {nw} words)")
        return
    old = cur.get("lyrics")
    was = f"{len(old['lines'])} lines" if isinstance(old, dict) and isinstance(old.get("lines"), list) else "none"
    value = block_text(lyr, top_indent(text), newline(text))
    new_text = replace_value(text, "lyrics", value) if "lyrics" in cur else append_member(text, "lyrics", value)
    diff = apply_edit(tpath, text, new_text, {"lyrics": lyr})
    print(f"   {tpath.name}: wrote aligned lyrics, {len(lyr['lines'])} lines / {nw} words (was {was}); "
          f"{len(diff)} changed lines, all other keys verified unchanged")


# ---------------------------------------------------------------- FLAC

def wav_pcm(path):
    """The interleaved s16le frames of the analysis WAV, read without ffmpeg."""
    with wave.open(str(path), "rb") as w:
        fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if fmt != WAV_FORMAT or w.getcomptype() != "NONE":
            sys.exit(f"{path} is {fmt}, expected {WAV_FORMAT} PCM")
        return w.readframes(w.getnframes())


def flac_check(path, ref, ref_name):
    """Fail loudly unless `path` is a lone 16-bit 44.1 kHz stereo FLAC stream that decodes to
    exactly the samples in `ref`, by ffmpeg and (when importable) by libsndfile."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,codec_name,sample_fmt,sample_rate,channels,bits_per_raw_sample",
                        "-of", "json", str(path)], capture_output=True, text=True, check=True)
    streams = json.loads(r.stdout)["streams"]
    got = [{k: str(s.get(k)) for k in FLAC_STREAM} for s in streams]
    if got != [FLAC_STREAM]:
        sys.exit(f"FLAC CHECK FAILED: {path} streams are {got}, expected exactly [{FLAC_STREAM}]")
    decoders = {}
    r = subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-i", str(path), "-map", "0:a:0",
                        "-f", "s16le", "-c:a", "pcm_s16le", "-"], capture_output=True, check=True)
    decoders["ffmpeg"] = r.stdout
    try:
        import soundfile
    except ImportError:
        soundfile = None
    if soundfile is not None:
        decoders["libsndfile"] = soundfile.read(str(path), dtype="int16", always_2d=True)[0].tobytes()
    for name, pcm in decoders.items():
        if pcm != ref:
            import numpy as np
            n = min(len(pcm), len(ref)) // 2
            a, b = np.frombuffer(pcm, "<i2", n), np.frombuffer(ref, "<i2", n)
            first = (int(np.argmax(a != b)) if (a != b).any() else n) // 2
            sys.exit(f"FLAC CHECK FAILED: {path} decoded by {name} is NOT sample-identical to {ref_name}: "
                     f"{len(pcm) // 4} vs {len(ref) // 4} frames, first difference at frame {first}")
    return f"{len(ref) // 4} frames ({len(ref) / 4 / 44100:.3f} s), sha256 {hashlib.sha256(ref).hexdigest()[:16]}, " \
           f"identical by {' + '.join(decoders)}"


def write_flac(slug, tpath):
    """Put a sample-exact FLAC next to the track's MP3 and point the track's url at it."""
    text = tpath.read_bytes().decode("utf-8")
    url = json.loads(text)["url"]
    if not url.startswith(LOCAL_PREFIX):
        sys.exit(f'{tpath.name}: url {url!r} is not under {LOCAL_PREFIX}; --flac only handles the local music folder')
    rel = PurePosixPath(unquote(url[len(LOCAL_PREFIX):]))
    ext = rel.suffix.lower()
    if ext not in (".mp3", ".flac") or not url.endswith(rel.suffix):
        sys.exit(f"{tpath.name}: url {url!r} does not end in .mp3 or .flac")
    folder = music_dir()  # workspace.py: LOCAL_MUSIC_PATH, else config musicDir, else ~/Music/Live
    mp3 = folder.joinpath(*rel.parts).with_suffix(rel.suffix if ext == ".mp3" else ".mp3")
    flac = mp3.with_suffix(".flac")
    if folder.resolve() not in flac.resolve().parents:
        sys.exit(f"{flac} is outside {folder}")
    wav = AUDIO / f"{slug}.wav"
    ref = wav_pcm(wav)
    ref_name = f"audio/{slug}.wav"

    if flac.exists():
        facts = flac_check(flac, ref, ref_name)
        print(f"   {flac.name}: already there ({flac.stat().st_size:,} bytes, not rewritten); {facts}")
    else:
        if not mp3.is_file():
            sys.exit(f"{mp3} not found: the FLAC is decoded from the track's MP3")
        with tempfile.TemporaryDirectory(prefix="export_app_") as tmp:
            tmpf = Path(tmp) / "track.flac"
            subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-n", "-i", str(mp3)] + FLAC_ENC + [str(tmpf)],
                           check=True)
            facts = flac_check(tmpf, ref, ref_name)
            with open(tmpf, "rb") as src, open(flac, "xb") as dst:  # "x": never replaces a file
                shutil.copyfileobj(src, dst, 1 << 20)
            if tmpf.read_bytes() != flac.read_bytes():
                sys.exit(f"{flac} does not match the file it was copied from")
        print(f"   {flac.name}: written ({flac.stat().st_size:,} bytes, MP3 {mp3.stat().st_size:,}); {facts}")

    new_url = url[:-len(rel.suffix)] + ".flac"
    if url == new_url:
        print(f"   {tpath.name}: url already points at the FLAC")
        return
    _, _, vs, ve = member(text, "url")
    raw = text[vs:ve]
    if not raw.endswith(rel.suffix + '"'):
        sys.exit(f"{tpath.name}: url text {raw} does not end in {rel.suffix}")
    new_text = text[:vs] + raw[:-len(rel.suffix) - 1] + '.flac"' + text[ve:]
    diff = apply_edit(tpath, text, new_text, {"url": new_url})
    print(f"   {tpath.name}: url -> {new_url} ({len(diff)} changed lines, all other keys verified unchanged)")


# ---------------------------------------------------------------- index

def fmt_index(items, nl):
    return "[" + nl + ("," + nl).join(f"  {json.dumps(i, ensure_ascii=False)}" for i in items) + nl + "]" + nl


def update_index(music, fname):
    path = music / "index.json"
    raw = path.read_bytes().decode("utf-8")
    items = json.loads(raw)
    if fname in items:
        print(f"   index.json already lists {fname}")
        return
    nl = "\r\n" if "\r\n" in raw else "\n"
    if fmt_index(items, nl) != raw:
        sys.exit("index.json formatting is not the plain one-per-line list this script can reproduce; add the entry by hand")
    title = lambda f: json.loads((music / f).read_bytes().decode("utf-8"))["title"].casefold()
    keys = [title(f) for f in items]
    if keys != sorted(keys):
        print("   WARNING: index.json was not in title order before this edit")
    pos = bisect.bisect_right(keys, title(fname))
    items.insert(pos, fname)
    path.write_bytes(fmt_index(items, nl).encode("utf-8"))
    after = items[pos - 1] if pos else "(start)"
    print(f"   index.json: inserted {fname} at position {pos} (after {after})")


def export(slug, track, repo, flac=False, force_sections=False, lyrics=False):
    """Export out/<slug>/events.json into the app checkout `repo` as public/music/<track>.json."""
    src = OUT / slug / "events.json"
    ev = json.loads(src.read_text(encoding="utf-8"))
    note = hand_timed_note(track) if lyrics else None
    if note:  # skipped, not refused: the rest of the export still runs
        print(f"--lyrics: {note}")
        lyrics = False
    if lyrics and not ev.get("lyrics"):  # before any write, so a refused run changes nothing
        sys.exit(f"{src}: no lyrics block; --lyrics has nothing to write for {slug}")
    music = repo / "public" / "music"
    if not music.is_dir():
        sys.exit(f"not a Fantasynth app checkout: {music} missing")
    meta = track_meta(slug)
    events_rel = f"events/{track}.events.json"

    exp = build_export(ev)
    dst = music / events_rel
    dst.parent.mkdir(exist_ok=True)
    prev = None
    if dst.exists():
        try:
            prev = json.loads(dst.read_bytes())
        except ValueError:
            pass
    if isinstance(prev, dict) and "exportedAt" in prev and prev == {**exp, "exportedAt": prev["exportedAt"]}:
        exp["exportedAt"] = prev["exportedAt"]  # nothing else changed: keep the stamp
    body = json.dumps(exp, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if dst.exists() and dst.read_bytes() == body:
        state = "unchanged"
    else:
        dst.write_bytes(body)
        state = "written"
    assert json.loads(dst.read_bytes()) == exp
    print(f"{slug} -> {dst.relative_to(repo)}: {len(body) / 1024:.0f} KB {state} (source {src.stat().st_size / 1024:.0f} KB)")

    bpm = float(ev["tempo"][0]["bpm"])
    bpm_s, gm_s = fmt_num(bpm, 2), fmt_num(ev["gridMarker"], 4)
    bpm_v, gm_v = json.loads(bpm_s), json.loads(gm_s)
    tpath = music / f"{track}.json"
    if tpath.exists():
        old_text = tpath.read_bytes().decode("utf-8")
        old = json.loads(old_text)
        new_text = edit_track(old_text, bpm_s, gm_s, events_rel)
        diff = apply_edit(tpath, old_text, new_text, {"bpm": bpm_v, "gridMarker": gm_v, "events": events_rel})
        print(f"   {tpath.name}: bpm {old.get('bpm')} -> {bpm_v}, gridMarker {old.get('gridMarker')} -> {gm_v}; "
              f"{len(diff)} changed/added lines, all other keys verified unchanged")
    else:
        if "title" not in meta:
            sys.exit(f'{tpath} does not exist, and config.json has no "title" and "file" for '
                     f"{slug} to create it")
        nl = "\r\n" if b"\r\n" in (music / "index.json").read_bytes() else "\n"
        text = new_track_text(meta, bpm_s, gm_s, events_rel, nl)
        new = json.loads(text, object_pairs_hook=no_dupes)
        assert new["bpm"] == bpm_v and new["gridMarker"] == gm_v and new["events"] == events_rel
        tpath.write_bytes(text.encode("utf-8"))
        print(f"   {tpath.name}: created (bpm {bpm_v}, gridMarker {gm_v})")
    write_sections(tpath, ev, force_sections)
    if lyrics:
        write_lyrics(tpath, ev)
    if flac:
        write_flac(slug, tpath)
    update_index(music, tpath.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", help="out/<slug>/events.json to export")
    ap.add_argument("--track", help="app track basename (public/music/<track>.json); default: the slug's "
                                    "appTrack in config.json")
    ap.add_argument("--all", action="store_true", help='export every slug in config.json that has an appTrack (except "all": false)')
    ap.add_argument("--app", help='the Fantasynth app checkout (default: FANTASYNTH_APP, else "app" in config.json)')
    ap.add_argument("--flac", action="store_true",
                    help="also write <name>.flac next to the track's MP3 in the local music folder, "
                         "verify it against audio/<slug>.wav, and point the track's url at it")
    ap.add_argument("--force-sections", action="store_true",
                    help="replace a track's existing, non-empty sections with the extracted ones")
    ap.add_argument("--lyrics", action="store_true",
                    help="replace the track's inline lyrics with the aligned ones from events.json "
                         "(per-word timing and confidence); refuses a track whose events have none, "
                         "skips a track marked handTimed in config.json")
    a = ap.parse_args()
    if a.all:
        if a.track:
            sys.exit("--track only makes sense with --slug")
        jobs = [(s, t["appTrack"]) for s, t in workspace.in_all().items() if t.get("appTrack")]
        if not jobs:
            sys.exit(f'--all exports every slug with an "appTrack" in {workspace.CONFIG}, and there are none')
    elif a.slug:
        track = a.track or workspace.app_track(a.slug)
        if not track:
            sys.exit(f"no app track for slug {a.slug}: pass --track, or set tracks.{a.slug}.appTrack "
                     f"in {workspace.CONFIG}")
        jobs = [(a.slug, track)]
    else:
        ap.error("pass --slug <slug> [--track <name>] or --all")
    if a.lyrics and a.all:
        # --all means every track that has aligned lyrics: an instrumental is skipped, by name
        has = lambda s: bool(json.loads((OUT / s / "events.json").read_text(encoding="utf-8")).get("lyrics"))
        skip = [s for s, _ in jobs if not has(s)]
        if skip:
            print(f"--lyrics: no lyrics block for {', '.join(skip)}; their lyrics are left alone")
        jobs = [(s, t, s not in skip) for s, t in jobs]
    else:
        jobs = [(s, t, a.lyrics) for s, t in jobs]
    app = app_root(a.app)
    for slug, track, lyrics in jobs:
        export(slug, track, app, flac=a.flac, force_sections=a.force_sections, lyrics=lyrics)


if __name__ == "__main__":
    main()
