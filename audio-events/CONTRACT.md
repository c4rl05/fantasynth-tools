# audio-events: layout and file contract

This is the canonical specification of the files the pipeline writes, and of the events
file Fantasynth reads. When the code and this file disagree, one of them is a bug.

Goal: turn a finished electronic track into a per-track event list (grid, sections, drum
hits, pitched notes, per-stem curves, aligned lyrics, vocal onsets) for syncing Fantasynth
visuals, and judge how reliable each layer is. Offline tooling. Fixed tempo, 4/4 and
DAW-made music are assumed.

## Layout

Code in the repo, state in the workspace. `workspace.py` resolves every path; no script
hard-codes one. The workspace is `AUDIO_EVENTS_WORKSPACE` when set, else
`<main checkout of this repo>/../../audio-events` if that folder already exists, else
`<main checkout of this repo>/../audio-events` (next to the clone).

| Path | What |
|---|---|
| `audio-events/scripts/` (repo) | One script per stage. Each takes `--slug <slug>` or `--all` |
| `audio-events/run_all.py` (repo) | Runs the stages in order, each in its venv |
| `audio-events/config.example.json` (repo) | Template for the workspace `config.json` |
| `<workspace>/config.json` | Per-machine and per-track settings, below. Never committed |
| `<workspace>/audio/<slug>.wav` | Input, 44.1 kHz stereo s16, from `run_all.py`'s decode stage |
| `<workspace>/venv-main/` | Python 3.13, torch 2.11.0+cu128. Beat This!, all-in-one-infer, audio-separator, ADTOF-pytorch, librosa, torchaudio (lyrics) |
| `<workspace>/venv-bp/` | Python 3.10, basic-pitch 0.4.0 (ONNX backend, CPU) |
| `<workspace>/models/` | Model checkpoints (audio-separator `--model_file_dir`, etc.) |
| `<workspace>/models/torch/` | `TORCH_HOME` of the lyrics stage: torchaudio `MMS_FA` + `WAV2VEC2_ASR_LARGE_LV60K_960H`, about 2.4 GB, downloaded on first use |
| `<workspace>/out/<slug>/` | All outputs for a track (below) |

`setup.ps1` builds both venvs from `requirements-main.txt` / `requirements-bp.txt`.

### `config.json`

Everything that names a track or a machine lives here, outside the repo. Copy
`config.example.json` into the workspace as `config.json` and edit it.

```jsonc
{
  "app": "C:/path/to/fantasynth",       // the Fantasynth checkout export_app.py writes into
  "musicDir": "C:/path/to/music",       // optional: where the source audio lives
  "tracks": {
    "track-a": {
      "file": "Track A.mp3",            // the audio file in musicDir
      "appTrack": "track_a",            // public/music/<appTrack>.json in the checkout
      "title": "Track A",               // display title, used when export_app.py creates the track file
      "bpm": 120, "gridMarker": 0.5,    // optional: known tempo and beat-0 time, for a diagnostic only
      "handTimed": "fitted by hand",    // optional: the lyrics stage must not align this track
      "pron": { "DJ": "dee jay" },      // optional: pronunciation fixes, {"<token>": "sub words"}
      "all": false                      // optional: leave this track out of --all (default true)
    }
  }
}
```

- Every key is optional. An unknown key or a value of the wrong type stops the run (a
  misspelt key would otherwise be a silent no-op); keys starting with `_` are ignored.
- `tracks` keys are the slugs. `--all` means every slug listed here except those with
  `"all": false`, in every stage script (`--slug` still reaches an excluded track);
  `export_app.py --all` takes the slugs with an `appTrack`, `measure/seek_formats_prep.py --all`
  those with a `file`. `report.py --all` is the exception: every analysed slug in `out/`.
- The music folder is `LOCAL_MUSIC_PATH` when set, else `musicDir`, else `~/Music/Live`.
- `bpm` / `gridMarker` are compared against the fitted grid in `grid_diag.json`. They are
  never used for fitting.
- `handTimed` and `pron` belong to the track file `appTrack` names. `pron` maps a display
  token (exactly, or case-folded without edge punctuation) to its spoken sub-words.
- `handTimed`: the reason string. A listed track is not aligned: the lyrics stage takes its
  word times verbatim from the track file (below), and `export_app.py --lyrics` never writes it.

The Fantasynth checkout is resolved only by the commands that read or write it
(`export_app.py`, `scripts/lyrics.py` when it reads a track file's lyric text, and
`run_all.py` when its `lyrics` stage will, so a bad path fails before the first stage):
`--app <path>`, else `FANTASYNTH_APP`, else `config.app`, else an error. A path without
`public/music/index.json` is an error. `run_all.py --app` passes the flag through. Importing
any module and running the tests never needs the checkout.

## Stage outputs (`<workspace>/out/<slug>/`)

All times are **seconds from the start of the WAV** in raw outputs. Only `events.json`
(the final assembly) converts to beats.

| File | Producer | Content |
|---|---|---|
| `raw_beatthis.json` | grid | `{beats:[s], downbeats:[s]}` from Beat This! `final0`, default (non-DBN) postprocessing |
| `raw_allin1.json` | grid | all-in-one-infer result: `{bpm, beats, downbeats, beat_positions, segments:[{start,end,label}]}` |
| `grid.json` | grid | fitted grid, below |
| `grid_diag.json` | grid | fit diagnostics: the fit, the integer-snap verdicts, the comparison with `bpm` / `gridMarker` from `config.json` when given |
| `stems/<stem>.wav` | separate | BS-Roformer-SW stems: `vocals, bass, drums, guitar, piano, other` |
| `stems/drums_<part>.wav` | separate | MDX23C DrumSep run on `stems/drums.wav`: `kick, snare, toms, hh, ride, crash` (DrumSep's own part names; `hh` is exported as the curve stem `hats`) |
| `raw_adtof.json` | drums | ADTOF-pytorch on `stems/drums.wav`: `{meta, classes:{<name>:[{t, vel}]}}` with names `kick, snare, tom, hihat, cymbal` (ADTOF's class names; the `tom` class fills the `toms` lane) |
| `raw_adtof_mix.json` | drums | the same on `audio/<slug>.wav` (the full mix) |
| `raw_drumsep_onsets.json` | drums | onset times per DrumSep sub-stem: `{meta, <part>:[{t, strength, db, dom}]}`: `db` = peak RMS in the 50 ms after the onset, `dom` = `db` minus the loudest other sub-stem there (the dominance the solo-DrumSep rules read) |
| `raw_notes_<stem>.json` | notes | Basic Pitch per stem: `{notes:[{t, dur, pitch, vel}]}` |
| `curves.npz` | curves | per stem arrays at 100 fps: `<stem>_rms_db`, `<stem>_centroid`, `<stem>_onset`, plus `fps`. `<stem>` is the stem FILE name, so the hi-hat stem is `hh` here |
| `raw_lyrics.json` | lyrics | aligned lyrics + vocal onsets, below. Always written, even for an instrumental |
| `lyrics_prior.json` | lyrics | the line-time prior, FROZEN on the first run: `{track, text: [line text], times: [[start, end] s]}`. Reused while the lyric text is unchanged, refrozen from the track file when it changes. Never regenerate it from a track file that already holds exported (aligned) lyrics |
| `lyrics_em/<model>_<source>.npz` | lyrics | cached CTC emissions, `model` in `mms, lv60k`, `source` in `mono, left, right`: `{em: [frames, V] log-probs, labels}` at 20 ms frames. Recomputed when `stems/vocals.wav` is newer |
| `events.json` | assemble | the events file, below |
| `check_*.mp3` | render | listening checks: original + clicks / synthesised drums / synthesised bass; vocals stem + word ticks (`check_lyrics_split`) / onset ticks (`check_vocal_onsets_split`) |
| `report.json` | report | the report card, incl. `lyrics` and `vocalOnsets` layers |

### `grid.json`

```json
{
  "bpm": 120.0, "bpmRaw": 119.98, "t0": 0.5,
  "beatPeriod": 0.5, "meter": 4, "downbeatPhase": 0,
  "nBeats": 400,
  "fit": { "residualMsMedian": 3.0, "residualMsP95": 10.0, "inliers": 390, "outliers": 5 },
  "downbeatVotes": { "beatthis": [90, 4, 2, 0], "allin1": [92, 3, 1, 0] },
  "sources": { "beatthis": 398, "allin1": 401 }
}
```

- `t0` = time of beat index 0 of the fitted grid, **chosen so that beat 0 is a downbeat**
  (after applying `downbeatPhase`). It may be negative if the music starts mid-bar.
- Beat i is at `t0 + i * beatPeriod`. Bar b starts at beat `4*b`.
- `bpm` is `bpmRaw` snapped to the nearest 0.01, or to the nearest integer when the
  integer fit's residual is not worse by more than 1 ms median (DAW tracks usually are
  integer; the default rule also accepts the integer when an audio comb finds the integer
  grid at least as sharp).
- `grid.json` also carries `"sections": [{ "startBar", "endBar", "label", "startRaw", "endRaw", "snapMs" }]`:
  all-in-one-infer segments with boundaries snapped to the nearest bar line of the fitted
  grid (`endBar` exclusive), the raw times kept, and `snapMs` = worst boundary shift. Labels
  are allin1's own (pop vocabulary); EDM relabelling happens later in `assemble`.

### `raw_lyrics.json`

```json
{
  "version": 1, "track": "example", "text": null,
  "models": ["MMS_FA", "WAV2VEC2_ASR_LARGE_LV60K_960H"], "sources": ["mono", "left", "right"],
  "margin": 1.5, "levelDbfs": -18.0, "prior": "track line times +-3.0s",
  "lines": [
    { "text": "la la", "ref": 10.0, "free": 40.0,
      "words": [
        { "w": "la", "t": 10.2, "d": 0.3, "conf": 0.8, "agree": 0.75, "rule": "onset-snap",
          "ctc": [10.25, 0.2] },
        { "w": "la", "t": 10.5, "d": 0.4, "conf": 0.9, "agree": 1.0, "rule": "ctc",
          "ctc": [10.5, 0.4] } ] } ],
  "extras": [ { "t": 60.0, "d": 2.0 } ],
  "onsets": [ { "t": 10.2, "s": 0.7, "kind": "flux" }, { "t": 12.0, "s": 0.35, "kind": "pitch" } ],
  "stats": { "words": 100, "tokens": 100, "lines": 20, "lowConf": 10, "medianConf": 0.8,
             "priorMovedLines": 2, "extrasSeconds": 5.0, "onsets": { "flux": 300, "pitch": 100 }, "...": "..." }
}
```

- Seconds from the start of the WAV, like every raw output. `track` is the
  `public/music/<track>.json` the text came from (null for `--text <file>`, which sets `text`).
- `t` / `d`: the refined span; `d > 0`, words ordered and non-overlapping. `d` comes from the
  rounded end minus the rounded start, so words that tile keep tiling. `ctc`: the raw CTC span
  `[t, d]` before refinement. `rule`: what set the start (`ctc`, `rest-onset`, `onset-snap`,
  `fricative`). `conf`: 0..1, `agree`: share of the four alternative alignments within 60 ms.
  `syl`: `[[t, d], ...]`, only when the token has more than one pronunciation sub-word.
- `ref`: the line's start in the frozen prior (`lyrics_prior.json`). `free`: present only when
  the unconstrained alignment put the line more than 1 s away from where the prior did: review
  those. `prior` is null when there were no line times, `--window 0`, or the windows admitted
  no path (the stage then warns and aligns freely).
- **Hand-timed** (the track has `handTimed` in `config.json`, and no `--text`): nothing is
  aligned. `lines` are the track file's own lines with their words' `w`, `t`, `d` verbatim
  (seconds, not rounded); rest words (`{"w": ""}`) and lines left empty are dropped. Words
  have ONLY `w`, `t`, `d` (no `conf`, `agree`, `rule`, `ctc`, `syl`); lines have no
  `ref`/`free`. `source` = `"hand-timed (public/music/<track>.json)"`, `handTimed` = the
  reason, `prior`/`models`/`sources`/`margin` = null, no `lyrics_prior.json` is written.
  `onsets` and `extras` are computed as for aligned words (`extras` against these words' spans).
- Words with no pronounceable letters are omitted; a line whose words were all omitted is
  omitted. `lines` / `extras` are `[]` without text; `onsets` is `[]` when the vocals stem is
  below -40 dBFS (the `notes` gate).
- A track whose lyric text should come from a track file that cannot be found is an error:
  the stage exits non-zero and writes nothing. Only an existing track file with no `lyrics`
  (an instrumental) produces empty `lines`.

## `events.json`

The file `assemble` writes, and (minus QA-only fields) the file Fantasynth reads.

Time is in **beats on the fitted grid**, and beat 0 is a downbeat. Seconds are
`gridMarker + b * 60 / bpm`. Arrays are columnar per lane, so they map straight onto typed
arrays.

```jsonc
{
  "version": 2,
  "source": { "audio": "audio/example.wav",
              "tools": { "beats": "...", "stems": "...", "drums": "...", "notes": "...", "lyrics": "..." } },
  "tempo": [{ "beat": 0, "bpm": 120 }],
  "gridMarker": 0.5,                   // seconds of beat 0
  "meter": 4,
  "lengthBeats": 400.0,
  "gridFit": { "residualMsMedian": 3.0, "residualMsP95": 10.0, "inliers": 390, "outliers": 5,
               "shiftToKicksMs": -10.0, "onBeatKicks": 200, "hatSwing": { } },
  "sections": [{ "startBeat": 0, "beats": 32, "label": "intro", "energy": 0.1,
                 "labelSource": "rule(kickFill=0.0, energy=0.1); allin1=intro" }],
  "drums": {
    "<kick|snare|toms|hatClosed|hatOpen|crash|ride>":
      { "b": [32, 33], "v": [0.9, 0.8], "c": [1, 0.67], "r": [1.2, 0], "swing": 0, "offGrid": 0, "source": "..." }
  },
  "notes": {
    "<bass|other|vocals|piano|guitar>":
      { "b": [32, 34], "len": [1.5, 0.5], "pitch": [36, 43], "v": [0.7, 0.6], "r": [2.0, 1.1], "swing": 0 }
  },
  "curves": {
    "samplesPerBeat": 24, "encoding": "u8-base64-p5p95",
    "stems": { "<mix|drums|bass|vocals|other|piano|guitar>": { "rms_db": "base64", "centroid": "base64", "onset": "base64" },
               "<kick|snare|toms|hats|ride|crash>": { "rms_db": "base64" } }
  },
  "lyrics": {                          // only when the lyrics stage placed some words
    "lines":  { "b": [40.0], "len": [6.0], "text": ["la la"] },
    "words":  { "b": [40.0, 40.6], "len": [0.6, 0.8], "v": [0.8, 0.9], "w": ["la", "la"], "line": [0, 0], "syl": [null, null] },
    "extras": { "b": [120.0], "len": [4.0] },
    "source": "forced alignment (MMS_FA + wav2vec2 LV60K, CTC, mono/L/R mixture)"
  },
  "vocalOnsets": { "b": [40.0, 44.0], "v": [0.7, 0.35], "kind": [0, 1] }   // only when the vocals stem is not silent
}
```

### Fields

| Field | Meaning |
|---|---|
| `version` | `2`. Version 2 renamed the drum lane `tom` to `toms` and the curve stem `hh` to `hats`, so the file uses Fantasynth's lane vocabulary. A reader should accept the version 1 spellings `drums.tom` and `curves.stems.hh` as aliases. `hatClosed` and `hatOpen` are finer than a single `hats` lane and stay separate in the file |
| `source` | provenance: the input WAV and one string per tool layer. `source.tools.lyrics` repeats `lyrics.source` |
| `tempo` | `[{beat, bpm}]`. One entry today (a rigid fit); a list so tempo changes can be added without a version bump |
| `gridMarker` | seconds of beat 0, which is a downbeat |
| `meter` | beats per bar, `4` |
| `lengthBeats` | the audio's length in beats |
| `gridFit` | diagnostics, kept in the exported copy: the grid fit's residuals and inliers, the shift applied to put `t0` on the kick attack (`shiftToKicksMs`, from `onBeatKicks` kicks), and the hat lane's swing statistics (`hatSwing`) |
| `sections[]` | `startBeat`, `beats`, `label` (`intro`, `build`, `drop`, `breakdown`, `groove`, `outro`), `energy` (0..1, drums+bass level, normalised per track), `labelSource` (how the label was reached; ends in `; energy split` for a fragment cut at a 4-bar mark and `; lyrics entry bar N` for a boundary moved onto a vocal entry). Sections tile the track from beat 0 |
| `b` | beat position. Drums and notes are snapped to the nearest 16th (straight or swung slot); a hit further than a third of a 16th from both keeps its raw position. Lyrics and vocal onsets are never snapped |
| `v` | velocity 0..1: ADTOF activation, or DrumSep level for a DrumSep-only hit; Basic Pitch velocity for notes |
| `c` | share of the drum detectors that agreed: of three (`0.33` means DrumSep alone), except `toms`, whose two votes are the two ADTOF passes |
| `r` | QA: snap residual in ms, `0` for an off-grid hit |
| `swing` | beats added to the 2nd and 4th 16th of each beat, for that lane |
| `offGrid`, lane `source` | QA: number of hits left off the grid; the lane's provenance |
| `len`, `pitch` | note length in beats; MIDI pitch |
| `curves` | `samplesPerBeat` samples per beat: sample k sits at beat k / 24. `encoding` `u8-base64-p5p95`: each feature is normalised per track from its 5th to its 95th percentile, clipped to 0..1, scaled to 0..255, stored as base64 bytes. Full-band stems carry `rms_db`, `centroid` and `onset`; drum sub-stems carry `rms_db` |
| `lyrics` | additive (`version` stays 2), omitted when empty. `lines`: one entry per sung line, `b` = its first word's start, `len` = to its last word's end, `text` = the line's display tokens joined by spaces. `words`: `w` = the display token exactly as the text has it (punctuation and case kept), `v` = alignment confidence 0..1, `line` = index into `lines`, `syl` = `null`, or `[[b, len], ...]` when the token was spelled as several sub-words (an acronym in `pron`, a number). `extras`: vocal activity no word covers (ad-libs, backing pads, repeats the text does not list), each at least 0.5 s. `source`: the aligner, or `hand-timed (public/music/<track>.json)` |
| `vocalOnsets` | additive, omitted when the vocals stem is silent. `v` = strength 0..1, `kind` 0 = spectral-flux onset, 1 = pitch jump of more than 0.8 semitone while voiced (a legato note change with little flux; fixed strength 0.35) |

Rules a reader can rely on:

- Lyrics and vocal onsets are converted to beats with the shifted grid (after the kick shift)
  and rounded to 3 decimals. `len` = rounded end minus rounded start, so words that tile in
  seconds still tile in beats.
- Aligned words are ordered, never overlap, and have `len > 0`. Hand-timed words are
  whatever the track file says, and may overlap slightly.
- `words.v` is present when every raw word has `conf` and absent when none has (hand-timed;
  a reader should treat a missing `v` as 1). A mix stops `assemble`.
- `lyrics.lines` has no `v`: every line start counts as 1.

| Events file | Comes from | Fantasynth lane |
|---|---|---|
| `drums.kick`, `drums.snare`, `drums.crash`, `drums.ride` | the same names upstream | one lane each |
| `drums.toms` | ADTOF's `tom` class | `toms` |
| `drums.hatClosed`, `drums.hatOpen` | one hat lane, split by decay | both merge into `hats` |
| `curves.stems.hats` | the DrumSep stem file `drums_hh.wav` (`hh` in `curves.npz`) | the `hats` curve |
| `lyrics.words` | `raw_lyrics.json` lines[].words[] (velocity = `v`, the confidence) | `words` |
| `lyrics.lines` | `raw_lyrics.json` lines[] | `lines` |
| `vocalOnsets` | `raw_lyrics.json` onsets[] (velocity = `v`, the strength) | `vocalOnsets` |

Only the events file is renamed: the stem WAV names and `DRUM_PARTS` stay DrumSep's, so
renaming a lane never forces a re-separation. `assemble.CURVE_STEM_RENAME` is the one place
the curve-stem rename happens, on the read side, so it applies to a cached `curves.npz`
without recomputing it.

## Into Fantasynth (`export_app.py`)

`export_app.py` writes into a Fantasynth checkout (resolved as above):

1. `public/music/events/<track>.events.json`: `events.json` as compact JSON, with the QA-only
   lane fields (`r`, `source`, `offGrid`) dropped and `exportedAt` added. `lyrics` and
   `vocalOnsets` keep every field they have today. The kept fields are `TOP_KEYS`,
   `LANE_KEEP`, `LYRICS_KEEP` and `VOCAL_ONSETS_KEEP`; an unknown top-level key, or an
   unknown field in a drum or note lane, `lyrics` or `vocalOnsets`, stops the export. The
   other top-level keys (`source`, `gridFit`, `sections`, `curves`, ...) are copied as they are. When
   nothing but `exportedAt` would change, the file is left alone, so a re-run is a no-op.
2. `public/music/<track>.json`: top-level `bpm` and `gridMarker` rewritten and `"events":
   "events/<track>.events.json"` added, by minimal text edits (the file is never
   re-serialised, so hand formatting and lyrics keep their bytes). The keys it may change are
   `MANAGED` (`url, bpm, gridMarker, events, sections, lyrics`); every other key is verified
   unchanged after the edit. A missing track file is created from `config.json`'s `title`
   and `file` (`url` = `/local-music/<file>`).
3. `"sections"` in the track file, only when it has none (or with `--force-sections`):
   `[{startBeat, beatLength, sectionType, label, params: {}}]`, with `sectionType` from
   `SECTION_TYPES` = `{intro: 0, build: 1, drop: 2, breakdown: 3, groove: 4, outro: 5}`.
   This table is fixed: never renumber, add new labels at the end.
4. `public/music/index.json`: the track file is added when missing, in title order.
5. With `--flac`: a 16-bit FLAC next to the source MP3 in the music folder (never
   overwriting an existing file), checked to decode sample-identically to
   `<workspace>/audio/<slug>.wav`, and the track's `url` switched to it.
6. With `--lyrics`: the track file's inline `lyrics` replaced by the aligned ones, in
   Fantasynth's format `{"lines": [{"words": [{"t", "d", "w", "c"}]}]}`: `t`/`d` in seconds
   from the EVENTS file's grid at full precision, rounded to 3 decimals (`d` never below
   0.001), `w` the display token, `c` the confidence (2 decimals; absent with no `words.v`).
   A gap of more than 2 s between lines (`REST_S`) and the tail to the song's end become an
   empty line `{"t", "d", "w": ""}` (the rest convention of hand-made track files: a line
   stays on screen until the next one starts). It refuses a slug whose events file has no
   `lyrics` block, before writing anything (under `--all` such a slug is skipped by name); a track file without inline lyrics gets them
   appended. A track with `handTimed` is never written: `--lyrics` skips it (also under
   `--all`, and before the refusal check), prints why, and the rest of the export runs.

## Rules

- Never commit audio, stems or anything under the workspace, including `config.json`.
  Never modify `audio/` except through the decode stage. Write only into your own stage's
  files.
- Ownership: `raw_lyrics.json` and `lyrics_em/` belong to the `lyrics` stage; `assemble` only
  reads them. The inline `lyrics` of a track file are hand-authored or exported, and only
  `export_app.py --lyrics` may rewrite them, never for a hand-timed track (there they are
  the SOURCE of the lyrics stage's times, not a copy of its output). The lyrics stage READS
  that text (and, on its first run only, its line times, frozen into `lyrics_prior.json`)
  and never writes the track file. After an export the track's times are the aligner's own,
  so they must never feed the prior again.
- No installs into `venv-main`/`venv-bp` unless your task says so; if a package is missing,
  report it instead of installing. To change the environment, edit `requirements-*.txt` and
  re-run `setup.ps1`.
- **If you do install into `venv-main` by hand, always pass**
  `-c audio-events\constraints-main.txt --extra-index-url https://download.pytorch.org/whl/cu128`.
  A bare install once replaced CUDA torch with torch 2.14 **CPU** via a torchvision
  dependency. After any install, confirm `torch.cuda.is_available()` is still True.
- The GPU can be shared: stages from different tracks can run concurrently when there is
  enough VRAM (tested on an RTX 5090, 32 GB).
