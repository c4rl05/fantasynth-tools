# audio-events

A finished track goes in. A per-track event list comes out: beat grid, sections, drum hits,
pitched notes, per-stem energy curves, word-aligned lyrics and vocal onsets
(`<workspace>/out/<slug>/events.json`). Fantasynth reads it to sync visuals to the music.
Everything runs offline, on a CUDA GPU (tested on an RTX 5090).

![The audio-events viewer](docs/viewer.png)

*The viewer on a test track: sections, sung words (text hidden here), vocal onsets, drum lanes, notes and stem curves, with the track renamed.*

Fixed tempo, 4/4 and DAW-made electronic music are assumed. The file format is specified in
[CONTRACT.md](CONTRACT.md). The full write-up, with the measurements behind each rule, is
[https://c4rl05.github.io/fantasynth-tools/audio-events/](https://c4rl05.github.io/fantasynth-tools/audio-events/) (source: [docs/index.html](docs/index.html)).

**Licences.** This repo's MIT licence covers its code only. The default pipeline downloads
models whose licences are non-commercial or not stated, so it is **not suitable for
commercial use** as shipped. See [Model weights and licences](#model-weights-and-licences).

## Code here, state in the workspace

The **code** lives in this repo, in `audio-events/`. Everything heavy lives in one
**workspace** folder outside the repo, shared by every checkout and worktree:

| Workspace path | What |
|---|---|
| `config.json` | your tracks and machine settings (below). Never committed |
| `venv-main/` | Python 3.13 + CUDA 12.8 torch: every stage except `notes` |
| `venv-bp/` | Python 3.10 + Basic Pitch on ONNX: the `notes` stage |
| `models/` | separator checkpoints (audio-separator); `models/torch/` holds the two lyric aligners (`TORCH_HOME` of the `lyrics` stage) |
| `audio/<slug>.wav` | decoded input. Never commit audio |
| `out/<slug>/` | stems, raw detections, curves, `events.json`, listening checks, `report.json` |
| `seektest/` | only while re-running the seek measurement (`measure/` creates it); safe to delete |

`workspace.py` resolves every path. The workspace is `AUDIO_EVENTS_WORKSPACE` when set, else
`<main checkout of this repo>/../../audio-events` if that folder already exists, else
`<main checkout of this repo>/../audio-events` (a folder next to your clone). Worktrees resolve
to the same folder, because the main checkout is found through git's common dir. Setting
`AUDIO_EVENTS_WORKSPACE` explicitly is the simplest choice for a new machine. Run
`python audio-events\workspace.py` to print every resolved path.

| Repo path (`audio-events/`) | What |
|---|---|
| `run_all.py` | the whole pipeline on one track |
| `scripts/*.py` | the stage scripts (`grid`, `separate`, `drums`, `notes`, `curves`, `lyrics`, `assemble`, `render_check`), each taking `--slug <slug>` or `--all`, plus `lyrics_align.py`, a library the `lyrics` stage imports. The `decode` stage has no script: `run_all.py` runs ffmpeg itself |
| `report.py` | the report card |
| `export_app.py` | `events.json` into a Fantasynth checkout's `public/music/` (the instrument's own file format; only useful with a Fantasynth checkout) |
| `config.example.json` | template for the workspace `config.json` |
| `serve.py`, `viewer.html` | the event viewer |
| `measure/` | media-element seek accuracy per audio format |
| `setup.ps1`, `requirements-*.txt`, `constraints-main.txt` | the two venvs |
| `workspace.py` | path and config resolution (stdlib only, works under both venvs) |
| `tests/` | unit tests (`unittest`) |

### `config.json`

Everything that names a track or a machine lives in `<workspace>/config.json`, never in the
repo. Start from the template:

```
copy audio-events\config.example.json <ws>\config.json
```

```jsonc
{
  "app": "C:/path/to/fantasynth",       // the Fantasynth checkout export_app.py writes into
  "musicDir": "C:/path/to/music",       // optional: where the source audio lives
  "tracks": {
    "my-track": {
      "file": "My Track.mp3",           // the audio file in musicDir
      "appTrack": "my_track",           // public/music/<appTrack>.json in the checkout
      "title": "My Track",              // display title, used when export_app.py creates the track file
      "bpm": 120, "gridMarker": 0.5,    // optional: known tempo and beat-0 time, diagnostic only
      "handTimed": "fitted by hand",    // optional: do not align this track's lyrics
      "pron": { "DJ": "dee jay" },      // optional: pronunciation fixes for the lyrics stage
      "all": false                      // optional: leave this track out of --all (default true)
    }
  }
}
```

- Every key is optional. An unknown key or a value of the wrong type is an error, so a
  misspelt key never passes silently; keys starting with `_` are ignored (room for notes).
- `--all` means every slug in `tracks` except those with `"all": false`, in every stage
  script; `--slug` still reaches an excluded track. `export_app.py --all` takes
  the slugs that have an `appTrack`, `measure/seek_formats_prep.py --all` those with a `file`.
  `report.py --all` is the exception: every analysed slug in `out/`.
- The music folder is `LOCAL_MUSIC_PATH` when set, else `musicDir`, else `~/Music/Live`.
- `appTrack` is the default `--track` for the `lyrics` stage and `export_app.py`.
  `handTimed` and `pron` apply to the track file that `appTrack` names.
- The Fantasynth checkout is only needed by `export_app.py`, by the `lyrics` stage when it
  reads a track file's lyric text, and by `run_all.py` when that stage will run (it checks
  the track file up front). It is `--app <path>`, else `FANTASYNTH_APP`, else
  `config.app`, else an error. A folder without `public/music/index.json` is refused.
  Fantasynth's source is not part of this repo; without a checkout, `events.json` and
  `--text` lyrics still work.

## Rebuilding the workspace

Nothing in the workspace is irreplaceable except `config.json`; every other folder is
rebuilt by code here. Two inputs live outside the repo: the source music, and third-party
download links.

| Workspace path | Size | Rebuilt by | Depends on |
|---|---|---|---|
| `venv-main/`, `venv-bp/` | several GB | `setup.ps1` + the pinned requirements | Python 3.13 and 3.10 installed (`py` launcher); the pins still on PyPI and the PyTorch cu128 index. Proven by a real install into an empty workspace, which reproduced `events.json` byte for byte. |
| `models/` | 1.1 GB | the `separate` stage, on first run | python-audio-separator's GitHub `model-configs` release (`BS-Roformer-SW.ckpt`, `MDX23C-DrumSep-aufr33-jarredou.ckpt` and their `.yaml`). **The one fragile input**: DrumSep's original release already went 404 once. Keep a backup of this folder outside the repo; the weights can't go in git (size, and unknown or non-commercial licences, below). |
| `models/torch/` | 2.4 GB | the `lyrics` stage, on first run | torchaudio's download of `MMS_FA` and `WAV2VEC2_ASR_LARGE_LV60K_960H` (pytorch.org). Only needed for tracks with lyric text. |
| `audio/` | about 40 MB per track | the `decode` stage | the source files, never in the repo |
| `out/` | about 1 GB per track | every stage of `run_all.py` | the steps above |

Also rebuilt on demand: the Beat This! and all-in-one weights in `~/.cache`, the FLAC copies
next to the MP3s (`export_app.py --flac`) and `seektest/` (`measure/`).

**Not quite deterministic.** all-in-one-infer varies from run to run on the same install:
beats by one 10 ms frame, a section boundary by about 0.2 s, and the downbeat vote shifts.
On the test tracks the rigid grid fit and bar snapping absorbed it. A track with a weak
downbeat could still come out with a different section or bar phase, so read its report
card after a re-run.

## Setup

Windows today: `setup.ps1` and the venv paths (`Scripts\python.exe`) assume it. The stage
scripts themselves are plain Python. From the root of this repo:

```
powershell -ExecutionPolicy Bypass -File audio-events\setup.ps1 -DryRun   # print what it would do
powershell -ExecutionPolicy Bypass -File audio-events\setup.ps1
```

It needs the `py` launcher with Python 3.13 and 3.10, git, and ffmpeg/ffprobe on PATH. It
creates `venv-main` and `venv-bp` in the workspace, or refreshes them to the pinned versions
when they exist. Then it checks that torch sees CUDA (and prints the device capability) and
that `basic_pitch` + `onnxruntime` import. It is idempotent and never deletes anything: a
venv with the wrong Python version stops it with a message.

## Run

From the root of this repo, with `<ws>` the workspace:

```
<ws>\venv-main\Scripts\python.exe audio-events\run_all.py "C:\path\to\music\My Track.mp3" --slug my-track
<ws>\venv-main\Scripts\python.exe audio-events\run_all.py "...mp3" --slug my-track --from assemble   # re-run later stages only
<ws>\venv-main\Scripts\python.exe audio-events\run_all.py "...mp3" --slug my-track --track my_track --app C:\path\to\fantasynth
<ws>\venv-main\Scripts\python.exe audio-events\serve.py                         # then http://127.0.0.1:8765/viewer.html?slug=my-track
<ws>\venv-main\Scripts\python.exe audio-events\export_app.py --slug my-track --track my_track
```

About 1.5 minutes per track, mostly all-in-one-infer. `run_all.py` runs each stage in its
own venv (`notes` in `venv-bp`, the rest in `venv-main`). `--track` names the track file
holding the lyric text (default: the slug's `appTrack`); `--app` is passed through to the
`lyrics` stage. When a track is known, `run_all.py` resolves the checkout and checks the
track file before the first stage runs, so a wrong path fails in a second, not after the
GPU stages.

Tests, from `audio-events/`:

```
<ws>\venv-main\Scripts\python.exe -m unittest discover -s tests
```

They use synthetic fixtures only: no audio, no models, no Fantasynth checkout.

### Viewer

`serve.py` serves `viewer.html` from `audio-events/` and `/out/...` and `/audio/...` from
the workspace, with HTTP Range support (Python's `http.server` has none, and `<audio>` stalls
without it). It binds to 127.0.0.1, port 8765 (`serve.py 8766` picks another). The viewer
draws the sections, drum lanes, piano roll and stem curves over a playhead. Its track menu
lists every slug in `out/` with an `events.json` (`serve.py` serves the list as
`/slugs.json`); `?slug=` selects one. Its audio menu switches between the original and the
listening checks:

| File | What you hear |
|---|---|
| `check_grid.mp3` | metronome, with a high beep on downbeats and a chime at section starts |
| `check_drums_split.mp3` | LEFT = original, RIGHT = drums synthesised from the events. On headphones a flam between the ears is a misplaced hit |
| `check_drums_mix.mp3` | the original ducked, with the synthesised drums on top |
| `check_bass_split.mp3` | LEFT = original, RIGHT = bass notes synthesised |
| `check_lyrics_split.mp3` | LEFT = vocals stem, RIGHT = a 40 ms tick at every aligned word start: 2 kHz, 3 kHz and louder on a line start, 700 Hz for a word with confidence below 0.6, soft ticks on the syllables inside a spelled word. A tick early or late against the word's first consonant is a timing error |
| `check_vocal_onsets_split.mp3` | LEFT = vocals stem, RIGHT = vocal onsets: 1.2 kHz for a flux onset, 600 Hz for a pitch jump |

The two vocal checks are written only when `events.json` has `lyrics` / `vocalOnsets`. The
viewer's audio menu does not list them yet: open `/out/<slug>/check_lyrics_split.mp3` from
`serve.py` directly. The viewer draws the aligned words (coloured by confidence, red low to
green high, text inside when there is room, a white bar at each line start, a faint band for
`extras`) and a vocal onset row (yellow flux, pink pitch jump, height by strength) above the
drum lanes. Chrome defers media loading in hidden tabs, so open the viewer in a normal tab.

### Export into Fantasynth

`export_app.py` writes the Fantasynth instrument's own music-file format, so it is only
useful alongside a Fantasynth checkout. Everything before it (the pipeline, `events.json`,
the report and the viewer) stands on its own.

It writes `public/music/events/<track>.events.json` into the Fantasynth
checkout (`--app`, `FANTASYNTH_APP` or `config.app`), points `public/music/<track>.json` at
it with minimal text edits, and adds sections only where the track has none. A re-run with
unchanged events is a no-op. `--flac` also writes a sample-exact FLAC next to the track's
MP3 in the music folder and switches the track's `url` to it; see
[Seek measurement](#seek-measurement-measure). With `--all --flac` that happens for every
slug with an `appTrack` whose track still points at an MP3. CONTRACT.md lists every key it
may touch.

`--lyrics` (opt-in) also replaces the track file's inline `"lyrics"` with the aligned ones:

```
<ws>\venv-main\Scripts\python.exe audio-events\export_app.py --slug my-track --lyrics
<ws>\venv-main\Scripts\python.exe audio-events\export_app.py --all --lyrics   # skips, by name, every slug without a lyrics block
```

It writes Fantasynth's lyric format, one entry per word, grouped into the lines the text
came in: `{"lines": [{"words": [{"t", "d", "w", "c"}]}]}`, with `t`/`d` in seconds of the
audio (`gridMarker + b * 60 / bpm` of the EVENTS file at full precision, rounded to 3
decimals; `d` never below 0.001), `w` the display token exactly as the text had it, and `c`
the alignment confidence (2 decimals). A gap of more than 2 s between lines (`REST_S`), and
the tail to the song's end, is written as an empty line `{"t", "d", "w": ""}`: the rest
convention of hand-made track files, because a line stays on screen until the next one
starts. It is a spliced text edit like the others, so the rest of the file keeps its bytes
and every other key is verified unchanged. It replaces the track's lyrics wholesale, with
one exception: a track with `handTimed` in `config.json` is **skipped**, with or without
`--all`, and the export says why (the rest of the export still runs). Its hand-fitted lyrics
are the source the lyrics stage read, so writing the events back over them would only round
them. The one refusal: a slug whose EVENTS file has no `lyrics` block is refused before
anything is written. A track file with no inline lyrics is not refused; it gets them
appended, which is how a `--text` track gets its lyrics. Seconds, not beats, so a later
`gridMarker` edit does not move them.

## Pipeline

| Stage | Tool | Output |
|---|---|---|
| decode | ffmpeg | `audio/<slug>.wav`, 44.1 kHz stereo s16 |
| grid | Beat This! + all-in-one-infer 3.1.0 | rigid fixed-tempo grid, downbeat phase by vote, raw sections |
| separate | BS-Roformer-SW (6 stems) then MDX23C DrumSep (6 drum stems), via audio-separator | `stems/*.wav` |
| drums | ADTOF-pytorch on the drum stem AND on the full mix; onsets per DrumSep sub-stem | three independent detectors per class |
| notes | Basic Pitch per pitched stem (venv-bp, Python 3.10, ONNX) | notes per stem |
| curves | librosa RMS / centroid / onset strength, 100 fps | `curves.npz` |
| lyrics | forced alignment of the KNOWN lyric text: torchaudio `MMS_FA` + `WAV2VEC2_ASR_LARGE_LV60K_960H` CTC on the vocals stem (mono, L, R), one Viterbi pass, signal refinement (a `handTimed` track skips all this and keeps its hand timings); vocal onsets from flux + pitch jumps | `raw_lyrics.json` (seconds), `lyrics_em/` (emission cache) |
| assemble | voting, grid alignment, swing, snapping, sections (refined by lyric vocal entries) | `events.json` |
| render | ffmpeg | `check_*.mp3` |
| report | `report.py` | `report.json` + a printed report card |

`report.py` measures `events.json` against the stems and the mix, not against the raw
detections, so it is an independent check: grid offset from the kick attack, lane
precision/recall at ±30 ms plus timing, open/closed hat separation, curve alignment, and
section boundaries with and without a change in the audio, and, when there are lyrics, word
starts against onset evidence of its own on the vocals stem (below). Each layer gets a verdict
(solid / mostly / check); the thresholds are constants at the top of the file. Run it alone
with `report.py --slug <slug>` or `--all`. On the test tracks it reproduces the independent
verification numbers; on a copy with kicks moved 40 ms it drops the grid and kick verdicts
to check.

Rules in `scripts/assemble.py`. Each one is there because a measurement on a test track
called for it; the docstrings give the numbers.

- **Drum hits:**
  - A drum hit needs 2 of the 3 detectors to agree.
  - A snare or hat heard by DrumSep alone is kept when its sub-stem is clearly the loudest drum there. Build rolls need this.
  - A hat that coincides with a kick needs DrumSep's vote, because ADTOF reads some kicks as hats.
  - Toms are only kept when both ADTOF passes agree and no kick falls in the same window.
  - Cymbals are only kept when the drum-stem pass is one of the votes.
- **Grid:** `t0` is shifted onto the median on-beat kick attack. The beat trackers land about 5 to 20 ms after the kick.
- **Grid seed:** the fit is seeded twice, from a sequential beat-index chain and from a phase-coherence period scan (±3%), and the coherence fit is kept only with strictly MORE inliers. One test track needed it: Beat This! inter-beat intervals, quantised to 20 ms frames, seeded 120.1 bpm, the chain miscounted through a drumless intro and breakdown, and the fit settled at 119.65 bpm with 40 of 439 inliers (the kicks say 121.000; 291 inliers). The strict guard matters: on another track the coherence seed alone lands at 136.77 against a true 136.
- **Swing:** estimated per lane. On one test track the hats sat +36.5 ms late on the 2nd and 4th 16ths while its snares were straight.
- **Open vs closed hats:** a hat is open when the hi-hat stem takes at least 150 ms to fall 12 dB before the next hit.
- **Sections:**
  - allin1 boundaries are snapped to the 8-bar phrase grid, whose offset is found by vote.
  - A section is split at a 4-bar mark when kick presence or stem level clearly changes.
  - With lyrics, a boundary exactly one bar from a vocal entry moves onto it (after the phrase snap, before the splits). An entry is the first word of a line that starts at least 4 beats after the previous word ends, or the first line. Pickup rule: an entry in the first half of bar k starts the section at bar k, one in the second half leads into bar k + 1. The first and last boundary never move, no section may drop under 2 bars, and a moved section's `labelSource` ends in `; lyrics entry bar N`. On four aligned test tracks this moved one boundary, by one bar onto a vocal pickup; phrase-snapped EDM sections already sit on the entries.
  - Labels come from kick fill and drums+bass energy: intro / build / drop / breakdown / groove / outro.
- **Lyrics and vocal onsets:** converted to beats on the shifted grid and rounded to 3 decimals, but never snapped: a singer is not quantised. A span's `len` comes from its rounded END minus its rounded start, so legato words that tile in seconds still tile in beats (rounding `b` and `len` separately broke the tiling).

## Lyrics stage

`scripts/lyrics.py`, with the method in `scripts/lyrics_align.py` (ported from
mexicat/pdoom-video, MIT; see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)). The lyric
**text** is known; the stage only finds **when** each word is sung, so there is nothing to
mishear.

```
<ws>\venv-main\Scripts\python.exe audio-events\scripts\lyrics.py --slug my-track
<ws>\venv-main\Scripts\python.exe audio-events\scripts\lyrics.py --slug my-track --track my_track --app C:\path\to\fantasynth
<ws>\venv-main\Scripts\python.exe audio-events\scripts\lyrics.py --slug my-track --text lyrics.txt --window 0
```

- `--track <name>`: the text comes from `public/music/<name>.json` in the Fantasynth
  checkout, its inline `lyrics` (word- or line-level; only the words are used, and the line
  times as the prior). Default: the slug's `appTrack` in `config.json`. `run_all.py --track`
  passes it through. A track file that cannot be found is an error: the stage exits
  non-zero and writes nothing. With no track named at all (and no `--text`), the stage
  writes vocal onsets only.
- `--text <file>`: a plain text file, one lyric line per line. Overrides `--track`, needs no
  checkout, and has no line times, so no prior.
- `--window <s>` (default 3.0): the **line-time prior**. Every letter of line i must lie within
  the track file's own time for that line ± this many seconds; `0` turns it off. Lyrics
  repeat: without it, on one test track, lines inside a repeated hook drifted up to 15 s onto
  other repeats of the same words. With it, 71% of that track's line starts landed within
  500 ms (95% within 1 s) of its coarse line times, and the unlisted repeats landed in
  `extras`. The stage also runs the free alignment, prints how many lines the prior moved, and
  keeps the free start as `free` on any line it moved by more than 1 s, for review. When the
  windows admit no path at all, it warns and aligns without them.
- The prior is **frozen** on the first run in `out/<slug>/lyrics_prior.json` (track, text,
  times) and reused while the lyric text is unchanged; a changed text refreezes it from the
  track file. Without this, after `export_app.py --lyrics` the track file holds the aligner's
  own output, which would become its own prior: a misplaced line could never move again. The
  stage warns when the track's lyrics already carry `c` (aligner confidences) and no frozen
  prior exists. If that happens, seed the prior from the track file's line times as they were
  before the first export.
- `handTimed` (in `config.json`): a listed track is **not aligned**. Its inline lyrics were
  fitted by hand, so the stage takes each word's `t`/`d` (seconds) and the line grouping
  straight from the track file, verbatim, dropping only rest words (`{"w": ""}`). No prior is
  frozen, no model runs. Its words carry no `conf`, so `events.json` has no `lyrics.words.v`
  (a reader treats every word as 1). Vocal onsets and `extras` are computed exactly as for
  aligned words; `raw_lyrics.json` gets `source: "hand-timed (public/music/<track>.json)"` and
  `handTimed: "<why>"`, which assemble carries into `lyrics.source`. `--text` ignores it.
- `pron` (per track in `config.json`): pronunciation fixes, `{"AGI": "ay gee i", "DJ": "dee jay"}`.
  A key matches a display token exactly, or case-folded without edge punctuation. A spelling
  of several sub-words gives the word one syllable span per sub-word (`syl`). Digits are
  spelled out on their own ("1998" as nineteen ninety eight). A token with no pronounceable
  letter is left out of the alignment. No test track needed one.
- Models download on first use into `<workspace>/models/torch` (`TORCH_HOME`, unless you already set one), about 2.4 GB for
  the two. Emissions are cached per model and channel in `out/<slug>/lyrics_em/` (4 to 8 MB per
  track, six files) and recomputed when `stems/vocals.wav` is newer. On the test GPU: about
  7 s of emissions per track (37 s including the first model load), every alignment under 1 s.
- A track without text still gets vocal onsets. A vocals stem below -40 dBFS (the `notes`
  gate) gets nothing. `raw_lyrics.json` is always written.

Method, in order:

1. **Emissions**: both character-level CTC models, each mapped onto one alphabet (blank, a-z,
   `'`), on the vocals stem as mono, left and right (a double-tracked chorus is often panned,
   so one side is closer to a single voice). The six are fused as a probability MIXTURE, so
   one model's confident "no" cannot veto another's "yes".
2. **One Viterbi pass over the whole song.** A garbage token sits between lines and at both
   ends, scoring 1.5 nats below each frame's best symbol, so it absorbs ad-libs, backing
   vocals and anything else not in the text while lyric letters win wherever they match.
3. **Refinement** of each start, because CTC has known biases: rest-onset (the voice's
   re-entry after a rest, at most 0.5 s back: unbounded, it moved starts up to 3.2 s early on
   a test track, onto backing vocals and reverb tails), onset-snap to a flux peak (rejecting
   peaks followed by frication, which belong to the previous word), and a fricative walk-back.
   Ends follow the voice (15 dB below the word's level for 60 ms); gaps under 30 ms close
   (legato). When the voice never drops that far, the end is capped at the CTC end + 1 s
   (`END_TAIL`) instead of running on to the next word: uncapped, one word on a test track
   lasted 27 s.
   The flux onsets come from `onset_strength(..., n_fft=1024)`. It pads its output by
   lag + n_fft // (2 hop) frames assuming its own default n_fft of 2048, so without the
   argument every vocal flux onset was about 31 ms late (inherited from pdoom-video's
   `vocal_feats.py`). That moved `vocalOnsets` and every onset-snap word start.
4. **Confidence** per word: `0.4 + 0.4 x agreement + 0.2 x posterior` (the CTC posterior,
   saturating at 0.5). Agreement = the share of four other alignments (each model alone on
   mono; both models on the left channel alone, and on the right) whose raw start is within
   60 ms.
5. **Extras**: vocal activity no word covers. **Vocal onsets**: log-mel flux peaks on active
   voice, plus pitch jumps (> 0.8 semitone while voiced) with no flux onset within 80 ms.

The report card's LYRICS layer never reads the pipeline's own `vocalOnsets` (the aligner snaps
starts to flux peaks, so they would vouch for themselves). It builds its own evidence on
`stems/vocals.wav`: level rises (6 dB over 100 ms, dated at the half-rise) and spectral-flux
peaks (legato words that tile have no level rise). Word-start precision P = share of starts
within 60 ms of either. Evidence this dense is near most instants, so it also reports the
CHANCE rate (a start placed at random in the voice) and judges on kappa =
(P - chance) / (1 - chance). Line entries (a word 250 ms or more after the previous one) are
scored against level rises alone, where chance is low. Coverage = share of active voice inside
a word (padded 100 ms) or an extra. **solid**: P >= 0.85, kappa >= 0.50, coverage >= 90%;
**mostly**: 0.70 / 0.30 / 75%; **check** on any unordered, overlapping or zero-length word or a
silent stem. Thresholds are provisional (set before the first real alignment). VOCAL ONSETS is
`info` only: counts by kind, and the share of pitch jumps with no Basic Pitch vocal note start
within 50 ms. That share was 73 to 85% on the voiced test tracks: the jumps add timing Basic
Pitch lacks.

Results on four test tracks with lyric text: every word placed. Word-start P 0.79 to 0.90
against a chance rate of 0.61 to 0.71, kappa 0.33 to 0.68, median start error within 7 ms,
coverage 0.97 or better. One track read solid, three mostly. The weakest, a heavily effected
vocal, had median confidence 0.61: treat an alignment like that as provisional. On one track
that already had word timings from a WhisperX transcription, scored on the same words against
the same evidence: forced alignment P 0.896 / kappa 0.676 / line entries on a vocal rise
0.75 (solid), WhisperX P 0.775 / kappa 0.296 / entries 0.459 (check). The two agreed on 68%
of starts within 100 ms; the disagreements clustered in a repeated ad-lib passage.

## events.json

Time is in beats on the fitted grid, and beat 0 is a downbeat. Seconds are
`gridMarker + b * 60 / bpm`. The full specification, with every field, is in
[CONTRACT.md](CONTRACT.md#eventsjson).

## Seek measurement (`measure/`)

How far `<audio>.currentTime` is from the audio actually playing after a seek, per format.
It is why `export_app.py --flac` points tracks at a FLAC: Chromium reports an MP3 25 ms ahead
after a seek (encoder delay), while a 16-bit FLAC seeks sample-exactly. The number belongs to
Chromium's decoder, so it is worth re-running when Chrome updates. `measure/` has its own
`package.json` (Playwright). Run `npm install` in `audio-events\measure` once, then
`npx playwright install chromium` there unless you capture with `--channel chrome`. The
track comes from the slug's `file` in `config.json`.

```
<ws>\venv-main\Scripts\python.exe audio-events\measure\seek_formats_prep.py --slug my-track   # per-format copies into <ws>\seektest
node audio-events\measure\seek_formats.mjs --slug my-track --channel chrome     # capture
<ws>\venv-main\Scripts\python.exe audio-events\measure\seek_formats_analyze.py seektest\runs\my-track_chrome_paused
```

## Model weights and licences

This repo's MIT licence covers **only the code in this repo**. It does not cover the
models the pipeline runs, their weights, or the audio you feed it.

Weights are **downloaded or installed at runtime, never vendored**: into `~/.cache/torch`,
`~/.cache/huggingface`, `<workspace>/models/`, `<workspace>/models/torch/`, or inside the
pip packages in the venvs. Nothing in this repo contains them.

| Component | Stage | Code licence | Weights licence |
|---|---|---|---|
| [Beat This!](https://github.com/CPJKU/beat_this), checkpoint `final0` | grid | MIT | MIT: "The code and the published model weights are released under the MIT license." |
| [all-in-one-infer](https://github.com/openmirlab/all-in-one-infer) 3.1.0, checkpoint `harmonix-all` (from [mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one)) | grid | MIT | No licence text. The Hugging Face repo the weights come from ([taejunkim/allinone](https://huggingface.co/taejunkim/allinone)) carries only a `license: mit` metadata tag; nothing addresses the terms of the Harmonix Set it was trained on |
| [Demucs](https://github.com/facebookresearch/demucs) `htdemucs`, run inside all-in-one-infer (via demucs-infer) | grid | MIT | Not stated by Meta. Trained on MUSDB18 plus 800 songs; MUSDB18's own terms limit its tracks to academic use |
| [madmom](https://github.com/CPJKU/madmom) (as madmom-infer 0.2.0, inside all-in-one-infer) | grid | BSD-2-Clause | madmom's model files are CC BY-NC-SA 4.0. all-in-one-infer uses only madmom's spectrogram code and DBN decoder, which (inferred from its imports) load none of them |
| [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) 0.47.0 | separate | MIT | per model, below |
| `BS-Roformer-SW.ckpt` | separate | n/a | Not stated. Rehosted without a licence; the original trainer is not identified |
| `MDX23C-DrumSep-aufr33-jarredou.ckpt` | separate | n/a | Not stated |
| [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch) at 85c192e | drums | **No licence file**, so no licence is granted | Bundled in the package, converted from the released weights of [ADTOF](https://github.com/MZehren/ADTOF), which is CC BY-NC-SA 4.0 |
| [Basic Pitch](https://github.com/spotify/basic-pitch) 0.4.0 | notes | Apache-2.0 | Bundled in the package; no separate statement, so presumably Apache-2.0 |
| torchaudio [`MMS_FA`](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.MMS_FA.html) | lyrics | torchaudio: BSD-2-Clause | **CC BY-NC 4.0** |
| torchaudio [`WAV2VEC2_ASR_LARGE_LV60K_960H`](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H.html) | lyrics | torchaudio: BSD-2-Clause | MIT |

Other dependencies (PyTorch, librosa, numpy, scipy, soundfile, mido, and so on) are
permissively licensed. The lyric alignment method is ported from mexicat/pdoom-video (MIT);
its notice is in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

**The default pipeline is not suitable for commercial use.** The `drums` stage runs
ADTOF-pytorch, which ships without a licence and bundles weights converted from a
CC BY-NC-SA 4.0 model. The `lyrics` stage runs `MMS_FA`, which is CC BY-NC 4.0. The
separator checkpoints and the Demucs weights state no licence at all. Whether a model's
licence terms carry over to the JSON the pipeline generates is unsettled. Check each
licence yourself before using the output for anything beyond personal, non-commercial work.

## Environment notes

- `venv-main`: Python 3.13 with torch 2.11.0+cu128. **Install only with
  `-c audio-events\constraints-main.txt` and the cu128 index** (setup.ps1 does). A bare
  `pip install` once swapped in torch 2.14 CPU through a torchvision dependency, silently.
  After any install, check `torch.cuda.is_available()`.
- `requirements-*.txt` are `pip freeze` of the working venvs. `adtof-pytorch` has no PyPI
  release and is pinned to commit 85c192e of its git repo. Every line is an exact pin except
  `onnx-weekly`, which is a floor: audio-separator requires it, it only ships dated
  dev builds that PyPI prunes after about a year, and nothing here imports it. Keep it a
  floor when re-freezing.
- `venv-bp` is Python 3.10 on purpose: on Windows, basic-pitch 0.4.0 depends on onnxruntime
  below Python 3.11 and on TensorFlow from 3.11 up.
- On a Blackwell GPU (tested on an RTX 5090), onnxruntime's ONNX separators silently fall
  back to CPU. The pipeline uses the PyTorch-native separator models.
- audio-separator 0.47.0 imports `audioread` without declaring it; `separate.py` stubs it.
- Python's `http.server` has no Range support, so `<audio>` never loads. Use `serve.py`.
- Pass `encoding="utf-8"` when reading JSON that can hold text. Windows' default code page
  turned a non-ASCII character in a lyric into mojibake once. Every JSON read on the lyric
  path (`lyrics.py`, `assemble.py`, `report.py`, `render_check.py`, `export_app.py`) names it;
  a write of plain `json.dumps` output is ASCII-escaped and safe either way.
- **Never commit audio, stems, outputs or `config.json`.** They belong in the workspace. The
  `.gitignore` files are only a safety net.
