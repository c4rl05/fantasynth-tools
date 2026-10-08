"""Run the whole audio -> events pipeline on one track.

    <workspace>\\venv-main\\Scripts\\python.exe music-events\\run_all.py "~\\Music\\Live\\My Track.mp3" [--slug name] [--from stage] [--track app_track] [--app path]

Paths come from workspace.py: the code runs from the repo, and audio/, out/, config.json and
the two venvs live in the workspace folder outside it.

The lyrics stage reads the lyric text from the app's track file: --track names it (else the
slug's "appTrack" in config.json), and --app names the app checkout (else FANTASYNTH_APP,
else "app" in config.json). When a track is known, the app is resolved and checked BEFORE
the first stage runs, so a wrong path fails in a second, not after the GPU stages.

Stages, in order (each writes into <workspace>/out/<slug>/, see CONTRACT.md):
    decode    ffmpeg -> <workspace>/audio/<slug>.wav (44.1 kHz stereo)
    grid      Beat This! + all-in-one-infer -> rigid grid, downbeats, raw sections
    separate  BS-Roformer-SW 6 stems, then MDX23C DrumSep on the drum stem
    drums     ADTOF on drum stem and on the mix, onsets per DrumSep sub-stem
    notes     Basic Pitch per pitched stem (runs in venv-bp)
    curves    per-stem RMS / centroid / onset strength at 100 fps
    lyrics    forced alignment of the known lyric text + vocal onsets (--track names the text)
    assemble  vote, snap, align grid to kicks, phrase-snap + label sections -> events.json
    render    listening checks (check_*.mp3)
    report    report card: measures events.json against the stems -> report.json (see report.py)
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

import workspace
from workspace import AUDIO, OUT, SCRIPTS, TOOL_DIR, VENV_BP, VENV_MAIN

STAGES = ["decode", "grid", "separate", "drums", "notes", "curves", "lyrics", "assemble", "render", "report"]
SCRIPT = {"grid": "grid.py", "separate": "separate.py", "drums": "drums.py", "notes": "notes.py",
          "curves": "curves.py", "lyrics": "lyrics.py", "assemble": "assemble.py", "render": "render_check.py"}


def slugify(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="input audio file (any format ffmpeg reads)")
    ap.add_argument("--slug", help="short name for out/<slug>/ (default: from the file name)")
    ap.add_argument("--from", dest="start", choices=STAGES, default="decode", help="resume from this stage")
    ap.add_argument("--track", help="app track (public/music/<track>.json) holding the lyric text, for the "
                                    "lyrics stage (default: the slug's appTrack in config.json)")
    ap.add_argument("--app", help="the Fantasynth app checkout, passed to the lyrics stage "
                                  '(default: FANTASYNTH_APP, else "app" in config.json)')
    args = ap.parse_args()
    slug = args.slug or slugify(Path(args.audio).stem)
    print(f"slug: {slug}")
    stages = STAGES[STAGES.index(args.start):]
    track = args.track or workspace.app_track(slug)
    if "lyrics" in stages and track:
        # the lyrics stage will read a track file: fail now, not after the GPU stages
        app = workspace.app_root(args.app)
        tfile = app / "public" / "music" / f"{track}.json"
        if not tfile.is_file():
            sys.exit(f"{tfile} not found: the lyrics stage reads track {track!r} from it (check --track / --app)")
        print(f"lyrics text: {tfile}")
        args.app = str(app)

    for stage in stages:
        t = time.time()
        if stage == "decode":
            AUDIO.mkdir(parents=True, exist_ok=True)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", args.audio,
                   "-ar", "44100", "-ac", "2", "-c:a", "pcm_s16le", str(AUDIO / f"{slug}.wav")]
        else:
            py = VENV_BP if stage == "notes" else VENV_MAIN
            # report.py sits beside run_all.py; the stage scripts live in scripts/
            script = TOOL_DIR / "report.py" if stage == "report" else SCRIPTS / SCRIPT[stage]
            cmd = [str(py), str(script), "--slug", slug]
            if stage == "lyrics" and args.track:
                cmd += ["--track", args.track]
            if stage == "lyrics" and args.app:
                cmd += ["--app", args.app]
        print(f"== {stage}")
        r = subprocess.run(cmd)
        if r.returncode:
            sys.exit(f"stage {stage} failed (exit {r.returncode}); fix and resume with --from {stage}")
        print(f"   {stage} done in {time.time() - t:.1f}s")
    print(f"\nevents: {OUT / slug / 'events.json'}")
    print(f"report: {OUT / slug / 'report.json'}")
    print(f"viewer: run `{VENV_MAIN} {TOOL_DIR / 'serve.py'}`, open http://127.0.0.1:8765/viewer.html?slug={slug}")


if __name__ == "__main__":
    main()
