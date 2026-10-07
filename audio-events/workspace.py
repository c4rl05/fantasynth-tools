"""Where the audio-events tooling keeps its heavy local state, its per-track config, and how
it finds the Fantasynth app checkout.

The CODE lives in this repo (audio-events/). Everything heavy, and every piece of per-track
data, lives in one WORKSPACE folder outside the repo, shared by every checkout and worktree:

    <workspace>/config.json    per-machine and per-track config (see config.example.json)
    <workspace>/venv-main/     Python 3.13 + CUDA torch (see setup.ps1)
    <workspace>/venv-bp/       Python 3.10 + Basic Pitch (ONNX)
    <workspace>/models/        separator checkpoints (audio-separator)
    <workspace>/audio/         decoded input WAVs (commercial audio: never commit)
    <workspace>/out/<slug>/    stems, raw detections, curves, events.json, checks

Workspace resolution:
  1. AUDIO_EVENTS_WORKSPACE, if set.
  2. <two levels above the MAIN checkout of this repo>/audio-events, but only if that
     folder already exists.
  3. Else <the folder holding the MAIN checkout>/audio-events: a sibling of the clone.
  The main checkout is found through git's common dir, so worktrees resolve to the same
  place; without git, this checkout is used. A checkout at a drive or filesystem root
  resolves to <root>/audio-events.

The app checkout (only commands that read or write the app resolve it, and only when they
need it, so importing these modules and running the tests never needs the app):
  1. --app <path> on the command line,
  2. FANTASYNTH_APP, if set,
  3. "app" in <workspace>/config.json,
  else an error. Whichever source wins, the folder must hold public/music/index.json.

MUSIC_DIR (the source audio, the folder the app's dev server serves as /local-music/):
LOCAL_MUSIC_PATH, else "musicDir" in config.json, else ~/Music/Live.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
SCRIPTS = TOOL_DIR / "scripts"

APP_ENV = "FANTASYNTH_APP"
APP_MARKER = Path("public") / "music" / "index.json"  # what makes a folder the app checkout


def _git(*args):
    try:
        out = subprocess.run(["git", *args], cwd=TOOL_DIR, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _repo_root():
    """The checkout this tool lives in (a worktree resolves to itself). The tool folder sits
    directly in the repo root, so without git that is its parent."""
    top = _git("rev-parse", "--show-toplevel")
    return Path(top) if top else TOOL_DIR.parent


def _main_checkout():
    """The main checkout, even when running from a worktree."""
    common = _git("rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common).parent if common else _repo_root()


def default_workspace(main):
    """The workspace when AUDIO_EVENTS_WORKSPACE is not set, for main checkout `main`:
    <main>/../../audio-events if that folder already exists, else <main>/../audio-events.
    Path.parent of a root is the root itself, so a checkout at a root never raises."""
    main = Path(main)
    far = main.parent.parent / "audio-events"
    if main.parent != main.parent.parent and far.is_dir():
        return far
    return main.parent / "audio-events"


TOOLS_REPO = _repo_root()
WORKSPACE = Path(os.environ.get("AUDIO_EVENTS_WORKSPACE") or default_workspace(_main_checkout()))
CONFIG = WORKSPACE / "config.json"

AUDIO = WORKSPACE / "audio"
OUT = WORKSPACE / "out"
MODELS = WORKSPACE / "models"
VENV_MAIN = WORKSPACE / "venv-main" / "Scripts" / "python.exe"
VENV_BP = WORKSPACE / "venv-bp" / "Scripts" / "python.exe"

# ---------------------------------------------------------------------------- config

CONFIG_KEYS = {"app", "musicDir", "tracks"}
# Per-track keys. Every one is optional; a command that needs one names the slug when it is missing.
#   file        the audio file name in MUSIC_DIR (export_app's new-track url, the seek test)
#   appTrack    the app track, public/music/<appTrack>.json (export target, lyric text source)
#   title       display title, used only when export_app has to create the track file
#   bpm, gridMarker   hand-measured grid, used only by grid.py's diagnostic comparison
#   handTimed   reason string: the track's inline lyrics were timed by hand, so the lyrics
#               stage takes them verbatim and export_app.py --lyrics never overwrites them
#   pron        pronunciation fixes for the aligner, {"<token>": "<spelling>"}
#   all         false leaves the slug out of every --all (stages, export_app, seek_formats_prep);
#               --slug still reaches it. Default true.
TRACK_TYPES = {"file": str, "appTrack": str, "title": str, "bpm": (int, float), "gridMarker": (int, float),
               "handTimed": str, "pron": dict, "all": bool}
TRACK_TYPE_NAMES = {k: {str: "a string", dict: "an object", bool: "true or false"}.get(v, "a number")
                    for k, v in TRACK_TYPES.items()}
TRACK_KEYS = set(TRACK_TYPES)


def _ignored(key):
    return key.startswith("_")  # "_comment" and friends, for notes inside the JSON


def load_config(path=None):
    """<workspace>/config.json as a dict ({} when the file does not exist). A file that does
    exist must be valid: unknown keys stop the run, since a misspelt key would otherwise be
    a silent no-op."""
    p = Path(path or CONFIG)
    if not p.exists():
        return {}
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        sys.exit(f"{p}: not valid JSON ({e})")
    if not isinstance(cfg, dict):
        sys.exit(f"{p}: expected a JSON object")
    stray = sorted(k for k in cfg if k not in CONFIG_KEYS and not _ignored(k))
    if stray:
        sys.exit(f"{p}: unknown top-level keys {stray}; known: {sorted(CONFIG_KEYS)}")
    for k in ("app", "musicDir"):
        if k in cfg and not isinstance(cfg[k], str):
            sys.exit(f'{p}: "{k}" must be a string (a path)')
    tracks = cfg.get("tracks", {})
    if not isinstance(tracks, dict):
        sys.exit(f'{p}: "tracks" must be an object keyed by slug')
    for slug, t in tracks.items():
        if _ignored(slug):
            continue
        if not isinstance(t, dict):
            sys.exit(f'{p}: tracks["{slug}"] must be an object')
        stray = sorted(k for k in t if k not in TRACK_KEYS and not _ignored(k))
        if stray:
            sys.exit(f'{p}: tracks["{slug}"] has unknown keys {stray}; known: {sorted(TRACK_KEYS)}')
        for k, v in t.items():
            want = TRACK_TYPES.get(k)
            ok = _ignored(k) or (want is bool and isinstance(v, bool)) or \
                (want is not None and want is not bool and isinstance(v, want) and not isinstance(v, bool))
            if not ok:
                sys.exit(f'{p}: tracks["{slug}"]["{k}"] must be {TRACK_TYPE_NAMES[k]}')
    return cfg


def tracks(path=None):
    """{slug: per-track config}, in file order."""
    return {s: t for s, t in load_config(path).get("tracks", {}).items() if not _ignored(s)}


def in_all(path=None):
    """{slug: per-track config} for --all: every track whose "all" is not false."""
    return {s: t for s, t in tracks(path).items() if t.get("all", True)}


def all_slugs(path=None):
    """What --all means for every stage: the slugs in config.json, in file order, except
    those with "all": false."""
    slugs = list(in_all(path))
    if not slugs:
        sys.exit(f'--all runs every slug in {Path(path or CONFIG)} "tracks" without "all": false, and there are none; '
                 f"pass --slug <slug>, or add tracks (see config.example.json)")
    return slugs


def track_conf(slug, path=None):
    """The config entry of one slug, {} when it has none."""
    return tracks(path).get(slug, {})


def app_track(slug, path=None):
    """The app track a slug maps to (tracks.<slug>.appTrack), or None."""
    return track_conf(slug, path).get("appTrack")


def _by_app_track(track, path=None):
    """The config entry whose appTrack is `track`, {} when none is."""
    if not track:
        return {}
    return next((t for t in tracks(path).values() if t.get("appTrack") == track), {})


def hand_timed(track, path=None):
    """The reason string when app track `track` is marked handTimed in config.json, else None.
    Keyed by the app track, not the slug: hand timing is a property of the track file."""
    return _by_app_track(track, path).get("handTimed") or None


def pron_table(track, path=None):
    """Pronunciation fixes for app track `track` (tracks.<slug>.pron), {} when none."""
    return dict(_by_app_track(track, path).get("pron") or {})


def music_dir(path=None):
    """The local music folder: LOCAL_MUSIC_PATH, else config musicDir, else ~/Music/Live."""
    env = os.environ.get("LOCAL_MUSIC_PATH")
    if env:
        return Path(env)
    cfg = load_config(path).get("musicDir")
    return Path(cfg).expanduser() if cfg else Path.home() / "Music" / "Live"


def app_root(cli=None, path=None):
    """The Fantasynth app checkout: --app, else FANTASYNTH_APP, else config "app", else an
    error. The first source that is set wins and must look like the app (it must hold
    public/music/index.json); an invalid one is an error, never a fall-through to the next."""
    sources = (("--app", cli), (APP_ENV, os.environ.get(APP_ENV)),
               (f'"app" in {Path(path or CONFIG)}', None))
    for i, (name, value) in enumerate(sources):
        if i == 2:
            value = load_config(path).get("app")
        if not value:
            continue
        root = Path(value).expanduser()
        if not (root / APP_MARKER).is_file():
            sys.exit(f"the app checkout from {name} is {root}, but {root / APP_MARKER} does not exist: "
                     f"point it at a Fantasynth app checkout")
        return root.resolve()
    sys.exit(f"no Fantasynth app checkout: pass --app <path>, set {APP_ENV}, or set \"app\" in "
             f"{Path(path or CONFIG)} (see config.example.json)")


if __name__ == "__main__":
    for name in ("TOOL_DIR", "TOOLS_REPO", "WORKSPACE", "CONFIG", "AUDIO", "OUT", "MODELS", "VENV_MAIN", "VENV_BP"):
        p = globals()[name]
        print(f"{name:<10} {p}{'' if Path(p).exists() else '   (missing)'}")
    print(f"{'MUSIC_DIR':<10} {music_dir()}")
    try:
        print(f"{'APP':<10} {app_root()}")
    except SystemExit as e:
        print(f"{'APP':<10} (unresolved: {e.code})")
    n = len(tracks())
    print(f"{'TRACKS':<10} {n} slug{'s' if n != 1 else ''} in config")
