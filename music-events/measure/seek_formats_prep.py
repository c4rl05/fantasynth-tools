"""Make per-format copies of test tracks for the media-element seek test.

    <workspace>\\venv-main\\Scripts\\python.exe music-events\\measure\\seek_formats_prep.py --slug track-a [--slug track-b]
    <workspace>\\venv-main\\Scripts\\python.exe music-events\\measure\\seek_formats_prep.py --all

Step 1 of 3 (then seek_formats.mjs, then seek_formats_analyze.py). Worth re-running when
Chrome updates: the MP3 seek offset is a property of Chromium's decoder, not of the file.

Reads each slug's "file" (tracks.<slug>.file in <workspace>/config.json) from MUSIC_DIR
(workspace.py); --all takes every slug in config.json that has a "file" (except "all": false). Writes ONLY into
<workspace>\\seektest\\ (never into the source music folder):
  <slug>.mp3   byte copy of the original (control)
  <slug>.flac  ffmpeg decode of the MP3 -> FLAC
  <slug>.wav   ffmpeg decode of the MP3 -> 16-bit PCM WAV, 44.1 kHz
  <slug>.opus  ffmpeg decode of the MP3 -> libopus 160k in Ogg (48 kHz, resampled by ffmpeg)
  <slug>.m4a   ffmpeg decode of the MP3 -> native AAC 256k in MP4 (faststart)
  <slug>.w48.wav  ffmpeg decode of the .opus file -> 16-bit WAV at 48 kHz: a pipeline-latency
               control at the Opus media rate (no seek error possible in PCM)
Prints ffprobe facts (duration, start_time, sample rate, size) for each.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # music-events/, for workspace.py
from workspace import CONFIG, WORKSPACE, in_all, music_dir, tracks  # noqa: E402

OUT = WORKSPACE / "seektest"
ENC = {
    "flac": ["-c:a", "flac", "-sample_fmt", "s16"],
    "wav": ["-c:a", "pcm_s16le"],
    "opus": ["-c:a", "libopus", "-b:a", "160k"],
    "m4a": ["-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart"],
}


def ff(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y"] + args, check=True)


def probe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(p)],
                       capture_output=True, text=True, check=True)
    j = json.loads(r.stdout)
    s = [x for x in j["streams"] if x["codec_type"] == "audio"][0]
    return {"codec": s["codec_name"], "sr": s["sample_rate"], "start_time": s.get("start_time"),
            "duration": j["format"].get("duration"), "bytes": int(j["format"]["size"]),
            "initial_padding": s.get("initial_padding"), "side_data": s.get("side_data_list")}


def files(slugs=None):
    """{slug: audio file name} from config.json: the given slugs (each must have a "file"),
    or every slug that has one."""
    conf = tracks()
    if slugs is None:
        out = {s: t["file"] for s, t in in_all().items() if t.get("file")}
        if not out:
            sys.exit(f'--all takes every slug with a "file" in {CONFIG}, and there are none')
        return out
    missing = [s for s in slugs if not conf.get(s, {}).get("file")]
    if missing:
        sys.exit(f'no "file" for {", ".join(missing)} in {CONFIG}: set tracks.<slug>.file')
    return {s: conf[s]["file"] for s in slugs}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug", action="append", help="a slug from config.json (repeatable)")
    g.add_argument("--all", action="store_true", help='every slug in config.json that has a "file"')
    a = ap.parse_args()
    todo = files(None if a.all else a.slug)
    live = music_dir()
    OUT.mkdir(exist_ok=True)
    assert OUT.resolve() != live.resolve()
    for slug, name in todo.items():
        src = live / name
        dst = OUT / f"{slug}.mp3"
        shutil.copyfile(src, dst)
        for ext, enc in ENC.items():
            ff(["-i", str(src), "-map", "0:a:0", "-map_metadata", "-1"] + enc + [str(OUT / f"{slug}.{ext}")])
        ff(["-i", str(OUT / f"{slug}.opus"), "-map", "0:a:0", "-map_metadata", "-1", "-c:a", "pcm_s16le",
            str(OUT / f"{slug}.w48.wav")])
        base = dst.stat().st_size
        for ext in ["mp3", "flac", "wav", "opus", "m4a", "w48.wav"]:
            p = OUT / f"{slug}.{ext}"
            info = probe(p)
            info["sizeVsMp3"] = round(info["bytes"] / base, 3)
            print(slug, ext, json.dumps(info))


if __name__ == "__main__":
    main()
