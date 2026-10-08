"""Basic Pitch on each separated pitched stem -> out/<slug>/raw_notes_<stem>.json.

Run with venv-bp (Python 3.10, ONNX backend). Stems quieter than SILENT_DBFS are skipped
so the model doesn't hallucinate notes from separation residue.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import Model, predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # music-events/, for workspace.py
from workspace import OUT, all_slugs  # noqa: E402

SILENT_DBFS = -40.0

# Per-stem settings. Default minimum_note_length (127.7 ms) would drop 16th notes at
# 127 BPM (118 ms), so it's lowered to a 32nd.
STEMS = {
    "bass":   dict(minimum_frequency=27.5, maximum_frequency=400.0, minimum_note_length=58, mono=True),
    "other":  dict(minimum_frequency=55.0, maximum_frequency=4200.0, minimum_note_length=58, mono=False),
    "vocals": dict(minimum_frequency=70.0, maximum_frequency=1400.0, minimum_note_length=80, mono=True),
    "piano":  dict(minimum_frequency=27.5, maximum_frequency=4200.0, minimum_note_length=58, mono=False),
    "guitar": dict(minimum_frequency=70.0, maximum_frequency=2000.0, minimum_note_length=58, mono=False),
}


def rms_dbfs(path):
    y, _ = sf.read(str(path), always_2d=True)
    return 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)


def monophonic(notes):
    """Keep one note at a time: when notes overlap, the louder one wins and the other is
    trimmed or dropped. Bass lines are monophonic; Basic Pitch often adds octave ghosts."""
    notes = sorted(notes, key=lambda n: (n["t"], -n["vel"]))
    out = []
    for n in notes:
        if out and n["t"] < out[-1]["t"] + out[-1]["dur"]:
            prev = out[-1]
            if n["vel"] > prev["vel"] and n["t"] - prev["t"] > 0.03:
                prev["dur"] = n["t"] - prev["t"]
                out.append(n)
            continue
        out.append(n)
    return out


def run(slug, model):
    d = OUT / slug
    for stem, cfg in STEMS.items():
        path = d / "stems" / f"{stem}.wav"
        if not path.exists():
            continue
        level = rms_dbfs(path)
        if level < SILENT_DBFS:
            print(f"{slug}/{stem}: {level:.1f} dBFS, skipped as silent")
            continue
        cfg = dict(cfg)
        mono = cfg.pop("mono")
        _, _, events = predict(str(path), model, onset_threshold=0.5, frame_threshold=0.3,
                               melodia_trick=True, **cfg)
        notes = [{"t": round(float(s), 4), "dur": round(float(e - s), 4), "pitch": int(p),
                  "vel": round(float(a), 3)} for s, e, p, a, _ in events]
        raw_count = len(notes)
        if mono:
            notes = monophonic(notes)
        (d / f"raw_notes_{stem}.json").write_text(json.dumps({"stem": stem, "levelDbfs": round(level, 1),
                                                                "mono": mono, "notes": notes}))
        print(f"{slug}/{stem}: {level:.1f} dBFS, {raw_count} notes"
              + (f" -> {len(notes)} after mono" if mono else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    model = Model(ICASSP_2022_MODEL_PATH)
    for s in (all_slugs() if args.all else [args.slug]):
        run(s, model)
