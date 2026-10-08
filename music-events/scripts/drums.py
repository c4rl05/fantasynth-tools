"""Stage `drums`: ADTOF-pytorch transcription + per-DrumSep-stem onset detection.

Writes (per CONTRACT.md):
  out/<slug>/raw_adtof.json          ADTOF on stems/drums.wav   {classes:{kick,snare,tom,hihat,cymbal:[{t,vel}]}, meta}
  out/<slug>/raw_adtof_mix.json      ADTOF on audio/<slug>.wav  (same shape; separation-vs-mix comparison)
  out/<slug>/raw_drumsep_onsets.json onsets per stems/drums_<part>.wav  {<part>:[{t,strength}], meta}
plus a run log out/<slug>/log_drums.json.

`vel` is the ADTOF sigmoid activation at the picked frame (0..1).
`strength` is the onset-envelope value at the onset divided by the 99th percentile of
that stem's own envelope (so ~1 = a typical loud hit on that stem).

Usage: drums.py --slug track-a | --all
"""
import argparse
import json
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import torch

from adtof_pytorch import (
    FRAME_RNN_THRESHOLDS,
    LABELS_5,
    PeakPicker,
    calculate_n_bins,
    create_frame_rnn_model,
    get_default_weights_path,
    load_audio_for_model,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # music-events/, for workspace.py
from workspace import AUDIO, OUT, WORKSPACE, all_slugs  # noqa: E402

ADTOF_FPS = 100
# LABELS_5 = [35, 38, 47, 42, 49] -> contract names
ADTOF_NAMES = {35: "kick", 38: "snare", 47: "tom", 42: "hihat", 49: "cymbal"}

DRUM_PARTS = ["kick", "snare", "toms", "hh", "ride", "crash"]
SR = 44100
HOP = 256            # 5.8 ms onset frames
FRAME = 2048         # RMS window for the loudness gate
GATE_LOOK_S = 0.05   # look this far after the onset for the hit's RMS peak
# Three gates, all on the hit's peak RMS in the GATE_LOOK_S after the onset:
#  own: within GATE_OWN_DB of this sub-stem's own loud level (99th pct of its frame RMS).
#  bus: within GATE_BUS_DB of the drums stem's loud level. The own gate alone passes
#       sub-stems that contain nothing BUT bleed (ride/crash on these tracks peak 30-45 dB
#       under the drum bus and produced hundreds of onsets); this is the floor for those.
#  dom: drop if another DrumSep sub-stem is more than GATE_DOM_DB louder in the same
#       window (crosstalk from a simultaneous hit; a real hat on a kick sits ~15-25 dB under).
GATE_OWN_DB = 24.0
GATE_BUS_DB = 40.0
GATE_DOM_DB = 30.0


# ---------------------------------------------------------------- ADTOF
def load_adtof(device):
    model = create_frame_rnn_model(calculate_n_bins())
    ckpt = torch.load(get_default_weights_path(), map_location="cpu")
    weights = ckpt.get("model_weights", ckpt)
    model.load_state_dict(weights, strict=True)  # fail loudly rather than run random weights
    return model.eval().to(device)


def run_adtof(model, device, wav):
    x = load_audio_for_model(str(wav)).to(device)
    with torch.no_grad():
        act = model(x)[0].float().cpu().numpy()  # [T, 5]
    picked = PeakPicker(thresholds=FRAME_RNN_THRESHOLDS, fps=ADTOF_FPS).pick(act, labels=LABELS_5)[0]
    classes = {}
    for ci, lab in enumerate(LABELS_5):
        hits = []
        for t in picked[int(lab)]:
            i = int(round(t * ADTOF_FPS))
            lo, hi = max(0, i - 1), min(len(act), i + 2)
            hits.append({"t": round(float(t), 4), "vel": round(float(act[lo:hi, ci].max()), 4)})
        classes[ADTOF_NAMES[int(lab)]] = hits
    return classes, act


# ---------------------------------------------------------------- DrumSep onsets
def rms_db_frames(y):
    rms = librosa.feature.rms(y=y, frame_length=FRAME, hop_length=HOP, center=True)[0]
    return 20 * np.log10(np.maximum(rms, 1e-9))


def drumsep_onsets(stem_dir):
    """Onsets for every drums_<part>.wav, gated (own / bus / dom, see constants)."""
    yb, _ = librosa.load(str(stem_dir / "drums.wav"), sr=SR, mono=True)
    bus_ref = float(np.percentile(rms_db_frames(yb), 99))
    ys = {p: librosa.load(str(stem_dir / f"drums_{p}.wav"), sr=SR, mono=True)[0] for p in DRUM_PARTS}
    lv = {p: rms_db_frames(ys[p]) for p in DRUM_PARTS}
    n = min(len(v) for v in lv.values())
    stack = np.stack([lv[p][:n] for p in DRUM_PARTS])
    look = int(round(GATE_LOOK_S * SR / HOP))

    parts, metas = {}, {}
    for pi, p in enumerate(DRUM_PARTS):
        y, rms_db = ys[p], lv[p]
        env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
        own_ref = float(np.percentile(rms_db, 99))
        gate_db = max(own_ref - GATE_OWN_DB, bus_ref - GATE_BUS_DB)
        env_ref = float(np.percentile(env, 99)) or 1.0
        frames = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=HOP, units="frames")
        kept, drop_level, drop_dom = [], 0, 0
        for f in frames:
            f = int(min(f, n - 1))
            local_db = float(rms_db[f:f + look + 1].max())
            if local_db < gate_db:
                drop_level += 1
                continue
            others = np.delete(stack[:, f:f + look + 1].max(axis=1), pi).max()
            dom = local_db - float(others)
            if dom < -GATE_DOM_DB:
                drop_dom += 1
                continue
            t = librosa.frames_to_time(f, sr=SR, hop_length=HOP)
            kept.append({"t": round(float(t), 4), "strength": round(float(env[f] / env_ref), 4),
                         "db": round(local_db, 1), "dom": round(dom, 1)})
        parts[p] = kept
        metas[p] = {"rms_db_p99": round(own_ref, 1), "gate_db": round(gate_db, 1),
                    "rms_db_track": round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)), 1),
                    "candidates": int(len(frames)), "gated_level": drop_level, "gated_dom": drop_dom}
    return parts, metas, bus_ref


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--all", action="store_true")
    args = ap.parse_args()
    slugs = all_slugs() if args.all else [args.slug]

    if not torch.cuda.is_available():
        sys.exit("CUDA not available in this venv; refusing to run ADTOF on CPU")
    device = "cuda"
    model = load_adtof(device)
    print(f"[drums] ADTOF params on {next(model.parameters()).device}", flush=True)

    for slug in slugs:
        out = OUT / slug
        log = {"gpu": torch.cuda.get_device_name(0), "adtof_device": str(next(model.parameters()).device)}
        for tag, wav, fname in (("stem", out / "stems" / "drums.wav", "raw_adtof.json"),
                                ("mix", AUDIO / f"{slug}.wav", "raw_adtof_mix.json")):
            torch.cuda.synchronize()
            t = time.perf_counter()
            classes, act = run_adtof(model, device, wav)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t
            doc = {"classes": classes,
                   "meta": {"source": str(wav.relative_to(WORKSPACE)).replace("\\", "/"), "fps": ADTOF_FPS,
                            "thresholds": dict(zip([ADTOF_NAMES[l] for l in LABELS_5], FRAME_RNN_THRESHOLDS)),
                            "vel": "sigmoid activation at the picked frame (max over +-1 frame)"}}
            (out / fname).write_text(json.dumps(doc))
            counts = {k: len(v) for k, v in classes.items()}
            log[f"adtof_{tag}"] = {"seconds": round(dt, 2), "frames": int(act.shape[0]), "counts": counts}
            print(f"[drums] {slug} ADTOF-{tag}: {dt:.1f}s {counts}", flush=True)

        t = time.perf_counter()
        parts, metas, bus_ref = drumsep_onsets(out / "stems")
        dt = time.perf_counter() - t
        doc = dict(parts)
        doc["meta"] = {"sr": SR, "hop": HOP,
                       "gate": {"own_db": GATE_OWN_DB, "bus_db": GATE_BUS_DB, "dom_db": GATE_DOM_DB,
                                "look_s": GATE_LOOK_S, "bus_rms_db_p99": round(bus_ref, 1)},
                       "stems": metas,
                       "strength": "onset envelope / its own 99th percentile",
                       "db": "peak frame RMS (dBFS) in the 50 ms after the onset",
                       "dom": "db minus the loudest OTHER DrumSep sub-stem in the same window"}
        (out / "raw_drumsep_onsets.json").write_text(json.dumps(doc))
        log["drumsep_onsets"] = {"seconds": round(dt, 2), "counts": {p: len(parts[p]) for p in DRUM_PARTS},
                                 "stems": metas}
        print(f"[drums] {slug} DrumSep onsets: {dt:.1f}s {log['drumsep_onsets']['counts']}", flush=True)
        (out / "log_drums.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
