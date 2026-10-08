"""Stage `separate`: BS-Roformer-SW 6-stem split, then MDX23C DrumSep on the drums stem.

Writes (per CONTRACT.md):
  out/<slug>/stems/{vocals,bass,drums,guitar,piano,other}.wav
  out/<slug>/stems/drums_{kick,snare,toms,hh,ride,crash}.wav
plus a run log out/<slug>/log_separate.json (device, runtime, peaks).

Usage: separate.py --slug track-a | --all
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import types

import numpy as np
import soundfile as sf
import torch

# audio-separator 0.47.0 imports `audioread` at module top but does not declare it (it used
# to arrive via librosa<1.0). Its only use is rerun_mp3() for the VR architecture, which
# this stage never calls, so a stub that fails loudly on use avoids an install into venv-main.
if "audioread" not in sys.modules:
    try:
        import audioread  # noqa: F401
    except ImportError:
        _stub = types.ModuleType("audioread")

        def _audio_open(*_a, **_k):
            raise RuntimeError("audioread is not installed (stubbed by separate.py)")

        _stub.audio_open = _audio_open
        sys.modules["audioread"] = _stub

from audio_separator.separator import Separator  # noqa: E402
from audio_separator.separator.uvr_lib_v5 import spec_utils as _spec_utils  # noqa: E402

# Keep stem levels exact. audio-separator rescales every stem whose peak exceeds
# normalization_threshold (each by its own factor) and writes int16, so on a mastered
# track the loud stems come out at different gains and no longer sum to the mix. We
# disable that gain stage (keeping its empty/non-finite validation) and write 32-bit
# float WAV instead, so stems are at mix level, may peak above 0 dBFS, and sum ~= mix.
_orig_normalize = _spec_utils.normalize


def _validate_only(wave, max_peak=1.0, min_peak=None):
    wave = np.asarray(wave)
    if wave.size == 0 or not np.isfinite(np.abs(wave).max()):
        return _orig_normalize(wave, max_peak, min_peak)  # raises InvalidAudioDataError
    return wave


_spec_utils.normalize = _validate_only


def _write_float(self, stem_path, stem_source):
    path = os.path.join(self.output_dir, stem_path)
    tmp = path + ".part"
    sf.write(tmp, np.asarray(stem_source, dtype=np.float32), self.sample_rate, subtype="FLOAT", format="WAV")
    os.replace(tmp, path)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # music-events/, for workspace.py
from workspace import AUDIO, MODELS, OUT, all_slugs  # noqa: E402

STEM_MODEL = "BS-Roformer-SW.ckpt"
STEMS = ["vocals", "bass", "drums", "guitar", "piano", "other"]

DRUM_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
DRUM_PARTS = ["kick", "snare", "toms", "hh", "ride", "crash"]


def make_separator(out_dir):
    # Gain normalization is disabled globally above; normalization_threshold only
    # matters for audio-separator's own validation now.
    return Separator(
        log_level=logging.INFO,
        model_file_dir=str(MODELS),
        output_dir=str(out_dir),
        output_format="WAV",
        normalization_threshold=1.0,
        sample_rate=44100,
    )


def model_device(sep):
    inst = sep.model_instance
    dev = getattr(inst, "torch_device", None)
    run = getattr(inst, "model_run", None)
    param_dev = None
    if run is not None:
        try:
            param_dev = str(next(run.parameters()).device)
        except StopIteration:
            pass
    return str(dev), param_dev


def instrument_names(sep):
    cfg = sep.model_instance.model_data_cfgdict
    return list(cfg.training.instruments)


def run_model(sep, model_file, expected, jobs, prefix):
    """jobs: list of (slug, input_wav, out_dir). Returns {slug: info}."""
    t = time.perf_counter()
    sep.load_model(model_file)
    load_s = time.perf_counter() - t
    names = instrument_names(sep)
    lower = {n.lower(): n for n in names}
    missing = [e for e in expected if e not in lower]
    if missing:
        raise RuntimeError(f"{model_file}: model instruments {names} lack {missing}")
    sep.model_instance.write_audio = types.MethodType(_write_float, sep.model_instance)
    dev, param_dev = model_device(sep)
    print(f"[separate] {model_file}: instruments={names} torch_device={dev} params_on={param_dev} load={load_s:.1f}s", flush=True)
    if not str(param_dev).startswith("cuda"):
        raise RuntimeError(f"{model_file} is not on CUDA (params on {param_dev})")

    results = {}
    for slug, in_wav, out_dir in jobs:
        out_dir.mkdir(parents=True, exist_ok=True)
        sep.output_dir = str(out_dir)
        sep.model_instance.output_dir = str(out_dir)
        custom = {lower[e]: f"{prefix}{e}" for e in expected}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        files = sep.separate(str(in_wav), custom_output_names=custom)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        peak_mb = torch.cuda.max_memory_allocated() / 2**20
        peaks = {}
        for e in expected:
            p = out_dir / f"{prefix}{e}.wav"
            if not p.exists():
                raise RuntimeError(f"expected output missing: {p} (got {files})")
            x, _ = sf.read(str(p), dtype="float32")
            peaks[e] = round(float(20 * np.log10(np.abs(x).max() + 1e-12)), 2)
        rescaled = [e for e, v in peaks.items() if v > 0.0]
        print(f"[separate] {slug} {model_file}: {dt:.1f}s, cuda peak alloc {peak_mb:.0f} MB, "
              f"stems peaking above 0 dBFS (kept, float WAV): {rescaled or 'none'}", flush=True)
        results[slug] = {
            "model": model_file, "seconds": round(dt, 2), "load_seconds": round(load_s, 2),
            "torch_device": dev, "params_device": param_dev,
            "cuda_peak_alloc_mb": round(peak_mb), "peak_dbfs": peaks, "over_0dbfs": rescaled,
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--slug")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--skip-stems", action="store_true", help="reuse existing stems, run DrumSep only")
    args = ap.parse_args()
    slugs = all_slugs() if args.all else [args.slug]

    if not torch.cuda.is_available():
        sys.exit("CUDA not available in this venv; refusing to separate on CPU")
    print(f"[separate] torch {torch.__version__}, device {torch.cuda.get_device_name(0)}", flush=True)

    sep = make_separator(OUT)
    log = {s: {} for s in slugs}

    if not args.skip_stems:
        jobs = [(s, AUDIO / f"{s}.wav", OUT / s / "stems") for s in slugs]
        for s, info in run_model(sep, STEM_MODEL, STEMS, jobs, prefix="").items():
            log[s]["stems"] = info

    jobs = [(s, OUT / s / "stems" / "drums.wav", OUT / s / "stems") for s in slugs]
    for s, info in run_model(sep, DRUM_MODEL, DRUM_PARTS, jobs, prefix="drums_").items():
        log[s]["drumsep"] = info

    for s in slugs:
        p = OUT / s / "log_separate.json"
        prev = json.loads(p.read_text()) if p.exists() else {}
        prev.update(log[s])
        prev["gpu"] = torch.cuda.get_device_name(0)
        p.write_text(json.dumps(prev, indent=2))


if __name__ == "__main__":
    main()
