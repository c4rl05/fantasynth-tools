"""handTimed (config.json): a track whose inline lyrics were timed by hand is not aligned. The
lyrics stage takes its words, times and line grouping verbatim (no confidences), assemble
writes them without `words.v` (the app's words lane then hits at 1), and export_app.py
--lyrics never overwrites them. Run from audio-events/:

    <venv-main>/Scripts/python.exe -m unittest discover -s tests

Every file here is a temp copy: nothing reads or writes the real public/music, workspace or
config.json.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

TOOL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOL))
sys.path.insert(0, str(TOOL / "scripts"))
import assemble as A  # noqa: E402
import export_app as E  # noqa: E402
import lyrics as L  # noqa: E402
import workspace  # noqa: E402

SR = 22050
# Laid out like a hand-fit track file: {t, d, w}, non-round float times, plus a rest line
# ({"w": ""}) that must not become a word. Invented words and times.
HAND = {"lines": [
    {"words": [{"t": 0.4291, "d": 0.25, "w": "one"},
               {"t": 0.6851, "d": 0.2, "w": "two"},
               {"t": 0.8987, "d": 0.42, "w": "three,"}]},
    {"words": [{"t": 1.4, "d": 1.6, "w": ""}]},
    {"words": [{"t": 1.3465, "d": 0.28, "w": "four"}]},
]}


def voice(spans, dur=5.0):
    """A sung-ish tone in each (start, end) span, silence elsewhere."""
    y = np.zeros(int(dur * SR), np.float32)
    for a, b in spans:
        i, j = int(a * SR), int(b * SR)
        t = np.arange(j - i) / SR
        y[i:j] = 0.3 * sum(np.sin(2 * np.pi * 220 * h * t) / h for h in range(1, 6))
    return y


def write_config(path, track, listed, app=None):
    """A config.json mapping slug s -> app track `track`, marked handTimed when `listed`."""
    entry = {"appTrack": track, **({"handTimed": "fitted by hand (test)"} if listed else {})}
    cfg = {"tracks": {"s": entry}, **({"app": str(app)} if app else {})}
    Path(path).write_text(json.dumps(cfg), encoding="utf-8")


class Sandbox:
    """A temp app checkout (public/music/<track>.json), config.json and out/<slug>/stems/vocals.wav."""

    def __init__(self, tmp, track, listed, lyrics=HAND, spans=((0.4, 1.7), (3.0, 4.2)), write_track=True):
        self.root = Path(tmp)
        self.repo, self.out = self.root / "repo", self.root / "out"
        music = self.repo / "public" / "music"
        music.mkdir(parents=True)
        (music / "index.json").write_text("[]\n", encoding="utf-8")
        if write_track:
            (music / f"{track}.json").write_text(json.dumps({"title": "T", "lyrics": lyrics}), encoding="utf-8")
        self.config = self.root / "config.json"
        write_config(self.config, track, listed)
        stems = self.out / "s" / "stems"
        stems.mkdir(parents=True)
        sf.write(str(stems / "vocals.wav"), np.stack([voice(spans)] * 2, axis=1), SR)

    def patches(self):
        return [mock.patch.object(L, "APP", self.repo), mock.patch.object(L, "OUT", self.out),
                mock.patch.object(workspace, "CONFIG", self.config)]


def run_stage(sb, track):
    ps = sb.patches()
    for p in ps:
        p.start()
    try:
        return L.run("s", track)
    finally:
        for p in ps:
            p.stop()


class LyricsStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sb = Sandbox(cls.tmp.name, "t", listed=True)
        # the aligner must never be reached for a hand-timed track
        with mock.patch.object(L, "all_emissions", side_effect=AssertionError("aligned a hand-timed track")), \
             mock.patch.object(L.LA, "align", side_effect=AssertionError("aligned a hand-timed track")):
            cls.doc = run_stage(cls.sb, "t")
        cls.raw = json.loads((cls.sb.out / "s" / "raw_lyrics.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_times_and_grouping_are_the_track_s_verbatim(self):
        want = [[w for w in ln["words"] if w["w"]] for ln in HAND["lines"]]
        want = [ws for ws in want if ws]  # the rest line is dropped, the others keep their grouping
        self.assertEqual([ln["words"] for ln in self.raw["lines"]], want)  # t/d not even rounded
        self.assertEqual([ln["text"] for ln in self.raw["lines"]], ["one two three,", "four"])

    def test_no_confidence_and_the_source_is_named(self):
        for ln in self.raw["lines"]:
            for w in ln["words"]:
                self.assertEqual(set(w), {"w", "t", "d"})
        self.assertEqual(self.raw["source"], "hand-timed (public/music/t.json)")
        self.assertEqual(self.raw["handTimed"], "fitted by hand (test)")
        self.assertIsNone(self.raw["prior"])
        self.assertFalse((self.sb.out / "s" / "lyrics_prior.json").exists())

    def test_onsets_and_extras_still_computed(self):
        self.assertTrue(self.raw["onsets"])
        # the voice at 3.0-4.2 s has no word: an extra. The words' own span (0.4-1.7 s) is covered.
        self.assertEqual(len(self.raw["extras"]), 1)
        x = self.raw["extras"][0]
        self.assertLess(abs(x["t"] - 3.0), 0.15)
        self.assertLess(abs(x["t"] + x["d"] - 4.2), 0.15)
        self.assertEqual(self.raw["stats"]["words"], 4)

    def test_extras_match_the_aligned_path_s_rule(self):
        """Same LA.extras call as for aligned words: word spans in seconds as start/end."""
        f = L.features(self.sb.out / "s" / "stems" / "vocals.wav")
        words = [dict(start=w["t"], end=w["t"] + w["d"]) for ln in self.raw["lines"] for w in ln["words"]]
        want = [dict(t=L.r3(a), d=L.r3(L.r3(b) - L.r3(a))) for a, b in L.LA.extras(words, f)]
        self.assertEqual(self.raw["extras"], want)

    def test_an_unlisted_track_is_still_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            sb = Sandbox(tmp, "t", listed=False)
            with mock.patch.object(L, "all_emissions", side_effect=RuntimeError("aligning")), \
                 self.assertRaisesRegex(RuntimeError, "aligning"):
                run_stage(sb, "t")

    def test_the_track_comes_from_the_slug_s_config_entry(self):
        """No --track: the slug's appTrack in config.json names the track file."""
        with tempfile.TemporaryDirectory() as tmp:
            sb = Sandbox(tmp, "t", listed=True)
            with mock.patch.object(L, "all_emissions", side_effect=AssertionError("aligned")):
                run_stage(sb, None)
            raw = json.loads((sb.out / "s" / "raw_lyrics.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["track"], "t")
            self.assertEqual(raw["stats"]["words"], 4)


class Assemble(unittest.TestCase):
    GRID = {"t0": 0.5, "beatPeriod": 0.5, "meter": 4}
    RAW = {"version": 1, "track": "t", "source": "hand-timed (public/music/t.json)",
           "lines": [{"text": "a b", "words": [{"w": "a", "t": 1.0, "d": 0.25}, {"w": "b", "t": 1.5, "d": 0.5}]},
                     {"text": "c", "words": [{"w": "c", "t": 4.0, "d": 1.0}]}],
           "extras": [{"t": 6.0, "d": 1.0}], "onsets": [{"t": 1.0, "s": 0.7, "kind": "flux"}]}

    def build(self, raw):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "raw_lyrics.json").write_text(json.dumps(raw), encoding="utf-8")
            return A.build_lyrics(A.Grid(dict(self.GRID)), Path(tmp))

    def test_words_without_conf_have_no_v(self):
        lyrics, onsets = self.build(self.RAW)
        self.assertEqual(list(lyrics["words"]), ["b", "len", "w", "line", "syl"])
        self.assertEqual(lyrics["words"]["b"], [1.0, 2.0, 7.0])
        self.assertEqual(lyrics["words"]["len"], [0.5, 1.0, 2.0])
        self.assertEqual(lyrics["lines"], {"b": [1.0, 7.0], "len": [2.0, 2.0], "text": ["a b", "c"]})
        self.assertEqual(lyrics["extras"], {"b": [11.0], "len": [2.0]})
        self.assertEqual(lyrics["source"], self.RAW["source"])
        self.assertEqual(onsets["b"], [1.0])
        # the exporter passes a v-less words block through
        self.assertNotIn("v", E.build_export({"version": 2, "lyrics": lyrics})["lyrics"]["words"])
        # and the lyric-entry refinement reads it as it reads aligned words
        self.assertEqual(A.lyric_entries(lyrics["words"]), [1.0, 7.0])  # 4 beats of rest before "c"

    def test_a_mix_of_aligned_and_hand_timed_words_is_refused(self):
        raw = json.loads(json.dumps(self.RAW))
        raw["lines"][0]["words"][0]["conf"] = 0.9
        with self.assertRaisesRegex(ValueError, "no conf"):
            self.build(raw)


def events(**extra):
    ev = {"version": 2, "source": {"tools": {}}, "tempo": [{"beat": 0, "bpm": 120.0}], "gridMarker": 0.5,
          "meter": 4, "lengthBeats": 64, "sections": [],
          "lyrics": {"lines": {"b": [2], "len": [3], "text": ["one two"]},
                     "words": {"b": [2, 3], "len": [0.5, 1], "v": [0.9, 0.4], "w": ["one", "two"],
                               "line": [0, 0], "syl": [None, None]},
                     "extras": {"b": [], "len": []}, "source": "forced alignment (test)"}}
    ev.update(extra)
    return ev


TRACK = {"title": "T", "bpm": 120, "gridMarker": 0.5,
         "lyrics": {"lines": [{"words": [{"t": 1.5, "d": 1.76, "w": "one two"}]}]}}


class ExportLyrics(unittest.TestCase):
    def export(self, listed, ev=None, all_=False):
        """Export slug s -> track `test` with --lyrics; returns (track before, track after)."""
        with tempfile.TemporaryDirectory() as tmp:
            out, repo = Path(tmp) / "out", Path(tmp) / "repo"
            (out / "s").mkdir(parents=True)
            (out / "s" / "events.json").write_text(json.dumps(ev or events()), encoding="utf-8")
            music = repo / "public" / "music"
            (music / "events").mkdir(parents=True)
            (music / "index.json").write_text('[\n  "test.json"\n]\n', encoding="utf-8")
            before = json.dumps(TRACK, indent=2) + "\n"
            (music / "test.json").write_text(before, encoding="utf-8")
            config = Path(tmp) / "config.json"
            write_config(config, "test", listed)
            with mock.patch.object(E, "OUT", out), mock.patch.object(workspace, "CONFIG", config):
                if all_:
                    argv = ["export_app.py", "--all", "--lyrics", "--app", str(repo)]
                    with mock.patch.object(sys, "argv", argv):
                        E.main()
                else:
                    E.export("s", "test", repo, lyrics=True)
            self.assertTrue((music / "events" / "test.events.json").exists())  # the rest of the export ran
            return json.loads(before), json.loads((music / "test.json").read_text(encoding="utf-8"))

    def test_an_unlisted_track_gets_the_aligned_lyrics(self):
        """The control: the same export does write when the track is not hand-timed."""
        before, after = self.export(listed=False)
        self.assertNotEqual(after["lyrics"], before["lyrics"])
        self.assertEqual(after["lyrics"], E.app_lyrics(events()))

    def test_a_hand_timed_track_keeps_its_lyrics(self):
        before, after = self.export(listed=True)
        self.assertEqual(after["lyrics"], before["lyrics"])

    def test_a_hand_timed_track_keeps_its_lyrics_under_all(self):
        before, after = self.export(listed=True, all_=True)
        self.assertEqual(after["lyrics"], before["lyrics"])
        _, control = self.export(listed=False, all_=True)
        self.assertNotEqual(control["lyrics"], before["lyrics"])

    def test_skipped_even_when_the_events_have_no_lyrics(self):
        """A hand-timed track is skipped, not refused: --lyrics never stops its export."""
        ev = events()
        del ev["lyrics"]
        before, after = self.export(listed=True, ev=ev)
        self.assertEqual(after["lyrics"], before["lyrics"])

    def test_write_lyrics_itself_refuses_a_hand_timed_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp) / "test.json"
            tpath.write_text(json.dumps(TRACK), encoding="utf-8")
            config = Path(tmp) / "config.json"
            write_config(config, "test", listed=True)
            before = tpath.read_bytes()
            with mock.patch.object(workspace, "CONFIG", config):
                E.write_lyrics(tpath, events())
            self.assertEqual(tpath.read_bytes(), before)

    def test_app_lyrics_without_v_writes_no_c(self):
        ev = events()
        del ev["lyrics"]["words"]["v"]
        words = [w for ln in E.app_lyrics(ev)["lines"] for w in ln["words"]]
        self.assertEqual([set(w) for w in words if w["w"]], [{"t", "d", "w"}] * 2)


class ExampleConfig(unittest.TestCase):
    """config.example.json, the committed template, is a valid config and says what it means."""

    def test_the_example_loads_and_marks_its_hand_timed_track(self):
        path = TOOL / "config.example.json"
        self.assertEqual(list(workspace.tracks(path)), ["track-a", "track-b"])
        self.assertEqual(workspace.all_slugs(path), ["track-a"])  # track-b is "all": false
        self.assertTrue(workspace.hand_timed("track_b", path))
        self.assertIsNone(workspace.hand_timed("track_a", path))
        self.assertEqual(workspace.app_track("track-a", path), "track_a")
        self.assertEqual(workspace.pron_table("track_b", path), {"DJ": "dee jay"})


if __name__ == "__main__":
    unittest.main()
