"""The events file's key vocabulary: assemble.py must emit the app's LANE names, not the
separator's or the detector's. Run from audio-events/:

    <venv-main>/Scripts/python.exe -m unittest discover -s tests

These names are a contract with the Fantasynth app's music lanes and with every consumer in
this folder (report.py, render_check.py, viewer.html), and nothing else here tests them:
a lane key typo would only show up as a silently empty lane after an export.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import assemble as A  # noqa: E402

# The lanes the app expects in events.json "drums", and the curve stems for a track
# separated by separate.py (DRUM_PARTS + the full-band stems + the mix).
DRUM_LANES = {"kick", "snare", "toms", "hatClosed", "hatOpen", "crash", "ride"}
CURVE_STEMS = {"mix", "vocals", "bass", "drums", "guitar", "piano", "other",
               "kick", "snare", "toms", "hats", "ride", "crash"}
GRID = {"t0": 0.0, "beatPeriod": 0.5, "meter": 4}


class DrumLanes(unittest.TestCase):
    def test_lane_names_are_the_app_s(self):
        """Every lane key, with ADTOF's singular "tom" class landing in "toms"."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            hits = lambda ts: [{"t": t, "vel": 0.9} for t in ts]
            (d / "raw_adtof.json").write_text(json.dumps({"classes": {
                "kick": hits([0.0, 0.5, 1.0, 1.5]), "snare": hits([0.5, 1.5]),
                "tom": hits([1.75]), "hihat": hits([0.25, 0.75]), "cymbal": hits([0.0])}}))
            (d / "raw_adtof_mix.json").write_text(json.dumps({"classes": {
                "tom": hits([1.75]), "cymbal": hits([0.0])}}))
            drums, stats, _ = A.build_drums(A.Grid(dict(GRID)), d)
        self.assertEqual(set(drums), DRUM_LANES)
        self.assertEqual(set(stats["hits"]), DRUM_LANES)
        # The tom voted by both ADTOF passes, away from any kick, reaches the "toms" lane.
        self.assertEqual(drums["toms"]["b"], [3.5])

    def test_no_drums_without_the_stem_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            drums, stats, _ = A.build_drums(A.Grid(dict(GRID)), Path(tmp))
        self.assertIsNone(drums)


class CurveStems(unittest.TestCase):
    def curves(self, npz_stems):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            n = 800
            arrays = {f"{s}_rms_db": np.linspace(-60, 0, n, dtype=np.float32) for s in npz_stems}
            np.savez_compressed(d / "curves.npz", fps=np.float32(100), **arrays)
            return A.build_curves(A.Grid(dict(GRID)), d, 8)

    def test_the_hi_hat_stem_is_exported_as_hats(self):
        """curves.npz keys follow the DrumSep stem FILE names (drums_hh.wav -> "hh"), which
        must not be renamed. assemble renames on the READ side, so a cached curves.npz
        exports the app's name without being recomputed."""
        npz_stems = (CURVE_STEMS - {"hats"}) | {"hh"}
        curves, per_beat = self.curves(npz_stems)
        self.assertEqual(set(curves["stems"]), CURVE_STEMS)
        self.assertNotIn("hh", curves["stems"])
        self.assertNotIn("hh", per_beat)
        self.assertEqual(set(curves["stems"]["hats"]), {"rms_db"})

    def test_the_rename_is_idempotent(self):
        """A curves.npz already written with the app's name passes straight through."""
        curves, _ = self.curves(CURVE_STEMS)
        self.assertEqual(set(curves["stems"]), CURVE_STEMS)

    def test_no_curves_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(A.build_curves(A.Grid(dict(GRID)), Path(tmp), 8), (None, None))


def word(w, t, d, conf=0.9, syl=None):
    out = {"w": w, "t": t, "d": d, "conf": conf, "agree": 1.0, "rule": "ctc", "ctc": [t, d]}
    if syl:
        out["syl"] = syl
    return out


class Lyrics(unittest.TestCase):
    RAW = {
        "version": 1, "track": "t", "margin": 1.5,
        "lines": [
            {"text": "one two", "words": [word("one", 1.0, 0.25), word("two", 1.5, 0.5, 0.4)]},
            {"text": "DJ", "words": [word("DJ", 5.0, 1.0, syl=[[5.0, 0.5], [5.5, 0.5]])]},
        ],
        "extras": [{"t": 10.0, "d": 2.0}],
        "onsets": [{"t": 1.0, "s": 0.7, "kind": "flux"}, {"t": 5.25, "s": 0.3, "kind": "pitch"}],
    }

    def build(self, raw, grid=None):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            if raw is not None:
                (d / "raw_lyrics.json").write_text(json.dumps(raw))
            return A.build_lyrics(grid or A.Grid(dict(GRID)), d)

    def test_beats_come_from_the_shifted_grid(self):
        """build_drums moves t0 onto the kicks; lyrics must use that grid, not grid.json's."""
        grid = A.Grid(dict(GRID))
        grid.t0 += 0.25  # as build_drums would
        lyrics, onsets = self.build(self.RAW, grid)
        self.assertEqual(lyrics["words"]["b"], [1.5, 2.5, 9.5])
        self.assertEqual(lyrics["lines"]["b"], [1.5, 9.5])
        self.assertEqual(onsets["b"], [1.5, 10.0])

    def test_non_ascii_lyrics_survive(self):
        """raw_lyrics.json is UTF-8; reading it with the platform encoding (cp1252 on
        Windows) turned an em dash into mojibake ("laâ€”") in a real export."""
        raw = dict(self.RAW, lines=[{"text": "la la— la’s", "words": [
            word("la", 1.0, 0.2), word("la—", 1.2, 0.2), word("la’s", 1.4, 0.3)]}], extras=[])
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "raw_lyrics.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
            lyrics, _ = A.build_lyrics(A.Grid(dict(GRID)), d)
        self.assertEqual(lyrics["words"]["w"], ["la", "la—", "la’s"])
        self.assertEqual(lyrics["lines"]["text"], ["la la— la’s"])

    def test_legato_words_still_tile_in_beats(self):
        """Words that tile in seconds must tile in beats: rounding b and len separately
        made b[i] + len[i] exceed b[i+1] by a rounding step (seen in real exports)."""
        # invented values; a period that is not a round number of seconds, so beats do not
        # come out exact (at 120 bpm they would and the test could not fail)
        grid = A.Grid({"t0": 0.25, "beatPeriod": 60 / 122, "meter": 4})
        t, raw_words = 3.0, []
        for i, dur in enumerate([0.31, 0.44, 0.23, 0.56, 0.31]):
            raw_words.append(word(f"w{i}", round(t, 3), round(round(t + dur, 3) - round(t, 3), 3)))
            t += dur
        raw = dict(self.RAW, lines=[{"text": "w", "words": raw_words}], extras=[])
        lyrics, _ = self.build(raw, grid)
        b, n = lyrics["words"]["b"], lyrics["words"]["len"]
        for i in range(len(b) - 1):
            self.assertLessEqual(round(b[i] + n[i], 3), b[i + 1])

    def test_lengths(self):
        lyrics, _ = self.build(self.RAW)
        self.assertEqual(lyrics["words"]["len"], [0.5, 1.0, 2.0])
        # A line runs from its first word's start to its last word's end.
        self.assertEqual(lyrics["lines"]["len"], [2.0, 2.0])
        self.assertEqual(lyrics["lines"]["text"], ["one two", "DJ"])
        self.assertEqual(lyrics["extras"], {"b": [20.0], "len": [4.0]})

    def test_word_fields(self):
        lyrics, onsets = self.build(self.RAW)
        w = lyrics["words"]
        self.assertEqual(w["w"], ["one", "two", "DJ"])
        self.assertEqual(w["v"], [0.9, 0.4, 0.9])
        self.assertEqual(w["line"], [0, 0, 1])
        # syl is null for a single-span word, [[b, len], ...] otherwise.
        self.assertEqual(w["syl"], [None, None, [[10.0, 1.0], [11.0, 1.0]]])
        self.assertEqual(lyrics["source"], A.LYRICS_SOURCE)
        self.assertEqual(onsets["v"], [0.7, 0.3])
        self.assertEqual(onsets["kind"], [0, 1])

    def test_not_snapped_and_rounded(self):
        raw = {"lines": [{"text": "x", "words": [word("x", 1.0123456, 0.1)]}], "extras": [], "onsets": []}
        lyrics, _ = self.build(raw)
        self.assertEqual(lyrics["words"]["b"], [2.025])  # 2.0246912 -> 3 dp, not the 16th grid

    def test_omitted_when_empty(self):
        self.assertEqual(self.build(None), (None, None))
        empty = {"lines": [], "extras": [{"t": 1.0, "d": 1.0}], "onsets": []}
        self.assertEqual(self.build(empty), (None, None))
        lyrics, onsets = self.build({**self.RAW, "onsets": []})
        self.assertIsNotNone(lyrics)
        self.assertIsNone(onsets)
        lyrics, onsets = self.build({**self.RAW, "lines": []})
        self.assertIsNone(lyrics)
        self.assertEqual(onsets["kind"], [0, 1])


def sections(*bounds):
    """Sections from boundary bars: sections(0, 8, 16) -> [0-8), [8-16)."""
    return [{"startBar": a, "endBar": z, "label": "x"} for a, z in zip(bounds, bounds[1:])]


def starts(secs):
    return [s["startBar"] for s in secs]


class VocalEntries(unittest.TestCase):
    def test_entries_are_line_starts_after_a_gap(self):
        words = {"b": [2.0, 3.0, 5.0, 10.0, 11.0, 30.0],
                 "len": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 "line": [0, 0, 1, 2, 2, 3]}
        # First line; line 1 follows 1 beat after line 0 ends; line 2 is 4 beats after
        # line 1 ends (6 -> 10); line 3 far later.
        self.assertEqual(A.lyric_entries(words), [2.0, 10.0, 30.0])

    def test_a_gap_inside_a_line_is_not_an_entry(self):
        words = {"b": [0.0, 20.0], "len": [1.0, 1.0], "line": [0, 0]}
        self.assertEqual(A.lyric_entries(words), [0.0])

    def test_pickup_rule(self):
        # pos 1 in bar 9 (beat 37): a long pickup, the section starts at bar 9.
        self.assertEqual(A.entry_bar(37.0), 9)
        # pos 3 in bar 9 (beat 39): a short pickup into bar 10.
        self.assertEqual(A.entry_bar(39.0), 10)
        self.assertEqual(A.entry_bar(37.99), 9)
        self.assertEqual(A.entry_bar(38.0), 10)
        self.assertEqual(A.entry_bar(-0.5), 0)

    def test_one_bar_away_moves_both_pickup_ways(self):
        secs, moved = A.apply_vocal_entries(sections(0, 8, 16, 24), [4 * 9 + 1])  # pos 1 -> bar 9
        self.assertEqual(starts(secs), [0, 9, 16])
        self.assertEqual(moved, 1)
        self.assertEqual(secs[0]["endBar"], 9)
        self.assertEqual(secs[1]["lyricsBar"], 9)
        secs, moved = A.apply_vocal_entries(sections(0, 8, 16, 24), [4 * 6 + 3])  # pos 3 -> bar 7
        self.assertEqual(starts(secs), [0, 7, 16])
        self.assertEqual((moved, secs[0]["endBar"]), (1, 7))

    def test_two_bars_away_does_not_move(self):
        secs, moved = A.apply_vocal_entries(sections(0, 8, 16, 24), [4 * 10])
        self.assertEqual((starts(secs), moved), ([0, 8, 16], 0))
        self.assertNotIn("lyricsBar", secs[1])

    def test_first_and_last_edges_never_move(self):
        secs, moved = A.apply_vocal_entries(sections(0, 8, 16), [4 * 1, 4 * 15])  # bars 1 and 15
        self.assertEqual((starts(secs), secs[-1]["endBar"], moved), ([0, 8], 16, 0))

    def test_no_section_under_two_bars(self):
        # Moving 8 -> 9 would leave [9, 10) one bar long.
        secs, moved = A.apply_vocal_entries(sections(0, 8, 10, 16), [4 * 9])
        self.assertEqual((starts(secs), moved), ([0, 8, 10], 0))
        # Moving 8 -> 7 would leave [6, 7) one bar long.
        secs, moved = A.apply_vocal_entries(sections(0, 6, 8, 16), [4 * 7])
        self.assertEqual((starts(secs), moved), ([0, 6, 8], 0))

    def test_a_boundary_on_an_entry_bar_stays(self):
        # Bar 8 is itself an entry; the entry at bar 9 must not pull it away.
        secs, moved = A.apply_vocal_entries(sections(0, 8, 16, 24), [4 * 8, 4 * 9])
        self.assertEqual((starts(secs), moved), ([0, 8, 16], 0))

    def test_label_source_records_the_move(self):
        grid_sections = sections(0, 8, 16, 24)
        out = A.relabel_sections(grid_sections, None, None, 96, vocal_entries=[4 * 9])
        self.assertEqual([s["startBeat"] for s in out], [0, 36, 64])
        self.assertIn("; lyrics entry bar 9", out[1]["labelSource"])
        self.assertNotIn("lyrics", out[0]["labelSource"])
        # Default None: unchanged behaviour.
        out = A.relabel_sections(grid_sections, None, None, 96)
        self.assertEqual([s["startBeat"] for s in out], [0, 32, 64])
        self.assertFalse(any("lyrics" in s["labelSource"] for s in out))


if __name__ == "__main__":
    unittest.main()
