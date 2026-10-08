"""export_app.py: the lyrics stage's blocks pass through the export, and --lyrics splices the
aligned words into a track file touching nothing but the "lyrics" value. Run from
music-events/:

    <venv-main>/Scripts/python.exe -m unittest discover -s tests

Every file here is a temp copy: nothing reads or writes the real public/music, and the
workspace config.json is replaced by a missing file for the whole module.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import export_app as E  # noqa: E402
import workspace  # noqa: E402

_NO_CONFIG = None


def setUpModule():
    """Never read the machine's real config.json: point CONFIG at a file that does not exist."""
    global _NO_CONFIG
    _NO_CONFIG = mock.patch.object(workspace, "CONFIG", Path(tempfile.gettempdir()) / "no-such-dir" / "config.json")
    _NO_CONFIG.start()


def tearDownModule():
    _NO_CONFIG.stop()

BPM, GM = 120.0, 0.5  # a beat is 0.5 s, beat 0 at 0.5 s

LYRICS = {
    "lines": {"b": [2, 10.5], "len": [3, 2], "text": ["one two three", "four five"]},
    "words": {"b": [2, 2.75, 4.1234, 10.5, 11.2], "len": [0.5, 0.25, 0.8766, 0.5, 1.3],
              "v": [0.856, 0.6, 0.9, 0.7, 0.955], "w": ["one", "two", "three", "four", "five"],
              "line": [0, 0, 0, 1, 1], "syl": [None, None, None, None, [[11.2, 0.6], [11.8, 0.7]]]},
    "extras": {"b": [20], "len": [4]},
    "source": "forced alignment (test)",
}
VOCAL_ONSETS = {"b": [1.98, 2.8, 6.0], "v": [0.71, 0.35, 0.5], "kind": [0, 1, 0]}


def events(**extra):
    ev = {"version": 2, "source": {"tools": {}}, "tempo": [{"beat": 0, "bpm": BPM}], "gridMarker": GM,
          "meter": 4, "lengthBeats": 64, "sections": [],
          "drums": {"kick": {"b": [0, 1], "v": [1, 1], "c": [1, 1], "swing": 0, "r": [0, 0]}}}
    ev.update(extra)
    return ev


def track_text(nl, with_lyrics=True):
    """A track file laid out like a hand-made app track file, hand-timed lyrics included."""
    lines = ['{',
             '  "title": "Test – Track",',
             '  "url": "/local-music/Test.flac",',
             '  "bpm": 120,',
             '  "gridMarker": 0.5,',
             '  "events": "events/test.events.json",',
             '  "key": "Gb Major",']
    if with_lyrics:
        lines += ['  "lyrics": {',
                  '    "lines": [',
                  '      {',
                  '        "words": [',
                  '          {',
                  '            "t": 1.5,',
                  '            "d": 1.76,',
                  '            "w": "one two three"',
                  '          }',
                  '        ]',
                  '      }',
                  '    ]',
                  '  },']
    lines += ['  "sections": [',
              '    {',
              '      "startBeat": 0,',
              '      "beatLength": 64',
              '    }',
              '  ]',
              '}']
    return nl.join(lines) + nl


class BuildExport(unittest.TestCase):
    def test_passes_lyrics_and_vocal_onsets_whole(self):
        out = E.build_export(events(lyrics=LYRICS, vocalOnsets=VOCAL_ONSETS))
        self.assertEqual(out["lyrics"], LYRICS)
        self.assertEqual(out["vocalOnsets"], VOCAL_ONSETS)
        self.assertEqual(list(out["lyrics"]["words"]), ["b", "len", "v", "w", "line", "syl"])
        self.assertNotIn("r", out["drums"]["kick"])  # the lane drops still apply

    def test_absent_blocks_stay_absent(self):
        out = E.build_export(events())
        self.assertNotIn("lyrics", out)
        self.assertNotIn("vocalOnsets", out)

    def test_unknown_fields_stop_the_export(self):
        bad_word = {**LYRICS, "words": {**LYRICS["words"], "ipa": ["aI"] * 5}}
        bad_block = {**LYRICS, "phonemes": {"b": []}}
        bad_onset = {**VOCAL_ONSETS, "hz": [220, 230, 240]}
        for ev in (events(lyrics=bad_word), events(lyrics=bad_block), events(vocalOnsets=bad_onset)):
            with self.assertRaises(SystemExit):
                E.build_export(ev)


class AppLyrics(unittest.TestCase):
    def test_seconds_grouping_and_rounding(self):
        lyr = E.app_lyrics(events(lyrics=LYRICS))
        sung = [ln for ln in lyr["lines"] if ln["words"][0]["w"]]
        self.assertEqual([[w["w"] for w in ln["words"]] for ln in sung], [["one", "two", "three"], ["four", "five"]])
        first, second, third = lyr["lines"][0]["words"]
        self.assertEqual(first, {"t": 1.5, "d": 0.25, "w": "one", "c": 0.86})  # 0.5 + 2 * 0.5; 0.856 -> 0.86
        self.assertEqual((second["t"], second["d"]), (1.875, 0.125))
        self.assertEqual((third["t"], third["d"]), (2.562, 0.438))  # 2.5617 / 0.4383 at 3 dp
        self.assertEqual(sung[1]["words"][1]["c"], 0.95)  # the float 0.955 is 0.95499..., so 0.95
        self.assertEqual(set(first), {"t", "d", "w", "c"})

    def test_rests_become_empty_lines(self):
        """The app shows a line until the next one starts, so a gap > REST_S (and the tail)
        is written as an empty line, as the hand-made files did with {"w": ""}."""
        lyr = E.app_lyrics(events(lyrics=LYRICS))
        texts = [" ".join(w["w"] for w in ln["words"]) for ln in lyr["lines"]]
        self.assertEqual(texts, ["one two three", "", "four five", ""])
        rest, tail = lyr["lines"][1]["words"][0], lyr["lines"][3]["words"][0]
        # line 0 ends at beat 5.0 = 3.0 s; line 1 starts at beat 10.5 = 5.75 s
        self.assertEqual((rest["t"], rest["d"]), (3.0, 2.75))
        # the song ends at lengthBeats 64 = 32.5 s; the last word ends at beat 12.5 = 6.75 s
        self.assertEqual((tail["t"], tail["d"]), (6.75, 25.75))

    def test_short_gaps_are_not_rests(self):
        ev = events(lyrics=LYRICS, lengthBeats=9)  # the last word ends at beat 8: a 0.5 s tail
        lines = dict(LYRICS["lines"], b=[2, 6])  # line 1 starts 0.5 s after line 0 ends
        words = dict(LYRICS["words"], b=[2, 2.75, 4.1234, 6, 6.7])
        ev["lyrics"] = dict(LYRICS, lines=lines, words=words)
        self.assertTrue(all(ln["words"][0]["w"] for ln in E.app_lyrics(ev)["lines"]))

    def test_uses_the_events_grid_at_full_precision(self):
        ev = events(lyrics=LYRICS, tempo=[{"beat": 0, "bpm": 122.0}], gridMarker=0.25)
        t = E.app_lyrics(ev)["lines"][0]["words"][0]["t"]
        self.assertEqual(t, round(0.25 + 2 * 60 / 122, 3))


class SpliceLyrics(unittest.TestCase):
    def splice(self, nl, with_lyrics=True):
        ev = events(lyrics=LYRICS)
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp) / "test.json"
            old = track_text(nl, with_lyrics).encode("utf-8")
            tpath.write_bytes(old)
            E.write_lyrics(tpath, ev)
            new = tpath.read_bytes()
            with mock.patch.object(Path, "write_bytes") as wb:
                E.write_lyrics(tpath, ev)  # unchanged events: a no-op
                wb.assert_not_called()
            self.assertEqual(tpath.read_bytes(), new)
        return old.decode("utf-8"), new.decode("utf-8"), E.app_lyrics(ev)

    def check_only_lyrics_changed(self, old, new, lyr, nl):
        self.assertEqual(json.loads(new)["lyrics"], lyr)
        _, _, ovs, ove = E.member(old, "lyrics")
        _, _, nvs, nve = E.member(new, "lyrics")
        self.assertEqual(old[:ovs], new[:nvs])  # every byte before the value
        self.assertEqual(old[ove:], new[nve:])  # and after it
        self.assertNotEqual(old[ovs:ove], new[nvs:nve])
        if nl == "\r\n":
            self.assertNotIn("\n", new.replace("\r\n", ""))
        else:
            self.assertNotIn("\r", new)
        # laid out like the hand-timed lyrics: one key per line, one indent deeper
        self.assertIn(f'{nl}            "t": 1.5,{nl}            "d": 0.25,{nl}', new)

    def test_crlf_track(self):
        old, new, lyr = self.splice("\r\n")
        self.check_only_lyrics_changed(old, new, lyr, "\r\n")

    def test_lf_track(self):
        old, new, lyr = self.splice("\n")
        self.check_only_lyrics_changed(old, new, lyr, "\n")

    def test_track_without_lyrics_gets_them_appended(self):
        old, new, lyr = self.splice("\n", with_lyrics=False)
        self.assertEqual(list(json.loads(new))[-1], "lyrics")
        self.assertEqual(json.loads(new)["lyrics"], lyr)
        last_ve = E.top_members(old)[-1][3]
        self.assertEqual(new[:last_ve], old[:last_ve])
        self.assertTrue(new.endswith("\n}\n"))

    def test_other_edits_may_not_touch_lyrics(self):
        old = track_text("\n")
        new = E.replace_value(old, "lyrics", '{"lines": []}')
        with self.assertRaises(AssertionError):
            E.verify_track(old, new, {"bpm": 120})


class Refusal(unittest.TestCase):
    def test_write_lyrics_refuses_without_a_lyrics_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            tpath = Path(tmp) / "test.json"
            tpath.write_bytes(track_text("\r\n").encode("utf-8"))
            before = tpath.read_bytes()
            with self.assertRaises(SystemExit) as cm:
                E.write_lyrics(tpath, events())
            self.assertIn("no lyrics block", str(cm.exception.code))
            self.assertEqual(tpath.read_bytes(), before)

    def test_export_refuses_before_writing_anything(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, repo = Path(tmp) / "out", Path(tmp) / "repo"
            (out / "s").mkdir(parents=True)
            (out / "s" / "events.json").write_text(json.dumps(events()), encoding="utf-8")
            music = repo / "public" / "music"
            music.mkdir(parents=True)
            (music / "test.json").write_bytes(track_text("\n").encode("utf-8"))
            before = sorted(p.relative_to(repo) for p in repo.rglob("*"))
            with mock.patch.object(E, "OUT", out), self.assertRaises(SystemExit) as cm:
                E.export("s", "test", repo, lyrics=True)
            self.assertIn("no lyrics block", str(cm.exception.code))
            self.assertEqual(sorted(p.relative_to(repo) for p in repo.rglob("*")), before)
            self.assertEqual((music / "test.json").read_bytes(), track_text("\n").encode("utf-8"))


if __name__ == "__main__":
    unittest.main()
