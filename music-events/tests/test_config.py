"""The workspace config.json and the app checkout: how --all, the app and the lyric text are
resolved, and that a wrong path fails loudly before anything is written. Run from
music-events/:

    <venv-main>/Scripts/python.exe -m unittest discover -s tests

Every file here is a temp copy: nothing reads the real config.json or app checkout.
"""
import json
import os
import subprocess
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
import export_app as E  # noqa: E402
import grid as G  # noqa: E402
import lyrics as L  # noqa: E402
import run_all as R  # noqa: E402
import workspace  # noqa: E402

CLEAN_ENV = {k: v for k, v in os.environ.items() if k not in (workspace.APP_ENV, "LOCAL_MUSIC_PATH")}


def make_app(root):
    """A folder that looks like the app: public/music/index.json."""
    music = Path(root) / "public" / "music"
    music.mkdir(parents=True)
    (music / "index.json").write_text("[]\n", encoding="utf-8")
    return Path(root)


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.config = self.tmp / "config.json"
        self._patches = [mock.patch.object(workspace, "CONFIG", self.config),
                         mock.patch.dict(os.environ, CLEAN_ENV, clear=True)]
        for p in self._patches:
            p.start()

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def write_config(self, cfg):
        self.config.write_text(json.dumps(cfg), encoding="utf-8")


class AppRoot(Tmp):
    def setUp(self):
        super().setUp()
        self.a, self.b, self.c = (make_app(self.tmp / n) for n in "abc")

    def test_order_flag_then_env_then_config(self):
        self.write_config({"app": str(self.c)})
        with mock.patch.dict(os.environ, {workspace.APP_ENV: str(self.b)}):
            self.assertEqual(workspace.app_root(str(self.a)), self.a.resolve())
            self.assertEqual(workspace.app_root(None), self.b.resolve())
        self.assertEqual(workspace.app_root(None), self.c.resolve())

    def test_none_set_is_an_error(self):
        with self.assertRaises(SystemExit) as cm:
            workspace.app_root(None)
        self.assertIn("--app", str(cm.exception.code))
        self.assertIn(workspace.APP_ENV, str(cm.exception.code))

    def test_a_folder_that_is_not_the_app_is_an_error_naming_its_source(self):
        bad = self.tmp / "not-the-app"
        bad.mkdir()
        self.write_config({"app": str(bad)})
        for cli, env, source in ((str(bad), None, "--app"), (None, str(bad), workspace.APP_ENV),
                                 (None, None, '"app" in')):
            with mock.patch.dict(os.environ, {workspace.APP_ENV: env} if env else {}), \
                 self.assertRaises(SystemExit) as cm:
                workspace.app_root(cli)
            self.assertIn(source, str(cm.exception.code))
            self.assertIn("index.json", str(cm.exception.code))

    def test_an_invalid_flag_does_not_fall_through_to_a_valid_env(self):
        with mock.patch.dict(os.environ, {workspace.APP_ENV: str(self.b)}), self.assertRaises(SystemExit):
            workspace.app_root(str(self.tmp / "missing"))


class DefaultWorkspace(unittest.TestCase):
    """MUSIC_EVENTS_WORKSPACE is handled at import; default_workspace is the rest of the rule."""

    def test_two_levels_up_when_that_folder_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            main = Path(tmp) / "group" / "clone"
            main.mkdir(parents=True)
            (Path(tmp) / "music-events").mkdir()
            self.assertEqual(workspace.default_workspace(main), Path(tmp) / "music-events")

    def test_a_sibling_of_the_clone_otherwise(self):
        with tempfile.TemporaryDirectory() as tmp:
            main = Path(tmp) / "group" / "clone"
            main.mkdir(parents=True)
            self.assertEqual(workspace.default_workspace(main), Path(tmp) / "group" / "music-events")
            (Path(tmp) / "music-events").write_text("a file, not a folder", encoding="utf-8")
            self.assertEqual(workspace.default_workspace(main), Path(tmp) / "group" / "music-events")

    def test_a_checkout_at_or_next_to_a_root(self):
        root = Path(Path.cwd().anchor or "/")
        self.assertEqual(workspace.default_workspace(root), root / "music-events")
        # one level below the root: two levels up IS the root, so only an existing
        # <root>/music-events could be taken, and it then equals the sibling anyway
        self.assertEqual(workspace.default_workspace(root / "clone"), root / "music-events")

    def test_the_env_var_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(CLEAN_ENV, MUSIC_EVENTS_WORKSPACE=tmp)
            out = subprocess.run([sys.executable, "-c", "import workspace; print(workspace.WORKSPACE)"],
                                 cwd=TOOL, env=env, capture_output=True, text=True, check=True)
            self.assertEqual(Path(out.stdout.strip()), Path(tmp))


class ConfigFile(Tmp):
    def test_missing_file_is_empty(self):
        self.assertEqual(workspace.load_config(), {})
        self.assertEqual(workspace.tracks(), {})

    def test_unknown_keys_stop_the_run(self):
        for cfg in ({"apps": "x"}, {"tracks": {"s": {"appTrak": "t"}}}):
            self.write_config(cfg)
            with self.assertRaises(SystemExit):
                workspace.load_config()

    def test_wrong_value_types_stop_the_run(self):
        for cfg in ({"app": 5}, {"musicDir": ["x"]}, {"tracks": {"s": {"bpm": "120"}}},
                    {"tracks": {"s": {"appTrack": None}}}, {"tracks": {"s": {"pron": ["DJ"]}}},
                    {"tracks": {"s": {"gridMarker": True}}}):
            self.write_config(cfg)
            with self.assertRaises(SystemExit) as cm:
                workspace.load_config()
            self.assertIn("must be", str(cm.exception.code))

    def test_underscore_keys_are_comments(self):
        self.write_config({"_comment": ["x"], "tracks": {"s": {"_note": "y", "appTrack": "t"}}})
        self.assertEqual(workspace.app_track("s"), "t")

    def test_music_dir_env_then_config_then_default(self):
        self.write_config({"musicDir": str(self.tmp / "cfg")})
        with mock.patch.dict(os.environ, {"LOCAL_MUSIC_PATH": str(self.tmp / "env")}):
            self.assertEqual(workspace.music_dir(), self.tmp / "env")
        self.assertEqual(workspace.music_dir(), self.tmp / "cfg")
        self.write_config({})
        self.assertEqual(workspace.music_dir(), Path.home() / "Music" / "Live")


class All(Tmp):
    CFG = {"tracks": {"track-b": {"appTrack": "b"}, "track-a": {"appTrack": "a"}, "seek-only": {"file": "x.mp3"}}}

    def test_all_is_the_config_slugs_in_file_order(self):
        self.write_config(self.CFG)
        self.assertEqual(workspace.all_slugs(), ["track-b", "track-a", "seek-only"])

    def test_all_false_is_left_out_of_every_all_but_reachable_by_slug(self):
        cfg = {"app": str(make_app(self.tmp / "app")), "tracks": {
            "in": {"appTrack": "a", "file": "a.mp3"},
            "out": {"appTrack": "b", "file": "b.mp3", "all": False},
            "on": {"appTrack": "c", "all": True}}}
        self.write_config(cfg)
        self.assertEqual(workspace.all_slugs(), ["in", "on"])
        calls = []
        with mock.patch.object(E, "export", lambda *a, **k: calls.append(a[:2])):
            for argv in (["--all"], ["--slug", "out"]):
                with mock.patch.object(sys, "argv", ["export_app.py", *argv]):
                    E.main()
        self.assertEqual(calls, [("in", "a"), ("on", "c"), ("out", "b")])
        ran = []
        with mock.patch.object(L, "run", lambda s, *a, **k: ran.append(s)):
            for argv in (["--all"], ["--slug", "out"]):
                with mock.patch.object(sys, "argv", ["lyrics.py", *argv]):
                    L.main()
        self.assertEqual(ran, ["in", "on", "out"])
        sys.path.insert(0, str(TOOL / "measure"))
        import seek_formats_prep as P
        self.assertEqual(P.files(None), {"in": "a.mp3"})
        self.assertEqual(P.files(["out"]), {"out": "b.mp3"})

    def test_all_must_be_a_boolean(self):
        for v in ("false", 0, None):
            self.write_config({"tracks": {"s": {"all": v}}})
            with self.assertRaises(SystemExit) as cm:
                workspace.load_config()
            self.assertIn("true or false", str(cm.exception.code))

    def test_all_without_tracks_is_an_error(self):
        with self.assertRaises(SystemExit) as cm:
            workspace.all_slugs()
        self.assertIn("--slug", str(cm.exception.code))

    def test_export_all_takes_every_slug_with_an_app_track(self):
        self.write_config(self.CFG)
        app = make_app(self.tmp / "app")
        calls = []
        with mock.patch.object(E, "export", lambda *a, **k: calls.append(a)), \
             mock.patch.object(sys, "argv", ["export_app.py", "--all", "--app", str(app)]):
            E.main()
        self.assertEqual(calls, [("track-b", "b", app.resolve()), ("track-a", "a", app.resolve())])

    def test_export_slug_track_defaults_to_the_config(self):
        self.write_config(dict(self.CFG, app=str(make_app(self.tmp / "app"))))
        calls = []
        with mock.patch.object(E, "export", lambda *a, **k: calls.append(a[:2])):
            for argv in (["--slug", "track-a"], ["--slug", "other", "--track", "t"]):
                with mock.patch.object(sys, "argv", ["export_app.py", *argv]):
                    E.main()
            with mock.patch.object(sys, "argv", ["export_app.py", "--slug", "seek-only"]), \
                 self.assertRaises(SystemExit):
                E.main()
        self.assertEqual(calls, [("track-a", "a"), ("other", "t")])

    def test_lyrics_all_runs_every_config_slug(self):
        self.write_config(self.CFG)
        ran = []
        with mock.patch.object(L, "run", lambda s, *a, **k: ran.append(s)), \
             mock.patch.object(sys, "argv", ["lyrics.py", "--all"]):
            L.main()
        self.assertEqual(ran, ["track-b", "track-a", "seek-only"])

    def test_export_track_meta_comes_from_the_config(self):
        self.write_config({"tracks": {"s": {"appTrack": "t", "title": "Artist - Title", "file": "Artist - Title.mp3"},
                                      "u": {"appTrack": "u"}}})
        self.assertEqual(E.track_meta("s"), {"track": "t", "title": "Artist - Title",
                                             "url": "/local-music/Artist - Title.mp3"})
        self.assertEqual(E.track_meta("u"), {"track": "u"})

    def test_grid_ground_truth_comes_from_the_config(self):
        self.write_config({"tracks": {"s": {"bpm": 124, "gridMarker": 0.5}, "u": {"bpm": 124}}})
        self.assertEqual(G.ground_truth("s"), {"bpm": 124.0, "gridMarker": 0.5})
        self.assertIsNone(G.ground_truth("u"))
        self.assertIsNone(G.ground_truth("nope"))


class LyricsMissingTrack(Tmp):
    """A named track whose file is missing is an error, and the stage writes nothing: never
    the old silent run that overwrote raw_lyrics.json with no words and exited 0."""

    def setUp(self):
        super().setUp()
        self.app = make_app(self.tmp / "app")
        self.out = self.tmp / "out"
        stems = self.out / "s" / "stems"
        stems.mkdir(parents=True)
        sr = 22050
        t = np.arange(sr * 2) / sr
        sf.write(str(stems / "vocals.wav"), np.stack([0.3 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1), sr)
        self.raw = self.out / "s" / "raw_lyrics.json"
        self.raw.write_text('{"previous": "run"}', encoding="utf-8")  # an earlier run's output
        self.before = sorted(p.relative_to(self.out) for p in self.out.rglob("*"))
        self._patches += [mock.patch.object(L, "OUT", self.out), mock.patch.object(L, "APP", None),
                          mock.patch.object(L, "APP_ARG", None)]
        for p in self._patches[2:]:
            p.start()

    def assert_nothing_written(self):
        self.assertEqual(sorted(p.relative_to(self.out) for p in self.out.rglob("*")), self.before)
        self.assertEqual(self.raw.read_text(encoding="utf-8"), '{"previous": "run"}')

    def test_named_track_missing(self):
        self.write_config({"app": str(self.app)})
        with self.assertRaises(SystemExit) as cm:
            L.run("s", "nope")
        self.assertIn("nope.json not found", str(cm.exception.code))
        self.assert_nothing_written()

    def test_config_track_missing(self):
        self.write_config({"app": str(self.app), "tracks": {"s": {"appTrack": "nope"}}})
        with self.assertRaises(SystemExit):
            L.run("s")
        self.assert_nothing_written()

    def test_hand_timed_track_missing(self):
        self.write_config({"app": str(self.app), "tracks": {"s": {"appTrack": "nope", "handTimed": "x"}}})
        with self.assertRaises(SystemExit):
            L.run("s")
        self.assert_nothing_written()

    def test_no_app_resolvable(self):
        self.write_config({"tracks": {"s": {"appTrack": "t"}}})
        with self.assertRaises(SystemExit) as cm:
            L.run("s")
        self.assertIn("--app", str(cm.exception.code))
        self.assert_nothing_written()

    def test_main_exits_non_zero(self):
        self.write_config({"app": str(self.app)})
        with mock.patch.object(sys, "argv", ["lyrics.py", "--slug", "s", "--track", "nope"]), \
             self.assertRaises(SystemExit) as cm:
            L.main()
        self.assertNotIn(cm.exception.code, (0, None))
        self.assert_nothing_written()

    def test_an_instrumental_still_gets_onsets(self):
        """The control: a track file WITHOUT lyrics is not an error (vocal onsets only)."""
        (self.app / "public" / "music" / "inst.json").write_text('{"title": "T"}', encoding="utf-8")
        self.write_config({"app": str(self.app), "tracks": {"s": {"appTrack": "inst"}}})
        doc = L.run("s")
        self.assertEqual(doc["lines"], [])
        self.assertNotEqual(self.raw.read_text(encoding="utf-8"), '{"previous": "run"}')

    def test_no_track_named_does_not_need_the_app(self):
        """No --track and no config entry: onsets only, and the app is never resolved."""
        with mock.patch.object(L, "app_root", side_effect=AssertionError("resolved the app")):
            doc = L.run("s")
        self.assertIsNone(doc["track"])


class RunAll(Tmp):
    def setUp(self):
        super().setUp()
        self.app = make_app(self.tmp / "app")
        (self.app / "public" / "music" / "t.json").write_text("{}", encoding="utf-8")

    def run_all(self, *argv):
        cmds = []
        done = mock.Mock(returncode=0)
        with mock.patch.object(R.subprocess, "run", lambda cmd, *a, **k: cmds.append(cmd) or done), \
             mock.patch.object(sys, "argv", ["run_all.py", "x.mp3", "--slug", "s", "--from", "lyrics", *argv]):
            R.main()
        return cmds

    def test_the_app_is_passed_to_the_lyrics_stage(self):
        cmds = self.run_all("--track", "t", "--app", str(self.app))
        lyr = next(c for c in cmds if c[1].endswith("lyrics.py"))
        self.assertEqual(lyr[lyr.index("--track") + 1], "t")
        self.assertEqual(lyr[lyr.index("--app") + 1], str(self.app.resolve()))
        self.assertTrue(all("--app" not in c for c in cmds if c is not lyr))

    def test_the_config_app_is_resolved_and_passed_too(self):
        self.write_config({"app": str(self.app), "tracks": {"s": {"appTrack": "t"}}})
        cmds = self.run_all()
        lyr = next(c for c in cmds if c[1].endswith("lyrics.py"))
        self.assertEqual(lyr[lyr.index("--app") + 1], str(self.app.resolve()))

    def test_a_missing_track_file_fails_before_any_stage(self):
        cmds = []
        with mock.patch.object(R.subprocess, "run", lambda cmd, *a, **k: cmds.append(cmd)), \
             mock.patch.object(sys, "argv", ["run_all.py", "x.mp3", "--slug", "s", "--from", "grid",
                                             "--track", "nope", "--app", str(self.app)]), \
             self.assertRaises(SystemExit):
            R.main()
        self.assertEqual(cmds, [])

    def test_no_pre_check_when_the_lyrics_stage_will_not_run(self):
        """--from assemble never reads a track file, so a missing one (or no app) is no error."""
        cmds = []
        done = mock.Mock(returncode=0)
        with mock.patch.object(R.subprocess, "run", lambda cmd, *a, **k: cmds.append(cmd) or done), \
             mock.patch.object(sys, "argv", ["run_all.py", "x.mp3", "--slug", "s", "--from", "assemble",
                                             "--track", "nope", "--app", str(self.tmp / "not-the-app")]):
            R.main()
        self.assertTrue(cmds)
        self.assertTrue(all(not c[1].endswith("lyrics.py") for c in cmds))

    def test_no_track_needs_no_app(self):
        cmds = self.run_all()
        lyr = next(c for c in cmds if c[1].endswith("lyrics.py"))
        self.assertNotIn("--app", lyr)


if __name__ == "__main__":
    unittest.main()
