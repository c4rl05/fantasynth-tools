"""Synthetic-signal tests for report.py: each check must pass on a correct events file and
fail on a broken one. Run from audio-events/:

    <venv-main>/Scripts/python.exe -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import report as R  # noqa: E402

SR = R.SR
BPM, T0 = 120.0, 0.5          # beat period 0.5 s
N_BEATS = 32
RNG = np.random.default_rng(7)


def events(lanes, t0=T0):
    return {"gridMarker": t0, "tempo": [{"beat": 0, "bpm": BPM}], "sections": [],
            "drums": {k: {"b": list(b), "c": [1.0] * len(b)} for k, b in lanes.items()}}


def noise_hits(times, tau, dur=T0 + N_BEATS * 0.5 + 1.0, amp=0.5):
    """Noise bursts with a sharp onset and an exponential decay (time constant tau s)."""
    y = RNG.normal(0, 1e-5, int(dur * SR))
    k = np.arange(int(min(1.0, 8 * tau) * SR))
    shape = amp * np.exp(-k / SR / tau)
    for t in times:
        i = int(round(t * SR))
        n = min(len(k), len(y) - i)
        y[i:i + n] += shape[:n] * RNG.normal(0, 1, n)
    return y


def label(_):
    return "x"


class Envelopes(unittest.TestCase):
    def test_rise_window_is_the_preceding_frames_inclusive(self):
        x = np.array([0, 0, 0, 10, 10, 10, 10, 10.0])
        r = R.rise_db(x, 2)  # min over [i-2, i]
        self.assertEqual(r[3], 10)
        self.assertEqual(r[4], 10)
        self.assertEqual(r[5], 0)

    def test_nearest_dist_is_signed_b_minus_a(self):
        d = R.nearest_dist([1.0, 2.0], [1.01, 1.97])
        np.testing.assert_allclose(d, [0.01, -0.03])
        self.assertTrue(np.isinf(R.nearest_dist([1.0], [])).all())

    def test_clusters_split_on_gaps(self):
        cl = R.clusters([1, 2, 3, 20, 21, 40], label, gap=4)
        self.assertEqual([c["n"] for c in cl], [3, 2, 1])
        self.assertEqual(cl[0]["bars"], [1, 3])


class Lanes(unittest.TestCase):
    beats = np.arange(N_BEATS, dtype=float)

    def setUp(self):
        g = R.Grid(events({}))
        self.env, self.dt = R.rms_env_db(noise_hits(g.time(self.beats), tau=0.05), 5, 1)

    def check(self, beats, t0=T0):
        e = events({"kick": beats}, t0)
        g = R.Grid(e)
        return R.lane_check("kick", g, *R.lane(e, g, ["kick"]), self.env, self.dt, label)

    def test_correct_lane_is_solid(self):
        res = self.check(self.beats)
        self.assertEqual((res["precision"], res["recall"], res["verdict"]), (1.0, 1.0, "solid"))
        self.assertLess(abs(res["timingMedianMs"]), 3)

    def test_lane_40ms_late_fails(self):
        res = self.check(self.beats + 0.040 / 0.5)
        self.assertEqual(res["recall"], 0.0)
        self.assertEqual(res["verdict"], "check")

    def test_20ms_late_passes_the_window_but_not_timing(self):
        res = self.check(self.beats, t0=T0 + 0.020)
        self.assertEqual(res["recall"], 1.0)
        self.assertGreater(res["timingMedianMs"], R.LANE_TIMING_MS)
        self.assertNotEqual(res["verdict"], "solid")

    def test_half_the_lane_missing_shows_in_recall_and_clusters(self):
        res = self.check(self.beats[:16])
        self.assertLess(res["recall"], 0.6)
        self.assertEqual(res["misses"]["clusters"][0]["bars"], [4, 7])

    def test_silent_stem_is_judged_on_precision_alone(self):
        e = events({"kick": self.beats})
        g = R.Grid(e)
        env, dt = R.rms_env_db(RNG.normal(0, 1e-5, int(18 * SR)), 5, 1)
        res = R.lane_check("kick", g, *R.lane(e, g, ["kick"]), env, dt, label)
        self.assertEqual((res["onsets"], res["verdict"]), (0, "check"))
        self.assertIsNotNone(res["note"])

    def test_extra_hits_between_beats_lower_precision(self):
        res = self.check(np.concatenate([self.beats, self.beats[:16] + 0.5]))
        self.assertLess(res["precision"], 0.7)
        self.assertEqual(res["verdict"], "check")


class Grid(unittest.TestCase):
    def test_attack_offset(self):
        g = R.Grid(events({}))
        attacks = R.kick_attacks(noise_hits(g.time(np.arange(N_BEATS)), tau=0.05))
        good = R.grid_check(g, attacks)
        self.assertEqual(good["verdict"], "solid")
        self.assertLess(abs(good["medianMs"]), 3)
        bad = R.grid_check(R.Grid(events({}, T0 + 0.020)), attacks)
        self.assertAlmostEqual(bad["medianMs"], -20, delta=3)
        self.assertEqual(bad["verdict"], "check")


class Hats(unittest.TestCase):
    def test_open_rings_longer(self):
        g = R.Grid(events({}))
        closed_b, open_b = np.arange(0, N_BEATS, 2.0), np.arange(1, N_BEATS, 2.0)
        y = noise_hits(g.time(closed_b), tau=0.01) + noise_hits(g.time(open_b), tau=0.2)
        env, dt = R.rms_env_db(y, R.HATS_RMS_MS, 1)
        good = R.hats_check(g, events({"hatClosed": closed_b, "hatOpen": open_b}), env, dt)
        self.assertEqual((good["auc"], good["verdict"]), (1.0, "solid"))
        swapped = R.hats_check(g, events({"hatClosed": open_b, "hatOpen": closed_b}), env, dt)
        self.assertEqual((swapped["auc"], swapped["verdict"]), (0.0, "check"))

    def test_peak_search_stops_at_the_next_hit(self):
        # a quiet closed hat with a loud one 30 ms later: the loud one's peak must not be taken
        g = R.Grid(events({}))
        quiet_b, loud_b = np.arange(0, N_BEATS, 2.0), np.arange(0, N_BEATS, 2.0) + 0.06
        y = noise_hits(g.time(quiet_b), tau=0.01, amp=0.1) + noise_hits(g.time(loud_b), tau=0.01)
        env, dt = R.rms_env_db(y, R.HATS_RMS_MS, 1)
        res = R.hats_check(g, events({"hatClosed": quiet_b, "hatOpen": loud_b}), env, dt)
        self.assertGreaterEqual(res["hatClosed"]["ringP10P90Ms"][0], 0)
        self.assertLessEqual(res["hatClosed"]["ringP10P90Ms"][1], 30)
        self.assertEqual(res["hatClosed"]["cutByNextHit"], 1.0)

    def test_too_few_is_not_a_verdict(self):
        g = R.Grid(events({}))
        env, dt = R.rms_env_db(noise_hits(g.time([0, 1]), tau=0.01), R.HATS_RMS_MS, 1)
        self.assertEqual(R.hats_check(g, events({"hatClosed": [0], "hatOpen": [1]}), env, dt)["verdict"], "n/a")


class Curves(unittest.TestCase):
    def test_a_late_curve_reads_positive(self):
        import base64
        g = R.Grid(events({}))
        y = noise_hits(g.time(np.arange(0, N_BEATS, 0.5)), tau=0.03)
        csum = np.concatenate([[0.0], np.cumsum(y * y)])
        dur = len(y) / SR

        def encode(t):  # written independently of report.curve_check
            s = np.round(t * SR).astype(int)
            a, z = np.clip(s - 1024, 0, len(y)), np.clip(s + 1024, 0, len(y))
            db = 10 * np.log10((csum[z] - csum[a]) / 2048 + 1e-10)
            lo, hi = np.percentile(db, [5, 95])
            u8 = np.round(np.clip((db - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
            return base64.b64encode(u8.tobytes()).decode()

        t = g.time(np.arange(N_BEATS * 24) / 24)
        for late_ms, lag in ((0, 0), (20.8, 1), (-20.8, -1)):
            e = events({})
            e["curves"] = {"samplesPerBeat": 24, "stems": {"drums": {"rms_db": encode(t - late_ms / 1000)}}}
            res = R.curve_check(g, e, "drums", csum, dur)
            self.assertEqual(res["bestLag"], lag)
            self.assertAlmostEqual(res["bestOffsetMs"], late_ms, delta=2)
        self.assertEqual(R.curve_check(g, events({}), "drums", csum, dur)["verdict"], "n/a")


class Sections(unittest.TestCase):
    def test_degenerate_section_lists(self):
        nb = 16
        F = {k: np.zeros(nb) for k in ("drums", "bass", "mix", "kick")}
        self.assertEqual(R.sections_check({"sections": []}, F, nb)["verdict"], "check")
        neg = {"sections": [{"startBeat": -4, "beats": 4, "label": "intro"}, {"startBeat": 0, "beats": 64, "label": "drop"}]}
        self.assertEqual(R.sections_check(neg, F, nb)["boundaries"][0]["skip"], "before the audio")
        dup = {"sections": [{"startBeat": 0, "beats": 32, "label": "intro"}, {"startBeat": 32, "beats": 0, "label": "drop"},
                            {"startBeat": 32, "beats": 32, "label": "drop"}]}
        self.assertEqual(R.sections_check(dup, F, nb)["emptySections"], [8])


    def test_boundaries_and_unmarked_changes(self):
        nb = 32
        drums = np.full(nb, -40.0)
        drums[8:] = -20.0            # a real change at bar 8, marked
        bass = np.full(nb, -20.0)
        bass[22:] = -35.0            # a strong change at bar 22, unmarked
        drums[27] = -40.0            # an isolated one-bar dip, not a section
        F = {"drums": drums, "bass": bass, "mix": np.zeros(nb), "kick": np.zeros(nb)}
        e = {"sections": [{"startBeat": 0, "beats": 32, "label": "intro"},
                          {"startBeat": 32, "beats": 32, "label": "drop"},
                          {"startBeat": 64, "beats": 64, "label": "groove"}]}  # relabel at bar 16, no change
        res = R.sections_check(e, F, nb)
        self.assertEqual([r["change"] for r in res["boundaries"]], [True, False])
        self.assertEqual(res["relabelNoChange"], [16])
        self.assertEqual([u["bar"] for u in res["unmarkedStrong"]], [22])
        self.assertEqual(res["oneBarDips"], {"drums": [27]})
        self.assertEqual(res["verdict"], "mostly")

    def test_downbeat(self):
        e = {"sections": [{"startBeat": 0}, {"startBeat": 32}]}
        kb = np.array([0, 1, 2, 3, 16, 17, 32, 48])
        self.assertEqual(R.downbeat_check(e, kb, np.array([64]))["verdict"], "solid")
        self.assertEqual(R.downbeat_check(e, kb + 1, np.array([65]))["verdict"], "check")
        e["sections"].append({"startBeat": 34})
        self.assertEqual(R.downbeat_check(e, kb, np.array([64]))["sectionStartsOffBar"], [34])


def tone(f0, dur, amp=0.3, attack=0.015, release=0.03):
    """A sung-vowel stand-in: 10 harmonics at 1/k, linear attack and release."""
    t = np.arange(int(dur * SR)) / SR
    y = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 11))
    env = np.minimum(1, np.minimum(t / attack, (dur - t) / release)).clip(0)
    return amp * env * y / 2


def voice(parts, dur=20.0):
    """parts: [(start s, f0, length s)] summed into a quiet noise floor."""
    y = RNG.normal(0, 1e-5, int(dur * SR))
    for t, f0, ln in parts:
        s = tone(f0, ln)
        i = int(round(t * SR))
        y[i:i + s.size] += s[: y.size - i]
    return y


def lyric_events(starts, lens, v=None, extras=(), onsets=None, notes=None):
    """starts/lens in seconds, converted to beats like assemble does."""
    e = events({})
    b = [(t - T0) / 0.5 for t in starts]
    e["lyrics"] = {"lines": {"b": [b[0]], "len": [1.0], "text": ["a line"]},
                   "words": {"b": b, "len": [d / 0.5 for d in lens], "v": list(v or [0.9] * len(b)),
                             "w": ["w"] * len(b), "line": [0] * len(b), "syl": [None] * len(b)},
                   "extras": {"b": [(t - T0) / 0.5 for t, _ in extras], "len": [d / 0.5 for _, d in extras]}}
    return e


# eight separated words (entries), then a legato phrase: four words that tile with no level
# dip, only a change of pitch at each boundary
SEP = [1.0 + 0.7 * k for k in range(8)]
LEG = [7.0 + 0.4 * k for k in range(4)]
LEG_F0 = [220.0, 330.0, 262.0, 392.0]


class Lyrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parts = [(t, 220.0, 0.35) for t in SEP]
        y = voice(parts)
        # phase-continuous and constant in level inside the phrase: no click, no dip, only the pitch moves
        f0 = np.repeat(LEG_F0, int(0.4 * SR))
        ph = 2 * np.pi * np.cumsum(f0) / SR
        i = int(round(LEG[0] * SR))
        fade = np.minimum(1, np.minimum(np.arange(f0.size), f0.size - np.arange(f0.size)) / (0.015 * SR))
        y[i:i + f0.size] += 0.15 * fade * sum(np.sin(h * ph) / h for h in range(1, 11))
        cls.y = y
        cls.ev = R.vocal_evidence(y)
        cls.starts = SEP + LEG
        cls.lens = [0.35] * 8 + [0.4] * 4

    def test_correct_alignment_is_solid_and_legato_is_not_penalised(self):
        res = R.lyrics_check(lyric_events(self.starts, self.lens), R.Grid(events({})), self.ev)
        self.assertEqual((res["precision"], res["verdict"]), (1.0, "solid"))
        self.assertGreater(res["kappa"], 0.8)
        self.assertGreaterEqual(res["coverage"], 0.95)
        self.assertEqual(res["entries"]["n"], 9)  # the 8 separated words + the phrase's first word
        self.assertEqual(res["entries"]["precision"], 1.0)
        # the legato boundaries have no level rise: flux alone confirms them
        g = R.Grid(events({}))
        self.assertTrue((np.abs(R.nearest_dist(LEG[1:], self.ev["rise"])) > R.LYR_WIN).all())
        self.assertTrue((np.abs(R.nearest_dist(LEG[1:], self.ev["flux"])) <= R.LYR_WIN).all())
        self.assertLess(abs(res["timingMedianMs"]), 20)
        self.assertEqual((res["overlaps"], res["unordered"], res["lowConf"]), (0, 0, 0))

    def test_late_alignment_fails(self):
        res = R.lyrics_check(lyric_events([t + 0.15 for t in self.starts], self.lens), R.Grid(events({})), self.ev)
        self.assertLess(res["precision"], 0.3)
        self.assertEqual(res["verdict"], "check")

    def test_uncovered_voice_and_extras(self):
        # the legato phrase left out of the words: a coverage hole, filled back by an extra
        g = R.Grid(events({}))
        res = R.lyrics_check(lyric_events(SEP, [0.35] * 8), g, self.ev)
        self.assertLess(res["coverage"], 0.7)
        self.assertEqual(res["uncovered"]["n"], 1)
        self.assertAlmostEqual(res["uncovered"]["longest"][0][0], (7.0 - T0) / 0.5, delta=0.3)
        self.assertEqual(res["verdict"], "check")
        res = R.lyrics_check(lyric_events(SEP, [0.35] * 8, extras=[(7.0, 1.6)]), g, self.ev)
        self.assertGreaterEqual(res["coverage"], 0.95)
        self.assertLess(res["coverageWords"], 0.7)
        self.assertEqual(res["uncovered"]["n"], 0)

    def test_sanity_overlap_and_low_confidence(self):
        lens = list(self.lens)
        lens[2] = 1.0  # runs into the next word
        v = [0.9] * 12
        v[5] = 0.3
        res = R.lyrics_check(lyric_events(self.starts, lens, v=v), R.Grid(events({})), self.ev)
        self.assertEqual((res["overlaps"], res["lowConf"], res["verdict"]), (1, 1, "check"))
        self.assertEqual(res["durationMs"]["min"], 350)

    def test_na_and_silent_stem(self):
        g = R.Grid(events({}))
        self.assertEqual(R.lyrics_check(events({}), g, self.ev)["verdict"], "n/a")
        silent = R.vocal_evidence(RNG.normal(0, 1e-5, int(5 * SR)))
        self.assertTrue(silent["silent"])
        res = R.lyrics_check(lyric_events(self.starts, self.lens), g, silent)
        self.assertEqual((res["verdict"], res["note"]), ("check", "vocals stem silent"))
        self.assertEqual(R.lyrics_check(lyric_events(self.starts, self.lens), g, None)["verdict"], "check")


class VocalOnsets(unittest.TestCase):
    def test_na_without_onsets(self):
        self.assertEqual(R.vocal_onsets_check(events({}), R.Grid(events({})))["verdict"], "n/a")

    def test_pitch_jumps_against_basic_pitch(self):
        g = R.Grid(events({}))
        e = events({})
        # Basic Pitch note starts at beats 0..7; note 3 was snapped 40 ms early (r = +40 ms)
        e["notes"] = {"vocals": {"b": [0, 1, 2, 3, 4, 5, 6, 7], "r": [0, 0, 0, 40, 0, 0, 0, 0],
                                 "len": [1] * 8, "pitch": [60] * 8, "v": [1] * 8}}
        # flux onsets on notes 0-2, a pitch jump on note 3's real time (b + r), and two pitch
        # jumps where Basic Pitch has nothing (beats 4.5 and 6.5)
        e["vocalOnsets"] = {"b": [0, 1, 2, 3 + 0.04 / 0.5, 4.5, 6.5], "v": [1] * 6, "kind": [0, 0, 0, 1, 1, 1]}
        res = R.vocal_onsets_check(e, g)
        self.assertEqual((res["verdict"], res["n"], res["flux"], res["pitch"]), ("info", 6, 3, 3))
        self.assertAlmostEqual(res["pitchNew"], 2 / 3, places=3)
        self.assertEqual(res["pitchNewN"], 2)
        self.assertEqual(res["notesMatched"], 0.5)
        self.assertEqual((res["notesMatchedFlux"], res["notesMatchedPitch"]), (0.375, 0.125))
        del e["notes"]
        self.assertIsNone(R.vocal_onsets_check(e, g)["notesMatched"])


if __name__ == "__main__":
    unittest.main()
