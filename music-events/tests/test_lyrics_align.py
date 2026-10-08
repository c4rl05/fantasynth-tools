"""Forced lyric alignment (scripts/lyrics_align.py). Run from music-events/:

    <venv-main>/Scripts/python.exe -m unittest discover -s tests

Synthetic emissions and features with known answers: the maths is checked without
torch, models or audio.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import lyrics_align as LA  # noqa: E402


def emissions(T, segments, blank=-0.1, other=-8.0):
    """[T, 28] log-probs: blank everywhere except `segments` [(t0, t1, char)]."""
    E = np.full((T, len(LA.ALPHA)), other)
    E[:, 0] = blank
    for t0, t1, ch in segments:
        E[t0:t1, :] = other
        E[t0:t1, LA.AIDX[ch]] = -0.05
    return E - np.logaddexp.reduce(E, axis=1, keepdims=True)


def frames(span):
    return round(span[0] / LA.FRAME), round(span[1] / LA.FRAME)


class Pronunciation(unittest.TestCase):
    def test_table_and_punctuation(self):
        table = {"agi": "ay gee i"}
        self.assertEqual(LA.pron("AGI,", table), ["ay", "gee", "i"])
        self.assertEqual(LA.pron("Don’t,"), ["don't"])
        self.assertEqual(LA.pron("self-upgrade"), ["self", "upgrade"])
        self.assertEqual(LA.pron("(oh)"), ["oh"])
        self.assertEqual(LA.pron("..."), [])

    def test_numbers_are_spelled_as_sung(self):
        self.assertEqual(LA.pron("1998"), ["nineteen", "ninety", "eight"])
        self.assertEqual(LA.pron("24/7"), ["twenty", "four", "seven"])
        self.assertEqual(LA.number_words(2005), ["two", "thousand", "five"])
        self.assertEqual(LA.number_words(1905), ["nineteen", "oh", "five"])
        self.assertEqual(LA.number_words(100), ["one", "hundred"])


class Viterbi(unittest.TestCase):
    def test_words_land_on_their_frames_and_junk_goes_to_star(self):
        E = emissions(60, [(3, 8, "z"), (8, 10, "q"),       # junk before
                           (12, 15, "h"), (15, 18, "i"),     # line 0 "hi"
                           (22, 30, "x"),                    # junk between lines
                           (33, 36, "g"), (36, 40, "o")])    # line 1 "go"
        sp, _ = LA.align(E, [["hi"], ["go"]])
        a, b = frames(sp[(0, 0)][0])
        self.assertEqual(a, 12)
        self.assertIn(b, (17, 18))
        a, b = frames(sp[(1, 0)][0])
        self.assertEqual(a, 33)
        # the mandatory STAR may take the last frame of the last word (known, 20 ms)
        self.assertIn(b, (39, 40))

    def test_numba_and_python_kernels_agree(self):
        rng = np.random.default_rng(3)
        E = np.log(rng.dirichlet(np.ones(len(LA.ALPHA)), size=80))
        Ex = np.concatenate([E, E.max(1, keepdims=True) - 1.5], axis=1)
        tgt, _ = LA.build_targets([["abc"], ["de"]], LA.pron)
        tgt = np.asarray(tgt, np.int64)
        lo, hi = np.zeros(len(tgt), np.int64), np.full(len(tgt), 79, np.int64)
        p1, s1 = LA._viterbi(Ex, tgt, lo, hi)
        p2, s2 = LA._viterbi_py(Ex, tgt, lo, hi)
        np.testing.assert_array_equal(p1, p2)
        self.assertAlmostEqual(s1, s2, places=6)

    def test_subwords_become_spans(self):
        E = emissions(50, [(10, 13, "e"), (13, 16, "m"), (18, 21, "e"), (21, 24, "l")])
        sp, _ = LA.align(E, [["ML"]], pron_fn=lambda t: LA.pron(t, {"ml": "em el"}))
        spans = sp[(0, 0)]
        self.assertEqual(len(spans), 2)
        self.assertEqual(frames(spans[0])[0], 10)
        self.assertEqual(frames(spans[1])[0], 18)

    def test_repeated_letters_need_a_blank(self):
        # "oo": two separate 'o' runs with a blank between them
        E = emissions(40, [(10, 13, "o"), (15, 18, "o")])
        sp, _ = LA.align(E, [["oo"]])
        a, b = frames(sp[(0, 0)][0])
        self.assertEqual(a, 10)
        self.assertGreaterEqual(b, 16)

    def test_unpronounceable_token_is_skipped(self):
        E = emissions(40, [(10, 13, "h"), (13, 16, "i")])
        sp, _ = LA.align(E, [["...", "hi"]])
        self.assertNotIn((0, 0), sp)
        self.assertIn((0, 1), sp)
        words = LA.word_table(sp, [["...", "hi"]])
        self.assertEqual([w["w"] for w in words], ["hi"])

    def test_text_longer_than_audio_is_an_error(self):
        E = emissions(5, [])
        with self.assertRaises(ValueError):
            LA.align(E, [["abcdefgh"]])


def synthetic_features(duration=4.0, voiced=((0.5, 1.5, 60.0), (2.0, 3.5, 60.0)), jump_at=None, jump_to=63.0):
    """5 ms features: silence except `voiced` (t0, t1, midi) notes at -20 dB."""
    hop = LA.FEAT_HOP / LA.FEAT_SR
    n = int(duration / hop)
    t = np.arange(n) * hop
    rms = np.full(n, -80.0)
    f0 = np.full(n, np.nan)
    on = np.full(n, 0.05)
    for t0, t1, m in voiced:
        sel = (t >= t0) & (t < t1)
        rms[sel] = -20.0
        mid = np.full(sel.sum(), m)
        if jump_at is not None:
            mid[t[sel] >= jump_at] = jump_to
        f0[sel] = 440.0 * 2 ** ((mid - 69) / 12)
        on[int(t0 / hop)] = 5.0  # a clear attack at each note start
    return dict(hop=hop, rms_db=rms, sib_ratio=np.full(n, -20.0), onset=on, f0=f0,
                voiced=~np.isnan(f0), vprob=(~np.isnan(f0)).astype(float))


class Signal(unittest.TestCase):
    def test_refine_keeps_order_and_tiles_legato(self):
        f = synthetic_features()
        words = [dict(li=0, ti=0, w="hey", start=0.56, end=1.2, conf=0.9, subs=[(0.56, 1.2)]),
                 dict(li=0, ti=1, w="you", start=2.06, end=3.3, conf=0.9, subs=[(2.06, 3.3)])]
        out = LA.refine(words, f)
        # the start of "hey" moves back to the attack at 0.5 (rest-onset or onset-snap)
        self.assertLess(abs(out[0]["start"] - 0.5), 0.03)
        self.assertIn(out[0]["start_rule"], ("rest-onset", "onset-snap"))
        # "hey" ends where the voice stops (1.5), not at the next word
        self.assertLess(abs(out[0]["end"] - 1.5), 0.03)
        for w in out:
            self.assertLess(w["start"], w["end"])
        self.assertLessEqual(out[0]["end"], out[1]["start"])

    def test_rest_onset_is_capped(self):
        # voice from 0.5 s, but CTC puts the word at 1.3 s: 0.8 s back is too far for a
        # held vowel, so rest-onset must not claim it
        f = synthetic_features(voiced=((0.5, 2.5, 60.0),))
        words = [dict(li=0, ti=0, w="oh", start=1.3, end=2.0, conf=0.9, subs=[(1.3, 2.0)])]
        out = LA.refine(words, f)
        self.assertNotEqual(out[0]["start_rule"], "rest-onset")
        self.assertGreater(out[0]["start"], 1.0)

    def test_word_end_is_capped_when_the_voice_never_drops(self):
        # the voice holds for 20 s after "hey" (a pad, a backing vocal) and the next word is
        # 20 s away: the word must end about END_TAIL after its CTC end, not at the next word
        f = synthetic_features(duration=24.0, voiced=((0.5, 21.0, 60.0),))
        words = [dict(li=0, ti=0, w="hey", start=0.52, end=1.2, conf=0.9, subs=[(0.52, 1.2)]),
                 dict(li=0, ti=1, w="you", start=21.0, end=21.5, conf=0.9, subs=[(21.0, 21.5)])]
        out = LA.refine(words, f)
        self.assertLessEqual(out[0]["end"], 1.2 + LA.END_TAIL + 0.01)

    def test_extras_are_voice_without_words(self):
        f = synthetic_features()
        words = [dict(start=0.5, end=1.5)]
        ex = LA.extras(words, f)
        self.assertEqual(len(ex), 1)
        a, b = ex[0]
        self.assertLess(abs(a - 2.0), 0.06)
        self.assertLess(abs(b - 3.5), 0.06)

    def test_pitch_jump_without_attack_is_an_onset(self):
        f = synthetic_features(voiced=((0.5, 3.0, 60.0),), jump_at=1.8)
        ev = LA.vocal_onsets(f)
        pitch = [t for t, s, k in ev if k == 1]
        self.assertTrue(any(abs(t - 1.8) < 0.06 for t in pitch), ev)
        flux = [t for t, s, k in ev if k == 0]
        self.assertTrue(any(abs(t - 0.5) < 0.03 for t in flux), ev)

    def test_confidence_uses_agreement(self):
        w = dict(li=0, ti=0, start=1.0, ctc_start=1.0, conf=0.5)
        alts = {"a": [dict(li=0, ti=0, start=1.02)], "b": [dict(li=0, ti=0, start=1.5)]}
        LA.confidence([w], alts)
        self.assertEqual(w["agree"], 0.5)
        self.assertAlmostEqual(w["conf_final"], 0.4 + 0.2 + 0.2, places=2)


if __name__ == "__main__":
    unittest.main()
